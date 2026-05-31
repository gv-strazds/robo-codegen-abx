import logging
from typing import Dict, Optional

import numpy as np

from multi_pick_strategy import ColorMatchStrategy
from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class _CapacityStub:
    """Minimal stub exposing the `.capacity` attribute the spatial-trigger
    scheduler reads from `primary_generator.position_generator.capacity`."""

    def __init__(self, n: int) -> None:
        self.capacity = n


class _FixedSpecsGenerator:
    """Pre-baked ItemSpec list wrapper that honors the scheduler API contract.

    Returns the first `count_range` items (or all items if count_range is
    None / not an int). Exposes `.position_generator.capacity` so
    `SpatialTriggeredItemScheduler` can resolve totals.
    """

    def __init__(self, items: list) -> None:
        self._items = list(items)
        self.position_generator = _CapacityStub(len(self._items))

    def generate(self, count_range=None, seed=None):
        if isinstance(count_range, int):
            n = max(0, min(count_range, len(self._items)))
        else:
            n = len(self._items)
        return list(self._items[:n])


class ColorMatchConveyorProximityStrategy(ColorMatchStrategy):
    """ColorMatchStrategy variant: pick the cube closest to the conveyor end
    next, regardless of color; assign first-unused matching-color target in
    target-list order.

    Pick selection ignores the spawn-order picking list and instead walks
    uncompleted picks sorted by current world Y (smaller = closer to -Y belt
    end). Once selected, the choice is latched until ``mark_pick_complete``
    or ``advance_pick_index`` clears it, so multi-tick pick phases see a
    stable target.

    Target assignment runs JIT in ``get_placing_target_name``: the first
    unused matching-color target in target-list order. Combined with a
    target list pre-sorted +Y -> -Y within each color group, this fills
    each color row from +Y to -Y in pick-completion order.
    """

    AXIS_IDX = 1   # conveyor moves along Y
    SIGN = -1      # belt edge is in -Y direction; smaller Y = more urgent

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        color_palette,
        has_color_fn=None,
    ) -> None:
        super().__init__(
            pick_objs, target_objs,
            color_palette=color_palette,
            has_color_fn=has_color_fn,
        )
        self._active_pick_name: Optional[str] = None
        self._latched_target_by_pick: Dict[str, str] = {}

    # ---- pick selection: dynamic by conveyor-end proximity ----

    def _pick_color(self, pick_name: str) -> Optional[str]:
        pick = self._pick_objs_by_name.get(pick_name)
        if pick is None:
            return None
        for c in self._color_palette:
            if self._has_color(pick, c):
                return c
        return None

    def _proximity_key(self, pick_obj) -> float:
        try:
            pos, _ = pick_obj.get_world_pose()
        except Exception:
            return float("inf")
        return -self.SIGN * float(pos[self.AXIS_IDX])

    def _pick_is_candidate(self, name: str) -> bool:
        return (name not in self._completed_picks
                and name not in self._permanently_unreachable_picks
                and name not in self._deferred_picks
                and self._has_target(name))

    def _select_next_pick(self) -> Optional[str]:
        best_name: Optional[str] = None
        best_key: Optional[float] = None
        for name in self._picking_order_item_names:
            if not self._pick_is_candidate(name):
                continue
            obj = self._pick_objs_by_name.get(name)
            if obj is None:
                continue
            key = self._proximity_key(obj)
            if best_key is None or key < best_key:
                best_key = key
                best_name = name
        return best_name

    def get_current_pick_name(self) -> Optional[str]:
        if self._targets_exhausted:
            return None
        active = self._active_pick_name
        if active is not None and self._pick_is_candidate(active):
            return active
        self._active_pick_name = self._select_next_pick()
        return self._active_pick_name

    def advance_pick_index(self) -> Optional[str]:
        # Only consume the cursor slot when there's actually a candidate to
        # advance to; otherwise the cursor outruns the picking-order list
        # during "waiting for more items" ticks, which makes
        # ``all_picks_done`` false-True before late replenishment items have
        # been processed (see UR10 controller ``is_done``).
        self._active_pick_name = None
        next_name = self.get_current_pick_name()
        if next_name is not None:
            self._current_pick_index += 1
        return next_name

    @property
    def all_picks_done(self) -> bool:
        # Override the base class's index-based check with a semantic one:
        # we're done when every name in the picking order is in the
        # completed-picks set. Picks marked permanently unreachable are
        # treated as "done" for completion purposes (they will never be
        # placed). This stays robust against the cursor and against
        # incremental additions to the picking-order list.
        names = self._picking_order_item_names
        if not names:
            return False
        for name in names:
            if name in self._completed_picks:
                continue
            if name in self._permanently_unreachable_picks:
                continue
            return False
        return True

    # ---- target assignment: JIT, first-unused matching-color in list order ----

    def _first_unused_matching_target(self, pick_name: str) -> Optional[str]:
        color = self._pick_color(pick_name)
        if color is None:
            return None
        occupied = self._currently_occupied_target_names()
        latched_by_others = {
            t for other, t in self._latched_target_by_pick.items()
            if other != pick_name and t is not None
        }
        for tgt in self._target_objs:
            tgt_name = tgt.name
            if not self._has_color(tgt, color):
                continue
            if tgt_name in self._permanently_unreachable_targets:
                continue
            if tgt_name in occupied:
                continue
            if tgt_name in latched_by_others:
                continue
            if not self.is_target_reachable(tgt_name):
                continue
            return tgt_name
        return None

    def _has_target(self, pick_name: str) -> bool:
        return self._first_unused_matching_target(pick_name) is not None

    def get_placing_target_name(self, pick_name: str) -> Optional[str]:
        if pick_name in self._completed_picks:
            return self._pairings_by_pick_name.get(pick_name)
        latched = self._latched_target_by_pick.get(pick_name)
        if latched is not None:
            occupied = self._currently_occupied_target_names()
            still_valid = (
                latched not in self._permanently_unreachable_targets
                and latched not in occupied
                and self.is_target_reachable(latched)
            )
            if still_valid:
                self._pairings_by_pick_name[pick_name] = latched
                return latched
            del self._latched_target_by_pick[pick_name]
        new_tgt = self._first_unused_matching_target(pick_name)
        self._pairings_by_pick_name[pick_name] = new_tgt
        return new_tgt

    # ---- latch hooks (no-ops in the default 9-phase tree, but harmless) ----

    def latch_current_target(self, pick_name: str) -> None:
        tgt = self._pairings_by_pick_name.get(pick_name)
        if tgt is None:
            tgt = self._first_unused_matching_target(pick_name)
        if tgt is None:
            return
        self._latched_target_by_pick[pick_name] = tgt
        self._pairings_by_pick_name[pick_name] = tgt

    def clear_target_latch(self, pick_name: str) -> None:
        self._latched_target_by_pick.pop(pick_name, None)

    def clear_all_target_latches(self) -> None:
        self._latched_target_by_pick.clear()

    def latch_current_pick(self, pick_name: str) -> None:
        self._active_pick_name = pick_name

    def clear_pick_latch(self, pick_name: Optional[str] = None) -> None:
        if pick_name is None or self._active_pick_name == pick_name:
            self._active_pick_name = None

    def clear_all_pick_latches(self) -> None:
        self._active_pick_name = None

    # ---- incremental spawning: extend without clobbering JIT state ----

    def add_incremental_picks(self, new_objs: list) -> None:
        self._extend_pick_objs(new_objs)
        for obj in new_objs:
            if obj.name not in self._picking_order_item_names:
                self._picking_order_item_names.append(obj.name)
        # New picks may unblock the BT if it was waiting for items.
        self._targets_exhausted = False


class TableTaskConveyorColorRows(UR10MultiPickPlaceTask):
    """Pick colored cubes from the slowly-moving conveyor and place them onto
    matching-color rectangular markers arranged in three color-coded rows on
    the cart, filling each row from +Y to -Y."""

    DEFAULT_TASK_NAME = "table_task_conveyor_color_rows"

    COLORS = ("red", "green", "blue")
    MARKERS_PER_ROW = 5
    INITIAL_COUNT = 5
    MIN_PER_COLOR = 3
    MAX_PER_COLOR = 5

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME

        from isaacsim.core.utils.stage import get_stage_units
        from env_config_values import (
            CART_SURFACE_CENTER,
            CONVEYOR_SURFACE_CENTER,
            CONVEYOR_SURFACE_TOP_Z,
            DEFAULT_CONVEYOR_SPEED,
        )
        from item_generation import (
            ItemSpec,
            SpatialTriggerConfig,
            SpatialTriggerRegion,
        )
        from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
        from table_setup import setup_two_tables
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        # Halved-default conveyor speed: the robot's pick-place cycle in full
        # sim cannot keep up with -0.015 m/s, leading to cubes drifting off
        # the belt before they are reached. Half-speed gives ~2x more cycle
        # margin per cube.
        conveyor_speed = DEFAULT_CONVEYOR_SPEED * 0.5

        # --- Seed-deterministic color allocation ---
        seed = kwargs.get("seed", None)
        rng = np.random.default_rng(seed)
        per_color_counts = {
            c: int(rng.integers(self.MIN_PER_COLOR, self.MAX_PER_COLOR + 1))
            for c in self.COLORS
        }
        total_picks = sum(per_color_counts.values())
        color_list = [c for c, n in per_color_counts.items() for _ in range(n)]
        rng.shuffle(color_list)
        initial_colors = color_list[: self.INITIAL_COUNT]
        replenish_colors = color_list[self.INITIAL_COUNT :]

        # --- Cube geometry ---
        cube_edge = 0.0515
        cube_scale = np.array([cube_edge, cube_edge, cube_edge]) / stage_units
        cube_half = cube_edge / 2

        # --- Pick spawn geometry (inside UR10 working radius) ---
        spawn_x = float(CONVEYOR_SURFACE_CENTER[0])
        feed_y = 0.85          # +Y feed point, ~1.0 m from robot base in XY
        row_spacing_y = 0.10   # initial row pitch
        jitter_xy = 0.01       # ±1 cm jitter on both axes
        spawn_z = CONVEYOR_SURFACE_TOP_Z + cube_half  # contact-on-spawn

        def _jitter():
            return float(rng.uniform(-jitter_xy, jitter_xy))

        initial_items = []
        for i, color in enumerate(initial_colors):
            y = feed_y - i * row_spacing_y + _jitter()
            x = spawn_x + _jitter()
            initial_items.append(
                ItemSpec(
                    asset_type="cube",
                    position=np.array([x, y, spawn_z]),
                    color=color,
                    scale=cube_scale,
                )
            )

        replenish_items = []
        for color in replenish_colors:
            y = feed_y + _jitter()
            x = spawn_x + _jitter()
            replenish_items.append(
                ItemSpec(
                    asset_type="cube",
                    position=np.array([x, y, spawn_z]),
                    color=color,
                    scale=cube_scale,
                )
            )

        primary_gen = _FixedSpecsGenerator(initial_items)
        replenishment_gen = _FixedSpecsGenerator(replenish_items)

        # --- Spatial-trigger config: spawn next at +Y feed point when region empties ---
        feed_region_x_half = 0.10
        feed_region_y_half = 0.06
        pick_trigger = SpatialTriggerConfig(
            region=SpatialTriggerRegion(
                min_x=spawn_x - feed_region_x_half,
                max_x=spawn_x + feed_region_x_half,
                min_y=feed_y - feed_region_y_half,
                max_y=feed_y + feed_region_y_half,
            ),
            initial_count=self.INITIAL_COUNT,
            items_per_batch=1,
            invert=True,
            trigger_delay=2.0,
            replenishment_generation_strategy=replenishment_gen,
        )

        # --- Targets: 3 color rows of 5 rect markers on the cart, +X-shifted ---
        rect_thickness = 0.002
        marker_scale = np.array([0.05, 0.04, rect_thickness]) / stage_units
        cart_x = float(CART_SURFACE_CENTER[0])
        cart_y = float(CART_SURFACE_CENTER[1])
        cart_z = float(CART_SURFACE_CENTER[2])
        marker_z = cart_z + 0.001 + rect_thickness / 2

        row_x_offsets = (0.10, 0.20, 0.30)  # +X-most row closest to robot
        col_y_spacing = 0.14
        col_y_center = cart_y - 0.10  # shift toward robot's "easy" Y band

        # Order: all reds (+Y→-Y), then all greens (+Y→-Y), then all blues (+Y→-Y)
        target_items = []
        for color, x_off in zip(self.COLORS, row_x_offsets):
            row_x = cart_x + x_off
            for col_idx in range(self.MARKERS_PER_ROW):
                # +Y→-Y order: col_idx 0 → +Y-most, col_idx 4 → -Y-most
                y = col_y_center + (self.MARKERS_PER_ROW - 1) / 2 * col_y_spacing \
                    - col_idx * col_y_spacing
                target_items.append(
                    ItemSpec(
                        asset_type="rect",
                        position=np.array([row_x, y, marker_z]),
                        color=color,
                        scale=marker_scale,
                    )
                )

        target_strategy = _FixedSpecsGenerator(target_items)

        # --- TaskSpec ---
        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick colored cubes (red, green, blue) arriving on the slowly-moving "
                "conveyor and place each onto a matching-color rectangular marker on "
                "the cart. The cart has three rows of 5 markers, one row per color, "
                "filled from +Y to -Y. Initially 5 cubes spawn in a row; new cubes "
                "spawn one at a time at the +Y feed point as the row drifts toward "
                "the robot."
            ),
            pick_generation_strategy=primary_gen,
            pick_count=total_picks,
            pick_spatial_trigger_config=pick_trigger,
            target_generation_strategy=target_strategy,
            target_count=len(target_items),
            conveyor_speed=conveyor_speed,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root,
                standard_objs=False,
                add_bin=False,
                conveyor_speed=conveyor_speed,
            ),
            seed=seed,
            scenario={
                "source": "conveyor",
                "destination": "cart",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cube"],
                "count": (
                    f"{total_picks} "
                    f"(red={per_color_counts['red']}, "
                    f"green={per_color_counts['green']}, "
                    f"blue={per_color_counts['blue']})"
                ),
                "arrangement": (
                    "5 initial cubes in a row along Y inside reach (Y ∈ "
                    "[0.45, 0.85]) with X/Y jitter; replenished one at a time "
                    "at the +Y feed point (Y ≈ 0.85)"
                ),
                "colors": "red, green, blue (independent 3-5 each, randomly interleaved)",
                "spawning": (
                    "SpatialTriggerConfig: initial_count=5, items_per_batch=1, "
                    "invert=True, trigger_delay=2.0s"
                ),
            },
            target_description={
                "type": "visible_rect_markers",
                "arrangement": (
                    "3 color rows of 5 markers each on the cart (+X-shifted toward "
                    "the robot); within each row, markers ordered +Y→-Y"
                ),
                "count": len(target_items),
                "colors": "row 0 red (+X-most), row 1 green, row 2 blue",
            },
            verification_description={
                "spatial_check": (
                    "default is_on_top via ColorMatchStrategy.is_pick_successfully_placed "
                    "(color match enforced)"
                ),
            },
            rationale={
                "create_strategy": (
                    "Picks must be processed in conveyor-end-proximity order "
                    "(closest to -Y belt edge first) regardless of color, while "
                    "targets must be color-matched and filled +Y→-Y within each "
                    "row. ColorMatchConveyorProximityStrategy selects picks JIT "
                    "by world Y and assigns the first-unused matching-color "
                    "target in target-list order (pre-sorted +Y→-Y per color)."
                ),
                "pick_count": (
                    "Pre-computed sum of seeded per-color 3-5 draws; "
                    "SpatialTriggeredItemScheduler needs the exact total to size "
                    "its initial + replenishment queues."
                ),
                "conveyor_speed": (
                    "Half of DEFAULT_CONVEYOR_SPEED (-0.0075 m/s) per the user's "
                    "'too fast' feedback at full default speed; falloff verification "
                    "auto-enables."
                ),
                "pick_spatial_trigger_config": (
                    "initial_count=5 spawns the user-requested initial row; "
                    "invert=True over a small region at the +Y feed point fires "
                    "when the previous spawn has drifted clear; trigger_delay=2 s "
                    "provides extra spacing so successive cubes never overlap."
                ),
            },
            implementation=TaskImplementationSpec(
                create_strategy=lambda picks, targets: ColorMatchConveyorProximityStrategy(
                    picks, targets, color_palette=list(self.COLORS),
                ),
                # Use the cortex-style BT so CheckPickReachable + IsPickReachableGuard
                # permanently flag cubes that drop off the belt edge; the strategy's
                # proximity scan excludes permanently-unreachable picks, so fallen
                # cubes are not re-selected.
                tree_factory=make_cortex_task_controller_tree,
                # Z-floor for pick reachability: cubes whose world Z falls below
                # this are flagged permanently unreachable by CheckPickReachable.
                pick_min_reachable_z=CONVEYOR_SURFACE_TOP_Z - 0.10,
                strategy_description={
                    "class": "ColorMatchConveyorProximityStrategy",
                    "pairing": "proximity_pick + color_match_target",
                    "details": (
                        "Pick selection: uncompleted cube with smallest world Y "
                        "(closest to -Y belt end). Target: JIT first-unused "
                        "matching-color target in target-list order. "
                        "color_palette=['red','green','blue']."
                    ),
                },
            ),
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
