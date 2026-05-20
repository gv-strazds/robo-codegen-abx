import logging
import random
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from typing_extensions import override

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from multi_pick_strategy import ColorSortStackBase, build_bin_geometry_check, compute_stacking_map

logger = logging.getLogger(__name__)


class ColorSortStackStrategy(ColorSortStackBase):
    """Sort picks by color into separate boxes, building round-robin stacks.

    Combines color-based routing with dynamic stacking: each placed cube becomes
    the target for the next cube in that stack (like SingleStackStrategy, but with
    multiple color-routed stacks).

    Args:
        pick_objs: List of pick objects.
        target_objs: List of target objects (bottom-layer markers, ordered by
            sort_color then stack position).
        sort_colors: Colors to sort (e.g. ["red", "green", "blue"]).
        stacks_per_box: Number of stack positions per box (e.g. 4 for 2x2).
        skip_colors: Colors to ignore (e.g. ["yellow"]).
        base_check_fn: Spatial check for bottom-layer placement.
    """

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        sort_colors: List[str],
        stacks_per_box: int = 4,
        skip_colors: Optional[List[str]] = None,
        base_check_fn: Optional[Callable] = None,
        stacking_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__(
            pick_objs=pick_objs,
            target_objs=target_objs,
            sort_colors=sort_colors,
            skip_colors=skip_colors,
            base_check_fn=base_check_fn,
            stacking_map=stacking_map,
        )
        self._stacks_per_box = stacks_per_box

    def _find_permanently_blocked(self) -> set:
        """Find pick indices that can never be reached due to skip-color cubes above.

        A cube is permanently blocked if any cube directly above it in the
        stacking_map is either a skip-color cube or is itself permanently blocked.
        Uses fixed-point iteration.
        """
        if not self._stacking_map:
            return set()

        # Build name → index lookup
        name_to_idx = {obj.name: i for i, obj in enumerate(self._pick_objs)}

        # Start with all skip-color cubes as "never completed"
        never_completed: set = set()
        for i, pick in enumerate(self._pick_objs):
            c = self._classify_pick(pick)
            if c is None:  # skip-color or unrecognized
                never_completed.add(pick.name)

        # Fixed-point: propagate — any cube with a never-completed cube above it
        # is also permanently blocked (and thus never completed)
        changed = True
        while changed:
            changed = False
            for name in list(self._stacking_map.keys()):
                if name in never_completed:
                    continue
                above = self._stacking_map.get(name, [])
                if any(a in never_completed for a in above):
                    never_completed.add(name)
                    changed = True

        # Return indices of sort-color cubes that are permanently blocked
        blocked_indices: set = set()
        for name in never_completed:
            idx = name_to_idx.get(name)
            if idx is not None:
                pick = self._pick_objs[idx]
                c = self._classify_pick(pick)
                if c is not None:  # it's a sort-color cube, but blocked
                    blocked_indices.add(idx)

        if blocked_indices:
            logger.info(
                "Permanently blocked cubes (under yellow): %d of %d sort-color cubes",
                len(blocked_indices), sum(len(v) for v in
                    {c: [i for i, p in enumerate(self._pick_objs)
                         if self._classify_pick(p) == c]
                     for c in self._sort_colors}.values()),
            )

        return blocked_indices

    @override
    def pair_picks_with_targets(self) -> Iterator[Tuple[int, Optional[int]]]:
        """Pair picks to targets: color routing + round-robin stacking."""
        self._truncate_target_objs(self._base_target_count)

        # Find cubes permanently blocked by yellow cubes above
        blocked = self._find_permanently_blocked()

        # Classify picks by sort color, excluding permanently blocked
        color_picks: Dict[str, List[int]] = {c: [] for c in self._sort_colors}
        for i, pick in enumerate(self._pick_objs):
            if i in blocked:
                continue
            c = self._classify_pick(pick)
            if c is not None:
                color_picks[c].append(i)

        # Assign round-robin to stack positions within each color's box
        stacks_idx: Dict[str, List[List[int]]] = {}
        for c in self._sort_colors:
            stacks_idx[c] = [[] for _ in range(self._stacks_per_box)]
            for i, pick_idx in enumerate(color_picks[c]):
                sp = i % self._stacks_per_box
                stacks_idx[c][sp].append(pick_idx)

        self._color_stacks = {
            c: [[self._pick_objs[pi].name for pi in stack] for stack in stacks_idx[c]]
            for c in self._sort_colors
        }

        # Compute max layers across all stacks
        self._max_layers = 0
        for c in self._sort_colors:
            for sp_list in stacks_idx[c]:
                self._max_layers = max(self._max_layers, len(sp_list))

        # Base markers ordered: sort_colors[0] stacks [0..N-1], sort_colors[1], ...
        paired: set = set()

        # Layer 0: each pick -> base marker
        for c_idx, c in enumerate(self._sort_colors):
            for sp in range(self._stacks_per_box):
                if stacks_idx[c][sp]:
                    pick_idx = stacks_idx[c][sp][0]
                    target_idx = c_idx * self._stacks_per_box + sp
                    paired.add(pick_idx)
                    yield (pick_idx, target_idx)

        # Upper layers: each pick -> the previously placed cube as target
        for layer in range(1, self._max_layers):
            for c in self._sort_colors:
                for sp in range(self._stacks_per_box):
                    if layer < len(stacks_idx[c][sp]):
                        pick_idx = stacks_idx[c][sp][layer]
                        prev_pick = self._pick_objs[stacks_idx[c][sp][layer - 1]]
                        tgt_start = self._extend_target_objs([prev_pick])
                        paired.add(pick_idx)
                        yield (pick_idx, tgt_start)

        # Unpaired picks (yellow + excess)
        for i in range(len(self._pick_objs)):
            if i not in paired:
                yield (i, None)

    def initialize_pairings(self) -> None:
        """Set picking order: all layer-0 placements first, then layer-1, etc."""
        super().initialize_pairings()

        ordered = []
        for layer in range(self._max_layers):
            for c in self._sort_colors:
                for sp in range(self._stacks_per_box):
                    if layer < len(self._color_stacks[c][sp]):
                        ordered.append(self._color_stacks[c][sp][layer])

        self._picking_order_item_names = ordered
        self._current_pick_index = 0
        self._build_layer_info()

    def get_recommended_ee_height(self, prim_geometry=None) -> Optional[float]:
        """Compute transport height to clear the tallest destination stack."""
        stacks = (
            self._color_stacks[c][sp]
            for c in self._sort_colors
            for sp in range(self._stacks_per_box)
            if sp < len(self._color_stacks.get(c, []))
        )
        return self._stack_clearance_height(stacks, prim_geometry)


class TableTaskSortAndStack(UR10MultiPickPlaceTask):
    """Sort cubes from a 6x5x3 stacked grid into 3 color-coded boxes on the cart.

    - Sources: 90 cubes in a 6x5 grid, 3 layers high, on the dropzone floor.
      Colors randomly assigned from red/green/blue/yellow.
    - Targets: 3 color-coded open boxes on the cart (red, green, blue), each
      with 4 bottom-layer markers (2x2 grid). Upper targets are placed cubes.
    - Strategy: ColorSortStackStrategy — routes by color to matching box,
      stacks round-robin at 4 positions per box. Yellow cubes skipped.
    """

    GRID_COLS = 6
    GRID_ROWS = 5
    NUM_LAYERS = 3
    COLOR_PALETTE = ["red", "green", "blue", "yellow"]
    SORT_COLORS = ["red", "green", "blue"]
    STACKS_PER_BOX = 4  # 2x2 grid of stacks per box

    def __init__(
        self,
        task_name: str = "table_task_sort_and_stack",
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import ItemSpec, resolve_count
        from table_setup import (
            CART_SURFACE_CENTER,
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            setup_two_tables,
            spawn_open_box,
        )
        from task_spec import TaskSpec

        stage_units = get_stage_units()

        # --- Cube specs ---
        cube_size = 0.0515 / stage_units

        # --- Cart / Box layout ---
        cart = CART_SURFACE_CENTER
        cx, cy, cart_z = cart[0], cart[1], cart[2]

        box_inner_size = np.array([0.15, 0.15]) / stage_units
        box_wall_height = 0.06 / stage_units
        box_wall = 0.01 / stage_units
        box_base_thickness = 0.01 / stage_units
        box_floor_z = cart_z + box_base_thickness + 0.001

        # 3 boxes in a row along Y on the cart
        box_y_offsets = {"red": -0.22, "green": 0.0, "blue": 0.22}
        box_colors_rgb = {
            "red": np.array([0.8, 0.3, 0.3]),
            "green": np.array([0.3, 0.8, 0.3]),
            "blue": np.array([0.3, 0.3, 0.8]),
        }

        box_specs = []
        for color in self.SORT_COLORS:
            bx, by = cx, cy + box_y_offsets[color]
            box_specs.append({
                "name": f"{color}_box",
                "center": np.array([bx, by, cart_z + box_wall_height / 2]),
                "center_xy": np.array([bx, by]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                # Verification height: much taller than visual walls (0.06m) to
                # accommodate stacks that grow well above the box walls.
                "height": 0.50 / stage_units,
                "match_labels": {"color": color},
                "z_tol": 0.03,
            })

        # --- Pick Strategy: 6x5x3 grid on dropzone, random colors, top-down ---
        dropzone_center = DROPZONE_CENTER_POINT
        grid_spacing = 0.06 / stage_units
        base_z = DROPZONE_Z + cube_size / 2  # center Z of bottom-layer cube

        class SortAndStackPickGenerator:
            """Generate cubes in a grid, stacked N layers, top-down order."""
            def __init__(self, center, cols, rows, num_layers, cube_size,
                         grid_spacing, base_z, colors):
                self.center = center
                self.cols = cols
                self.rows = rows
                self.num_layers = num_layers
                self.cube_size = cube_size
                self.grid_spacing = grid_spacing
                self.base_z = base_z
                self.colors = colors

            def generate(self, count_range=None, seed=None):
                rng = random.Random(seed)
                items = []
                # Top-down: layer N-1 first, then N-2, ..., 0
                for layer in range(self.num_layers - 1, -1, -1):
                    for row in range(self.rows):
                        for col in range(self.cols):
                            x = self.center[0] + (col - (self.cols - 1) / 2) * self.grid_spacing
                            y = self.center[1] + (row - (self.rows - 1) / 2) * self.grid_spacing
                            z = self.base_z + layer * self.cube_size
                            items.append(ItemSpec(
                                asset_type="cube",
                                position=np.array([x, y, z]),
                                color=rng.choice(self.colors),
                                scale=np.array([self.cube_size] * 3),
                            ))

                count = resolve_count(count_range, capacity=len(items), seed=seed)
                if count is not None and count < len(items):
                    items = items[:count]
                return items

        pick_strategy = SortAndStackPickGenerator(
            center=dropzone_center,
            cols=self.GRID_COLS,
            rows=self.GRID_ROWS,
            num_layers=self.NUM_LAYERS,
            cube_size=cube_size,
            grid_spacing=grid_spacing,
            base_z=base_z,
            colors=self.COLOR_PALETTE,
        )

        # --- Target Strategy: 4 bottom-layer markers per box (2x2 grid) ---
        marker_scale = np.array([0.04, 0.04, 0.001]) / stage_units
        stack_spacing = 0.055 / stage_units  # center-to-center between stacks

        class BoxFloorMarkerGenerator:
            """Generate hidden markers on the floor of each box (2x2 grid)."""
            def __init__(self, box_specs, floor_z, marker_scale, stack_spacing,
                         stacks_per_box):
                self.box_specs = box_specs
                self.floor_z = floor_z
                self.marker_scale = marker_scale
                self.stack_spacing = stack_spacing
                self.stacks_per_box = stacks_per_box

            def generate(self, count_range=None, seed=None):
                targets = []
                grid_side = int(self.stacks_per_box ** 0.5)  # 2 for 4 stacks
                for bspec in self.box_specs:
                    for row in range(grid_side):
                        for col in range(grid_side):
                            ox = (col - (grid_side - 1) / 2) * self.stack_spacing
                            oy = (row - (grid_side - 1) / 2) * self.stack_spacing
                            targets.append(ItemSpec(
                                asset_type="marker",
                                position=np.array([
                                    bspec["center_xy"][0] + ox,
                                    bspec["center_xy"][1] + oy,
                                    self.floor_z,
                                ]),
                                scale=self.marker_scale,
                                hidden=True,
                            ))
                return targets

        target_strategy = BoxFloorMarkerGenerator(
            box_specs=box_specs,
            floor_z=box_floor_z,
            marker_scale=marker_scale,
            stack_spacing=stack_spacing,
            stacks_per_box=self.STACKS_PER_BOX,
        )

        # --- Strategy factory ---
        sort_colors = list(self.SORT_COLORS)
        stacks_per_box = self.STACKS_PER_BOX

        # Build combined base check across all boxes
        box_checks = []
        for bs in box_specs:
            box_checks.append(build_bin_geometry_check({
                "center_xy": bs["center_xy"],
                "inner_size": bs["inner_size"],
                "floor_z": bs["floor_z"],
                "height": 0.50,  # generous for growing stacks
                "z_tol": bs.get("z_tol", 0.03),
            }))

        def _combined_base_check(pick_obj, target_obj=None,
                                 bb_cache=None, obj_scale=None):
            return any(
                chk(pick_obj, target_obj, bb_cache=bb_cache, obj_scale=obj_scale)
                for chk in box_checks
            )

        def _strategy_factory(picks, targets):
            stacking_map = compute_stacking_map(picks)
            return ColorSortStackStrategy(
                pick_objs=picks,
                target_objs=targets,
                sort_colors=sort_colors,
                stacks_per_box=stacks_per_box,
                skip_colors=["yellow"],
                base_check_fn=_combined_base_check,
                stacking_map=stacking_map,
            )

        # --- Workspace setup ---
        def _workspace_setup(scene, assets_root):
            setup_two_tables(scene, assets_root, standard_objs=False, add_bin=False)
            for bspec in box_specs:
                spawn_open_box(
                    scene, name=bspec["name"], center=bspec["center"],
                    inner_size=box_inner_size, wall_height=box_wall_height,
                    wall_thickness=box_wall, base_thickness=box_base_thickness,
                    color=box_colors_rgb[bspec["match_labels"]["color"]],
                )

        # --- TaskSpec ---
        pick_count = kwargs.pop("pick_count", None)
        kwargs.pop("target_count", None)
        total_cubes = self.GRID_COLS * self.GRID_ROWS * self.NUM_LAYERS

        if pick_count is None:
            pick_count = total_cubes

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick red, green, and blue cubes from a 6x5x3 stacked grid "
                "and sort them into matching color-coded boxes on the cart, "
                "stacking cubes on top of previously placed cubes."
            ),
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            virtual_target_generation_strategy=target_strategy,
            create_strategy=_strategy_factory,
            setup_workspace=_workspace_setup,
            box_verification_info={"box_specs": box_specs},
            containment_check=True,
            stacking_enabled=True,
            ee_height_for_move=0.35 / stage_units,
            scenario={
                "source": "dropzone_grid",
                "destination": "boxes_on_cart",
                "workspace": "two_tables_custom_boxes",
            },
            pick_description={
                "asset_types": ["cube"],
                "count": total_cubes,
                "arrangement": (
                    f"{self.GRID_COLS}x{self.GRID_ROWS} grid on dropzone, "
                    f"{self.NUM_LAYERS} layers high"
                ),
                "colors": "RandomChoice(['red', 'green', 'blue', 'yellow'])",
                "generation_order": "top-down (layer 2 → 1 → 0)",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "2x2 bottom-layer markers per box",
                "count": self.STACKS_PER_BOX * len(self.SORT_COLORS),
                "virtual": True,
                "containers": {
                    "count": 3,
                    "layout": "3 color-coded boxes in a row on cart",
                    "capacity_per_box": f"{self.STACKS_PER_BOX} stacks",
                },
            },
            strategy_description={
                "class": "ColorSortStackStrategy",
                "pairing": "color_sort_stack",
                "details": (
                    f"Routes R/G/B cubes to matching boxes. "
                    f"Within each box, {self.STACKS_PER_BOX} stacks filled "
                    f"round-robin. Upper targets are placed cubes. "
                    f"Yellow cubes skipped."
                ),
            },
            verification_description={
                "containment_check": True,
                "match_labels": "color-based: red_box accepts only red cubes, etc.",
            },
            rationale={
                "create_strategy": (
                    "Custom ColorSortStackStrategy combines color routing with "
                    "dynamic stacking — each placed cube becomes the target for "
                    "the next cube in that stack. Yellow cubes are distractors."
                ),
                "containment_check": "Items placed inside boxes — box-boundary verification confirms correct placement",
                "virtual_target_generation_strategy": "Only bottom-layer markers per box — upper targets are placed cubes",
                "stacking_enabled": (
                    "Source cubes are stacked 3 layers high — stacking_map enforces "
                    "top-down pick order and permanently blocked cubes (under yellow) "
                    "are excluded from pairings"
                ),
                "ee_height_for_move": "Base 0.35m + dynamic increase via strategy to clear growing stacks",
            },
        )

        super().__init__(task_spec=spec, offset=offset, **kwargs)
