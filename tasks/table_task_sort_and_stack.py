import logging
import random
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from typing_extensions import override

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from multi_pick_strategy import ColorSortStackBase, build_bin_geometry_check, compute_stacking_map

logger = logging.getLogger(__name__)


class ColorSortRelocateStackStrategy(ColorSortStackBase):
    """Per-color-stack-count variant with no skip colors.

    All colors are sorted to their own stacks. Each color can have a
    different number of stacks (e.g., 4 stacks per R/G/B box, 6 stacks
    for relocated yellow cubes on the dropzone).

    Args:
        pick_objs: List of pick objects.
        target_objs: List of target objects (base markers, ordered by color
            then stack position).
        sort_colors: All colors to sort (e.g. ["red", "green", "blue", "yellow"]).
        stacks_per_color: Dict mapping each color to its number of stacks.
        base_check_fn: Spatial check for bottom-layer placement.
        stacking_map: Source stacking relationships.
    """

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        sort_colors: List[str],
        stacks_per_color: Dict[str, int],
        base_check_fn: Optional[Callable] = None,
        stacking_map: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        super().__init__(
            pick_objs=pick_objs,
            target_objs=target_objs,
            sort_colors=sort_colors,
            skip_colors=[],
            base_check_fn=base_check_fn,
            stacking_map=stacking_map,
        )
        self._stacks_per_color = dict(stacks_per_color)

        # Cumulative target index offsets per color
        self._color_target_offset: Dict[str, int] = {}
        offset = 0
        for c in sort_colors:
            self._color_target_offset[c] = offset
            offset += stacks_per_color[c]

    @override
    def pair_picks_with_targets(self) -> Iterator[Tuple[int, Optional[int]]]:
        """Pair picks to targets with per-color stack counts."""
        self._truncate_target_objs(self._base_target_count)

        # All colors are sortable — no permanently blocked cubes
        color_picks: Dict[str, List[int]] = {c: [] for c in self._sort_colors}
        for i, pick in enumerate(self._pick_objs):
            c = self._classify_pick(pick)
            if c is not None:
                color_picks[c].append(i)

        # Round-robin with per-color stack counts.  Stacks store pick names
        # for name-based bookkeeping; we still emit (pick_idx, tgt_idx)
        # pairs because the base ``pair_picks_with_targets`` contract is
        # index-based.
        stacks_idx: Dict[str, List[List[int]]] = {}
        for c in self._sort_colors:
            n = self._stacks_per_color[c]
            stacks_idx[c] = [[] for _ in range(n)]
            for i, pick_idx in enumerate(color_picks[c]):
                stacks_idx[c][i % n].append(pick_idx)

        self._color_stacks = {
            c: [[self._pick_objs[pi].name for pi in stack] for stack in stacks_idx[c]]
            for c in self._sort_colors
        }

        self._max_layers = 0
        for c in self._sort_colors:
            for sp_list in stacks_idx[c]:
                self._max_layers = max(self._max_layers, len(sp_list))

        paired: set = set()

        # Layer 0: each pick -> base marker
        for c in self._sort_colors:
            base_off = self._color_target_offset[c]
            for sp in range(self._stacks_per_color[c]):
                if stacks_idx[c][sp]:
                    pick_idx = stacks_idx[c][sp][0]
                    paired.add(pick_idx)
                    yield (pick_idx, base_off + sp)

        # Upper layers: each pick -> previously placed cube as target
        for layer in range(1, self._max_layers):
            for c in self._sort_colors:
                for sp in range(self._stacks_per_color[c]):
                    if layer < len(stacks_idx[c][sp]):
                        pick_idx = stacks_idx[c][sp][layer]
                        prev_pick = self._pick_objs[stacks_idx[c][sp][layer - 1]]
                        tgt_start = self._extend_target_objs([prev_pick])
                        paired.add(pick_idx)
                        yield (pick_idx, tgt_start)

        # Unpaired (unrecognized colors, if any)
        for i in range(len(self._pick_objs)):
            if i not in paired:
                yield (i, None)

    def initialize_pairings(self) -> None:
        """Pick order: all layer-0 placements first, then layer-1, etc."""
        super().initialize_pairings()

        ordered = []
        for layer in range(self._max_layers):
            for c in self._sort_colors:
                for sp in range(self._stacks_per_color[c]):
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
            for sp in range(self._stacks_per_color[c])
            if sp < len(self._color_stacks.get(c, []))
        )
        return self._stack_clearance_height(stacks, prim_geometry)


class TableTaskSortAndStack(UR10MultiPickPlaceTask):
    """Sort cubes from a 6x5x3 stacked grid into 3 color-coded boxes on the cart,
    and relocate yellow cubes to 6 stacks on the dropzone.

    - Sources: 90 cubes in a 6x5 grid, 3 layers high, on the dropzone floor.
      Colors randomly assigned from red/green/blue/yellow.
    - Targets: 3 color-coded open boxes on the cart (red, green, blue), each
      with 4 bottom-layer markers (2x2 grid). Plus 6 yellow-cube stacks on
      the dropzone floor (2x3 grid, to the right of (+X) and closer to the
      robot (-Y) than the source pile, non-overlapping in X and Y).
    - Strategy: ColorSortRelocateStackStrategy — routes by color to matching
      destination (boxes for R/G/B, floor stacks for yellow), stacks
      round-robin. No cubes are skipped.
    """

    DEFAULT_TASK_NAME = "table_task_sort_and_stack"

    GRID_COLS = 6
    GRID_ROWS = 5
    NUM_LAYERS = 3
    COLOR_PALETTE = ["red", "green", "blue", "yellow"]
    SORT_COLORS = ["red", "green", "blue", "yellow"]
    BOX_STACKS_PER_BOX = 4      # 2x2 grid of stacks per R/G/B box
    YELLOW_NUM_STACKS = 6       # 2x3 grid for yellow cubes
    YELLOW_STACKS_COLS = 2      # along Y (horizontal rows)
    YELLOW_STACKS_ROWS = 3      # along X (depth from robot)

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import ItemSpec, resolve_count
        from table_setup import (
            CART_SURFACE_CENTER,
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            setup_two_tables,
            spawn_open_box,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

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

        box_colors = ["red", "green", "blue"]
        box_y_offsets = {"red": -0.22, "green": 0.0, "blue": 0.22}
        box_colors_rgb = {
            "red": np.array([0.8, 0.3, 0.3]),
            "green": np.array([0.3, 0.8, 0.3]),
            "blue": np.array([0.3, 0.3, 0.8]),
        }

        box_specs = []
        for color in box_colors:
            bx, by = cx, cy + box_y_offsets[color]
            box_specs.append({
                "name": f"{color}_box",
                "center": np.array([bx, by, cart_z + box_wall_height / 2]),
                "center_xy": np.array([bx, by]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": 0.50 / stage_units,
                "match_labels": {"color": color},
                "z_tol": 0.03,
            })

        # --- Yellow stacks layout: 2x3 grid on the dropzone floor ---
        # Positioned to the right (+X) and closer to the robot (-Y)
        # than the source pile. No X or Y overlap with source grid.
        # Source grid extents: X ~ [-0.11, 0.19], Y ~ [0.57, 0.81]
        dropzone_center = DROPZONE_CENTER_POINT
        yellow_center_x = dropzone_center[0] + 0.26   # to the right (+X), clear of source
        yellow_center_y = dropzone_center[1] - 0.21   # closer to robot (-Y), clear of source
        yellow_stack_spacing = 0.06 / stage_units
        yellow_floor_z = DROPZONE_Z + 0.001

        yellow_positions = []
        for row in range(self.YELLOW_STACKS_ROWS):
            for col in range(self.YELLOW_STACKS_COLS):
                x = yellow_center_x + (row - (self.YELLOW_STACKS_ROWS - 1) / 2) * yellow_stack_spacing
                y = yellow_center_y + (col - (self.YELLOW_STACKS_COLS - 1) / 2) * yellow_stack_spacing
                yellow_positions.append(np.array([x, y, yellow_floor_z]))

        # Virtual verification region covering all yellow stacks
        yellow_region_spec = {
            "name": "yellow_stacks_region",
            "center_xy": np.array([yellow_center_x, yellow_center_y]),
            "floor_z": yellow_floor_z,
            "inner_size": np.array([0.20, 0.14]) / stage_units,
            "height": 0.50 / stage_units,
            "match_labels": {"color": "yellow"},
            "z_tol": 0.03,
        }

        # Combined box_specs for verification (R/G/B boxes + yellow region)
        all_verification_specs = box_specs + [yellow_region_spec]

        # --- Source grid: 6x5x3 on dropzone, random colors, top-down ---
        grid_spacing = 0.06 / stage_units
        base_z = DROPZONE_Z + cube_size / 2

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

        # --- Target Strategy: box markers (R/G/B) + yellow floor markers ---
        marker_scale = np.array([0.04, 0.04, 0.001]) / stage_units
        stack_spacing = 0.055 / stage_units  # center-to-center within boxes

        class CombinedMarkerGenerator:
            """Generate box floor markers (R/G/B) then yellow floor markers."""
            def __init__(self, box_specs, box_floor_z, marker_scale,
                         stack_spacing, stacks_per_box,
                         yellow_positions):
                self.box_specs = box_specs
                self.box_floor_z = box_floor_z
                self.marker_scale = marker_scale
                self.stack_spacing = stack_spacing
                self.stacks_per_box = stacks_per_box
                self.yellow_positions = yellow_positions

            def generate(self, count_range=None, seed=None):
                targets = []
                # R/G/B box markers
                grid_side = int(self.stacks_per_box ** 0.5)
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
                                    self.box_floor_z,
                                ]),
                                scale=self.marker_scale,
                                hidden=True,
                            ))
                # Yellow floor markers
                for pos in self.yellow_positions:
                    targets.append(ItemSpec(
                        asset_type="marker",
                        position=pos,
                        scale=self.marker_scale,
                        hidden=True,
                    ))
                return targets

        target_strategy = CombinedMarkerGenerator(
            box_specs=box_specs,
            box_floor_z=box_floor_z,
            marker_scale=marker_scale,
            stack_spacing=stack_spacing,
            stacks_per_box=self.BOX_STACKS_PER_BOX,
            yellow_positions=yellow_positions,
        )

        # --- Strategy factory ---
        sort_colors = list(self.SORT_COLORS)
        stacks_per_color = {
            "red": self.BOX_STACKS_PER_BOX,
            "green": self.BOX_STACKS_PER_BOX,
            "blue": self.BOX_STACKS_PER_BOX,
            "yellow": self.YELLOW_NUM_STACKS,
        }

        # Build spatial checks for R/G/B boxes + yellow region
        all_region_checks = []
        for spec in all_verification_specs:
            all_region_checks.append(build_bin_geometry_check({
                "center_xy": spec["center_xy"],
                "inner_size": spec["inner_size"],
                "floor_z": spec["floor_z"],
                "height": 0.50,
                "z_tol": spec.get("z_tol", 0.03),
            }))

        def _combined_base_check(pick_obj, target_obj=None,
                                 bb_cache=None, obj_scale=None):
            return any(
                chk(pick_obj, target_obj, bb_cache=bb_cache, obj_scale=obj_scale)
                for chk in all_region_checks
            )

        def _strategy_factory(picks, targets):
            stacking_map = compute_stacking_map(picks)
            return ColorSortRelocateStackStrategy(
                pick_objs=picks,
                target_objs=targets,
                sort_colors=sort_colors,
                stacks_per_color=stacks_per_color,
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
        total_base_markers = (
            self.BOX_STACKS_PER_BOX * len(box_colors) + self.YELLOW_NUM_STACKS
        )

        if pick_count is None:
            pick_count = total_cubes

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick red, green, and blue cubes from a 6x5x3 stacked grid "
                "and sort them into matching color-coded boxes on the cart, "
                "stacking cubes on top of previously placed cubes. "
                "Yellow cubes are relocated to 6 stacks on the dropzone floor "
                "(to the right (+X) and closer to the robot (-Y) than the source pile)."
            ),
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            setup_workspace=_workspace_setup,
            box_verification_info={"box_specs": all_verification_specs},
            containment_check=True,
            stacking_enabled=True,
            scenario={
                "source": "dropzone_grid",
                "destination": "boxes_on_cart_and_dropzone_stacks",
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
                "generation_order": "top-down (layer 2 -> 1 -> 0)",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": (
                    "2x2 bottom-layer markers per R/G/B box + "
                    "3x2 floor markers for yellow stacks"
                ),
                "count": total_base_markers,
                "virtual": True,
                "containers": {
                    "count": 3,
                    "layout": "3 color-coded boxes in a row on cart",
                    "capacity_per_box": f"{self.BOX_STACKS_PER_BOX} stacks",
                },
                "yellow_stacks": {
                    "count": self.YELLOW_NUM_STACKS,
                    "layout": (
                        f"{self.YELLOW_STACKS_COLS}x{self.YELLOW_STACKS_ROWS} "
                        f"grid on dropzone floor"
                    ),
                },
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                create_strategy=_strategy_factory,
                ee_height_for_move=0.35 / stage_units,
                strategy_description={
                    "class": "ColorSortRelocateStackStrategy",
                    "pairing": "color_sort_relocate_stack",
                    "details": (
                        f"Routes R/G/B cubes to matching boxes ({self.BOX_STACKS_PER_BOX} "
                        f"stacks per box, round-robin). Yellow cubes relocated to "
                        f"{self.YELLOW_NUM_STACKS} stacks on the dropzone. "
                        f"Upper targets are placed cubes. No cubes skipped."
                    ),
                },
            ),
            verification_description={
                "containment_check": True,
                "match_labels": (
                    "color-based: red_box accepts only red cubes, etc. "
                    "yellow_stacks_region accepts only yellow cubes."
                ),
            },
            rationale={
                "create_strategy": (
                    "Custom ColorSortRelocateStackStrategy extends "
                    "ColorSortStackStrategy with per-color stack counts. "
                    "Yellow cubes are relocated to separate stacks instead "
                    "of being skipped, so no cubes are permanently blocked."
                ),
                "containment_check": (
                    "R/G/B items placed inside boxes, yellow items placed "
                    "in a virtual region on the dropzone — box-boundary "
                    "verification confirms correct placement for all colors"
                ),
                "virtual_target_generation_strategy": (
                    "Bottom-layer markers for boxes and floor markers for "
                    "yellow stacks — upper targets are placed cubes"
                ),
                "stacking_enabled": (
                    "Source cubes are stacked 3 layers high — stacking_map "
                    "enforces top-down pick order. All cubes are reachable "
                    "since yellow cubes are moved rather than skipped."
                ),
                "ee_height_for_move": (
                    "Base 0.35m + dynamic increase via strategy to clear "
                    "growing stacks"
                ),
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
