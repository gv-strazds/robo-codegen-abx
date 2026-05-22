import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTaskRedBlueCubesToGreenGrid(UR10MultiPickPlaceTask):
    """Pick 9 red cubes from a 3x3 grid in the bin and 3 blue cubes from a
    row on the cart, then place them onto green square rectangle markers
    arranged in a 3x4 grid on the dropzone (bin cubes placed first)."""

    DEFAULT_TASK_NAME = "table_task_red_blue_cubes_to_green_grid"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME

        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import (
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
            ItemSpec,
        )
        from table_setup import (
            BIN_X_COORD,
            BIN_Y_COORD,
            CART_SURFACE_CENTER,
            DROPZONE_X,
            DROPZONE_Y,
            DROPZONE_Z,
            ITEM_SPAWN_REFERENCE_Z,
            setup_two_tables,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()
        cube_edge = 0.0515
        cube_scale = np.array([cube_edge, cube_edge, cube_edge]) / stage_units
        cube_half = cube_edge / 2

        # --- Bin source: 3x3 grid of red cubes ---
        # +0.02 drop margin so cubes settle on the bin floor by gravity.
        bin_pick_z = ITEM_SPAWN_REFERENCE_Z + cube_half + 0.02
        bin_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, bin_pick_z]),
            rows=3,
            cols=3,
            spacing_x=0.08,
            spacing_y=0.08,
            randomize=False,
        )

        # --- Cart source: row of 3 blue cubes ---
        # X offset of -0.18 keeps the row clear of the bin, which sits at +X
        # relative to CART_SURFACE_CENTER.  Mirrors the green-cubes-row task.
        cart_pick_z = ITEM_SPAWN_REFERENCE_Z + cube_half + 0.001
        cart_pos_gen = GridPositionGenerator(
            center=np.array([
                CART_SURFACE_CENTER[0] - 0.18,
                CART_SURFACE_CENTER[1],
                cart_pick_z,
            ]),
            rows=3,
            cols=1,
            spacing_x=0.0,
            spacing_y=0.08,
            randomize=False,
        )

        # --- Combined pick generator: bin cubes first, then cart cubes ---
        _cube_scale = cube_scale

        class _CombinedPickGenerator:
            """Generate 9 red bin cubes followed by 3 blue cart cubes.

            Returning the items in this fixed order guarantees that the
            default sequential pairing places all bin cubes before any cart
            cubes — pick[0..8] -> target[0..8] (bin reds), pick[9..11] ->
            target[9..11] (cart blues).
            """

            def __init__(self, bin_pg, cart_pg):
                self._bin_pos_gen = bin_pg
                self._cart_pos_gen = cart_pg

            def generate(self, count_range=None, seed=None):
                bin_positions = self._bin_pos_gen.get_positions(9, seed=seed)
                cart_positions = self._cart_pos_gen.get_positions(3, seed=seed)
                items = []
                for i in range(9):
                    items.append(ItemSpec(
                        asset_type="cube",
                        position=bin_positions[i],
                        scale=_cube_scale,
                        color="red",
                        name=f"bin_cube_{i}",
                    ))
                for j in range(3):
                    items.append(ItemSpec(
                        asset_type="cube",
                        position=cart_positions[j],
                        scale=_cube_scale,
                        color="blue",
                        name=f"cart_cube_{j}",
                    ))
                return items

        pick_strategy = _CombinedPickGenerator(bin_pos_gen, cart_pos_gen)

        # --- Target: 3x4 grid of green rectangles on the dropzone ---
        # "rect" -> FixedCuboid (static): cubes rest on it without jitter
        # (per learnings Issue 16).
        RECT_HEIGHT = 0.002
        dx = -0.12
        dy = 0.13
        grid_rows = 3  # along Y
        grid_cols = 4  # along X
        center_grid_x = DROPZONE_X + (grid_cols - 1) * dx / 2
        center_grid_y = DROPZONE_Y + (grid_rows - 1) * dy / 2
        center_grid_z = DROPZONE_Z + 0.001 + RECT_HEIGHT / 2

        target_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, center_grid_z]),
            rows=grid_rows,
            cols=grid_cols,
            spacing_x=dx,
            spacing_y=dy,
            randomize=False,
        )
        target_scale = np.array([0.06, 0.06, RECT_HEIGHT]) / stage_units
        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("rect"),
            color_strategy=FixedValue("green"),
            scale_strategy=FixedValue(target_scale),
        )

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick 9 red cubes from a 3x3 grid in the bin and 3 blue "
                "cubes from a row on the cart, then place them onto green "
                "square rectangle markers arranged in a 3x4 grid on the "
                "dropzone (bin cubes placed first, then cart cubes)."
            ),
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root, standard_objs=False, add_bin=True,
            ),
            scenario={
                "source": "bin_and_cart",
                "destination": "dropzone_grid",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cube"],
                "count": 12,
                "arrangement": (
                    "9 cubes in a 3x3 grid in the pick bin + 3 cubes in a "
                    "single row along Y on the cart surface (offset in -X "
                    "to clear the bin)"
                ),
                "colors": "9 red (bin) + 3 blue (cart)",
            },
            target_description={
                "type": "visible_markers",
                "marker_color": "green",
                "arrangement": (
                    "3x4 grid (3 rows along Y, 4 cols along X) on the "
                    "dropzone, square rectangles 6 cm x 6 cm"
                ),
                "count": 12,
            },
            verification_description={"spatial_check": "is_on_top (default)"},
            rationale={
                "combined_pick_generator": (
                    "TaskSpec accepts one pick_generation_strategy, so a "
                    "custom generator class produces both bin (3x3 red) "
                    "and cart (row of 3 blue) cubes in one list.  Bin "
                    "items appear first in the list, so default sequential "
                    "pairing places them before the cart items — the user's "
                    "'bin cubes placed first' requirement falls out of "
                    "generator ordering, no custom strategy needed."
                ),
                "cart_starts_empty": (
                    "standard_objs=False (Issue 14) — the default cart "
                    "decoration props would collide with the blue-cube row."
                ),
                "visible_green_markers": (
                    "User explicitly named target color AND shape ('square "
                    "green markers'), so visible 'rect' markers (Issue 15) "
                    "rather than hidden virtual targets.  Static 'rect' "
                    "(FixedCuboid) rather than dynamic 'cube' to avoid "
                    "jitter under placed cubes (Issue 16)."
                ),
                "ee_height_for_move": (
                    "Raised to 0.45 m so the EE clears the remaining cart "
                    "cubes during transport (Issue 3) until all 3 blue "
                    "cubes have been picked."
                ),
                "create_strategy": (
                    "Default sequential pairing — pick[i] -> target[i]. "
                    "Pick order is fixed by the combined generator (bin "
                    "first), so no custom strategy is required."
                ),
            },
            implementation=TaskImplementationSpec(
                ee_height_for_move=0.45 / stage_units,
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                },
            ),
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
