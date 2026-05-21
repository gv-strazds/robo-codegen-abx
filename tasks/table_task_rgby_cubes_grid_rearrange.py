import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTaskRgbyCubesGridRearrange(UR10MultiPickPlaceTask):
    """Pick 12 cubes (3 each of red, green, blue, yellow) pre-arranged in a
    4x3 grid on the dropzone and place them to form a more compact 3x4 grid
    on the cart."""

    DEFAULT_TASK_NAME = "table_task_rgby_cubes_grid_rearrange"

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
            SequentialChoice,
        )
        from table_setup import (
            CART_SURFACE_CENTER,
            DROPZONE_CENTER_POINT,
            ITEM_SPAWN_REFERENCE_Z,
            setup_two_tables,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()
        cube_edge = 0.0515
        cube_scale = np.array([cube_edge, cube_edge, cube_edge]) / stage_units
        cube_half = cube_edge / 2

        # --- Pick: 12 cubes in a 4x3 grid on the dropzone ---
        # 3 of each colour, organised one colour per row of the source grid.
        PICK_SPACING = 0.075
        pick_z = ITEM_SPAWN_REFERENCE_Z + cube_half + 0.001

        pick_pos_gen = GridPositionGenerator(
            center=np.array(
                [DROPZONE_CENTER_POINT[0], DROPZONE_CENTER_POINT[1], pick_z]
            ),
            rows=4,
            cols=3,
            spacing_x=PICK_SPACING,
            spacing_y=PICK_SPACING,
            randomize=False,
        )
        pick_color_seq = (
            ["red"] * 3 + ["green"] * 3 + ["blue"] * 3 + ["yellow"] * 3
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("cube"),
            scale_strategy=FixedValue(cube_scale),
            color_strategy=SequentialChoice(pick_color_seq, loop=False),
        )

        # --- Target: 12 hidden virtual markers in a 3x4 grid on the cart ---
        # Tighter spacing than the pick grid -> "more compact" 3x4 arrangement.
        RECT_THICKNESS = 0.002
        TARGET_SPACING = 0.060
        target_z = CART_SURFACE_CENTER[2] + 0.001 + RECT_THICKNESS / 2
        marker_scale = np.array([0.06, 0.06, RECT_THICKNESS]) / stage_units

        target_pos_gen = GridPositionGenerator(
            center=np.array(
                [CART_SURFACE_CENTER[0], CART_SURFACE_CENTER[1], target_z]
            ),
            rows=3,
            cols=4,
            spacing_x=TARGET_SPACING,
            spacing_y=TARGET_SPACING,
            randomize=False,
        )
        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("marker"),
            color_strategy=FixedValue("white"),
            scale_strategy=FixedValue(marker_scale),
            hidden_strategy=FixedValue(True),
        )

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick 12 cubes (3 each of red, green, blue, yellow) pre-"
                "arranged in a 4x3 grid on the dropzone and place them to "
                "form a more compact 3x4 grid on the cart."
            ),
            pick_generation_strategy=pick_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root, standard_objs=False, add_bin=False,
            ),
            scenario={
                "source": "dropzone_grid",
                "destination": "cart_grid",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cube"],
                "count": 12,
                "arrangement": (
                    "4x3 grid (4 rows along Y, 3 cols along X) on the dropzone"
                ),
                "colors": (
                    "3 red + 3 green + 3 blue + 3 yellow, "
                    "organized one color per row"
                ),
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": (
                    "3x4 grid (3 rows along Y, 4 cols along X) on the cart "
                    "surface"
                ),
                "count": 12,
                "virtual": True,
                "spacing": (
                    f"{TARGET_SPACING} m (more compact than pick spacing "
                    f"{PICK_SPACING} m)"
                ),
            },
            verification_description={"spatial_check": "is_on_top (default)"},
            rationale={
                "cart_starts_empty": (
                    "standard_objs=False and add_bin=False — placements need "
                    "the full cart surface, and per Issue 14 default cart "
                    "decorations would collide with the placement grid."
                ),
                "hidden_virtual_targets": (
                    "Per Issue 15: user described arrangement (3x4 grid) but "
                    "not target geometry, so virtual targets stay hidden and "
                    "the final scene shows just the cubes forming the grid."
                ),
                "create_strategy": (
                    "Default sequential pairing — pick[i] -> target[i]. "
                    "Colors do not need to match specific cart positions "
                    "since the task is a rearrangement, not a sort."
                ),
            },
            implementation=TaskImplementationSpec(
                ee_height_for_move=0.45 / stage_units,
                virtual_target_generation_strategy=target_strategy,
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                },
            ),
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
