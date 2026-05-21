import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTaskGreenCubesRowToYellowGrid(UR10MultiPickPlaceTask):
    """Pick 6 green cubes pre-arranged in a single row on the cart and place
    them onto yellow rectangular markers in a 2x3 grid on the dropzone."""

    DEFAULT_TASK_NAME = "table_task_green_cubes_row_to_yellow_grid"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME

        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import FixedValue, GridPositionGenerator, ItemGenerator
        from table_setup import (
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

        # --- Pick: 6 green cubes in a single row on the cart, along Y ---
        # Offset in -X from the cart center keeps the row clear of the bin,
        # which sits at +X relative to CART_SURFACE_CENTER.
        row_center_x = CART_SURFACE_CENTER[0] - 0.18
        row_center_y = CART_SURFACE_CENTER[1]
        pick_z = ITEM_SPAWN_REFERENCE_Z + cube_half + 0.001

        pick_pos_gen = GridPositionGenerator(
            center=np.array([row_center_x, row_center_y, pick_z]),
            rows=6,
            cols=1,
            spacing_x=0.0,
            spacing_y=0.08,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("cube"),
            scale_strategy=FixedValue(cube_scale),
            color_strategy=FixedValue("green"),
        )

        # --- Target: 2x3 grid of yellow rectangle markers on the dropzone ---
        RECT_HEIGHT = 0.002
        dx = -0.18
        dy = 0.15
        grid_rows = 2  # along Y
        grid_cols = 3  # along X
        center_grid_x = DROPZONE_X + (grid_cols - 1) * dx / 2
        center_grid_y = DROPZONE_Y + (grid_rows - 1) * dy / 2
        center_grid_z = DROPZONE_Z + 0.001 + RECT_HEIGHT / 2

        target_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, center_grid_z]),
            rows=grid_rows,
            cols=grid_cols,
            spacing_x=dx,
            spacing_y=dy,
        )
        target_scale = np.array([0.10, 0.10, RECT_HEIGHT]) / stage_units
        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("rect"),
            color_strategy=FixedValue("yellow"),
            scale_strategy=FixedValue(target_scale),
        )

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick 6 green cubes pre-arranged in a single row on the cart "
                "and place them onto yellow rectangles arranged in a 2x3 grid "
                "on the dropzone."
            ),
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root, standard_objs=False, add_bin=False,
            ),
            scenario={
                "source": "cart_surface_row",
                "destination": "dropzone_grid",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cube"],
                "count": 6,
                "arrangement": "single row along Y on the cart surface (away from the bin)",
                "colors": "green",
            },
            target_description={
                "type": "visible_markers",
                "marker_color": "yellow",
                "arrangement": "2x3 grid (2 rows along Y, 3 cols along X) on the dropzone",
                "count": 6,
            },
            verification_description={"spatial_check": "is_on_top (default)"},
            rationale={
                "source": (
                    "Pick items spawn on the cart surface (not the bin) using the "
                    "ITEM_SPAWN_REFERENCE_Z + half_height drop pattern, mirroring "
                    "the red cart balls in tasks/table_task_3c.py."
                ),
                "create_strategy": (
                    "Default sequential pairing — all cubes identical, all markers "
                    "identical, no matching needed."
                ),
                "spatial_check_fn": (
                    "Default is_on_top is sufficient: cube XY-overlaps marker and "
                    "rests within 2 cm of marker top."
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
