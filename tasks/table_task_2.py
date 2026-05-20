import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTask2(UR10MultiPickPlaceTask):
    """Task using UR10 robot to pick cubes and place them on blue cubes arranged in a
     3x4 grid on the dropzone surface."""

    DEFAULT_TASK_NAME = "table_task_2"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        # Lazily import Isaac utilities to avoid import-order issues
        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import FixedValue, GridPositionGenerator, ItemGenerator
        from table_setup import DROPZONE_X, DROPZONE_Y, DROPZONE_Z, setup_two_tables
        from task_spec import TaskImplementationSpec, TaskSpec

        # --- Define Generation Strategies ---

        CUBE_SIZE = 0.0515
        # Resolve object size for calculations
        stage_units = get_stage_units()  # Assuming strict context
        expected_scale = np.array([CUBE_SIZE, CUBE_SIZE, CUBE_SIZE]) / stage_units

        # Pick Strategy: Line of 7 cubes
        spacing_y = expected_scale[1] + (0.01 / stage_units)
        rows = 7
        start_y = (DROPZONE_Y - 0.1) / stage_units

        total_len_y = (rows - 1) * spacing_y
        center_y = start_y + total_len_y / 2

        center_x = (DROPZONE_X + 0.18) / stage_units
        center_z = CUBE_SIZE / 2  # cubes position z is usually half-height

        pick_pos_gen = GridPositionGenerator(
            center=np.array([center_x, center_y, center_z]),
            rows=rows,
            cols=1,
            spacing_x=0.0,
            spacing_y=spacing_y,
        )

        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("cube"),
            scale_strategy=FixedValue(expected_scale),
            color_strategy=None,  # Random
        )

        # Target Strategy: 3x4 Grid
        dx = -0.15
        dy = 0.15
        start_grid_x = DROPZONE_X
        start_grid_y = DROPZONE_Y

        grid_w = 3
        grid_l = 4

        # initial positions for target objects
        center_grid_x = start_grid_x + (grid_w - 1) * dx / 2
        center_grid_y = start_grid_y + (grid_l - 1) * dy / 2
        center_grid_z = DROPZONE_Z + CUBE_SIZE / 2 + 0.001 # small offset to avoid init in collision
        # target cubes will be spawned slightly above the dropzone surface.

        target_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, center_grid_z]),
            rows=grid_l,
            cols=grid_w,
            spacing_x=dx,
            spacing_y=dy,
        )

        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("cube"),
            scale_strategy=FixedValue(expected_scale),
            color_strategy=FixedValue(np.array([0, 0, 1])),  # Blue
        )

        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick multiple cubes and place them on blue cubes arranged in a 3x4 grid.",
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            scenario={"source": "table", "destination": "dropzone_grid", "workspace": "two_tables"},
            pick_description={"asset_types": ["cube"], "count": 7, "arrangement": "7x1 line on table surface"},
            target_description={"type": "visible_markers", "arrangement": "3x4 grid on dropzone", "count": 12},
            implementation=TaskImplementationSpec(
                strategy_description={"class": "MultiPickStrategy", "pairing": "sequential"},
            ),
            verification_description={"spatial_check": "is_on_top (default)"},
            rationale={"create_strategy": "Default sequential pairing — simple placement without color or type matching"},
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
