import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTask5(UR10MultiPickPlaceTask):
    """Pick green cubes from the dropzone table and place onto 7 red rectangles arranged in a circle."""

    DEFAULT_TASK_NAME = "table_task_5"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        # Lazily import Isaac utilities and scene helpers
        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import (
            CircularPositionGenerator,
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
        )
        from table_setup import (
            DROPZONE_CENTER_POINT,
            DROPZONE_X,
            DROPZONE_Y,
            setup_two_tables,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        BLOCK_SIZE = 0.0515

        # --- Strategies ---
        stage_units = get_stage_units()
        expected_scale = np.array([BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE]) / stage_units

        # Pick Strategy: 3x2 Grid of Green Cubes on Dropzone
        # Grid width=3, length=2.
        dx = -0.15
        dy = 0.15
        start_grid_x = DROPZONE_X
        start_grid_y = DROPZONE_Y

        grid_w = 3
        grid_l = 2
        center_grid_x = start_grid_x + (grid_w - 1) * dx / 2
        center_grid_y = start_grid_y + (grid_l - 1) * dy / 2
        pick_z = DROPZONE_CENTER_POINT[2] + 0.001 + expected_scale[2] / 2

        pick_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, pick_z]),
            rows=grid_l,
            cols=grid_w,
            spacing_x=dx,
            spacing_y=dy,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("cube"),
            scale_strategy=FixedValue(expected_scale),
            color_strategy=FixedValue("green"),
        )

        # Target Strategy: Circular, 7 pos, radius 0.18. Red Rects.
        RECT_HEIGHT = expected_scale[2]*1.5 / stage_units
        target_scale = expected_scale.copy()
        target_scale[2] = RECT_HEIGHT
        target_z = DROPZONE_CENTER_POINT[2] + RECT_HEIGHT / 2
        target_pos_gen = CircularPositionGenerator(
            center=np.array(
                [DROPZONE_CENTER_POINT[0], DROPZONE_CENTER_POINT[1], target_z]
            ),
            radius=0.18,
            z_offset=0.0,
            count=7,
        )

        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("rect"),
            color_strategy=FixedValue("red"),
            scale_strategy=FixedValue(target_scale),
        )

        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick green cubes from the dropzone table and place them onto red rectangles arranged in a circle.",
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            scenario={"source": "dropzone", "destination": "dropzone_circle", "workspace": "two_tables"},
            pick_description={"asset_types": ["cube"], "count": 6, "arrangement": "3x2 grid on dropzone surface", "colors": "green (fixed)"},
            target_description={"type": "visible_markers", "arrangement": "circle (r=0.18m, 7 positions) on dropzone", "count": 7},
            implementation=TaskImplementationSpec(
                strategy_description={"class": "MultiPickStrategy", "pairing": "sequential"},
            ),
            verification_description={"spatial_check": "is_on_top (default)"},
            rationale={"create_strategy": "Default sequential pairing — cubes placed on circle markers without matching"},
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
