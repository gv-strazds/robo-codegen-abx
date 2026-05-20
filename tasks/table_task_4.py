import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTask4(UR10MultiPickPlaceTask):
    """Task using UR10 robot to pick cubes from a bin and place them onto yellow rectangles arranged in a circle."""

    DEFAULT_TASK_NAME = "table_task_4"

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
        from item_generation import (
            CircularPositionGenerator,
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
        )
        from table_setup import (
            BIN_X_COORD,
            BIN_Y_COORD,
            DROPZONE_CENTER_POINT,
            setup_two_tables,
            ITEM_SPAWN_REFERENCE_Z,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        # --- Generation Strategies ---
        stage_units = get_stage_units()
        expected_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units

        # Pick Strategy: 3x3 Grid of Cubes in Bin
        pick_z = ITEM_SPAWN_REFERENCE_Z + expected_scale[2] / 2 + 0.025
        pick_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=3,
            cols=3,
            spacing_x=0.08,
            spacing_y=0.08,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("cube"),
            scale_strategy=FixedValue(expected_scale),
            color_strategy=None,  # Random
        )

        # Target Strategy: Circular Pattern of Yellow Rectangles
        # radius=0.2, center=DROPZONE_CENTER_POINT
        RECT_HEIGHT = 0.002
        target_z = DROPZONE_CENTER_POINT[2] + 0.001 + RECT_HEIGHT / 2

        target_pos_gen = CircularPositionGenerator(
            center=np.array(
                [DROPZONE_CENTER_POINT[0], DROPZONE_CENTER_POINT[1], target_z]
            ),
            radius=0.2,
            z_offset=0.0,
            count=7,
        )

        target_scale = np.array([0.025, 0.025, RECT_HEIGHT]) / stage_units

        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("rect"),
            color_strategy=FixedValue("yellow"),
            scale_strategy=FixedValue(target_scale),
        )

        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick cubes from the bin and place them onto yellow rectangles arranged in a circle on the dropzone table.",
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            scenario={"source": "bin", "destination": "dropzone_circle", "workspace": "two_tables"},
            pick_description={"asset_types": ["cube"], "count": 9, "arrangement": "3x3 grid in pick bin"},
            target_description={"type": "visible_markers", "arrangement": "circle (r=0.2m, 7 positions) on dropzone", "count": 7},
            implementation=TaskImplementationSpec(
                strategy_description={"class": "MultiPickStrategy", "pairing": "sequential"},
            ),
            verification_description={"spatial_check": "is_on_top (default)"},
            rationale={"create_strategy": "Default sequential pairing — cubes placed on circle markers without matching"},
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
