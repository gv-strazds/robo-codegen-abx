import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTask3(UR10MultiPickPlaceTask):
    """Task using UR10 robot to pick balls from a bin and place them onto disc targets arranged in a grid."""

    DEFAULT_TASK_NAME = "table_task_3"

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
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
            SequentialChoice,
        )
        from table_setup import (
            BIN_X_COORD,
            BIN_Y_COORD,
            DROPZONE_X,
            DROPZONE_Y,
            DROPZONE_Z,
            setup_two_tables,
            ITEM_SPAWN_REFERENCE_Z,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        # --- Generation Strategies ---
        stage_units = get_stage_units()
        expected_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units

        # Pick Strategy: 3x3 Grid of Balls in Bin
        pick_z = ITEM_SPAWN_REFERENCE_Z + expected_scale[2] / 2 + 0.02
        pick_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=3,
            cols=1,
            spacing_x=0.08,
            spacing_y=0.08,
        )

        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("ball"),
            scale_strategy=FixedValue(expected_scale),
            color_strategy=None,  # Random
        )

        # Target Strategy: 3x4 Grid of Discs in Dropzone
        dx = -0.15
        dy = 0.15
        start_grid_x = DROPZONE_X
        start_grid_y = DROPZONE_Y
        grid_w = 3
        grid_l = 4

        center_grid_x = start_grid_x + (grid_w - 1) * dx / 2
        center_grid_y = start_grid_y + (grid_l - 1) * dy / 2
        center_grid_z = DROPZONE_Z + 0.001 + expected_scale[2] / 2

        target_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, center_grid_z]),
            rows=grid_l,
            cols=grid_w,
            spacing_x=dx,
            spacing_y=dy,
        )

        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("disc"),
            color_strategy=SequentialChoice(
                ["purple", "cyan", "black", "yellow"], loop=True
            ),
            scale_strategy=FixedValue(expected_scale),
        )

        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick balls from the bin and place them onto disc targets arranged in a 3x4 grid on the dropzone table.",
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            scenario={"source": "bin", "destination": "dropzone_grid", "workspace": "two_tables"},
            pick_description={"asset_types": ["ball"], "count": 3, "arrangement": "3x1 column in pick bin"},
            target_description={"type": "visible_markers", "arrangement": "3x4 grid on dropzone", "count": 12},
            verification_description={"spatial_check": "is_on_top (default)"},
            rationale={"create_strategy": "Default sequential pairing — balls placed on disc targets without matching"},
            implementation=TaskImplementationSpec(
                strategy_description={"class": "MultiPickStrategy", "pairing": "sequential"},
                # Balls are dropped into the bin via gravity and bounce/roll for a
                # short time. In teleport mode they would otherwise be picked
                # mid-roll, retaining linear/angular velocity, and roll off the
                # disc targets after teleport-placement. Wait for them to settle.
                startup_delay_seconds=1.5,
            ),
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
