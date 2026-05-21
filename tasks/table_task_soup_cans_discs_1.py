import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from task_verification import is_on_top, is_vertical

logger = logging.getLogger(__name__)


class TableTaskSoupCansDiscs1(UR10MultiPickPlaceTask):
    """Pick soup cans from the bin and place them onto a 2x3 grid of colored disc markers."""

    DEFAULT_TASK_NAME = "table_task_soup_cans_discs_1"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        from isaacsim.core.utils import rotations
        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import (
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
            SequentialChoice,
        )
        from pxr import Gf
        from table_setup import (
            BIN_X_COORD,
            BIN_Y_COORD,
            DROPZONE_X,
            DROPZONE_Y,
            DROPZONE_Z,
            ITEM_SPAWN_REFERENCE_Z,
            setup_two_tables,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        # --- Pick strategy: 3x3 grid of upright soup cans in the bin ---
        default_orientation = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        )

        pick_z = ITEM_SPAWN_REFERENCE_Z + 0.0515 / 2 + 0.025

        pick_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=3,
            cols=3,
            spacing_x=0.08,
            spacing_y=0.08,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("soup_can"),
            orientation_strategy=FixedValue(default_orientation),
            color_strategy=None,
        )

        # --- Target strategy: 2x3 grid of disc markers, cycling red/yellow/blue ---
        # Use the static "fixed_disc" (FixedCylinder) — dynamic disc primitives
        # under cans on the kinematic table can jitter from uneven contact.
        DISC_THICKNESS = 0.002
        DISC_DIAMETER = 0.07
        dx = -0.15
        dy = 0.15
        start_grid_x = DROPZONE_X
        start_grid_y = DROPZONE_Y
        grid_cols = 2   # along X
        grid_rows = 3   # along Y
        center_grid_x = start_grid_x + (grid_cols - 1) * dx / 2
        center_grid_y = start_grid_y + (grid_rows - 1) * dy / 2
        center_grid_z = DROPZONE_Z + 0.001 + DISC_THICKNESS / 2

        target_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, center_grid_z]),
            rows=grid_rows,
            cols=grid_cols,
            spacing_x=dx,
            spacing_y=dy,
        )

        target_scale = (
            np.array([DISC_DIAMETER, DISC_DIAMETER, DISC_THICKNESS]) / stage_units
        )

        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("fixed_disc"),
            color_strategy=SequentialChoice(["red", "yellow", "blue"], loop=True),
            scale_strategy=FixedValue(target_scale),
        )

        def _soup_can_spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None,
                                    log_failure=False):
            on_top = is_on_top(pick_obj, target_obj, bb_cache=bb_cache,
                               obj_scale=obj_scale, log_failure=log_failure)
            vertical = is_vertical(
                pick_obj, obj_scale=obj_scale, max_tilt_deg=15,
                log_failure=log_failure,
            )
            return on_top and vertical

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick soup cans from the bin and place them onto colored disc markers "
                "(red, yellow, blue) arranged in a 2x3 grid on the dropzone."
            ),
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            spatial_check_fn=_soup_can_spatial_check,
            scenario={"source": "bin", "destination": "dropzone_grid", "workspace": "two_tables"},
            pick_description={
                "asset_types": ["soup_can"],
                "count": 9,
                "arrangement": "3x3 grid in pick bin",
                "colors": "USD asset default",
                "orientation": "upright (-90° X)",
            },
            target_description={
                "type": "visible_markers",
                "asset_type": "disc",
                "arrangement": "2x3 grid (3 rows x 2 cols) on dropzone",
                "count": 6,
                "colors": "SequentialChoice(['red','yellow','blue'], loop=True)",
            },
            implementation=TaskImplementationSpec(
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                },
            ),
            verification_description={"spatial_check": "is_on_top + is_vertical"},
            rationale={
                "create_strategy": (
                    "Default sequential pairing — all picks are same type, no matching needed. "
                    "9 picks vs. 6 targets means 3 picks overflow with no target and are not placed."
                ),
                "spatial_check_fn": (
                    "Soup cans must rest on their disc marker (is_on_top) and remain upright "
                    "(is_vertical) after placement."
                ),
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
