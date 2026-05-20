import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from task_verification import is_on_top, is_vertical

logger = logging.getLogger(__name__)


class TableTaskCrackerBoxes1(UR10MultiPickPlaceTask):
    """Pick cracker boxes from the pick bin and place onto targets.

    Similar to TableTaskSoupCans1, but spawns USD "cracker_box" objects
    (same asset/orientation as in setup_two_tables) and arranges them 4x1
    within the pick bin.
    """

    DEFAULT_TASK_NAME = "table_task_cracker_boxes_1"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        from isaacsim.core.utils import rotations

        # Lazily import Isaac utilities to avoid import-order issues
        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import FixedValue, GridPositionGenerator, ItemGenerator
        from pxr import Gf
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

        stage_units = get_stage_units()

        # --- Strategies ---
        # Pick Strategy: 4x1 Grid. Cracker Box.
        # Orientation: -90 deg X.
        default_orientation = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        )

        pick_z = ITEM_SPAWN_REFERENCE_Z + 0.0515 / 2 + 0.025

        pick_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=4,
            cols=1,
            spacing_x=0.08,
            spacing_y=0.08,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("cracker_box"),
            orientation_strategy=FixedValue(default_orientation),
            color_strategy=None,
        )

        # Target Strategy: 3x4 Grid. Green Rect.
        # spacing_x = -0.18 m fits 3 cracker boxes (~0.164 m world X footprint
        # each) side-by-side across one row with ~1.6 cm clearance.
        RECT_HEIGHT = 0.002
        dx = -0.18
        dy = 0.15
        start_grid_x = DROPZONE_X
        start_grid_y = DROPZONE_Y
        grid_w = 3
        grid_l = 4
        center_grid_x = start_grid_x + (grid_w - 1) * dx / 2
        center_grid_y = start_grid_y + (grid_l - 1) * dy / 2
        center_grid_z = DROPZONE_Z + 0.001 + RECT_HEIGHT / 2

        target_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, center_grid_z]),
            rows=grid_l,
            cols=grid_w,
            spacing_x=dx,
            spacing_y=dy,
        )

        target_scale = np.array([0.1, 0.1, RECT_HEIGHT]) / stage_units

        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("rect"),
            color_strategy=FixedValue("green"),
            scale_strategy=FixedValue(target_scale),
        )

        def _cracker_box_spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None,
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
            task_description="Pick cracker boxes from the bin and place them onto thin green rectangles arranged in a 3x4 grid in the dropzone.",
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            spatial_check_fn=_cracker_box_spatial_check,
            scenario={"source": "bin", "destination": "dropzone_grid", "workspace": "two_tables"},
            pick_description={
                "asset_types": ["cracker_box"],
                "count": 4,
                "arrangement": "4x1 column in pick bin",
                "colors": "USD asset default",
                "orientation": "upright (-90° X)",
            },
            target_description={
                "type": "visible_markers",
                "arrangement": "3x4 grid on dropzone",
                "count": 12,
            },
            implementation=TaskImplementationSpec(
                ee_height_for_move=0.45 / stage_units,
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                },
            ),
            verification_description={"spatial_check": "is_on_top + is_vertical"},
            rationale={
                "create_strategy": "Default sequential pairing \u2014 all items are same type, no matching needed",
                "spatial_check_fn": "Cracker boxes must be placed on the target marker (is_on_top) and remain upright (is_vertical) after placement",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
