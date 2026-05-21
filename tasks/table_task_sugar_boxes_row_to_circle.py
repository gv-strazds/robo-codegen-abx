import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from task_verification import is_on_top, is_vertical

logger = logging.getLogger(__name__)


class TableTaskSugarBoxesRowToCircle(UR10MultiPickPlaceTask):
    """Pick 6 upright sugar boxes from a row in the bin and place them onto a
    circle of 6 thin white rectangular markers on the dropzone, keeping each
    box vertical."""

    DEFAULT_TASK_NAME = "table_task_sugar_boxes_row_to_circle"

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
            CircularPositionGenerator,
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
        )
        from pxr import Gf
        from table_setup import (
            BIN_X_COORD,
            BIN_Y_COORD,
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            ITEM_SPAWN_REFERENCE_Z,
            setup_two_tables,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        # --- Pick strategy: 6 sugar boxes upright in a single column inside the bin ---
        # Same -90° X recipe as TableTaskCrackerBoxes1 (works for cracker_box,
        # sugar_box, mustard_bottle, soup_can per item_generation.py).
        upright_orientation = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        )

        # Bin Y inner dimension ≈ 0.393 m. Upright sugar box footprint along Y
        # is ~0.045 m, so 5 * 0.060 + 0.045 ≈ 0.345 m fits 6 boxes comfortably.
        pick_z = ITEM_SPAWN_REFERENCE_Z + 0.0515 / 2 + 0.025
        pick_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=6,
            cols=1,
            spacing_x=0.0,
            spacing_y=0.060,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("sugar_box"),
            orientation_strategy=FixedValue(upright_orientation),
            color_strategy=None,
        )

        # --- Target strategy: 6 white rectangular markers in a circle on the dropzone ---
        RECT_THICKNESS = 0.002
        marker_scale = np.array([0.06, 0.06, RECT_THICKNESS]) / stage_units
        marker_z = DROPZONE_Z + 0.001 + RECT_THICKNESS / 2
        target_pos_gen = CircularPositionGenerator(
            center=np.array([DROPZONE_CENTER_POINT[0], DROPZONE_CENTER_POINT[1], marker_z]),
            radius=0.18,
            count=6,
            randomize=False,
        )
        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("marker"),
            color_strategy=FixedValue("white"),
            scale_strategy=FixedValue(marker_scale),
            hidden_strategy=FixedValue(True),
        )

        def _sugar_box_spatial_check(
            pick_obj, target_obj, bb_cache=None, obj_scale=None, log_failure=False,
        ):
            on_top = is_on_top(
                pick_obj, target_obj, bb_cache=bb_cache,
                obj_scale=obj_scale, log_failure=log_failure,
            )
            vertical = is_vertical(
                pick_obj, obj_scale=obj_scale, max_tilt_deg=15,
                log_failure=log_failure,
            )
            return on_top and vertical

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick 6 upright sugar boxes from a row in the bin and place them "
                "onto a circle of 6 white markers on the dropzone, keeping each "
                "box vertical."
            ),
            pick_generation_strategy=pick_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            spatial_check_fn=_sugar_box_spatial_check,
            scenario={
                "source": "bin",
                "destination": "dropzone_circle",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["sugar_box"],
                "count": 6,
                "arrangement": "6x1 column (single row along Y) in pick bin",
                "colors": "USD asset default (yellow)",
                "orientation": "upright (-90° X)",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "circle (r=0.18 m, 6 positions, evenly spaced) on dropzone",
                "count": 6,
                "virtual": True,
            },
            verification_description={"spatial_check": "is_on_top + is_vertical"},
            rationale={
                "create_strategy": (
                    "Default sequential pairing — all picks are identical sugar "
                    "boxes and all targets are identical markers."
                ),
                "spatial_check_fn": (
                    "Sugar boxes must rest on the marker (is_on_top) AND remain "
                    "upright (is_vertical, 15° tilt tolerance) after placement."
                ),
                "pick_arrangement": (
                    "spacing_y=0.060 keeps the 6-box row inside the bin's Y "
                    "inner dimension (~0.393 m) with margin on each end."
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
