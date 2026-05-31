import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from task_verification import is_on_top, is_vertical

logger = logging.getLogger(__name__)


class TableTaskSugarBoxGrid(UR10MultiPickPlaceTask):
    """Pick incrementally-spawned sugar boxes from the bin and place them
    vertically in a 3x4 grid on the dropzone."""

    DEFAULT_TASK_NAME = "table_task_sugar_box_grid"

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
            ConveyorPositionGenerator,
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
            SpatialTriggerConfig,
            SpatialTriggerRegion,
        )
        from pxr import Gf
        from table_setup import (
            BIN_INNER_REGION,
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

        # --- Pick strategy: sugar boxes spawned one at a time at bin center ---
        upright_orientation = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        )

        pick_z = ITEM_SPAWN_REFERENCE_Z + 0.0515 / 2 + 0.025

        pick_pos_gen = ConveyorPositionGenerator(
            center_x=BIN_X_COORD,
            center_y=BIN_Y_COORD,
            z=pick_z,
            spacing=0.0,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("sugar_box"),
            orientation_strategy=FixedValue(upright_orientation),
            color_strategy=None,
        )

        # Spatial trigger: spawn next box 2.2s after the previous one
        # leaves the bin region (picked up and transported away).
        pick_trigger = SpatialTriggerConfig(
            region=SpatialTriggerRegion(
                min_x=BIN_INNER_REGION.min_x,
                max_x=BIN_INNER_REGION.max_x,
                min_y=BIN_INNER_REGION.min_y,
                max_y=BIN_INNER_REGION.max_y,
            ),
            initial_count=1,
            items_per_batch=1,
            invert=True,
            trigger_delay=2.2,
        )

        # --- Target strategy: 3x4 grid of hidden markers on the dropzone ---
        RECT_HEIGHT = 0.002
        dx = -0.12
        dy = 0.10
        grid_w = 3
        grid_l = 4
        center_grid_x = DROPZONE_X + (grid_w - 1) * dx / 2
        center_grid_y = DROPZONE_Y + (grid_l - 1) * dy / 2
        center_grid_z = DROPZONE_Z + 0.001 + RECT_HEIGHT / 2

        target_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, center_grid_z]),
            rows=grid_l,
            cols=grid_w,
            spacing_x=dx,
            spacing_y=dy,
            randomize=False,
        )

        target_scale = np.array([0.06, 0.06, RECT_HEIGHT]) / stage_units

        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("marker"),
            color_strategy=FixedValue("green"),
            scale_strategy=FixedValue(target_scale),
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
                "Pick incrementally-spawned sugar boxes from the bin center "
                "and place them vertically onto a 3x4 grid of green markers "
                "on the dropzone. A new box spawns 2.2s after the previous "
                "one is removed from the bin."
            ),
            pick_generation_strategy=pick_strategy,
            pick_count=12,
            pick_spatial_trigger_config=pick_trigger,
            # Sentinel non-zero conveyor_speed gates spatial-trigger
            # replenishment; physical conveyor must stay stationary, so the
            # physics-side speed is pinned to 0.0 explicitly below (overriding
            # the ambient passthrough that would otherwise inherit this sentinel).
            conveyor_speed=1e-6,
            conveyor_falloff_enabled=False,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root, conveyor_speed=0.0
            ),
            spatial_check_fn=_sugar_box_spatial_check,
            scenario={
                "source": "bin",
                "destination": "dropzone_grid",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["sugar_box"],
                "count": 12,
                "arrangement": "incremental, one at a time at bin center",
                "colors": "USD asset default (yellow)",
                "orientation": "upright (-90° X)",
                "spawning": "SpatialTriggerConfig: invert=True on bin region, 2.2s trigger_delay",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "3x4 grid on dropzone (sequential order)",
                "count": 12,
                "virtual": True,
            },
            verification_description={"spatial_check": "is_on_top + is_vertical"},
            rationale={
                "create_strategy": (
                    "Default sequential pairing — all items are identical "
                    "sugar boxes, no matching needed."
                ),
                "spatial_check_fn": (
                    "Sugar boxes must rest on the target marker (is_on_top) "
                    "and remain upright (is_vertical, 15° tilt) after placement."
                ),
                "conveyor_speed": (
                    "Set to 1e-6 (sentinel) to allow SpatialTriggeredItemScheduler "
                    "replenishment; physical conveyor is stationary."
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
