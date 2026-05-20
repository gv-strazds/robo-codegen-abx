import logging
from typing import Optional

import numpy as np

from bottle_conveyor_pick_strategy import BottleConveyorPickStrategy
from multi_pickplace_task import UR10MultiPickPlaceTask
from task_verification import is_vertical, is_within

logger = logging.getLogger(__name__)


class TableTaskBottlesToConveyor(UR10MultiPickPlaceTask):
    """Pick bottles from the pick bin (2 stacked layers) and place into pads on the conveyor.

    - Sources (pick bin): 4x2 grid stacked 2 layers high (16 madara_bottles total).
      The bin is filled to capacity; only 9 are picked.
    - Targets (conveyor): 9 madara_pads in a row along Y with positional jitter.
    - Strategy: BottleConveyorPickStrategy — JIT proximity target selection
      (lowest-Y reachable unoccupied pad each tick) with stacking_map for
      top-down pick ordering and HORIZONTAL_DROP_QUAT drop orientation.
    """

    DEFAULT_TASK_NAME = "table_task_bottles_to_conveyor"

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
            LayeredPositionGenerator,
        )
        from pxr import Gf
        from table_setup import setup_two_tables
        from env_config_values import (
            BIN_X_COORD,
            BIN_Y_COORD,
            CONVEYOR_END_Y,
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            ITEM_SPAWN_REFERENCE_Z,
            DEFAULT_CONVEYOR_SPEED,
            make_z_reachability_check,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        # --- Pick Strategy ---
        # 4x3 grid in bin, stacked 2 layers high (24 bottles total).
        # Upright orientation: -90 deg X rotation (standard for USD assets).
        default_orientation = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        )

        pick_z = ITEM_SPAWN_REFERENCE_Z + (0.03 / stage_units) / 2 + 0.025

        base_pick_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=2,
            cols=4,
            spacing_x=0.08,
            spacing_y=0.15,
            randomize=False,
        )

        # Full bottle height: rest_height + top_surface_height ≈ 0.063 + 0.072 = 0.135m
        bottle_layer_height = 0.135

        pick_pos_gen = LayeredPositionGenerator(
            base_generator=base_pick_gen,
            num_layers=2,
            layer_height=bottle_layer_height,
        )

        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("madara_bottle"),
            orientation_strategy=FixedValue(default_orientation),
            scale_strategy=FixedValue(np.array([1.0, 1.0, 1.0])),
            color_strategy=None,
        )

        # --- Target Strategy ---
        # 9 madara_pads in a row along the conveyor (Y axis) with jitter.
        pad_z = DROPZONE_Z + 0.002
        spacing = 0.10 / stage_units
        # Conveyor speed: None → stationary; set to DEFAULT_CONVEYOR_SPEED
        # (import from env_config_values) to make the belt move.
        conveyor_speed: Optional[float] = DEFAULT_CONVEYOR_SPEED * 0.85
        conveyor_offset = 0.6 if conveyor_speed else 0.2

        target_pos_gen = ConveyorPositionGenerator(
            center_x=DROPZONE_CENTER_POINT[0],
            center_y=DROPZONE_CENTER_POINT[1]+conveyor_offset,
            z=pad_z,
            spacing=spacing,
            jitter_x=0.02 / stage_units,
            jitter_y=0.005 / stage_units,
        )

        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("madara_pad"),
            scale_strategy=FixedValue(None),
        )

        # --- Verification ---
        def _bottle_spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None):
            return (
                is_within(pick_obj, target_obj, bb_cache, obj_scale)
                and is_vertical(pick_obj, obj_scale=obj_scale, max_tilt_deg=15)
            )

        # --- Strategy factory: JIT proximity target selection + stacking-driven pick order ---
        def _strategy_factory(picks, targets):
            from multi_pick_strategy import compute_stacking_map
            stacking_map = compute_stacking_map(picks)
            return BottleConveyorPickStrategy(
                pick_objs=picks, target_objs=targets,
                conveyor_axis="y", conveyor_sign=-1,
                conveyor_end=CONVEYOR_END_Y,
                stacking_map=stacking_map,
            )

        MAX_TARGETS = 9
        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick bottles from 2 stacked layers in the bin and place them into carrier pads in a row on the conveyor.",
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            target_count=MAX_TARGETS,
            conveyor_speed=conveyor_speed,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root, conveyor_speed=(conveyor_speed or 0.0)
            ),
            spatial_check_fn=_bottle_spatial_check,
            stacking_enabled=True,
            scenario={"source": "bin", "destination": "conveyor_row", "workspace": "two_tables"},
            pick_description={
                "asset_types": ["madara_bottle"],
                "count": 16,
                "arrangement": "4x2 grid in bin, stacked 2 layers high (layer_height=0.135m)",
                "colors": "USD asset default",
                "orientation": "upright (-90° X)",
            },
            target_description={
                "type": "carrier_pads",
                "arrangement": "row of 9 along conveyor (Y axis) with jitter",
                "count": MAX_TARGETS,
                "spacing": "0.10m with jitter_x=0.02m, jitter_y=0.005m",
            },
            implementation=TaskImplementationSpec(
                create_strategy=_strategy_factory,
                target_reachable_fn=make_z_reachability_check(),
                strategy_description={
                    "class": "BottleConveyorPickStrategy",
                    "pairing": "JIT lowest-Y target selection; stacking_map for top-down pick order",
                    "details": (
                        "Target side: per-tick selection of the lowest-Y reachable"
                        " unoccupied pad (closest to the -Y fall-off edge / robot),"
                        " inherited from ConveyorProximityStrategy._jit_select."
                        " Pick side: stacking_map drives top-layer-first picking via"
                        " _apply_stacking_order.  Bottle-on-side drop orientation"
                        " (HORIZONTAL_DROP_QUAT) preserved from BottlePickStrategy."
                    ),
                },
            ),
            verification_description={
                "spatial_check": "is_within + is_vertical",
            },
            rationale={
                "create_strategy": (
                    "BottleConveyorPickStrategy combines JIT proximity target"
                    " selection with bottle drop orientation.  The prior"
                    " BottlePickStrategy used MultiPickStrategy's sequential"
                    " default pairing, which after _apply_stacking_order /"
                    " _reassign_targets_by_picking_order sent the first (topmost)"
                    " bottle to the +Y-end pad instead of the lead -Y pad."
                    " JIT proximity re-pairs every tick, so the first bottle now"
                    " correctly targets the lead pad nearest the robot."
                ),
                "target_reachable_fn": (
                    "make_z_reachability_check() — the conveyor is moving and"
                    " pads will fall off the -Y edge over time.  Without the Z"
                    " filter, _jit_select would keep returning a pad that has"
                    " physically dropped below the belt surface."
                ),
                "stacking_enabled": (
                    "Bottles are stacked 2 layers high — stacking_map enforces"
                    " top-layer picked first.  Base ConveyorProximityStrategy"
                    " (unlike DynamicTopPickStrategy) has no top-Z pick selection,"
                    " so stacking_map remains the source of pick ordering."
                ),
                "spatial_check_fn": "Bottles must be within their carrier pad (is_within) and remain upright (is_vertical) after placement",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
