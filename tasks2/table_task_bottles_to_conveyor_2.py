import logging
from typing import Optional

import numpy as np

from dynamic_top_pick_strategy import DynamicTopPickStrategy
from multi_pickplace_task import UR10MultiPickPlaceTask
from task_verification import is_vertical, is_within

logger = logging.getLogger(__name__)


class TableTaskBottlesToConveyor2(UR10MultiPickPlaceTask):
    """Pick bottles from the pick bin (2 stacked layers) and place into pads on the conveyor.

    Uses the cortex-style behaviour tree (MotionCommand-based,
    threshold-checked completion) and ``DynamicTopPickStrategy`` for JIT
    selection on both sides:
      - Pick side: highest-Z settled bottle each tick, with post-grasp
        latching.
      - Target side (inherited from ConveyorProximityStrategy):
        lowest-Y reachable unoccupied pad each tick, with place-phase
        latching and re-latch-on-falloff.

    - Sources (pick bin): 4x2 grid stacked 2 layers high (16 madara_bottles total).
      The bin is filled to capacity; only 10 are picked.
    - Targets (conveyor): initial row of madara_pads in a row along Y with positional
      jitter, replenished one at a time at the +Y spawn end (with x-jiggle) whenever
      the conveyor has carried every existing pad past Y_THRESHOLD.  Replenishment
      uses ``SpatialTriggerConfig`` and is suppressed when the conveyor is stationary.
    - Falloff monitor captures spatial-check snapshots 2 cm before the edge.
    """

    def __init__(
        self,
        task_name: str = "table_task_bottles_to_conveyor_2",
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        from isaacsim.core.utils import rotations
        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import (
            ConveyorPositionGenerator,
            FixedValue,
            GridPositionGenerator,
            IncrementalGenerationConfig,
            ItemGenerator,
            LayeredPositionGenerator,
            SpatialTriggerConfig,
            SpatialTriggerRegion,
        )
        from pxr import Gf
        from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
        from table_setup import setup_two_tables
        from env_config_values import (
            BIN_INNER_REGION,
            BIN_X_COORD,
            BIN_Y_COORD,
            CONVEYOR_END_Y,
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            ITEM_SPAWN_REFERENCE_Z,
            DEFAULT_CONVEYOR_SPEED,
            Region2D,
            make_z_reachability_check,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        # --- Pick Strategy ---
        # 4x2 grid in bin, stacked 2 layers high (16 bottles total).
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
        # 10 madara_pads in a row along the conveyor (Y axis) with jitter.
        pad_z = DROPZONE_Z + 0.002
        spacing = 0.10 / stage_units
        # Conveyor speed: None → stationary; set to DEFAULT_CONVEYOR_SPEED
        # (import from env_config_values) to make the belt move.
        conveyor_speed: Optional[float] = DEFAULT_CONVEYOR_SPEED * 0.85
        conveyor_offset = 0.6 if conveyor_speed else 0.2

        # Shift the pad row slightly toward the -X edge of the VISIBLE
        # conveyor belt so the robot has more wrist clearance during
        # the place descent.  Note: the invisible collision surface
        # ``conveyor_surface`` (scale [0.7, 1.6, 0.01], centre
        # X=0.125) extends X ∈ [-0.225, +0.475], but the VISIBLE belt
        # mesh (``/Isaac/Props/Conveyors/ConveyorBelt_A05.usd``,
        # rotated 90° about Z and scaled [0.8, 0.8, 0.7]) is much
        # narrower — empirically its -X edge sits around X ≈ 0.0
        # relative to the robot base.  Therefore shifts below ~0.0
        # put pads visibly off the belt (even though the collision
        # surface still supports them).
        #
        # Workaround for a stale place-phase posture config that, on
        # longer reaches, rotates the wrist into collision with the
        # belt's +X edge.  Proper fix: per-cycle posture selection
        # (follow-up).
        TARGET_ROW_X = DROPZONE_CENTER_POINT[0] - 0.12  # ≈ -0.08
        TARGET_ROW_CENTER_Y = DROPZONE_CENTER_POINT[1] + conveyor_offset
        target_pos_gen = ConveyorPositionGenerator(
            center_x=TARGET_ROW_X,
            center_y=TARGET_ROW_CENTER_Y,
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

        # --- Strategy factory: JIT top-settled pick selection ---
        # Bottles arriving incrementally can land on top of the bottle the
        # robot has already committed to.  DynamicTopPickStrategy scans
        # live world-frame Z every tick and retargets CortexMoveToPick to
        # the new highest *settled* bottle, so a bottle still dropping or
        # rolling does not cause a redirect.  Post-grasp the pick is
        # latched (via LatchCurrentPick in the cortex tree) so the arm
        # keeps carrying the bottle it actually grasped.
        #
        # pick_region: bottles that have been displaced from the bin (e.g.
        # bounced off a pad during a failed placement and landed on the
        # conveyor, or got knocked over the bin wall) are excluded from
        # selection — they are no longer reachable by the pick approach.
        # Add a ~5 cm margin around the inner footprint so a bottle
        # tilted against a wall is still considered pickable.
        _region_margin = 0.05
        _pick_region = Region2D(
            min_x=BIN_INNER_REGION.min_x - _region_margin,
            max_x=BIN_INNER_REGION.max_x + _region_margin,
            min_y=BIN_INNER_REGION.min_y - _region_margin,
            max_y=BIN_INNER_REGION.max_y + _region_margin,
        )

        def _strategy_factory(picks, targets):
            return DynamicTopPickStrategy(
                pick_objs=picks,
                target_objs=targets,
                # No min_pick_z: bottles rest *inside* the bin, whose inner
                # floor sits below ITEM_SPAWN_REFERENCE_Z in world space — a naive
                # tabletop floor would mark every bin-resting bottle
                # "below min_pick_z" and deadlock selection.  The pick_region
                # XY filter catches bottles that actually escape the bin.
                pick_region=_pick_region,
                conveyor_axis="y",
                conveyor_sign=-1,
                conveyor_end=CONVEYOR_END_Y,
            )

        # Initial row: same row of pads as before (spacing/jitter unchanged).
        # MAX_TARGETS now caps total spawns including replenishment.
        INITIAL_ROW_COUNT = 6  # row of 6 pads — same as the prior all-up-front layout
        MAX_TARGETS = 12       # cap including replenishment; CLI --target-count overrides

        # --- Spatial-trigger replenishment for targets ---
        # ``Y_THRESHOLD`` sits half a slot below the row's lead so the lead
        # initial pad (with up to ``jitter_y`` of noise) is reliably above
        # the threshold at task start — otherwise the predicate fires on
        # the first tick and a 7th pad spawns immediately on top of the row.
        #
        # ``SPAWN_Y`` sits one spacing-step *above the threshold* (not above
        # the row's lead).  When the trigger fires, the lead pad has just
        # crossed below ``Y_THRESHOLD``; releasing the new pad at
        # ``Y_THRESHOLD + spacing`` produces a clean one-spacing gap to that
        # lead pad, matching the initial row's pad-to-pad spacing.
        ROW_LEAD_Y = TARGET_ROW_CENTER_Y + (INITIAL_ROW_COUNT - 1) / 2.0 * spacing
        Y_THRESHOLD = ROW_LEAD_Y - 0.5 * spacing
        SPAWN_Y = Y_THRESHOLD + spacing  # = ROW_LEAD_Y + 0.5 * spacing

        target_pos_gen_replenish = ConveyorPositionGenerator(
            center_x=TARGET_ROW_X,
            center_y=SPAWN_Y,
            z=pad_z,
            spacing=0.0,                 # all replenishment pads at the same Y
            jitter_x=0.02 / stage_units,
            jitter_y=0.002 / stage_units,
        )
        target_strategy_replenish = ItemGenerator(
            position_generator=target_pos_gen_replenish,
            asset_type_strategy=FixedValue("madara_pad"),
            scale_strategy=FixedValue(None),
        )

        target_trigger = SpatialTriggerConfig(
            region=SpatialTriggerRegion(max_y=Y_THRESHOLD),
            initial_count=INITIAL_ROW_COUNT,
            replenishment_generation_strategy=target_strategy_replenish,
            invert=False,                 # spawn iff no pad has y > Y_THRESHOLD
            items_per_batch=1,
            min_spawn_interval=0.5,       # let a freshly spawned pad enter the region
        )

        # --- Incremental pick generation (time-based, unchanged) ---
        # Spawn bottles in batches every 0.5s; start the BT after 6 bottles.
        inc_config = IncrementalGenerationConfig(
            items_per_batch=3,
            batch_interval=0.5,
            bt_start_threshold=6,
        )

        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick bottles from 2 stacked layers in the bin and place them into carrier pads in a row on the conveyor (cortex-style BT). Bottles are spawned incrementally (1 every 0.5s); BT starts after 3 bottles.",
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            target_count=MAX_TARGETS,
            conveyor_speed=conveyor_speed,
            conveyor_falloff_snapshot_margin=0.02,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root, conveyor_speed=(conveyor_speed or 0.0)
            ),
            spatial_check_fn=_bottle_spatial_check,
            stacking_enabled=False,
            pick_incremental_config=inc_config,
            target_spatial_trigger_config=target_trigger,
            scenario={"source": "bin", "destination": "conveyor_row", "workspace": "two_tables"},
            pick_description={
                "asset_types": ["madara_bottle"],
                "count": 16,
                "arrangement": (
                    "4x2 grid in bin, stacked 2 layers high (layer_height=0.135m);"
                    " stacking is incidental — not enforced via stacking_map"
                ),
                "colors": "USD asset default",
                "orientation": "upright (-90° X)",
                "incremental": "1 bottle every 0.5s, BT starts after 3 bottles spawned",
            },
            target_description={
                "type": "carrier_pads",
                "arrangement": (
                    f"initial row of {INITIAL_ROW_COUNT} along conveyor (Y axis) with"
                    " jitter, replenished one at a time at the +Y spawn end (with"
                    " x-jiggle) whenever the conveyor has carried every existing pad"
                    " past Y_THRESHOLD"
                ),
                "count": MAX_TARGETS,
                "spacing": "0.10m with jitter_x=0.005m, jitter_y=0.005m",
                "replenishment": "spatial-trigger; max_y region predicate; conveyor-paused → no replenishment",
            },
            implementation=TaskImplementationSpec(
                create_strategy=_strategy_factory,
                tree_factory=make_cortex_task_controller_tree,
                target_reachable_fn=make_z_reachability_check(),
                # Tighter pick-approach close than the conveyor-friendly 5 mm
                # default: bottles are picked from a stationary bin and placed
                # into snug socket pads on the conveyor.  Any suction-tip air
                # gap at grasp time becomes a pendulum offset during the
                # upright→horizontal wrist rotation, which then mis-centres
                # the bottle into the socket and wedges it against the rim.
                # 2 mm forces a flush close before SUCCESS propagates to
                # CortexCloseGripper.
                pick_approach_p_thresh=0.002,
                strategy_description={
                    "class": "DynamicTopPickStrategy",
                    "pairing": "JIT on both sides — top-Z pick + conveyor-edge proximity target",
                    "details": (
                        "Pick selection: every tick, return the uncompleted bottle with"
                        " the highest settled world-frame Z (settled = bounded XY and Z"
                        " net drift across a sliding window).  Mid-motion CortexMoveToPick"
                        " redirects automatically when a newly-arrived bottle settles"
                        " higher than the current target.  Post-grasp, LatchCurrentPick"
                        " pins the committed pick name so the lift/place phases keep"
                        " carrying the bottle the gripper actually closed around."
                        " Target selection (inherited from ConveyorProximityStrategy):"
                        " per-tick pick the lowest-Y unoccupied reachable pad (closest"
                        " to the -Y fall-off edge).  LatchPlacementTarget pins the pad"
                        " at start of the place phase; if that pad then falls off the"
                        " belt, the latch is cleared and the strategy re-selects the"
                        " next most-urgent survivor so the carried bottle is redirected"
                        " rather than dropped."
                    ),
                    "tree": "cortex-style (MotionCommand-based, threshold-checked completion)",
                },
            ),
            verification_description={
                "spatial_check": "is_within + is_vertical",
            },
            rationale={
                "create_strategy": (
                    "DynamicTopPickStrategy extends ConveyorProximityStrategy"
                    " adding JIT height-aware pick selection on top of the"
                    " inherited conveyor-edge proximity target selection."
                    " Bottles land in the bin unpredictably (top of stacks"
                    " shift as new bottles arrive); pads arrive on a moving"
                    " belt and fall off the -Y edge over time.  Both sides"
                    " need JIT reselection to stay optimal, and both sides"
                    " need latching (pick after grasp, target during place)"
                    " to avoid mid-cycle swaps."
                ),
                "stacking_enabled": (
                    "Disabled — JIT selection supersedes the stacking_map-based"
                    " pick ordering.  The physical stacking of bottles in the bin"
                    " is still handled implicitly: higher-Z bottles are picked first."
                ),
                "target_reachable_fn": (
                    "make_z_reachability_check() rejects any pad whose world Z"
                    " has dropped below the belt surface (TARGET_MIN_REACHABLE_Z),"
                    " complementing the fall-off monitor's Y-edge detection."
                ),
                "conveyor_falloff_snapshot_margin": (
                    "0.02 — snapshot the pick/target spatial check 2 cm before"
                    " the pad reaches the belt edge so a PASSED state is"
                    " captured while the pad is still visible to AABB queries,"
                    " not lost to timing noise on the edge itself."
                ),
                "spatial_check_fn": "Bottles must be within their carrier pad (is_within) and remain upright (is_vertical) after placement",
                "tree_factory": "Cortex-style BT with MotionCommand-based behaviours for compatibility with Cortex motion planner",
                "pick_incremental_config": "Incremental spawning simulates bottles arriving on the line; BT starts early (after 3) so robot begins working while bottles are still appearing",
                "target_spatial_trigger_config": (
                    "Initial row of pads matches the prior all-up-front layout."
                    " Replenishment fires whenever no existing pad has y > Y_THRESHOLD"
                    " (i.e., the conveyor has carried the row past the threshold);"
                    " a fresh pad is dropped at the +Y spawn end with x-jiggle."
                    " Stationary belt → replenishment is suppressed and"
                    " more_targets_expected flips to False so the BT can complete"
                    " on the initial row."
                ),
            },
            # place_posture_config=None,
        )

        super().__init__(task_spec=spec, offset=offset, **kwargs)
