import logging
import random
from typing import List, Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from task_verification import is_on_top, is_vertical

logger = logging.getLogger(__name__)


def _plan_bursts(
    total: int,
    min_size: int,
    max_size: int,
    slot_offsets: List[float],
    seed: Optional[int],
) -> List[List[float]]:
    """Plan a sequence of bursts as lists of X-offsets chosen from ``slot_offsets``.

    Each burst has between ``min_size`` and ``max_size`` items (capped by the
    remaining count and by ``len(slot_offsets)``).  Within a burst, the
    occupied slots are a random subset of ``slot_offsets``, preserving the
    ordering of the row (sorted ascending) so the visual spread across
    conveyor X stays left-to-right.  Returned list flattened sums to ``total``.
    """
    rng = random.Random(seed)
    remaining = int(total)
    bursts: List[List[float]] = []
    while remaining > 0:
        max_for_burst = min(max_size, remaining, len(slot_offsets))
        if min_size < max_for_burst:
            size = rng.randint(min_size, max_for_burst)
        else:  # avoid an error if remaining was < min_size
            size = max_for_burst
        chosen = sorted(rng.sample(slot_offsets, size))
        bursts.append(chosen)
        remaining -= size
    return bursts


class TableTaskSoupCans2(UR10MultiPickPlaceTask):
    """Place soup cans onto red rectangles arriving in 1-3 item bursts on a moving conveyor.

    Picks are the same 3x3 grid of YCB soup cans in the KLT bin as in
    ``TableTaskSoupCans1``.  Targets are not pre-placed in a grid; instead, thin
    red rectangles are released at the far-Y end of the drop zone (the same
    spawn location used for picks in ``TableTaskConveyorTypeSort``) in bursts
    of 1-3 rectangles per spawn interval, with burst size randomized per
    interval.  The conveyor runs at the library-default belt speed, carrying
    each burst toward the robot.
    """

    TOTAL_TARGETS = 18 # 9
    # Each burst may occupy 1 or 2 of the 3 row-positions across conveyor X.
    # Capped at 2 so the robot can keep up with the incoming bursts at the
    # chosen spawn interval.
    MIN_BURST = 1
    MAX_BURST = 3
    ROW_SLOTS = 3
    # Belt moves ~1.5 cm/s (DEFAULT_CONVEYOR_SPEED = -0.015 m/s); 6 s between
    # bursts gives ~9 cm travel, just shy of the 10 cm rectangle length so
    # successive bursts are visually tight without fully overlapping.
    TARGET_BATCH_INTERVAL = 6.0

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        super().pre_step(time_step_index=time_step_index, simulation_time=simulation_time)
        # Seed each newly-spawned target rectangle with a linear velocity
        # matching the belt so the dynamic cuboid starts moving with the
        # conveyor immediately (avoids the transient where surface velocity
        # has not yet coupled through friction — same trick
        # TableTaskConveyorTypeSort uses on picks).
        # Also apply a high-friction physics material so the rect doesn't
        # rotate or drift on the belt surface.
        try:
            from isaacsim.core.prims import SingleRigidPrim
        except Exception:
            return
        belt_v = self._task_spec.conveyor_speed if self._task_spec else None
        if not belt_v:
            return
        if not hasattr(self, "_target_belt_velocity_initialized"):
            self._target_belt_velocity_initialized: set = set()
        if not hasattr(self, "_target_friction_material"):
            try:
                from isaacsim.core.api.materials import PhysicsMaterial
                self._target_friction_material = PhysicsMaterial(
                    prim_path="/World/Physics_Materials/target_rect",
                    dynamic_friction=0.8,
                    static_friction=1.0,
                    restitution=0.0,
                )
            except Exception:
                self._target_friction_material = None
        targets = getattr(self, "_target_objs", None) or []
        velocity = np.array([0.0, belt_v, 0.0])
        for prim in targets:
            name = getattr(prim, "name", None)
            if name is None or name in self._target_belt_velocity_initialized:
                continue
            try:
                SingleRigidPrim(prim_path=prim.prim_path).set_linear_velocity(velocity)
                if self._target_friction_material is not None:
                    prim.apply_physics_material(self._target_friction_material)
                # Add damping and mass to reduce drift, rotation, and jitter
                from isaacsim.core.utils import prims as prim_utils
                from pxr import PhysxSchema, UsdPhysics
                usd_prim = prim_utils.get_prim_at_path(prim.prim_path)
                if usd_prim and usd_prim.IsValid():
                    physx_api = PhysxSchema.PhysxRigidBodyAPI.Apply(usd_prim)
                    physx_api.CreateLinearDampingAttr(5.0)
                    physx_api.CreateAngularDampingAttr(10.0)
                    mass_api = UsdPhysics.MassAPI.Apply(usd_prim)
                    mass_api.CreateMassAttr(0.5)  # 500 g
                self._target_belt_velocity_initialized.add(name)
            except Exception:
                pass

    def __init__(
        self,
        task_name: str = "table_task_soup_cans_2",
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        import functools

        from isaacsim.core.utils import rotations
        from isaacsim.core.utils.stage import get_stage_units
        from pxr import Gf

        from env_config_values import (
            BIN_X_COORD,
            BIN_Y_COORD,
            CONVEYOR_END_Y,
            CONVEYOR_SURFACE_TOP_Z,
            DEFAULT_CONVEYOR_SPEED,
            DROPZONE_CENTER_POINT,
            DROPZONE_Y,
            DROPZONE_Z,
            ITEM_SPAWN_REFERENCE_Z,
            make_z_reachability_check,
        )
        from conveyor_proximity_strategy import ConveyorProximityStrategy
        from item_generation import (
            FixedValue,
            GridPositionGenerator,
            IncrementalGenerationConfig,
            ItemGenerator,
            ItemSpec,
        )
        from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
        from table_setup import setup_two_tables
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        default_orientation = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        )

        # --- Pick strategy: identical to TableTaskSoupCans1 (3x3 soup cans in bin) --
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

        # --- Target strategy: custom burst generator -----------------------------
        # Spawn rectangles at the far-Y end of the drop zone (same landmark used
        # for picks in TableTaskConveyorTypeSort: twice the drop-zone's half-depth
        # past its centerline along +Y).
        dropzone_half_depth = DROPZONE_CENTER_POINT[1] - DROPZONE_Y
        far_y = DROPZONE_CENTER_POINT[1] + 2.0 * dropzone_half_depth
        centerline_x = DROPZONE_CENTER_POINT[0]

        # Targets must be dynamic (not FixedCuboid) so the conveyor's surface
        # velocity can carry them via friction — hence "cube" asset_type
        # (DynamicCuboid) rather than "rect" (FixedCuboid).  1 cm now works
        # reliably against the thicker 1 cm conveyor collision surface (see
        # CONVEYOR_SURFACE_THICKNESS in env_config_values).
        RECT_HEIGHT = 0.01
        rect_scale = np.array([0.1, 0.1, RECT_HEIGHT]) / stage_units
        # Spawn rect bottom ~1 mm above the conveyor collision surface top
        # so there's no free-fall gap on spawn (which used to cause initial
        # penetration into the thin 1 mm surface).  The conveyor surface
        # top is defined in env_config_values as ``CONVEYOR_SURFACE_TOP_Z``.
        target_z = CONVEYOR_SURFACE_TOP_Z + 0.001 + RECT_HEIGHT / 2
        # Fixed 3-slot row across conveyor X; each burst fills 1-2 of these
        # slots (randomly chosen).  0.14 m half-width keeps the three 0.1 m
        # rectangles inside the ~0.42 m wide drop-zone X extent with margin.
        dx_max = 0.14
        row_slots = self.ROW_SLOTS
        slot_offsets = [
            centerline_x + (i - (row_slots - 1) / 2) * (2 * dx_max / (row_slots - 1))
            for i in range(row_slots)
        ]

        total_targets = self.TOTAL_TARGETS
        min_burst = self.MIN_BURST
        max_burst = self.MAX_BURST

        class BurstRectTargetGenerator:
            """Emit ``total`` thin red rectangles pre-grouped into random bursts.

            Bursts are planned in ``generate()`` from the seed passed by the
            scheduler (same seed the rest of the task uses), so mock and
            real simulation paths see the same plan.  Each burst occupies
            1-2 of the three fixed slots across conveyor X; the exact X
            positions used are a random subset of ``slot_offsets`` per burst.
            """

            def __init__(self):
                self.bursts: List[List[float]] = []

            @property
            def group_sizes(self) -> List[int]:
                return [len(b) for b in self.bursts]

            def generate(self, count_range=None, seed=None):
                effective_total = total_targets
                if count_range is not None:
                    if isinstance(count_range, int):
                        effective_total = count_range
                    elif isinstance(count_range, tuple):
                        effective_total = count_range[0]
                self.bursts = _plan_bursts(
                    total=effective_total,
                    min_size=min_burst,
                    max_size=max_burst,
                    slot_offsets=slot_offsets,
                    seed=seed,
                )
                specs: List[ItemSpec] = []
                idx = 0
                for burst in self.bursts:
                    for x in burst:
                        specs.append(ItemSpec(
                            asset_type="cube",
                            position=np.array([x / stage_units,
                                                far_y / stage_units,
                                                target_z / stage_units]),
                            scale=rect_scale,
                            color="red",
                            name=f"target_rect_{idx}",
                        ))
                        idx += 1
                return specs[:effective_total]

        target_strategy = BurstRectTargetGenerator()

        # --- Incremental config (initial items_per_batch is overwritten in
        # set_up_scene to match the first planned group's size) ------------------
        target_inc_config = IncrementalGenerationConfig(
            items_per_batch=1,
            batch_interval=self.TARGET_BATCH_INTERVAL,
            bt_start_threshold=1,
        )

        # --- Verification: on-target + upright ---------------------------------
        def _soup_can_spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None,
                                    log_failure=False):
            on_top = is_on_top(pick_obj, target_obj, bb_cache=bb_cache,
                               obj_scale=obj_scale, log_failure=log_failure)
            vertical = is_vertical(
                pick_obj, obj_scale=obj_scale, max_tilt_deg=15,
                log_failure=log_failure,
            )
            return on_top and vertical

        conveyor_speed = DEFAULT_CONVEYOR_SPEED * 1.3

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick soup cans from the bin and place them upright onto thin"
                " red rectangles that arrive in 1-3 item bursts on a moving"
                " conveyor."
            ),
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            target_count=total_targets,
            target_incremental_config=target_inc_config,
            conveyor_speed=conveyor_speed,
            conveyor_falloff_snapshot_margin=0.02,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root, conveyor_speed=conveyor_speed,
            ),
            spatial_check_fn=_soup_can_spatial_check,
            scenario={
                "source": "bin",
                "destination": "conveyor_dynamic_targets",
                "workspace": "two_tables_moving_conveyor",
            },
            pick_description={
                "asset_types": ["soup_can"],
                "count": 9,
                "arrangement": "3x3 grid in pick bin",
                "colors": "USD asset default",
                "orientation": "upright (-90 deg X)",
            },
            target_description={
                "type": "visible_red_rectangles_on_conveyor",
                "arrangement": (
                    "spawned dynamically at far-Y end of drop zone; each"
                    " burst fills 1 or 2 of three fixed X-row slots on the"
                    " conveyor (randomly chosen per burst); moving conveyor"
                    " carries each burst toward the robot"
                ),
                "count": total_targets,
                "incremental": (
                    f"1-{max_burst} per {self.TARGET_BATCH_INTERVAL} s"
                    f" interval across {row_slots} fixed X slots;"
                    " BT starts after the first burst is released"
                ),
            },
            implementation=TaskImplementationSpec(
                target_reachable_fn=make_z_reachability_check(),
                create_strategy=lambda pick_objs, target_objs: ConveyorProximityStrategy(
                    pick_objs, target_objs,
                    conveyor_axis="y",
                    conveyor_sign=-1,
                    conveyor_end=CONVEYOR_END_Y,
                ),
                tree_factory=functools.partial(
                    make_cortex_task_controller_tree,
                    # Tighten the Z-axis check during DownToInsert so the
                    # can is released within ~5 mm of the pad's top surface
                    # rather than the ~2 cm allowed by the loose 3D
                    # threshold (steady-state RmpFlow Z error otherwise
                    # leaves the can hovering and visibly bouncing on
                    # release).
                    down_to_insert_z_thresh=0.005,
                ),
                # Disable the module-level place posture for this task — see
                # rationale["place_posture_config"] below.
                place_posture_config=None,
                strategy_description={
                    "class": "ConveyorProximityStrategy",
                    "pairing": (
                        "JIT selection by conveyor-edge proximity: each pick is"
                        " paired with the lowest-Y reachable unoccupied target"
                        " at the time of selection; target is latched at start"
                        " of place phase to avoid mid-descent swaps"
                    ),
                    "tree": "cortex-style (MotionCommand-based, threshold-checked completion)",
                },
            ),
            verification_description={"spatial_check": "is_on_top + is_vertical"},
            rationale={
                "create_strategy": (
                    "ConveyorProximityStrategy: targets ride a moving conveyor"
                    " toward a -Y fall-off edge, so the most urgent (lowest-Y)"
                    " unoccupied target should always be preferred over safer"
                    " just-arrived targets. JIT selection maximises the chance"
                    " that imminent drop-offs are filled in time; latching at"
                    " start of place phase keeps mid-descent tracking stable."
                ),
                "target_generation_strategy": (
                    "Custom BurstRectTargetGenerator so burst sizes can be"
                    " pre-planned with a seeded RNG and positions within each"
                    " burst spread across conveyor X to avoid overlap"
                ),
                "target_incremental_config": (
                    "Enables time-based target spawning; items_per_batch is"
                    " overwritten per burst via a scheduler wrapper installed"
                    " in set_up_scene"
                ),
                "conveyor_speed": (
                    "DEFAULT_CONVEYOR_SPEED — user requested default belt"
                    " speed for the moving conveyor variant"
                ),
                "spatial_check_fn": (
                    "Soup cans must land on the target rectangle (is_on_top)"
                    " and remain upright (is_vertical) after placement"
                ),
                "tree_factory": (
                    "Cortex-style BT with MotionCommand-based behaviours."
                    " Adds down_to_insert_z_thresh=0.005 so the descent"
                    " SUCCESS check requires the EE to be within 5 mm"
                    " of the commanded Z, releasing the can close to"
                    " contact rather than at the ~2 cm hover allowed by"
                    " the default loose 3D threshold."
                ),
                "place_posture_config": (
                    "Explicitly disables the null-space posture bias"
                    " (place_posture_config=None) — the module-level"
                    " default is hand-tuned for the bottle task and its"
                    " biased arm configuration is not appropriate for"
                    " soup cans being placed upright on thin rectangles."
                ),
            },
        )

        super().__init__(task_spec=spec, offset=offset, **kwargs)

    # ------------------------------------------------------------------
    # Scene setup: plan bursts + wrap scheduler
    # ------------------------------------------------------------------

    def set_up_scene(self, scene) -> None:
        # Pre-invoke the generator so its burst plan is materialized using
        # the finalized seed; this lets us size the scheduler's initial
        # batch to match the first burst.  The scheduler re-invokes
        # generate() internally with the same seed so the resulting item
        # sequence matches.
        generator = self._task_spec.target_generation_strategy
        generator.generate(
            count_range=self._task_spec.target_count, seed=self._task_spec.seed,
        )
        group_sizes = generator.group_sizes
        logger.info(
            "TableTaskSoupCans2: planned target bursts = %s (sum=%d)",
            group_sizes, sum(group_sizes),
        )
        if self._task_spec.target_incremental_config is not None:
            self._task_spec.target_incremental_config.items_per_batch = group_sizes[0]

        super().set_up_scene(scene)

        # Wrap the incremental target scheduler so subsequent releases pull
        # their batch size from the pre-planned burst sequence.  The
        # scheduler lives on the spawner staged by set_up_scene.
        spawner = self._configurator._staged_spawner
        if spawner is not None and spawner.target_scheduler is not None:
            spawner.target_scheduler = _BurstBatchSchedulerWrapper(
                spawner.target_scheduler, group_sizes, start_group_idx=1,
            )


class _BurstBatchSchedulerWrapper:
    """Proxy around ``IncrementalItemScheduler`` that overrides ``tick()``.

    Before each underlying tick, the wrapper mutates
    ``inner._config.items_per_batch`` to the next pre-planned burst size so
    each batch_interval releases 1-3 targets according to the plan.  All
    other scheduler attributes/methods are delegated to the inner instance.
    """

    def __init__(self, inner, group_sizes, start_group_idx=0):
        self._inner = inner
        self._group_sizes = list(group_sizes)
        self._group_idx = int(start_group_idx)

    def tick(self, current_time):
        if self._group_idx < len(self._group_sizes):
            self._inner._config.items_per_batch = self._group_sizes[self._group_idx]
        released = self._inner.tick(current_time)
        if released:
            self._group_idx += 1
        return released

    # Delegate everything else to the wrapped scheduler.
    def __getattr__(self, name):
        return getattr(self._inner, name)
