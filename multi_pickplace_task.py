# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# from abc import ABC, abstractmethod
import logging
import select
import sys

import random

from pathlib import Path
from typing import Callable, List, Optional, Union

import isaacsim.core.api.tasks as tasks  # tasks.BaseTask, tasks.PickPlace,
import numpy as np
from asset_data_utils import PrimGeometry  # noqa: F401 — used by task subclasses
from isaacsim.core.api.scenes.scene import Scene

from isaacsim.core.utils import prims as prim_utils

# from isaacsim.core.api.tasks import BaseTask

from isaacsim.core.utils.bounds import create_bbox_cache  # noqa: F401
from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
from isaacsim.core.utils.string import find_unique_string_name
from isaacsim.cortex.framework.cortex_utils import get_assets_root_path_or_die
from isaacsim.cortex.framework.robot import CortexUr10
from multi_pick_strategy import MultiPickStrategy
from task_context import TaskContext  # noqa: F401 — used by TaskController

from robot_controllers import UR10MultiPickPlaceController  # noqa: F401

logger = logging.getLogger(__name__)


def _build_scheduler_for_side(
    *,
    inc_config,
    spatial_config,
    generation_strategy,
    count_range,
    seed,
    side_label: str,
):
    """Build a pick or target scheduler from the two mutually-exclusive configs.

    Returns ``IncrementalItemScheduler`` / ``SpatialTriggeredItemScheduler``
    or ``None`` when neither config is set.  ``side_label`` (``"pick"`` /
    ``"target"``) is used only for the conflict error message.
    """
    if inc_config is not None and spatial_config is not None:
        raise ValueError(
            f"TaskSpec.{side_label}_incremental_config and "
            f"TaskSpec.{side_label}_spatial_trigger_config "
            "are mutually exclusive; configure only one."
        )
    if spatial_config is not None:
        from item_generation import SpatialTriggeredItemScheduler
        return SpatialTriggeredItemScheduler(
            primary_generator=generation_strategy,
            config=spatial_config,
            count_range=count_range,
            seed=seed,
        )
    if inc_config is not None:
        from item_generation import IncrementalItemScheduler
        return IncrementalItemScheduler(
            generator=generation_strategy,
            config=inc_config,
            count_range=count_range,
            seed=seed,
        )
    return None


def _check_stdin_enter():
    """Return True if ENTER was pressed (non-blocking)."""
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()  # consume the line
        return True
    return False


class UR10MultiPickPlaceTask(tasks.BaseTask):
    """Bridges from IsaacSim BaseTask API
     (used by isaacsim.core.api.world to interface with tasks added to a running simulation scene)
      to our MultiPickPlaceController API
      via a TaskContext instance and forwarding calls to task_step().

    Args:
        name (str, optional): [description]. Defaults to "ur10_stacking".
        offset (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    DEFAULT_TASK_NAME: str = "ur10_stacking"

    def _customize_spec(self, spec: "TaskSpec") -> "TaskSpec":
        """Hook for subclasses to amend a TaskSpec before it reaches __init__.

        v1 task subclasses build a TaskSpec inside their own __init__ and call
        ``spec = self._customize_spec(spec)`` just before delegating to
        ``super().__init__``.  v2 cortex-tree subclasses override this hook to
        return ``dataclasses.replace(spec, tree_factory=..., ...)``.  Default
        is identity.
        """
        return spec

    def __init__(
        self,
        task_name: Optional[str] = None,
        task_spec: Optional["TaskSpec"] = None,
        offset: Optional[np.ndarray] = None,
        pick_count: Optional[Union[int, tuple]] = None,
        target_count: Optional[Union[int, tuple]] = None,
        seed: Optional[int] = None,
        randomize: Optional[bool] = None,
        assets_root_path: Optional[str] = None,
        teleport_mode: bool = False,
        incremental_checks: bool = True,
        pause_after_cycle: bool = False,
        dynamic_pick_interval: Optional[float] = None,
        dynamic_target_interval: Optional[float] = None,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        # When a TaskSpec is provided, extract generation-related params from it.
        # Individual params (offset, teleport_mode, etc.) are still passed directly.
        if task_spec is not None:
            task_name = task_spec.task_name
            pick_generation_strategy = task_spec.pick_generation_strategy
            target_generation_strategy = task_spec.target_generation_strategy
            # Resolve pick_count / target_count / seed from the TaskSpec when
            # the caller didn't supply them; otherwise mirror the caller's
            # value back onto the TaskSpec so downstream reads (incremental
            # schedulers, virtual target generation, TaskController) see the
            # CLI-provided value instead of the spec's literal default.
            if pick_count is None:
                pick_count = task_spec.pick_count
            else:
                task_spec.pick_count = pick_count
            if target_count is None:
                target_count = task_spec.target_count
            else:
                task_spec.target_count = target_count
            if seed is None:
                seed = task_spec.seed
            else:
                task_spec.seed = seed

            # CLI override for incremental pick/target batch intervals.
            # Silently ignored when the spec has no matching scheduler config.
            if (dynamic_pick_interval is not None
                    and task_spec.pick_incremental_config is not None):
                task_spec.pick_incremental_config.batch_interval = float(dynamic_pick_interval)
            if (dynamic_target_interval is not None
                    and task_spec.target_incremental_config is not None):
                task_spec.target_incremental_config.batch_interval = float(dynamic_target_interval)
        else:
            pick_generation_strategy = None
            target_generation_strategy = None

        super().__init__(
            name=task_name, offset=offset
        )  # super=isaacsim.core.api.tasks.BaseTask
        self._task_spec = task_spec
        self._robot = None
        self._teleport_mode = teleport_mode
        self._incremental_checks = incremental_checks
        self._containment_check = False
        self._pause_after_cycle = pause_after_cycle
        self._paused_for_inspection = False
        self._pause_cycle_count = 0
        self._prev_completed_picks = set()
        self._pending_incremental_pick_names: List[str] = []
        # Optional callback fired once per failed incremental verification
        # check.  Signature: ``(check: PlacementCheck, sim_time: float) -> None``.
        self._on_incremental_check_fail: Optional[Callable] = None
        # Public, concise, natural-language description of the task scenario.
        # Derived from TaskSpec.task_description, with a generic default fallback.
        self.task_description: Optional[str] = (
            (task_spec.task_description if task_spec else None)
            or "UR10 multi-object pick-and-place task"
        )
        self._assets_root_path = (
            assets_root_path
            if assets_root_path is not None
            else get_assets_root_path_or_die()
        )

        # Create SimulationConfigurator to manage scene objects and verification
        from simulation_configurator import SimulationConfigurator
        self._configurator = SimulationConfigurator(
            pick_generation_strategy=pick_generation_strategy,
            target_generation_strategy=target_generation_strategy,
            pick_count=pick_count,
            target_count=target_count,
            seed=seed,
            assets_root_path=self._assets_root_path,
        )

        # Apply randomization override if provided
        if randomize is not None:
            pick_gen = self._configurator.pick_generation_strategy
            if pick_gen and hasattr(pick_gen.position_generator, "randomize"):
                pick_gen.position_generator.randomize = randomize
            target_gen = self._configurator.target_generation_strategy
            if target_gen and hasattr(target_gen.position_generator, "randomize"):
                target_gen.position_generator.randomize = randomize

        # Apply task_spec settings for containment, stacking, ee_height
        if task_spec is not None:
            if task_spec.containment_check:
                self._containment_check = True
            if task_spec.stacking_enabled:
                self._stacking_enabled = True
            impl = task_spec.implementation
            if impl is not None and impl.ee_height_for_move is not None:
                self._ee_height_for_move = impl.ee_height_for_move

        self._last_simulation_time = 0.0

        # BT startup delay: hold off ticking the BT for this many sim-time
        # seconds after the BT-start gate first opens, so gravity-spawned
        # items can settle before the first pick (especially in teleport mode).
        self._startup_delay_seconds: Optional[float] = (
            task_spec.implementation.startup_delay_seconds
            if task_spec is not None and task_spec.implementation is not None
            else None
        )
        self._bt_gate_open_time: Optional[float] = None
        self._startup_delay_logged: bool = False

        # Conveyor fall-off monitor (constructed in post_reset when enabled)
        self._falloff_monitor = None

        # Track if this task has completed (either placed all picks or exhausted targets)
        self._task_done = False
        self._task_exit = False
        self._steps_after_done = 100
        self._extra_step_countdown = self._steps_after_done

        # TODO: WARNING -THIS IS A HACK!
        file_dir = Path(__file__).parent.absolute()
        self._ur10_asset_path = str(
            file_dir.joinpath("SimEnvs/Collected_ur10_bin_filling/ur10_bin_filling.usd")
        )
        logger.warning(f"USD base file={self._ur10_asset_path}")

        return

    # Pick / target object lists are held by SimulationConfigurator
    # during set_up_scene; the strategy takes over once it's built (and
    # the context exposes them as ``pick_objs`` / ``target_objs``).
    # These read-only forwarders hide the handoff.

    @property
    def _pick_objs(self):
        if self._configurator is not None:
            return self._configurator._pick_objs
        return self._task_context.pick_objs

    @property
    def _target_objs(self):
        if self._configurator is not None:
            return self._configurator._target_objs
        return self._task_context.target_objs

    def set_up_scene(self, scene: Scene) -> None:
        """Loads the stage USD and adds the robot and task objects to the World's scene.

        Args:
            scene (Scene): The world's scene.
        """
        # super().set_up_scene(scene)
        self._scene = scene  # isaacsim/core/api/tasks/base_task.py

        # INCLUDED in ur10_table_scene .usd  #scene.add_default_ground_plane() z_position=-0.5)
        self._robot = self.set_robot()
        # Register with CortexWorld so commanders are stepped automatically
        from isaacsim.cortex.framework.cortex_world import CortexWorld
        cortex_world = CortexWorld.instance()
        if cortex_world is not None:
            cortex_world.add_robot(self._robot)
        else:
            scene.add(self._robot)
        self._task_objects[self._robot.name] = self._robot
        self.setup_workspace(scene)

        spec = self._task_spec
        pick_scheduler = _build_scheduler_for_side(
            inc_config=spec.pick_incremental_config if spec else None,
            spatial_config=spec.pick_spatial_trigger_config if spec else None,
            generation_strategy=spec.pick_generation_strategy if spec else None,
            count_range=spec.pick_count if spec else None,
            seed=spec.seed if spec else None,
            side_label="pick",
        )
        if pick_scheduler is not None:
            initial_items = pick_scheduler.get_initial_batch()
            new_prims = self._configurator.prim_factory.create_picks(initial_items, scene)
            for prim in new_prims:
                self._configurator._pick_objs.append(prim)
                self._task_objects[prim.name] = prim
            logger.info(
                "Incremental generation: spawned %d/%d initial pick objects",
                len(initial_items), pick_scheduler.total_count,
            )
        else:
            self._configurator.add_source_objects(scene)
            for prim in self._pick_objs:
                self._task_objects[prim.name] = prim

        target_scheduler = _build_scheduler_for_side(
            inc_config=spec.target_incremental_config if spec else None,
            spatial_config=spec.target_spatial_trigger_config if spec else None,
            generation_strategy=spec.target_generation_strategy if spec else None,
            count_range=spec.target_count if spec else None,
            seed=spec.seed if spec else None,
            side_label="target",
        )
        if target_scheduler is not None:
            initial_targets = target_scheduler.get_initial_batch()
            new_prims = self._configurator.prim_factory.create_targets(initial_targets, scene)
            for prim in new_prims:
                self._configurator._target_objs.append(prim)
                self._task_objects[prim.name] = prim
            logger.info(
                "Incremental generation: spawned %d/%d initial target objects",
                len(initial_targets), target_scheduler.total_count,
            )
        else:
            self._configurator.add_target_objects(scene)
            for prim in self._target_objs:
                self._task_objects[prim.name] = prim
        self._move_task_objects_to_their_frame()

        self._configurator.stage_spawner_from(
            pick_scheduler=pick_scheduler,
            target_scheduler=target_scheduler,
            scene=scene,
            conveyor_speed_fn=(
                (lambda: spec.conveyor_speed) if spec is not None else None
            ),
        )

    def setup_workspace(self, scene) -> None:
        """Set up the workspace (tables, bins, etc.) before object generation.

        When a TaskSpec with setup_workspace is available, calls it.
        Subclasses may still override for backward compatibility.
        """
        if self._task_spec is not None and self._task_spec.setup_workspace is not None:
            self._task_spec.setup_workspace(scene, self._assets_root_path)

    def add_source_objects(self, scene) -> None:
        """Delegate to configurator, then register objects with BaseTask tracking."""
        self._configurator.add_source_objects(scene)
        for prim in self._pick_objs:
            self._task_objects[prim.name] = prim

    def add_target_objects(self, scene) -> None:
        """Delegate to configurator, then register objects with BaseTask tracking."""
        self._configurator.add_target_objects(scene)
        for prim in self._target_objs:
            self._task_objects[prim.name] = prim

    def set_robot(self) -> CortexUr10:
        """Create and configure the CortexUr10 robot.

        Loads the scene USD (which includes the UR10 prim at /World/Scene/ur10),
        then wraps it as a CortexUr10 with built-in MotionCommander and
        SurfaceGripper.

        Returns:
            CortexUr10 instance.
        """
        ur10_robot_name = find_unique_string_name(
            initial_name="my_ur10",
            is_unique_fn=lambda x: not self.scene.object_exists(x),
        )
        add_reference_to_stage(usd_path=self._ur10_asset_path, prim_path="/World/Scene")
        self._ur10_robot = CortexUr10(
            name=ur10_robot_name, prim_path="/World/Scene/ur10",
        )

        self._ur10_robot.set_joints_default_state(
            positions=np.array(
                [-np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2, 0]
            )
        )
        return self._ur10_robot

    def _create_strategy_with(self, pick_objs, target_objs) -> MultiPickStrategy:
        """Strategy factory that forwards the (possibly combined) object lists.

        Called by TaskController.create_strategy() with the target list that
        may include virtual targets appended after scene targets.
        """
        impl = self._task_spec.implementation if self._task_spec is not None else None
        if impl is not None and impl.create_strategy is not None:
            return impl.create_strategy(pick_objs, target_objs)
        return MultiPickStrategy(pick_objs=pick_objs, target_objs=target_objs)

    def _get_placement_constraints_fn(self):
        """Hook for task-specific placement constraints.

        When a TaskSpec with placement_constraints_fn is available, uses it.
        Subclasses may still override for backward compatibility.

        Return a callable(pick_index, target_index) -> (bool, str), or None
        to use the strategy's default placement_constraints_satisfied.
        """
        if self._task_spec is not None and self._task_spec.placement_constraints_fn is not None:
            return self._task_spec.placement_constraints_fn
        return None

    def _get_box_verification_info(self):
        """Extract box verification info from task_spec.

        Returns dict with 'box_specs' list where each spec has per-box
        'inner_size' and 'height', or None if not configured.
        """
        if self._task_spec is not None and self._task_spec.box_verification_info is not None:
            return self._task_spec.box_verification_info
        return None

    @staticmethod
    def _adjust_box_specs_for_physics(box_specs):
        """Adjust box specs to account for physics settling of container prims.

        Box specs with a ``prim_path`` and ``spawn_position`` are updated:
        the prim's current world position is queried and the delta from the
        original spawn position is applied to ``floor_z`` and ``center_xy``.

        Returns a new list of (possibly adjusted) spec dicts; originals are
        not mutated.
        """
        from isaacsim.core.prims import SingleXFormPrim
        adjusted = []
        for spec in box_specs:
            prim_path = spec.get("prim_path")
            spawn_pos = spec.get("spawn_position")
            if prim_path is not None and spawn_pos is not None:
                try:
                    prim = SingleXFormPrim(prim_path=prim_path)
                    current_pos, _ = prim.get_world_pose()
                    spawn_pos = np.asarray(spawn_pos, dtype=float)
                    delta = current_pos - spawn_pos
                    if np.abs(delta[:3]).max() > 1e-4:
                        spec = dict(spec)  # shallow copy before mutating
                        spec["floor_z"] = spec["floor_z"] + delta[2]
                        spec["center_xy"] = np.asarray(spec["center_xy"], dtype=float) + delta[:2]
                        logger.debug(
                            f"Adjusted box '{spec['name']}' for physics settling: "
                            f"delta_xy=({delta[0]:.4f}, {delta[1]:.4f}), delta_z={delta[2]:.4f}"
                        )
                except Exception as e:
                    logger.warning(f"Could not query prim '{prim_path}' for box spec adjustment: {e}")
            adjusted.append(spec)
        return adjusted

    def get_task_spec(self) -> "TaskSpec":
        """Synthesize a TaskSpec from the current task state.

        Returns a TaskSpec that captures this task's configuration as a
        standalone description.  Callable fields wrap the task instance's
        methods so they work when invoked later with spawned objects.

        Can be called at any point after ``__init__`` — does not require
        scene setup or object spawning.
        """
        from task_spec import TaskImplementationSpec, TaskSpec

        task_ref = self

        # Wrap setup_workspace as (scene, assets_root_path) -> None
        def _workspace_setup(scene, assets_root_path):
            old_assets = task_ref._assets_root_path
            task_ref._assets_root_path = assets_root_path
            try:
                task_ref.setup_workspace(scene)
            finally:
                task_ref._assets_root_path = old_assets

        # Spatial check from TaskSpec (or None → defaults to is_on_top downstream)
        spatial_check = (
            self._task_spec.spatial_check_fn
            if self._task_spec is not None and self._task_spec.spatial_check_fn is not None
            else None
        )

        # Detect placement constraints: _get_placement_constraints_fn already
        # checks TaskSpec first, then falls back to subclass override
        placement_constraints = self._get_placement_constraints_fn()

        from dataclasses import replace
        source_impl = self._task_spec.implementation if self._task_spec is not None else None
        base_impl = source_impl if source_impl is not None else TaskImplementationSpec()
        impl = replace(
            base_impl,
            create_strategy=self._create_strategy_with,
            ee_height_for_move=getattr(self, "_ee_height_for_move", None),
        )

        return TaskSpec(
            task_name=self._name,
            task_description=self.task_description or "",
            pick_generation_strategy=self._configurator.pick_generation_strategy,
            target_generation_strategy=self._configurator.target_generation_strategy,
            pick_count=self._configurator._pick_count,
            target_count=(
                self._task_spec.target_count if self._task_spec
                else self._configurator._target_count
            ),
            seed=self._configurator._seed,
            setup_workspace=_workspace_setup,
            spatial_check_fn=spatial_check,
            placement_constraints_fn=placement_constraints,
            containment_check=self._containment_check,
            box_verification_info=self._get_box_verification_info(),
            stacking_enabled=getattr(self, "_stacking_enabled", False),
            scenario=self._task_spec.scenario if self._task_spec else None,
            pick_description=self._task_spec.pick_description if self._task_spec else None,
            target_description=self._task_spec.target_description if self._task_spec else None,
            verification_description=self._task_spec.verification_description if self._task_spec else None,
            rationale=self._task_spec.rationale if self._task_spec else None,
            pick_incremental_config=(
                self._task_spec.pick_incremental_config if self._task_spec else None
            ),
            target_incremental_config=(
                self._task_spec.target_incremental_config if self._task_spec else None
            ),
            pick_spatial_trigger_config=(
                self._task_spec.pick_spatial_trigger_config if self._task_spec else None
            ),
            target_spatial_trigger_config=(
                self._task_spec.target_spatial_trigger_config if self._task_spec else None
            ),
            conveyor_speed=(
                self._task_spec.conveyor_speed if self._task_spec else None
            ),
            implementation=impl,
        )

    def post_reset(self) -> None:
        """Reset task state and create TaskController → TaskContext → BT controller."""
        # Note: Gripper reset logic moved to UR10MultiPickPlaceController.reset()
        from task_controller import TaskController

        my_ur10 = self._robot
        # Reset task completion flag on reset
        self._task_done = False
        self._task_exit = False
        self._extra_step_countdown = self._steps_after_done

        # Re-arm the BT startup-delay gate so a re-run honors the delay again.
        self._bt_gate_open_time = None
        self._startup_delay_logged = False

        # Create TaskController (policy layer).  Per-task tunables (postures,
        # hover heights, watchdog timeouts, virtual-target generation, tree
        # factory, etc.) are read from the task spec inside the controller.
        # ``ee_height_for_move`` is passed explicitly because some subclasses
        # recompute it from scene state at runtime — when set, it wins over
        # the spec field.
        task_ref = self
        impl_spec = self._task_spec.implementation if self._task_spec is not None else None
        self._policy = TaskController(
            strategy_factory=lambda picks, targets: task_ref._create_strategy_with(picks, targets),
            prim_geometry=self._configurator._prim_geometry,
            ee_height_for_move=getattr(self, "_ee_height_for_move", None),
            task_spec=self._task_spec,
            impl_spec=impl_spec,
            configurator=self._configurator,  # used only if impl_spec.use_curobo == True
        )

        # Create strategy → TaskContext → BT controller.
        # ``create_strategy`` merges scene + virtual targets internally; the
        # strategy is the sole owner of the combined list afterwards.
        self._policy.create_strategy(self._pick_objs, self._target_objs)
        # Wire target reachability predicate (e.g. z-threshold for conveyor tasks)
        if impl_spec is not None and impl_spec.target_reachable_fn is not None:
            self._policy.strategy.set_target_reachable_fn(impl_spec.target_reachable_fn)
        self._task_context = self._policy.create_task_context(
            robot=my_ur10,
            teleport_mode=self._teleport_mode,
        )
        self._task_controller = self._policy.create_bt_controller(
            name="ur10_stacking_controller",
            fake_fast=self._teleport_mode,
            show_status=getattr(self, '_show_status', False),
        )

        staged = self._configurator._staged_spawner
        if staged is not None:
            if staged.more_picks_expected:
                self._policy.strategy.more_items_expected = True
            if staged.more_targets_expected:
                self._policy.strategy.more_targets_expected = True

        # Hand off all runtime state from the (setup-only) configurator
        # to the (long-lived) TaskContext, then drop the configurator.
        spec = self._task_spec
        strategy = self._policy.strategy
        verifier = self._configurator.build_verifier(
            strategy=strategy,
            spatial_check_fn=(spec.spatial_check_fn if spec is not None else None),
            placement_constraints_fn=self._get_placement_constraints_fn(),
            containment_check=self._containment_check,
            box_verification_info=self._get_box_verification_info(),
            adjust_box_specs_fn=self._adjust_box_specs_for_physics,
            on_incremental_check_fail=self._on_incremental_check_fail,
        )
        self._task_context.verifier = verifier
        # Wire the verifier's frozen-target callback into the strategy so
        # retargeting skips targets already claimed by a passing snapshot.
        strategy.set_frozen_target_names_fn(verifier.frozen_target_names)
        self._task_context.spawner = self._configurator._staged_spawner
        self._task_context._prim_geometry = self._configurator._prim_geometry
        self._configurator = None

        # Conveyor fall-off snapshot monitor. Auto-on when the task specifies
        # a non-zero conveyor_speed; explicit TaskSpec.conveyor_falloff_enabled
        # overrides. When disabled, no monitor is created and the existing
        # verification behavior is unchanged.
        self._falloff_monitor = None
        if self._task_spec is not None and self._task_spec.falloff_is_enabled():
            from conveyor_falloff_monitor import ConveyorFalloffMonitor
            from env_config_values import CONVEYOR_END_Y
            end_y = (
                self._task_spec.conveyor_end_y
                if self._task_spec.conveyor_end_y is not None
                else CONVEYOR_END_Y
            )
            strategy = self._policy.strategy
            self._falloff_monitor = ConveyorFalloffMonitor(
                strategy=strategy,
                target_objs_ref=strategy.target_objs,
                conveyor_end_y=end_y,
                snapshot_margin=self._task_spec.conveyor_falloff_snapshot_margin,
                hide_after=self._task_spec.conveyor_falloff_hide_after,
                on_snapshot=self._run_snapshot_verification,
                on_available_lost=self._record_available_target_lost,
                get_half_extent_y=self._falloff_half_extent_y,
            )
            logger.info(
                "Conveyor fall-off monitor enabled: end_y=%.3f margin=%.3f "
                "snapshot_threshold_y=%.3f hide_after=%s",
                end_y,
                self._task_spec.conveyor_falloff_snapshot_margin,
                end_y + self._task_spec.conveyor_falloff_snapshot_margin,
                self._task_spec.conveyor_falloff_hide_after,
            )

        return

    def _falloff_half_extent_y(self, prim) -> float:
        """Return a conservative Y half-extent for ``prim`` for fall-off triggering.

        Uses the cached ``PrimGeometry.local_half_extents`` on the task
        context (populated during scene setup) — taking the max of
        horizontal axes (X, Y) so the estimate is orientation-agnostic
        around Z (correct for rects and upright cans regardless of
        Z-rotation). Falls back to ``prim._local_half_extents``
        (LightweightObj mocks) or 0.0 as a last resort.
        """
        name = getattr(prim, "name", None)
        if name is not None:
            geom = self._task_context._prim_geometry.get(name)
            if geom is not None and getattr(geom, "local_half_extents", None) is not None:
                he = geom.local_half_extents
                try:
                    return float(max(he[0], he[1]))
                except Exception:
                    pass
        local_he = getattr(prim, "_local_half_extents", None)
        if local_he is not None:
            try:
                return float(max(local_he[0], local_he[1]))
            except Exception:
                pass
        return 0.0

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        """Called by CortexWorld.step() before monitors/behaviors/commanders.

        Args:
            time_step_index (int): Current simulation step index.
            simulation_time (float): Current simulation time.
        """
        super().pre_step(
            time_step_index=time_step_index, simulation_time=simulation_time
        )
        self._last_simulation_time = simulation_time
        # Publish current sim time to TaskContext so WaitForCycleTime (and
        # any other time-aware behaviour) reads CortexWorld's clock.
        # Also forward the cycle-time gate setting from the CLI (set on the
        # task instance by run_task.py); cheap per-tick attribute set.
        if self._task_context is not None:
            self._task_context.simulation_time = float(simulation_time)
            self._task_context.min_cycle_time_s = float(
                getattr(self, "_min_cycle_time_s", 0.0) or 0.0
            )

        # Strategy notification stays on the task because the spawner is
        # deliberately strategy-unaware; halt spawning once the task is
        # done so we don't admit items that can no longer be placed.
        spawner = self._task_context.spawner if self._task_context is not None else None
        if spawner is not None and not self._task_done:
            ctx = self._task_context
            result = spawner.tick(
                simulation_time,
                live_picks=ctx.pick_objs,
                live_targets=ctx.target_objs,
            )
            strategy = self._policy.strategy
            if result.new_picks:
                for prim in result.new_picks:
                    self._task_objects[prim.name] = prim
                strategy.add_incremental_picks(result.new_picks)
            if result.new_targets:
                for prim in result.new_targets:
                    self._task_objects[prim.name] = prim
                strategy.add_incremental_targets(result.new_targets)
            self._update_more_expected_flags(spawner, result)

        # Poll the conveyor fall-off monitor. Runs after incremental spawn so
        # newly released targets are picked up on the next tick, but still
        # before physics updates in this step. Skipped once the task is done
        # since no further placements can be committed.
        if self._falloff_monitor is not None and not self._task_done:
            self._falloff_monitor.poll(simulation_time)

        return

    def _update_more_expected_flags(self, spawner, result) -> None:
        """Clear ``strategy.more_*_expected`` once schedulers exhaust or stall.

        Spatial schedulers paired with a stationary conveyor cannot
        replenish, so we also clear the flag in that case to let the BT
        complete on items already in flight (see pt_task_behaviours.py:74-76).
        """
        strategy = self._policy.strategy
        if not strategy.more_items_expected and not strategy.more_targets_expected:
            return
        for sched, all_released, flag_attr, label in (
            (spawner.pick_scheduler, result.all_picks_released,
             "more_items_expected", "pick"),
            (spawner.target_scheduler, result.all_targets_released,
             "more_targets_expected", "target"),
        ):
            if sched is None or not getattr(strategy, flag_attr):
                continue
            if all_released:
                logger.info(
                    "Incremental generation complete: all %d %s objects spawned",
                    sched.total_count, label,
                )
                setattr(strategy, flag_attr, False)
            elif spawner.is_spatial_scheduler_inert(sched):
                logger.info(
                    "Spatial-trigger %s scheduler is inert (conveyor stationary); "
                    "clearing %s.", label, flag_attr,
                )
                setattr(strategy, flag_attr, False)

    def task_step(self):
        # BT-start gate: only *configured* schedulers participate.  A missing
        # scheduler must NOT default to "ready" — that silently collapses
        # the OR to always-True and renders ``bt_start_threshold`` inert on
        # single-sided tasks (e.g. picks-only incremental spawning).
        spawner = self._task_context.spawner if self._task_context is not None else None
        if spawner is not None and (spawner.has_pick_scheduler or spawner.has_target_scheduler):
            if not spawner.bt_should_start(self._last_simulation_time):
                return

        # Startup-delay gate: once the BT-start gate has opened, optionally
        # hold off ticking the BT for ``startup_delay_seconds`` of sim time so
        # gravity-spawned items can settle before the first pick.  Configured
        # per task via TaskSpec.startup_delay_seconds.  Applies in both
        # teleport and real-sim modes; default (None) preserves prior behavior.
        if self._startup_delay_seconds is not None and self._startup_delay_seconds > 0:
            if self._bt_gate_open_time is None:
                self._bt_gate_open_time = self._last_simulation_time
            elapsed = self._last_simulation_time - self._bt_gate_open_time
            if elapsed < self._startup_delay_seconds:
                return
            if not self._startup_delay_logged:
                logger.info(
                    "Task '%s': starting BT after %.2fs settling delay",
                    self._name, self._startup_delay_seconds,
                )
                self._startup_delay_logged = True

        # If task already completed, do nothing
        if self._task_done:
            if self._extra_step_countdown > 0:
                self._extra_step_countdown -= 1
            else:
                if not self._task_exit:
                    try:
                        logger.warning(f"Task '{self._name}' signaling ready to exit.")
                    except Exception:
                        pass
                self._task_exit = True
                return

        # If paused for inspection, check for ENTER to resume; skip BT tick.
        # Deferred incremental checks are also held until resume so the user
        # can reposition objects during the pause before verification runs.
        if self._paused_for_inspection:
            if _check_stdin_enter():
                self._paused_for_inspection = False
                logger.warning("=== Resuming task... ===")
            else:
                return  # keep sim running but don't advance BT

        # Run deferred incremental checks from the previous step (or after
        # a pause has been resumed).  Deferred by at least one step so that
        # the physics engine has processed any pose changes (e.g. teleport
        # via set_world_pose) and AABB queries return up-to-date bounding
        # boxes.
        if self._incremental_checks and self._pending_incremental_pick_names:
            pick_names = self._pending_incremental_pick_names
            self._pending_incremental_pick_names = []
            self._check_incremental(pick_names)

        # If controller reports done (all picks placed or targets exhausted), mark task finished
        if self._task_controller.is_done():
            # One-time warning log to signal task completion, and stop the
            # conveyor immediately so placed pairs still on the belt don't
            # keep traveling (and potentially fall off the edge unseen)
            # during the extra-step countdown before task exit.
            if not self._task_done:
                try:
                    logger.warning(f"Task '{self._name}' has finished.")
                except Exception:
                    pass
                try:
                    self.stop_conveyor()
                except Exception:
                    logger.exception("stop_conveyor at task-done failed")
            self._task_done = True
            return

        # Tick the behaviour tree (ContextMonitor refreshes blackboard from TaskContext,
        # then the orchestration tree runs the appropriate phase)
        self._task_controller.forward()

        # Detect newly completed picks and queue them for incremental check
        # on the next step (after the physics engine has updated).
        current_completed = set(self._task_context.strategy.completed_picks)
        new_picks = current_completed - self._prev_completed_picks
        if new_picks:
            if self._incremental_checks:
                self._pending_incremental_pick_names.extend(new_picks)
            self._prev_completed_picks = current_completed
            if self._pause_after_cycle:
                self._pause_cycle_count += 1
                self._paused_for_inspection = True
                logger.warning(
                    f"=== Cycle {self._pause_cycle_count} complete. "
                    f"Press ENTER in terminal to continue... ==="
                )
        return

    def _check_incremental(self, pick_names):
        """Forward to the task verifier for incremental per-pick verification."""
        self._task_context.verifier.verify_incremental(
            pick_names, simulation_time=self._last_simulation_time,
        )

    def _run_snapshot_verification(self, pick_name: str, target_name: str,
                                    simulation_time: float) -> None:
        """Forward to the task verifier (called by ConveyorFalloffMonitor)."""
        self._task_context.verifier.run_snapshot_verification(
            pick_name, target_name, simulation_time,
        )

    def _record_available_target_lost(self, target_name: str,
                                       simulation_time: float) -> None:
        """Forward to the task verifier (called by ConveyorFalloffMonitor)."""
        self._task_context.verifier.record_available_target_lost(
            target_name, simulation_time,
        )

    def check_groundtruth_task_success(self) -> (bool, list[str]):
        """Ground-truth success check; merges frozen snapshots with live verification.

        See :meth:`task_verifier.TaskVerifier.verify_final` for the
        verification semantics, including box-containment vs marker mode,
        snapshot/live merge, and lost-target info lines.
        """
        if not self._pick_objs or not self._target_objs:
            logger.warning("Ground-truth check skipped: picks or targets are missing.")
            return True, []
        return self._task_context.verifier.verify_final()

    def _format_failure_reasons(self, reasons_dict: dict[str, str]):
        if reasons_dict:
            reasons = [f"{key}: {val}" for (key, val) in _failure_reasons]
            reasons_msg = f", due to:"
            for i, reason in enumerate(reasons):
                if len(reasons) > 1:
                    reasons_msg += f"({i}) "
                reasons_msg += reason
        else:
            reasons_msg = "!"
        return f"Task failure{reasons_msg}"

    def check_groundtruth_task_cannot_succeed(self) -> bool:
        """Check, using privileged access to simulator APIs and semanic labels assigned to prims,
        whether the current state of the simulation is such that it is no longer possible to achieve task success.

        If the state of the simulation does not correspond to the task success conditions from the task_description,
        this method should log a warning message that states what conditions are not met, and return False.

        Sub-classes can override this method to perform task specific customized checks.
        """
        _task_is_doomed_to_fail_ = False
        _failure_reasons: dict[str, str] = {}
        if _task_is_doomed_to_fail_:
            failure_msg = self._format_failure_reasons(_failure_reasons)
            logger.warning(failure_msg)
            return True
        return False

    def _get_bb_cache(self):
        # Each call returns a fresh cache; extsMock provides a stub
        # ``create_bbox_cache`` so the mock path resolves identically.
        from isaacsim.core.utils.bounds import create_bbox_cache
        return create_bbox_cache()

    # BEGIN ---  merged base class methods: isaacsim.core.api.tasks.Stacking(=BaseStacking)
    def set_params(
        self,
        obj_name: Optional[str] = None,
        obj_position: Optional[str] = None,
        obj_orientation: Optional[str] = None,
        target_name: Optional[str] = None,
        target_position: Optional[str] = None,
        target_orientation: Optional[str] = None,
    ) -> None:
        """[summary]

        Args:
            obj_name (Optional[str], optional): [description]. Defaults to None.
            obj_position (Optional[str], optional): [description]. Defaults to None.
            obj_orientation (Optional[str], optional): [description]. Defaults to None.
        """
        if obj_name is not None:
            self._task_objects[obj_name].set_local_pose(
                position=obj_position, orientation=obj_orientation
            )
        if target_name is not None:
            self._task_objects[target_name].set_local_pose(
                position=target_position, orientation=target_orientation
            )
        return

    def get_params(self) -> dict:
        """[summary]

        Returns:
            dict: [description]
        """
        params_representation = dict()
        params_representation["robot_name"] = {
            "value": self._robot.name,
            "modifiable": False,
        }
        return params_representation

    def get_obj_names(self) -> List[str]:
        """[summary]

        Returns:
            List[str]: [description]
        """
        return [obj.name for obj in self._pick_objs]

    def get_target_names(self) -> List[str]:
        """[summary]

        Returns:
            List[str]: [description]
        """
        return [obj.name for obj in self._target_objs]

    def calculate_metrics(self) -> dict:
        """[summary]

        Raises:
            NotImplementedError: [description]

        Returns:
            dict: [description]
        """
        raise NotImplementedError

    def is_done(self) -> bool:
        """Return True when the task has finished processing.

        This is set when the controller completes all picks or reports
        exhaustion of target objects.
        """
        return bool(getattr(self, "_task_done", False))

    # get_observations() MODIFIED from base class(isaacsim.core.api.tasks.Stacking) method
    def get_observations(self) -> dict:
        """Return robot state + per-pick target pairing observations.

        Raw robot observations (joint positions, EE position) are computed
        here; target pairing info is augmented by the TaskController.
        """
        joints_state = self._robot.get_joints_state()
        end_effector_position, _ = self._robot.end_effector.get_local_pose()
        observations = {
            self._robot.name: {
                "joint_positions": joints_state.positions,
                "end_effector_position": end_effector_position,
            }
        }
        # Augment with per-pick target pairing info via TaskController
        return self._policy.augment_observations(observations, self._pick_objs)

    # END ---  merge base class methods: isaacsim.core.api.tasks.Stacking

    def reorder_pick_sequence(
        self, new_order_names: List[str], preserve_current: bool = True
    ) -> None:
        """Public hook to reorder the controller's pick sequence.

        Args:
            new_order_names: New list of pick object names (order to follow).
            preserve_current: If True, keeps the current pick as the active one in the new order.
        """
        controller = self._task_controller
        current_name = None
        if preserve_current and hasattr(controller, "get_current_pick_name"):
            current_name = controller.get_current_pick_name()
        controller.reorder_picks(new_order_names, current_pick_name=current_name)

    def can_exit(self) -> bool:
        """Return True when the task is ready to exit.

        This is set when the controller completes all picks and any extra simulation steps have been completed.
        """
        return bool(getattr(self, "_task_exit", False))

    def stop_conveyor(self) -> None:
        """Zero the conveyor surface velocity if this task uses a moving conveyor."""
        if self._task_spec is None or not self._task_spec.conveyor_speed:
            return
        try:
            from isaacsim.core.utils import prims
            from pxr import Gf, PhysxSchema

            usd_prim = prims.get_prim_at_path("/World/conveyor_surface")
            if usd_prim is None or not usd_prim.IsValid():
                return
            surface_velocity = PhysxSchema.PhysxSurfaceVelocityAPI(usd_prim)
            surface_velocity.GetSurfaceVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            logger.info("Conveyor stopped.")
        except Exception as e:
            logger.warning("Could not stop conveyor: %s", e)
