"""TaskController: policy layer wrapping MultiPickStrategy.

Owns strategy creation, TaskContext construction, BT controller creation,
and observation augmentation (adding target pairing info to raw observations).

Separated from UR10MultiPickPlaceTask so that execution policy (pairing,
sequencing, target selection) is decoupled from simulation setup and
the Isaac Sim BaseTask lifecycle.
"""
import logging
from typing import Callable, Optional

from task_spec import TaskImplementationSpec, TaskSpec

logger = logging.getLogger(__name__)


class TaskController:
    """Policy layer that wraps a MultiPickStrategy and owns the task context.

    Responsibilities:
    - Create and initialize the strategy via a factory callable.
    - Create the TaskContext that bridges strategy to hardware/robot.
    - Create the BT controller that executes the pick-place tree.
    - Augment raw observations with target pairing information.

    Per-task tunables (postures, hover heights, watchdog timeouts, etc.)
    are read from ``impl_spec`` rather than threaded through individual
    constructor kwargs.  Construction-time wiring that does not live on
    the spec pair (the strategy factory closure, the live geometry cache,
    a possibly subclass-overridden ``ee_height_for_move``) stays as
    explicit kwargs.

    Args:
        strategy_factory: ``(pick_objs, target_objs) -> MultiPickStrategy``.
            Closes over per-task scene state and is built fresh by the
            caller, so it can't live on the spec pair.
        prim_geometry: Dict mapping prim names to ``PrimGeometry``.
            Owned by ``SimulationConfigurator``, populated as objects
            spawn — also not on the spec pair.
        ee_height_for_move: Optional override for transport height (scene
            units).  Defaults to reading ``impl_spec.ee_height_for_move``;
            an explicit value here wins.
        task_spec: The scene-side ``TaskSpec``.  Used by the controller
            for ``target_count`` / ``seed`` when generating virtual targets.
        impl_spec: The policy-side ``TaskImplementationSpec``.  Holds
            per-task tunables (postures, hover heights, watchdog timeouts,
            strategy factory, tree factory, etc.).  May be ``None`` for
            unit tests that exercise only strategy wiring; in that case
            all tunables fall back to library defaults.
    """

    def __init__(
        self,
        strategy_factory: Callable,
        prim_geometry: Optional[dict] = None,
        ee_height_for_move: Optional[float] = None,
        task_spec: Optional[TaskSpec] = None,
        impl_spec: Optional[TaskImplementationSpec] = None,
        configurator=None,
    ):
        self._strategy_factory = strategy_factory
        self._prim_geometry = prim_geometry if prim_geometry is not None else {}
        self._ee_height_for_move = ee_height_for_move
        self._task_spec = task_spec
        self._impl_spec = impl_spec
        self._configurator = configurator
        self._strategy = None
        self._task_context = None
        self._bt_controller = None

    # ------------------------------------------------------------------
    # Strategy
    # ------------------------------------------------------------------

    def _generate_virtual_targets(self, pick_objs, scene_target_objs):
        """Generate virtual target objects from the impl spec's virtual strategy.

        If ``impl_spec.virtual_target_generation_strategy`` is set,
        generates target items and converts them to ``LightweightObj``
        instances with geometry cached in ``self._prim_geometry``.  These
        targets are never spawned as USD prims — they exist only as
        Python pose/geometry holders for the pairing strategy.

        Args:
            pick_objs: List of pick objects (for pick-aware callables).
            scene_target_objs: List of scene target objects.

        Returns:
            List of LightweightObj instances (empty if no virtual strategy).
        """
        impl = self._impl_spec
        virtual_strategy = impl.virtual_target_generation_strategy if impl is not None else None
        if virtual_strategy is None:
            return []

        from task_context_base import create_lightweight_objs_from_items

        if hasattr(virtual_strategy, 'generate'):
            # ItemGenerator-style: call generate() with count and seed
            spec = self._task_spec
            items = virtual_strategy.generate(
                count_range=spec.target_count if spec is not None else None,
                seed=spec.seed if spec is not None else None,
            )
        elif callable(virtual_strategy):
            # Plain callable: call with (pick_objs, scene_target_objs)
            items = virtual_strategy(pick_objs, scene_target_objs)
        else:
            logger.warning("virtual_target_generation_strategy is neither a generator nor callable; skipping")
            return []

        virtual_objs = create_lightweight_objs_from_items(
            items, prefix="virtual_target", prim_geometry_out=self._prim_geometry,
        )
        logger.info(f"Generated {len(virtual_objs)} virtual target objects")
        return virtual_objs

    def create_strategy(self, pick_objs, target_objs):
        """Create and initialize the pairing strategy.

        If a virtual target strategy is configured on the task spec,
        virtual targets are appended in place to ``target_objs`` before
        creating the strategy.  Scene targets keep their original
        indices.  Mutating in place keeps the configurator's setup-time
        list and the strategy's runtime list as a single shared
        reference; otherwise incremental spawn would compute offsets
        against a stale list (see TableTaskIncrementalTargets crash).

        Args:
            pick_objs: List of pick objects (prims or LightweightObj).
            target_objs: List of target objects.

        Returns:
            The initialized MultiPickStrategy instance.
        """
        virtual_objs = self._generate_virtual_targets(pick_objs, target_objs)
        if virtual_objs:
            target_objs.extend(virtual_objs)
        self._strategy = self._strategy_factory(pick_objs, target_objs)
        self._strategy.initialize_pairings()
        return self._strategy

    @property
    def strategy(self):
        return self._strategy

    # ------------------------------------------------------------------
    # TaskContext
    # ------------------------------------------------------------------

    def create_task_context(self, robot, teleport_mode=False):
        """Create a TaskContext with the current strategy.

        Args:
            robot: Robot articulation object.
            teleport_mode: If True, skip physics and teleport objects.

        Returns:
            The constructed TaskContext instance.
        """
        from task_context import TaskContext

        impl = self._impl_spec if self._impl_spec is not None else TaskImplementationSpec()
        kwargs = dict(
            robot_articulation=robot,
            strategy=self._strategy,
            teleport_mode=teleport_mode,
            prim_geometry=self._prim_geometry,
            pick_posture_config=impl.pick_posture_config,
            place_posture_config=impl.place_posture_config,
            place_hover_above_z=impl.place_hover_above_z,
            place_approach_distance=impl.place_approach_distance,
            pick_min_reachable_z=impl.pick_min_reachable_z,
            pick_max_reachable_radius_xy=impl.pick_max_reachable_radius_xy,
            pick_approach_p_thresh=impl.pick_approach_p_thresh,
            pick_approach_std_dev=impl.pick_approach_std_dev,
            move_timeout_s=impl.move_timeout_s,
            approach_timeout_s=impl.approach_timeout_s,
            insert_timeout_s=impl.insert_timeout_s,
            grasp_offset_local_overrides=impl.grasp_offset_local_overrides,
            impl_spec=self._impl_spec,
            configurator=self._configurator,
        )
        if self._ee_height_for_move is not None:
            kwargs["ee_height_for_move"] = self._ee_height_for_move
        self._task_context = TaskContext(**kwargs)
        return self._task_context

    @property
    def task_context(self):
        return self._task_context

    # ------------------------------------------------------------------
    # BT Controller
    # ------------------------------------------------------------------

    def create_bt_controller(self, name, fake_fast=False, show_status=False):
        """Create a BT controller wrapping the task context.

        Args:
            name: Controller name.
            fake_fast: If True, skip physics delays in the BT.
            show_status: If True, log RUNNING transitions for diagnostics.

        Returns:
            The constructed UR10MultiPickPlaceController instance.
        """
        from robot_controllers import UR10MultiPickPlaceController

        self._bt_controller = UR10MultiPickPlaceController(
            name=name,
            task_context=self._task_context,
            fake_fast=fake_fast,
            tree_factory=(self._impl_spec.tree_factory if self._impl_spec is not None else None),
            show_status=show_status,
        )
        self._bt_controller.reset()
        return self._bt_controller

    @property
    def bt_controller(self):
        return self._bt_controller

    # ------------------------------------------------------------------
    # Observation augmentation
    # ------------------------------------------------------------------

    def augment_observations(self, observations, pick_objs):
        """Add target pairing info to raw robot observations.

        For each pick object, adds its current position/orientation plus
        the target name, position, and orientation from the strategy's
        pairing mapping.

        Args:
            observations: Dict with robot observations (modified in-place).
            pick_objs: List of pick objects.

        Returns:
            The augmented observations dict.
        """
        strategy = self._strategy
        task_context = self._task_context

        for current_obj in pick_objs:
            obj_position, obj_orientation = current_obj.get_local_pose()
            paired_target = strategy.pairings_by_pick_name.get(current_obj.name)

            if paired_target is not None:
                drop_orient = task_context.get_end_effector_orientation_for_drop(
                    current_obj.name,
                    task_context.get_placing_target_name(current_obj.name),
                )
                target_name, target_position, target_orientation = (
                    task_context.get_placing_info(current_obj.name, drop_orient)
                )
            else:
                target_name, target_position, target_orientation = None, None, None

            observations[current_obj.name] = {
                "position": obj_position,
                "orientation": obj_orientation,
                "target_name": target_name,
                "target_position": target_position,
                "target_orientation": target_orientation,
            }
        return observations
