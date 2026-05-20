"""TaskContext: encapsulates all simulation state needed by the py_trees task tree.

Provides query methods for ContextMonitorBehaviour and mutation methods for
task-level behaviours, decoupling the behavior tree from direct task/scene access.
"""
import logging
from typing import Optional

import numpy as np

from isaacsim.core.utils.stage import get_stage_units

from multi_pick_strategy import MultiPickStrategy
from task_context_base import POSTURE_UNSET, TaskContextBase

logger = logging.getLogger(__name__)


class TaskContext(TaskContextBase):
    """Holds references to robot and strategy, provides state query / mutation
    methods for the py_trees task behaviour tree.

    Automatically creates the appropriate commander adapters based on the robot type:
    - CortexUr10 (has .arm): uses CortexArmAdapter / CortexGripperAdapter

    Args:
        robot_articulation: Robot object (CortexUr10 or legacy UR10).
        strategy: MultiPickStrategy instance owning pairing and pick iteration.
        gripper: Optional gripper; defaults to robot_articulation.gripper.
    """

    def __init__(
        self,
        robot_articulation,
        strategy: MultiPickStrategy,
        gripper=None,
        teleport_mode: bool = False,
        ee_height_for_move: float = None,
        prim_geometry: Optional[dict] = None,
        pick_posture_config=POSTURE_UNSET,
        place_posture_config=POSTURE_UNSET,
        place_hover_above_z: Optional[float] = None,
        place_approach_distance: Optional[float] = None,
        pick_min_reachable_z: Optional[float] = None,
        pick_max_reachable_radius_xy: Optional[float] = None,
        pick_approach_p_thresh: Optional[float] = None,
        pick_approach_std_dev: Optional[float] = None,
        move_timeout_s: Optional[float] = None,
        approach_timeout_s: Optional[float] = None,
        insert_timeout_s: Optional[float] = None,
        grasp_offset_local_overrides: Optional[dict] = None,
        impl_spec=None,
        configurator=None,  # used only if impl_spec.use_curobo == True
    ) -> None:
        if ee_height_for_move is None:
            ee_height_for_move = 0.3 / get_stage_units()

        # Determine commander adapters based on robot type and teleport mode
        arm_commander = None
        gripper_commander = None

        if teleport_mode:
            # Teleport mode: use null commanders so no motion commands reach the robot
            from robot_controllers.cortex_adapters import NullArmCommander, NullGripperCommander
            arm_commander = NullArmCommander()
            gripper_commander = NullGripperCommander()
        elif hasattr(robot_articulation, 'arm'):
            # CortexUr10 path — robot has MotionCommander and SurfaceGripper
            from robot_controllers.cortex_adapters import CortexArmAdapter, CortexGripperAdapter
            arm_commander = CortexArmAdapter(robot_articulation.arm)
            gripper_commander = CortexGripperAdapter(robot_articulation.suction_gripper)
        else:
            logger.warning("Robot does not have Cortex .arm; no commanders available")

        super().__init__(
            robot=robot_articulation,
            strategy=strategy,
            gripper=gripper,
            ee_height_for_move=ee_height_for_move,
            teleport_mode=teleport_mode,
            prim_geometry=prim_geometry,
            arm_commander=arm_commander,
            gripper_commander=gripper_commander,
            pick_posture_config=pick_posture_config,
            place_posture_config=place_posture_config,
            place_hover_above_z=place_hover_above_z,
            place_approach_distance=place_approach_distance,
            pick_min_reachable_z=pick_min_reachable_z,
            pick_max_reachable_radius_xy=pick_max_reachable_radius_xy,
            pick_approach_p_thresh=pick_approach_p_thresh,
            pick_approach_std_dev=pick_approach_std_dev,
            move_timeout_s=move_timeout_s,
            approach_timeout_s=approach_timeout_s,
            insert_timeout_s=insert_timeout_s,
            grasp_offset_local_overrides=grasp_offset_local_overrides,
        )

    def get_current_sim_time(self) -> float:
        """Read sim time from the live World singleton; fall back to wall-clock."""
        try:
            from isaacsim.core.api.world import World
            world = World.instance()
            if world is not None:
                return float(world.current_time)
        except Exception:
            pass
        return super().get_current_sim_time()
