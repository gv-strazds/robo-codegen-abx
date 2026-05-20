"""MockTaskContext: test-friendly TaskContext that doesn't require Isaac Sim.

Provides configurable mock state for unit/integration testing of the
py_trees task behaviour tree.
"""
import logging
from typing import List, Optional

import numpy as np

from robot_controllers.mock_robot import (
    MockArmCommander,
    MockGripperCommander,
)

from multi_pick_strategy import MultiPickStrategy
from task_context_base import POSTURE_UNSET, LightweightObj, TaskContextBase

logger = logging.getLogger(__name__)

# Backward-compatible alias — tests and other modules import MockPickObj from here.
MockPickObj = LightweightObj


class MockJointsState:
    """Minimal mock for robot joints state."""
    def __init__(self, num_joints: int = 6):
        self.positions = np.zeros(num_joints)


class MockEndEffector:
    """Minimal mock for robot end effector."""
    def __init__(self):
        self._position = np.array([0.0, 0.0, 0.3])

    def get_local_pose(self):
        return self._position.copy(), np.array([1.0, 0.0, 0.0, 0.0])


class MockRobotForContext:
    """Minimal mock robot that satisfies TaskContext requirements."""

    def __init__(self, name: str = "mock_robot", num_joints: int = 6):
        self._name = name
        self._num_joints = num_joints
        self.end_effector = MockEndEffector()
        self.gripper = _MockGripperMinimal()

    @property
    def name(self) -> str:
        return self._name

    def get_joints_state(self) -> MockJointsState:
        return MockJointsState(self._num_joints)


class _MockGripperMinimal:
    """Minimal gripper for MockRobotForContext."""
    def __init__(self):
        self.joint_opened_positions = np.array([0.04, 0.04])

    def set_joint_positions(self, positions):
        pass

    def open(self):
        pass

    def close(self):
        pass


class MockTaskContext(TaskContextBase):
    """Test-friendly TaskContext that doesn't require Isaac Sim.

    Provides the same interface as TaskContext but with configurable mock state.

    Args:
        pick_names: List of pick object names.
        target_names: List of target object names.
        pick_positions: Optional dict of name -> position arrays.
        target_positions: Optional dict of name -> position arrays.
        prim_geometry: Optional dict of name -> PrimGeometry for cached geometry.
        strategy: Optional MultiPickStrategy; if None, a default is created.
    """

    def __init__(
        self,
        pick_names: Optional[List[str]] = None,
        target_names: Optional[List[str]] = None,
        pick_positions: Optional[dict] = None,
        target_positions: Optional[dict] = None,
        prim_geometry: Optional[dict] = None,
        gripper=None,
        strategy: Optional[MultiPickStrategy] = None,
        pick_posture_config=POSTURE_UNSET,
        place_posture_config=POSTURE_UNSET,
        teleport_mode: bool = False,
        mock_mode: bool = True,
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
        impl_spec = None,
        configurator = None,   # used only if impl_spec.use_curobo == True
    ) -> None:
        if pick_names is None:
            pick_names = ["pick_0", "pick_1", "pick_2"]
        if target_names is None:
            target_names = ["target_0", "target_1", "target_2"]

        pick_positions = pick_positions or {}
        target_positions = target_positions or {}

        default_pick_pos = [
            np.array([0.5, 0.0, 0.05]),
            np.array([0.5, 0.1, 0.05]),
            np.array([0.5, 0.2, 0.05]),
        ]
        default_target_pos = [
            np.array([-0.5, 0.0, 0.05]),
            np.array([-0.5, 0.1, 0.05]),
            np.array([-0.5, 0.2, 0.05]),
        ]

        pick_objs = []
        for i, name in enumerate(pick_names):
            pos = pick_positions.get(name, default_pick_pos[i] if i < len(default_pick_pos) else np.array([0.5, 0.0, 0.05]))
            pick_objs.append(MockPickObj(name, position=pos))

        target_objs = []
        for i, name in enumerate(target_names):
            pos = target_positions.get(name, default_target_pos[i] if i < len(default_target_pos) else np.array([-0.5, 0.0, 0.05]))
            target_objs.append(MockPickObj(name, position=pos))

        # Create default strategy if none provided
        if strategy is None:
            strategy = MultiPickStrategy(
                pick_objs=pick_objs,
                target_objs=target_objs,
            )
            strategy.initialize_pairings()

        robot = MockRobotForContext()

        # Create mock commanders for the Cortex-aligned interface
        arm_cmdr = MockArmCommander()
        gripper_cmdr = MockGripperCommander()

        super().__init__(
            robot=robot,
            strategy=strategy,
            gripper=gripper,
            ee_height_for_move=0.3,
            teleport_mode=teleport_mode,
            prim_geometry=prim_geometry,
            arm_commander=arm_cmdr,
            gripper_commander=gripper_cmdr,
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
        self._mock_mode = mock_mode

    @property
    def mock_mode(self) -> bool:
        """Default True — the pure-Python mock harness.

        The mock arm simulates motion via a tick-based countdown whose
        timing does not align with the wall-clock ``py_trees.timers.Timer``
        that gates the grip/release waits.  As a result ``VerifyGrasp``'s
        pose-deviation check is not a reliable signal in mock mode and
        must short-circuit (see ``pt_cortex_perception_behaviours``).

        Tests that want to exercise the pose-check logic directly can
        pass ``mock_mode=False`` to the constructor.
        """
        return self._mock_mode
