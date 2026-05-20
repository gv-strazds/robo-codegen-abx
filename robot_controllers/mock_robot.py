"""Mock implementations of robot interfaces for testing.

These classes implement the protocols defined in robot_controllers.robot_interfaces,
allowing controllers to be tested without the full IsaacSim simulator.

Includes both legacy mocks (ArticulationAction-based) and commander mocks
(IArmCommander/IGripperCommander) for the Cortex migration.
"""
from typing import Optional, Tuple
import numpy as np

# Import ArticulationAction from extsMock
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.cortex.framework.motion_commander import PosePq


class MockGripper:
    """Mock gripper for testing.
    
    Implements the IGripper protocol without IsaacSim dependencies.
    """
    
    def __init__(self, num_dofs: int = 8):
        """Initialize mock gripper.
        
        Args:
            num_dofs: Total number of DOFs in the robot articulation
                     (used for return action sizing)
        """
        self._num_dofs = num_dofs
        self._is_closed = False
        # ParallelGripper-compatible attributes for duck-typing
        self.joint_opened_positions = np.array([0.04, 0.04])
        self.joint_closed_positions = np.array([0.0, 0.0])
        self._current_positions = self.joint_opened_positions.copy()
    
    def forward(self, action: str) -> ArticulationAction:
        """Compute action for 'open' or 'close' command.
        
        Args:
            action: Either "open" or "close"
            
        Returns:
            ArticulationAction with None joint positions (no-op)
        """
        if action == "open":
            self.open()
        elif action == "close":
            self.close()
        else:
            raise ValueError(f"Unknown gripper action: {action}")
        return ArticulationAction(joint_positions=[None] * self._num_dofs)
    
    def open(self) -> None:
        """Open the gripper."""
        self._is_closed = False
        self._current_positions = self.joint_opened_positions.copy()
    
    def close(self) -> None:
        """Close the gripper."""
        self._is_closed = True
        self._current_positions = self.joint_closed_positions.copy()
    
    def set_joint_positions(self, positions: np.ndarray) -> None:
        """Set gripper joint positions (ParallelGripper compatibility).
        
        Args:
            positions: Array of joint positions
        """
        self._current_positions = np.array(positions)
    
    def get_joint_positions(self) -> np.ndarray:
        """Get current gripper joint positions.
        
        Returns:
            Array of current joint positions
        """
        return self._current_positions.copy()
    
    def is_closed(self) -> bool:
        """Check if gripper is closed."""
        return self._is_closed
    
    def is_open(self) -> bool:
        """Check if gripper is open."""
        return not self._is_closed
    
    def update(self) -> None:
        """Update gripper state (no-op for mock)."""
        pass


class MockEndEffectorController:
    """Mock end-effector controller for testing.
    
    Implements the IEndEffectorController protocol without IsaacSim dependencies.
    Returns simple mock joint positions for any target pose.
    """
    
    def __init__(self, num_joints: int = 6):
        """Initialize mock end-effector controller.
        
        Args:
            num_joints: Number of robot joints
        """
        self._num_joints = num_joints
        self._last_target_position: Optional[np.ndarray] = None
        self._last_target_orientation: Optional[np.ndarray] = None
    
    def forward(
        self,
        target_end_effector_position: np.ndarray,
        target_end_effector_orientation: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """Compute joint actions to reach target end-effector pose.
        
        For testing, returns zero joint positions. A more sophisticated
        mock could implement simple inverse kinematics.
        
        Args:
            target_end_effector_position: Target EE position [x, y, z]
            target_end_effector_orientation: Target EE orientation quaternion
            
        Returns:
            ArticulationAction with mock joint positions
        """
        self._last_target_position = target_end_effector_position
        self._last_target_orientation = target_end_effector_orientation
        
        # Return zeros - a simple mock response
        return ArticulationAction(joint_positions=np.zeros(self._num_joints))
    
    def reset(self) -> None:
        """Reset controller state."""
        self._last_target_position = None
        self._last_target_orientation = None


class MockArticulationController:
    """Mock articulation controller for testing.
    
    Implements the IArticulationController protocol without IsaacSim dependencies.
    Records applied actions for test verification.
    """
    
    def __init__(self):
        """Initialize mock articulation controller."""
        self.applied_actions: list = []
        self.last_action: Optional[ArticulationAction] = None
    
    def apply_action(self, action: ArticulationAction) -> None:
        """Apply computed action to the robot.
        
        Records the action for later verification in tests.
        
        Args:
            action: ArticulationAction to apply
        """
        self.last_action = action
        self.applied_actions.append(action)
    
    def clear_history(self) -> None:
        """Clear recorded action history."""
        self.applied_actions.clear()
        self.last_action = None


class MockRobotArticulation:
    """Mock robot articulation for testing.

    Implements the IRobotArticulation protocol without IsaacSim dependencies.
    Provides mock gripper and articulation controller.
    """

    def __init__(self, name: str = "mock_robot", num_joints: int = 6):
        """Initialize mock robot articulation.

        Args:
            name: Robot name identifier
            num_joints: Number of robot joints
        """
        self._name = name
        self._num_joints = num_joints
        self._gripper = MockGripper(num_dofs=num_joints + 2)
        self._articulation_controller = MockArticulationController()

    @property
    def name(self) -> str:
        """Robot name identifier."""
        return self._name

    @property
    def gripper(self) -> MockGripper:
        """Gripper attached to robot."""
        return self._gripper

    def get_articulation_controller(self) -> MockArticulationController:
        """Get the articulation controller for applying actions."""
        return self._articulation_controller


# ---------------------------------------------------------------------------
# Commander-based mocks (Cortex-aligned)
# ---------------------------------------------------------------------------

class MockArmCommander:
    """Mock arm commander implementing IArmCommander.

    Tracks the latest target and simulates movement via a tick countdown.
    When ``tick()`` is called each frame, the countdown decrements;
    at zero the arm "arrives" and ``get_fk_p()`` returns the target.
    """

    def __init__(self, initial_position: Optional[np.ndarray] = None,
                 ticks_per_move: int = 2):
        self._position = initial_position if initial_position is not None else np.zeros(3)
        self._orientation: Optional[np.ndarray] = None
        self._target_position: Optional[np.ndarray] = None
        self._target_orientation: Optional[np.ndarray] = None
        self._ticks_per_move = ticks_per_move
        self._ticks_remaining = 0
        # For test inspection
        self.last_target_position: Optional[np.ndarray] = None
        self.last_target_orientation: Optional[np.ndarray] = None
        self.send_count: int = 0

    def send_ee_target(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        self._target_position = np.array(position)
        self._target_orientation = np.array(orientation) if orientation is not None else None
        self._ticks_remaining = self._ticks_per_move
        self.last_target_position = self._target_position
        self.last_target_orientation = self._target_orientation
        self.send_count += 1

    def send_motion_command(self, cmd) -> None:
        """Send a MotionCommand by extracting target pose and delegating to send_ee_target."""
        if hasattr(cmd, 'target_pose') and cmd.target_pose is not None:
            self.send_ee_target(cmd.target_pose.p, cmd.target_pose.q)
        elif hasattr(cmd, 'target_position') and cmd.target_position is not None:
            self.send_ee_target(cmd.target_position)

    def tick(self) -> None:
        """Simulate one physics step.  Call after each BT tick in mock mode."""
        if self._ticks_remaining > 0:
            self._ticks_remaining -= 1
            if self._ticks_remaining == 0 and self._target_position is not None:
                self._position = self._target_position.copy()
                self._orientation = (
                    self._target_orientation.copy()
                    if self._target_orientation is not None else None
                )

    def get_fk_p(self) -> np.ndarray:
        return self._position.copy()

    def get_fk_pq(self) -> PosePq:
        q = self._orientation.copy() if self._orientation is not None else np.array([1.0, 0.0, 0.0, 0.0])
        return PosePq(self._position.copy(), q)

    @property
    def is_moving(self) -> bool:
        return self._ticks_remaining > 0

    def reset(self) -> None:
        self._target_position = None
        self._target_orientation = None
        self._ticks_remaining = 0
        self.send_count = 0


class MockGripperCommander:
    """Mock gripper commander implementing IGripperCommander."""

    def __init__(self):
        self._is_closed = False
        self.open_count: int = 0
        self.close_count: int = 0
        # Test hook for VerifyGrasp / grasp-failure injection.  ``None``
        # → ``grasp_state()`` returns ``"unknown"`` (the safe default,
        # matching production adapters without feedback).  Set to
        # ``"holding"`` / ``"empty"`` to force that return value.
        self.grasp_state_override: object = None  # Optional[str]

    def open(self) -> None:
        self._is_closed = False
        self.open_count += 1

    def close(self) -> None:
        self._is_closed = True
        self.close_count += 1

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def grasp_state(self) -> str:
        if self.grasp_state_override is not None:
            return str(self.grasp_state_override)
        return "unknown"

    def reset(self) -> None:
        self._is_closed = False
        self.open_count = 0
        self.close_count = 0
        self.grasp_state_override = None


class MockCortexRobot:
    """Mock robot with Cortex-style .arm and .suction_gripper attributes.

    Provides the interface that CortexUr10 exposes so that behaviours
    and contexts can be tested without the real robot.
    """

    def __init__(self, name: str = "mock_cortex_robot", num_joints: int = 6,
                 initial_position: Optional[np.ndarray] = None):
        self._name = name
        self._num_joints = num_joints
        self.arm = MockArmCommander(initial_position=initial_position)
        self.suction_gripper = MockGripperCommander()

    @property
    def name(self) -> str:
        return self._name
