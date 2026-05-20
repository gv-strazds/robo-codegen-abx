"""Abstract interfaces for robot components used by controllers.

These protocols enable dependency injection and mock implementations,
allowing controllers to be tested without the full IsaacSim simulator.

Two interface families coexist during the Cortex migration:
- Legacy: IGripper, IEndEffectorController, IArticulationController (ArticulationAction-based)
- Commander: IArmCommander, IGripperCommander (MotionCommand-based, Cortex-aligned)
"""
from typing import Protocol, Optional, Tuple, runtime_checkable
import numpy as np

from isaacsim.cortex.framework.motion_commander import PosePq

# Import ArticulationAction - available in both extsMock and real IsaacSim
try:
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:
    # Fallback for environments where isaacsim is not available
    ArticulationAction = None  # type: ignore


@runtime_checkable
class IGripper(Protocol):
    """Protocol for gripper controllers.
    
    Defines the minimal interface required for gripper operations
    used by PickPlaceController and UR10MultiPickPlaceController.
    """
    
    def forward(self, action: str) -> "ArticulationAction":
        """Compute articulation action for 'open' or 'close' command.
        
        Args:
            action: Either "open" or "close"
            
        Returns:
            ArticulationAction to be applied to the robot
        """
        ...
    
    def open(self) -> None:
        """Open the gripper (release held object)."""
        ...
    
    def close(self) -> None:
        """Close the gripper (grasp object)."""
        ...


@runtime_checkable  
class IEndEffectorController(Protocol):
    """Protocol for end-effector motion controllers.
    
    Abstracts motion planners like RMPFlowController that compute
    joint-space trajectories to reach Cartesian end-effector poses.
    """
    
    def forward(
        self,
        target_end_effector_position: np.ndarray,
        target_end_effector_orientation: Optional[np.ndarray] = None,
    ) -> "ArticulationAction":
        """Compute joint actions to reach target end-effector pose.
        
        Args:
            target_end_effector_position: Target position [x, y, z]
            target_end_effector_orientation: Target orientation quaternion [w, x, y, z]
            
        Returns:
            ArticulationAction with computed joint positions/velocities
        """
        ...
    
    def reset(self) -> None:
        """Reset controller state."""
        ...


@runtime_checkable
class IArticulationController(Protocol):
    """Protocol for robot articulation controllers.
    
    Abstracts the interface for applying computed actions to robot joints.
    """
    
    def apply_action(self, action: "ArticulationAction") -> None:
        """Apply computed action to the robot.
        
        Args:
            action: ArticulationAction containing joint commands
        """
        ...


@runtime_checkable
class IRobotArticulation(Protocol):
    """Protocol for robot articulation interface.
    
    Abstracts the robot's top-level interface including gripper access
    and articulation controller retrieval.
    """
    
    @property
    def name(self) -> str:
        """Robot name identifier."""
        ...
    
    @property
    def gripper(self) -> IGripper:
        """Gripper attached to robot."""
        ...
    
    def get_articulation_controller(self) -> IArticulationController:
        """Get the articulation controller for applying actions."""
        ...


# Type alias for optional end-effector controller parameter
EndEffectorControllerType = Optional[IEndEffectorController]


# ---------------------------------------------------------------------------
# Commander interfaces (Cortex-aligned)
# ---------------------------------------------------------------------------

@runtime_checkable
class IArmCommander(Protocol):
    """High-level end-effector motion interface (Cortex-aligned).

    Wraps either a Cortex MotionCommander or the legacy
    cspace_controller + articulation_controller pattern.
    """

    def send_motion_command(self, cmd) -> None:
        """Send a MotionCommand to the robot arm.

        The MotionCommand can carry target_pose, approach_params, and
        posture_config for the motion planner.

        Args:
            cmd: A MotionCommand instance.
        """
        ...

    def get_fk_p(self) -> np.ndarray:
        """Return the current end-effector position [x, y, z]."""
        ...

    def get_fk_pq(self) -> PosePq:
        """Return the current end-effector pose as a PosePq."""
        ...


@runtime_checkable
class IGripperCommander(Protocol):
    """Direct open/close gripper interface (Cortex-aligned)."""

    def open(self) -> None:
        """Open the gripper."""
        ...

    def close(self) -> None:
        """Close the gripper."""
        ...

    @property
    def is_closed(self) -> bool:
        """Return True if the gripper is closed."""
        ...

    def grasp_state(self) -> str:
        """Report whether the gripper appears to be holding an object.

        Best-effort; implementations without feedback return ``"unknown"``.
        Valid return values:

        - ``"holding"``: Gripper appears to be grasping something.
        - ``"empty"``: Gripper is closed but no object is detected.
        - ``"unknown"``: No reliable feedback available — caller should
          defer to pose-based checks.

        ``VerifyGrasp`` and similar consumers must treat ``"unknown"`` as
        "fall through to the pose-based deviation check" so adapters
        lacking contact sensors do not unconditionally fail.
        """
        ...
