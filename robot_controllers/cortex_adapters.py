"""Adapters that wrap Cortex robot components to satisfy IArmCommander/IGripperCommander.

Also provides LegacyArmAdapter for backward compatibility with the old
cspace_controller + articulation_controller pattern during migration,
and NullArmCommander / NullGripperCommander for teleport mode.
"""
from typing import Optional

import numpy as np

from isaacsim.cortex.framework.motion_commander import MotionCommand, PosePq


class CortexArmAdapter:
    """Wraps a Cortex MotionCommander (robot.arm) as IArmCommander."""

    def __init__(self, motion_commander):
        """Args:
            motion_commander: A MotionCommander instance (robot.arm).
        """
        self._mc = motion_commander

    def send_ee_target(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        if orientation is not None:
            cmd = MotionCommand(target_pose=PosePq(position, orientation))
        else:
            cmd = MotionCommand(target_position=position)
        self._mc.send(cmd)

    def send_motion_command(self, cmd) -> None:
        self._mc.send(cmd)

    def get_fk_p(self) -> np.ndarray:
        return self._mc.get_fk_p()

    def get_fk_pq(self) -> PosePq:
        return self._mc.get_fk_pq()


class CortexGripperAdapter:
    """Wraps a Cortex SurfaceGripper (robot.suction_gripper) as IGripperCommander."""

    def __init__(self, suction_gripper):
        self._gripper = suction_gripper
        self._is_closed = False

    def open(self) -> None:
        self._gripper.open()
        self._is_closed = False

    def close(self) -> None:
        self._gripper.close()
        self._is_closed = True

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def grasp_state(self) -> str:
        """Best-effort grasp state for a ``SurfaceGripper``.

        ``SurfaceGripper`` exposes an ``is_closed()`` physical state query
        (as distinct from the command-state ``self._is_closed`` tracked
        above), but it only reports Open/Closed — it cannot distinguish
        "closed on an object" from "closed on air".  Return ``"unknown"``
        to defer to the pose-based check in ``VerifyGrasp``; a follow-up
        step may tighten this if the C++ layer exposes contact.
        """
        return "unknown"


class LegacyArmAdapter:
    """Wraps legacy cspace_controller + articulation_controller as IArmCommander.

    Used for backward compatibility during the Cortex migration.
    """

    def __init__(self, cspace_controller, articulation_controller, robot=None):
        self._cspace = cspace_controller
        self._artic = articulation_controller
        self._robot = robot

    def send_ee_target(
        self,
        position: np.ndarray,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        action = self._cspace.forward(
            target_end_effector_position=position,
            target_end_effector_orientation=orientation,
        )
        self._artic.apply_action(action)

    def send_motion_command(self, cmd) -> None:
        if hasattr(cmd, 'target_pose') and cmd.target_pose is not None:
            self.send_ee_target(cmd.target_pose.p, cmd.target_pose.q)
        elif hasattr(cmd, 'target_position') and cmd.target_position is not None:
            self.send_ee_target(cmd.target_position)

    def get_fk_p(self) -> np.ndarray:
        if self._robot is not None and hasattr(self._robot, 'end_effector'):
            pos, _ = self._robot.end_effector.get_local_pose()
            return np.array(pos)
        return np.zeros(3)

    def get_fk_pq(self) -> PosePq:
        if self._robot is not None and hasattr(self._robot, 'end_effector'):
            pos, orient = self._robot.end_effector.get_local_pose()
            return PosePq(np.array(pos), np.array(orient))
        return PosePq(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))


class LegacyGripperAdapter:
    """Wraps legacy gripper.forward() + articulation_controller as IGripperCommander."""

    def __init__(self, gripper, articulation_controller):
        self._gripper = gripper
        self._artic = articulation_controller
        self._is_closed = False

    def open(self) -> None:
        action = self._gripper.forward(action="open")
        self._artic.apply_action(action)
        self._is_closed = False

    def close(self) -> None:
        action = self._gripper.forward(action="close")
        self._artic.apply_action(action)
        self._is_closed = True

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def grasp_state(self) -> str:
        """Legacy articulation grippers expose no contact feedback."""
        return "unknown"


class NullArmCommander:
    """No-op arm commander for teleport mode.  Accepts all commands silently."""

    def send_motion_command(self, cmd) -> None:
        pass

    def get_fk_p(self) -> np.ndarray:
        return np.zeros(3)

    def get_fk_pq(self) -> PosePq:
        return PosePq(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))


class NullGripperCommander:
    """No-op gripper commander for teleport mode.  Accepts all commands silently."""

    def __init__(self):
        self._is_closed = False

    def open(self) -> None:
        self._is_closed = False

    def close(self) -> None:
        self._is_closed = True

    @property
    def is_closed(self) -> bool:
        return self._is_closed

    def grasp_state(self) -> str:
        """Teleport mode has no physics — grasp success is never in doubt."""
        return "unknown"
