import numpy as np
from typing import Optional, Tuple, Union
from isaacsim.cortex.framework import math_util
from isaacsim.cortex.framework.smoothed_command import SmoothedCommand, TargetAdapter
from isaacsim.core.utils.rotations import quat_to_rot_matrix

class ApproachParams(object):
    """Parameters describing how to approach a target (in position). They generally describe a
    funnel approaching the target from a particular direction.

    The approach direction is a 3D vector pointing in the direction of approach. It's magnitude
    defines the max offset from the position target the intermediate approach target will be shifted
    by. The std dev defines the length scale a radial basis (Gaussian) weight function that defines
    what fraction of the shift we take. The radial basis function is defined on the orthogonal
    distance to the line defined by the target and the direction vector.

    Intuitively, the normalized vector direction of the direction vector defines which direction to
    approach from, and it's magnitude defines how far back we want the end effector to come in from.
    The std dev defines how tighly the end-effector approaches along that line. Small std dev is
    tight around that approach line, large std dev is looser. A good value is often between 1 and 3
    cm (values of .01-.03 in meters).

    See calc_shifted_approach_target() for the specific implementation of how these parameters are
    used.

    Args:
        direction: The direction vector describing the direction to approach from.
        std_dev: The radial basis std dev characterizing how tightly to follow the approach
            direction.
    """

    def __init__(self, direction: np.ndarray, std_dev: float):
        self.direction = direction
        self.std_dev = std_dev

    def __repr__(self):
        return f"ApproachParams(direction={self.direction}, std_dev={self.std_dev})"

    def __str__(self):
        return "{direction: %s, std_dev %s}" % (str(self.direction), str(self.std_dev))


class PosePq:
    """A pose represented internally as a position p and quaternion orientation q.

    Args:
        p: The pose position
        q: The pose orientation as a quaternion.
    """

    def __init__(self, p: np.ndarray, q: np.ndarray):
        self.p = p
        self.q = q

    def __repr__(self):
        return f"PosePq(p={self.p}, q={self.q})"

    def as_tuple(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the pose as a (p,q) tuple"""
        return self.p, self.q

    def to_T(self) -> np.ndarray:
        """Returns the pose as a homogeneous transform matrix T."""
        return math_util.pack_Rp(quat_to_rot_matrix(self.q), self.p)


class MotionCommand:
    """Contains information about a motion command: an end-effector target (either full pose or
    position only), optional approach parameters, and an optional posture configuration.

    The target pose is a full position and orientation target. The approach params define how the
    end-effector should approach that target (see ApproachParams above). And the posture config
    defines how the system should resolve redundancy and generally posture the arm throughout the
    movement.

    Users should set either target_pose or target_position, but not both. target_pose defines a full
    pose target for the end-effector; target_position defines a postion-only end-effector allowing
    the arm to move through the nullspace. That nullspace can be optionally biased by the posture
    configuration.

    Args:
        target_pose: A full pose end-effector target. Set this or target_position, but not both.
        target_position: A position-only end-effector target. Set this or target_pose, but not both.
        approach_params: Optional parameters describing how the end-effector should approach the
            target.
        posture_config: A configuration of all joints commanded by the MotionCommander to bias the
            motion in the null space of the target.

    Raises:
        TypeError if either both target_pose and target_position are set or neither of them are set.
    """

    def __init__(
        self,
        target_pose: Optional[PosePq] = None,
        target_position: Optional[np.ndarray] = None,
        approach_params: Optional[np.ndarray] = None,
        posture_config: Optional[np.ndarray] = None,
    ):
        if target_pose is not None:
            if target_position is not None:
                raise TypeError("Cannot specify both a full pose and a position only command.")
            self.target_pose = target_pose
        else:
            if target_position is None:
                raise TypeError("Must specify either a full pose or position only command.")
            self.target_pose = PosePq(target_position, None)

        self.approach_params = approach_params
        self.posture_config = posture_config

    @property
    def has_approach_params(self) -> bool:
        """Determines whether approach parameters have been specified.

        Returns: True if they've been set, False otherwise.
        """
        return self.approach_params is not None

    @property
    def has_posture_config(self) -> bool:
        """Determines whether a posture config has been specified.

        Returns: True if it's been set, False otherwise.
        """
        return self.posture_config is not None


def calc_shifted_approach_target(target_T: np.ndarray, eff_T: np.ndarray, approach_params: np.ndarray) -> np.ndarray:
    """Calculates how the target should be shifted to implement the approach given the current
    end-effector position.

    Args:
        target_T: Final target pose as a homogeneous transform matrix.
        eff_T: Current end effector pose as a homogeneous transform matrix.
        approach_params: The approach parameters.

    Returns: The shifted target position.
    """
    target_R, target_p = math_util.unpack_T(target_T)
    eff_R, eff_p = math_util.unpack_T(eff_T)

    direction = approach_params.direction
    std_dev = approach_params.std_dev

    v = eff_p - target_p
    an = math_util.normalized(direction)
    norm = np.linalg.norm
    dist = norm(v - np.dot(v, an) * an)
    dist += 0.5 * norm(target_R - eff_R) / 3
    alpha = 1.0 - np.exp(-0.5 * dist * dist / (std_dev * std_dev))
    shifted_target_p = target_p - alpha * direction

    return shifted_target_p


class MotionCommandAdapter(TargetAdapter):
    """A simple adapter class to extract the target information to pass into the SmoothedCommand
    object.

    Args:
        command: The motion command being adapted.
    """

    def __init__(self, command: MotionCommand):
        self.command = command

    def get_position(self) -> np.ndarray:
        """Extract the position vector from the target pose.

        Returns: The position vector.
        """
        return self.command.target_pose.p

    def has_rotation(self) -> bool:
        """Determines whether there's a specified orientation in the target pose.

        Returns: True if the commanded target orientation has been set, False otherwise.
        """
        return self.command.target_pose.q is not None

    def get_rotation_matrix(self) -> np.array:
        """Converts the target pose orientation to a rotation matrix.

        Note that this method doesn't verify whether the rotation is set. Use has_rotation() to
        verify it's been set before calling this method.

        Returns: The 3x3 rotation matrix for the target orientation.
        """
        return quat_to_rot_matrix(self.command.target_pose.q)
