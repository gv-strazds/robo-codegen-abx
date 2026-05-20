import numpy as np
from scipy.spatial.transform import Rotation

# Note: Isaac Sim uses (w, x, y, z)
# Scipy uses (x, y, z, w)

def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]])

def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]])

def euler_angles_to_quat(euler_angles: np.ndarray) -> np.ndarray:
    """Converts euler angles (in XYZ convention?) to quaternion (w,x,y,z).
    IsaacSim usually assumes radians. specific convention might vary but XYZ is reasonable default.
    """
    # Assuming XYZ convention for now, can adjust if needed.
    r = Rotation.from_euler('xyz', euler_angles)
    q = r.as_quat()
    return xyzw_to_wxyz(q)

def matrix_to_euler_angles(mat: np.ndarray) -> np.ndarray:
    """Converts rotation matrix to euler angles."""
    r = Rotation.from_matrix(mat)
    return r.as_euler('xyz')

def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    """Converts quaternion (w,x,y,z) to rotation matrix."""
    q_scipy = wxyz_to_xyzw(q)
    r = Rotation.from_quat(q_scipy)
    return r.as_matrix()

def rot_matrix_to_quat(mat: np.ndarray) -> np.ndarray:
    """Converts rotation matrix to quaternion (w,x,y,z)."""
    r = Rotation.from_matrix(mat)
    q = r.as_quat()
    return xyzw_to_wxyz(q)


def gf_quat_to_np_array(orientation) -> np.ndarray:
    """Converts a pxr Quaternion type to numpy array [w, x, y, z]."""
    quat = np.zeros(4)
    quat[1:] = orientation.GetImaginary()
    quat[0] = orientation.GetReal()
    return quat


def gf_rotation_to_np_array(orientation) -> np.ndarray:
    """Converts a pxr Rotation type to numpy array [w, x, y, z]."""
    return gf_quat_to_np_array(orientation.GetQuat())
