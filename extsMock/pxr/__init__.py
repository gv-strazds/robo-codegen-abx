"""Minimal mock of pxr (Pixar USD) for testing without full USD installation.

Provides Gf.Vec3d, Gf.Rotation, Gf.Quatd, and stub UsdGeom/UsdPhysics modules.
"""
import math
import numpy as np
from types import ModuleType


class _Vec3d:
    """Minimal mock of Gf.Vec3d."""

    def __init__(self, x=0.0, y=0.0, z=0.0):
        self._v = np.array([float(x), float(y), float(z)])

    def GetNormalized(self):
        n = np.linalg.norm(self._v)
        if n < 1e-12:
            return _Vec3d(0, 0, 0)
        normed = self._v / n
        return _Vec3d(*normed)

    def GetLength(self):
        return float(np.linalg.norm(self._v))

    def __getitem__(self, i):
        return self._v[i]

    def __repr__(self):
        return f"Vec3d({self._v[0]}, {self._v[1]}, {self._v[2]})"


class _Quatd:
    """Minimal mock of Gf.Quatd — double-precision quaternion."""

    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        if isinstance(x, _Vec3d):
            # Quatd(real, Vec3d_imaginary)
            self._w = float(w)
            self._imag = np.array([x[0], x[1], x[2]])
        else:
            self._w = float(w)
            self._imag = np.array([float(x), float(y), float(z)])

    def GetReal(self):
        return self._w

    def GetImaginary(self):
        return self._imag.copy()

    def __repr__(self):
        return f"Quatd({self._w}, {self._imag[0]}, {self._imag[1]}, {self._imag[2]})"


class _Rotation:
    """Minimal mock of Gf.Rotation — axis-angle rotation."""

    def __init__(self, axis, angle_degrees):
        """Create rotation from axis (Vec3d) and angle in degrees."""
        ax = np.array([axis[0], axis[1], axis[2]], dtype=float)
        n = np.linalg.norm(ax)
        if n > 1e-12:
            ax = ax / n
        self._axis = ax
        self._angle_deg = float(angle_degrees)
        self._angle_rad = math.radians(self._angle_deg)

        # Compute quaternion: q = [cos(a/2), sin(a/2)*axis]
        half = self._angle_rad / 2.0
        self._w = math.cos(half)
        self._xyz = math.sin(half) * self._axis

    def GetQuat(self):
        """Return the rotation as a Quatd."""
        return _Quatd(self._w, _Vec3d(*self._xyz))

    def GetAngle(self):
        return self._angle_deg

    def GetAxis(self):
        return _Vec3d(*self._axis)

    def __repr__(self):
        return f"Rotation(axis={self._axis}, angle={self._angle_deg})"


class _GfModule:
    """Namespace mock for pxr.Gf."""
    Vec3d = _Vec3d
    Rotation = _Rotation
    Quatd = _Quatd
    Quatf = _Quatd  # Alias — float precision not needed for mocks


Gf = _GfModule()

# Stub modules for UsdGeom and UsdPhysics
UsdGeom = ModuleType("pxr.UsdGeom")
UsdPhysics = ModuleType("pxr.UsdPhysics")

# Common UsdGeom classes used by asset_utils
class _XformOp:
    PrecisionFloat = "float"
    PrecisionDouble = "double"

class _Xformable:
    def __init__(self, *a, **kw):
        pass
    def AddScaleOp(self, *a, **kw):
        return None

UsdGeom.XformOp = _XformOp
UsdGeom.Xformable = _Xformable

# Stub PhysxSchema module (used by table_setup.py)
PhysxSchema = ModuleType("pxr.PhysxSchema")

# Stub Usd, Sdf, UsdShade modules for completeness
Usd = ModuleType("pxr.Usd")
Sdf = ModuleType("pxr.Sdf")
