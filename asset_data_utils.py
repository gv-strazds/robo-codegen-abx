"""Pure-Python / numpy asset metadata and geometry utilities.

This module intentionally avoids any IsaacSim / pxr imports so it can be
loaded by mock tasks, unit tests, and other code paths that run without the
IsaacSim SimulationApp. IsaacSim-dependent functionality (scene creation,
AABB computation from live prims, semantic labels) lives in ``asset_utils``.
"""

import json
import numpy as np
import os
from dataclasses import dataclass, field
from typing import List, Optional

from colors import *  # noqa: F401,F403  (kept for parity with asset_utils)

import logging
logger = logging.getLogger(__name__)


# String keys of primitive asset types. The matching class mapping
# (``PRIMS_MAP``) lives in ``asset_utils`` because the classes themselves
# come from IsaacSim.
PRIM_TYPES = {
    "cube",
    "disc",
    "ball",
    "cylinder",
    "capsule",
    "cone",
    "rect",
    "marker",
}


@dataclass(frozen=True)
class AssetSymmetry:
    """Rotational-symmetry description for an asset.

    Consumed by ``perception_utils.compute_place_pose`` to decide how
    much of a measured grasp-time jostle to correct for at drop time.
    Rotations about a symmetry axis are physically unobservable and
    must be projected out before applying the inverse-delta correction
    (otherwise the wrist chases unreachable poses for rolling
    cylinders).

    First pass supports:
      - kind='full'            — full SO(3); every rotation is a symmetry (sphere).
      - kind='continuous_axis' — rotation about a fixed item-local axis is a symmetry
                                 (cylinder / cone / disc / cylindrical bottle).

    Future extensions (e.g. 'discrete_cube' for cube face symmetries,
    'discrete_axis_n' for n-fold prism symmetry) can be added without
    breaking callers; consumers should treat unknown kinds as 'none'.

    axis_local: Unit vector in the item's **native** local frame (pre-spawn-
        rotation).  Required when kind == 'continuous_axis'; ignored for
        kind == 'full'.
    """

    kind: str
    axis_local: Optional[np.ndarray] = None


@dataclass
class AssetMetaData:
    """Lightweight description of an asset type.

    Fields
    - asset_type: Name of the object type (e.g., "cube", "disc").
    - is_primitive: Derived flag indicating whether this refers to a primitive type in PRIM_TYPES.
    - usd_path: Optional path to a USD file when not a primitive.
    - color: Named color string (used when creating primitives).
    - is_a: Classification labels from specific to abstract (e.g., ["soup_can", "can", "container"]).
    - grasp_height: Optional override for Z from object origin to EE contact point.
    - grasp_approach_axis: Object-local axis the gripper approaches along.
    - rest_height: Optional override for Z from origin to bottom resting surface.
    - top_surface_height: Optional override for Z from origin to top surface (for stacking).

    Rules
    - If asset_type is in PRIM_TYPES, treat as a primitive (is_primitive=True).
    - Otherwise, usd_path must be provided and is_primitive=False.
    """

    asset_type: str
    is_primitive: bool = field(init=False)
    is_local_asset: bool = False
    pick_axis: int = 1  # default axis used for calculating ee_offset for pick operations
    usd_path: Optional[str] = None
    color: str = "blue"
    is_a: List[str] = field(default_factory=list)

    # Grasp geometry (optional overrides — computed from live prim if None)
    grasp_height: Optional[float] = None      # Z from object origin to EE contact point
    grasp_approach_axis: str = "z"             # object-local axis the gripper approaches along

    # Placement geometry (optional overrides — computed from live prim if None)
    rest_height: Optional[float] = None        # Z from origin to bottom resting surface
    top_surface_height: Optional[float] = None # Z from origin to top surface (for stacking)

    # Asset-level default grasp-offset: optional 3-vector in the
    # object's **native** local frame (pre-spawn-rotation, unscaled)
    # that shifts the grasp point away from the geometric center.
    # Composed with the geometry-derived ``[0, 0, grasp_height]``
    # offset by ``perception_utils`` and ``TaskContextBase``.  ``None``
    # means "no asset-level offset" (legacy behavior).  Per-task
    # overrides on ``TaskImplementationSpec.grasp_offset_local_overrides``
    # take precedence over this default.
    default_grasp_offset: Optional[np.ndarray] = None

    # Rotational symmetry (optional).  Consumed by place-time orientation
    # correction; see ``AssetSymmetry`` and ``get_asset_symmetry``.  ``None``
    # means "no known symmetry" — correction stays disabled for this asset.
    symmetry: Optional["AssetSymmetry"] = None

    # Natural orientation axes in the asset's **native** local frame
    # (pre-spawn-rotation).  Signed unit vectors.
    #
    # ``up_axis_local``: the local axis that should point along world +Z when
    # the asset is correctly placed under its standard spawn orientation.
    # Used by ``task_verification.is_vertical`` / ``is_horizontal``: rotating
    # this vector by the current world quaternion gives a world-frame up
    # direction whose angle to +Z is the tilt.
    #
    # ``front_axis_local``: the local axis that should point along the
    # "forward" direction (currently unused by is_vertical/is_horizontal;
    # reserved for future place-time orientation correction).
    #
    # Defaults assume native-upright geometry (long axis already +Z, front +X).
    # YCB assets that are spawned with a -90° X rotation to stand upright must
    # set ``up_axis_local=[0, -1, 0]`` so the local up vector lands on +Z after
    # the spawn rotation.
    up_axis_local: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0])
    )
    front_axis_local: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0])
    )

    def __post_init__(self):
        # Determine primitive vs. USD-backed asset, and validate inputs.
        if self.asset_type in PRIM_TYPES:
            self.is_primitive = True
        elif self.usd_path is not None and len(str(self.usd_path)) > 0:
            self.is_primitive = False
        else:
            raise ValueError(
                "AssetMetaData requires asset_type in PRIM_TYPES or a non-empty usd_path"
            )

        # Normalize the natural-axis fields and reject zero vectors.  Stored
        # back as numpy arrays so callers can rely on ``np.ndarray`` dtype.
        self.up_axis_local = _validate_unit_axis(self.up_axis_local, "up_axis_local")
        self.front_axis_local = _validate_unit_axis(self.front_axis_local, "front_axis_local")

        # Validate optional default_grasp_offset: must be a 3-vector when set.
        if self.default_grasp_offset is not None:
            self.default_grasp_offset = _validate_offset_vec3(
                self.default_grasp_offset, "default_grasp_offset",
            )


def _validate_unit_axis(axis, field_name: str) -> np.ndarray:
    """Coerce ``axis`` to a normalized 3-vector. Reject zero vectors."""
    vec = np.asarray(axis, dtype=float).reshape(-1)
    if vec.shape[0] != 3:
        raise ValueError(f"AssetMetaData.{field_name} must be a 3-vector, got shape {vec.shape}")
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        raise ValueError(f"AssetMetaData.{field_name} must be non-zero")
    return vec / norm


def _validate_offset_vec3(vec, field_name: str) -> np.ndarray:
    """Coerce a non-normalized 3-vector (offsets can have any magnitude, including 0)."""
    arr = np.asarray(vec, dtype=float).reshape(-1)
    if arr.shape[0] != 3:
        raise ValueError(f"{field_name} must be a 3-vector, got shape {arr.shape}")
    return arr


ITEMS_MAP = {
    "ball": AssetMetaData(
        asset_type="ball",
        symmetry=AssetSymmetry(kind="full"),
    ),
    "disc": AssetMetaData(
        asset_type="disc",
        symmetry=AssetSymmetry(
            kind="continuous_axis",
            axis_local=np.array([0.0, 0.0, 1.0]),
        ),
    ),
    "cylinder": AssetMetaData(
        asset_type="cylinder",
        pick_axis=2,
        symmetry=AssetSymmetry(
            kind="continuous_axis",
            axis_local=np.array([0.0, 0.0, 1.0]),
        ),
    ),
    "cone": AssetMetaData(
        asset_type="cone",
        pick_axis=2,
        symmetry=AssetSymmetry(
            kind="continuous_axis",
            axis_local=np.array([0.0, 0.0, 1.0]),
        ),
    ),
    "madara_pad": AssetMetaData(
        asset_type="madara_pad",
        usd_path="SimEnvs/assets/madara_pad.usd",
        color="red",
        is_local_asset=True,
    ),
    "m_pad_black": AssetMetaData(
        asset_type="m_pad_black",
        usd_path="SimEnvs/assets/m_pad_BLACK.usd",
        color="black",
        is_local_asset=True,
    ),
    "m_pad_red": AssetMetaData(
        asset_type="m_pad_red",
        usd_path="SimEnvs/assets/m_pad_RED.usd",
        color="black",
        is_local_asset=True,
    ),
    "m_pad_green": AssetMetaData(
        asset_type="m_pad_green",
        usd_path="SimEnvs/assets/m_pad_GREEN.usd",
        color="black",
        is_local_asset=True,
    ),
    "m_pad_blue": AssetMetaData(
        asset_type="m_pad_blue",
        usd_path="SimEnvs/assets/m_pad_BLUE.usd",
        color="blue",
        is_local_asset=True,
    ),
    "madara_bottle": AssetMetaData(
        asset_type="madara_bottle",
        pick_axis=2, # by default pick from obj relative z direction
        usd_path="SimEnvs/assets/bottle_v3.usd",
        color="green",
        is_local_asset=True,
        # Native Z-long (confirmed by pick_axis=2).  Cylindrical → rotation
        # about the long axis is a symmetry.
        symmetry=AssetSymmetry(
            kind="continuous_axis",
            axis_local=np.array([0.0, 0.0, 1.0]),
        ),
        # Grasp 2.5 cm above the bottle's geometric center (toward the cap)
        # so the gripper closes around the neck rather than the wider body.
        default_grasp_offset=np.array([0.0, 0.0, 0.025]),
    ),
    "m_bottle_red": AssetMetaData(
        asset_type="m_bottle_red",
        pick_axis=2, # by default pick from obj relative z direction
        usd_path="SimEnvs/assets/bottle_v3_RED.usd",
        color="red",
        is_local_asset=True,
        # Native Z-long (confirmed by pick_axis=2).  Cylindrical → rotation
        # about the long axis is a symmetry.
        symmetry=AssetSymmetry(
            kind="continuous_axis",
            axis_local=np.array([0.0, 0.0, 1.0]),
        ),
        default_grasp_offset=np.array([0.0, 0.0, 0.025]),
    ),
    "m_bottle_blue": AssetMetaData(
        asset_type="m_bottle_blue",
        pick_axis=2, # by default pick from obj relative z direction
        usd_path="SimEnvs/assets/bottle_v3_BLUE.usd",
        color="blue",
        is_local_asset=True,
        # Native Z-long (confirmed by pick_axis=2).  Cylindrical → rotation
        # about the long axis is a symmetry.
        symmetry=AssetSymmetry(
            kind="continuous_axis",
            axis_local=np.array([0.0, 0.0, 1.0]),
        ),
        default_grasp_offset=np.array([0.0, 0.0, 0.025]),
    ),
    "m_bottle_green": AssetMetaData(
        asset_type="m_bottle_green",
        pick_axis=2, # by default pick from obj relative z direction
        usd_path="SimEnvs/assets/bottle_v3_GREEN.usd",
        color="green",
        is_local_asset=True,
        # Native Z-long (confirmed by pick_axis=2).  Cylindrical → rotation
        # about the long axis is a symmetry.
        symmetry=AssetSymmetry(
            kind="continuous_axis",
            axis_local=np.array([0.0, 0.0, 1.0]),
        ),
        default_grasp_offset=np.array([0.0, 0.0, 0.025]),
    ),
    "m_bottle_white": AssetMetaData(
        asset_type="m_bottle_white",
        pick_axis=2, # by default pick from obj relative z direction
        usd_path="SimEnvs/assets/bottle_v3_WHITE.usd",
        color="white",
        is_local_asset=True,
        # Native Z-long (confirmed by pick_axis=2).  Cylindrical → rotation
        # about the long axis is a symmetry.
        symmetry=AssetSymmetry(
            kind="continuous_axis",
            axis_local=np.array([0.0, 0.0, 1.0]),
        ),
        default_grasp_offset=np.array([0.0, 0.0, 0.025]),
    ),
    "soup_can": AssetMetaData(
        asset_type="soup_can",
        usd_path="/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        color="red",
        # YCB native Y-long (spawn-rotated -90° about X to stand upright).
        # Cylindrical → rotation about the native Y axis is a symmetry.
        symmetry=AssetSymmetry(
            kind="continuous_axis",
            axis_local=np.array([0.0, 1.0, 0.0]),
        ),
        # -90° X rotation maps native -Y → world +Z; up_axis_local picks the
        # signed local direction that lands at world up after the spawn.
        up_axis_local=np.array([0.0, -1.0, 0.0]),
        front_axis_local=np.array([0.0, 0.0, 1.0])
    ),
    "cracker_box": AssetMetaData(
        asset_type="cracker_box",
        usd_path="/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        color="red",
        # YCB native Y-long; spawn-rotated -90° X to stand upright.
        up_axis_local=np.array([0.0, -1.0, 0.0]),
        front_axis_local=np.array([0.0, 0.0, 1.0])
    ),
    "sugar_box": AssetMetaData(
        asset_type="sugar_box",
        usd_path="/Isaac/Props/YCB/Axis_Aligned_Physics/004_sugar_box.usd",
        color="yellow",
        # YCB native Y-long; spawn-rotated -90° X to stand upright.
        up_axis_local=np.array([0.0, -1.0, 0.0]),
        front_axis_local=np.array([0.0, 0.0, 1.0])
    ),
    "mustard_bottle": AssetMetaData(
        asset_type="mustard_bottle",
        usd_path="/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
        color="yellow",
        # YCB native Y-long; spawn-rotated -90° X to stand upright.
        up_axis_local=np.array([0.0, -1.0, 0.0]),
        front_axis_local=np.array([0.0, 0.0, 1.0])
    ),
    "mug_black": AssetMetaData(
        asset_type="mug_black",
        usd_path="/Isaac/Props/Mugs/SM_Mug_B1.usd",
        color="black",
    ),
    "mug_black_green": AssetMetaData(
        asset_type="mug_black_green",
        usd_path="/Isaac/Props/Mugs/SM_Mug_A2.usd",
        color="black",
    ),
    "mug_yellow": AssetMetaData(
        asset_type="mug_yellow",
        usd_path="/Isaac/Props/Mugs/SM_Mug_C1.usd",
        color="yellow",
    ),
    "mug_blue": AssetMetaData(
        asset_type="mug_blue",
        usd_path="/Isaac/Props/Mugs/SM_Mug_D1.usd",
        color="blue",
    ),
    "factory_bolt_m16": AssetMetaData(
        asset_type="factory_bolt_m16",
        usd_path="/Isaac/IsaacLab/Factory/factory_bolt_m16.usd",
        color="white",
    ),
    "gear_large": AssetMetaData(
        pick_axis=2, # by default pick from obj relative z direction
        asset_type="gear_large",
        usd_path="/Isaac/IsaacLab/Factory/factory_gear_large.usd",
        color="white",
    ),
    "gear_medium": AssetMetaData(
        asset_type="gear_medium",
        usd_path="/Isaac/IsaacLab/Factory/factory_gear_medium.usd",
        color="light_blue",
    ),
    "gear_small": AssetMetaData(
        asset_type="gear_small",
        usd_path="/Isaac/IsaacLab/Factory/factory_gear_small.usd",
        color="white",
    ),
    "gear_base": AssetMetaData(
        asset_type="gear_base",
        usd_path="/Isaac/IsaacLab/Factory/factory_gear_base.usd",
        color="white",
    ),
    "factory_hole_8mm": AssetMetaData(
        asset_type="factory_hole_8mm",
        usd_path="/Isaac/IsaacLab/Factory/factory_hole_8mm.usd",
        color="gray",
    ),
    "factory_peg_8mm": AssetMetaData(
        asset_type="factory_peg_8mm",
        usd_path="/Isaac/IsaacLab/Factory/factory_peg_8mm.usd",
        color="yellow",
    ),
    "nut_m16_yellow": AssetMetaData(
        asset_type="nut_m16_yellow",
        usd_path="/Isaac/IsaacLab/Factory/factory_nut_m16.usd",
        color="yellow",
    ),
    "nut_m16_green": AssetMetaData(
        asset_type="nut_m16_green",
        usd_path="/Isaac/IsaacLab/Mimic/nut_pour_task/nut_pour_assets/factory_m16_nut_green.usd",
        color="green",
    ),
    # "teddy_bear": AssetMetaData(
    #     asset_type="teddy_bear",
    #     usd_path="/Isaac/IsaacLab/Objects/teddy_bear.usd",
    #     color="brown",
    # ),
    # -------------- bins and sorting trays ------------------
    "KLT_Bin": AssetMetaData(
        asset_type="KLT_Bin",
        usd_path="/Isaac/Props/KLT_Bin/small_KLT.usd",
        color="violet",
    ),
    "sorting_bin_blue": AssetMetaData(
        asset_type="sorting_bin_blue",
        usd_path="/Isaac/IsaacLab/Mimic/nut_pour_task/nut_pour_assets/sorting_bin_blue.usd",
        color="blue",
    ),
    "sorting_bin_black": AssetMetaData(
        asset_type="sorting_bin_black",
        usd_path="/Isaac/IsaacLab/Mimic/exhaust_pipe_task/exhaust_pipe_assets/black_sorting_bin.usd",
        color="black",
    ),
    "sorting_beaker_red": AssetMetaData(
        asset_type="sorting_beaker_red",
        usd_path="/Isaac/IsaacLab/Mimic/nut_pour_task/nut_pour_assets/sorting_beaker_red.usd",
        color="red",
    ),
    "sorting_bowl_yellow": AssetMetaData(
        asset_type="sorting_bowl_yellow",
        usd_path="/Isaac/IsaacLab/Mimic/nut_pour_task/nut_pour_assets/sorting_bowl_yellow.usd",
        color="yellow",
    ),
#/data/gstrazds/isaacsim_assets/Assets/Isaac/5.0/Isaac/IsaacLab/Mimic/nut_pour_task/nut_pour_assets/sorting_bowl_yellow.usd
#/data/gstrazds/isaacsim_assets/Assets/Isaac/5.0/Isaac/IsaacLab/Mimic/exhaust_pipe_task/exhaust_pipe_assets/black_sorting_bin.usd
}


@dataclass
class PrimGeometry:
    """Cached intrinsic geometry for a loaded prim instance.

    All values are in object-local frame and stage units.
    Static after computation — does not change during simulation.
    """
    grasp_height: float        # Z from origin to EE grasp point
    rest_height: float         # Z from origin to bottom surface (positive = origin above bottom)
    top_surface_height: float  # Z from origin to top surface (positive = top above origin)
    local_half_extents: np.ndarray  # [half_x, half_y, half_z] in local frame
    needs_aabb_scale_correction: bool  # True for cuboid prims with double-scale bug
    # Orientation (quaternion [w, x, y, z]) the above heights/extents were
    # computed against.  Populated by ``lookup_prim_geometry`` with the
    # ``orientation`` argument it received — i.e. the task's intended spawn
    # orientation.  Consumed by ``perception_utils.compute_place_pose`` to
    # un-rotate the drop EE when the item is measured to be jostled off
    # this reference orientation at grasp time.  ``None`` means the
    # reference is unknown — correction is skipped.
    reference_orientation: Optional[np.ndarray] = None
    # Effective rotational symmetry for this spawned instance.  Populated
    # by ``lookup_prim_geometry`` via ``get_asset_symmetry`` (asset-type
    # lookup + scale-preservation check).  ``None`` means "no known
    # symmetry for this instance" — ``compute_place_pose`` leaves the
    # drop EE orientation unchanged (current-behavior preservation).
    symmetry: Optional[AssetSymmetry] = None

    # Asset-level default grasp offset for this spawned instance.
    # 3-vector in the object's local frame (scaled to the instance,
    # pre-orientation).  Populated from ``AssetMetaData.default_grasp_offset``
    # by ``lookup_prim_geometry`` / ``compute_prim_geometry``.  Default
    # is the zero vector — preserves legacy "grasp at center" behavior.
    # Per-task overrides live on ``TaskContextBase`` and take precedence
    # at lookup time (see ``TaskContextBase._grasp_offset_world``).
    default_grasp_offset: np.ndarray = field(
        default_factory=lambda: np.zeros(3)
    )


def get_asset_symmetry(
    asset_type: str,
    obj_scale: Optional[np.ndarray] = None,
) -> Optional[AssetSymmetry]:
    """Return the effective symmetry for a spawned instance of ``asset_type``.

    Reads ``AssetMetaData.symmetry`` from ``ITEMS_MAP``, then applies a
    scale-preservation check that can downgrade to ``None`` when
    non-uniform scaling breaks the symmetry:
      - 'full' requires ``sx == sy == sz`` (else returns None — a
        non-uniformly scaled sphere is an ellipsoid, not rotationally
        symmetric).
      - 'continuous_axis' requires uniform scale in the plane orthogonal
        to ``axis_local`` (else returns None — a non-uniformly scaled
        cylinder cross-section is an ellipse).

    Returns None when the asset is not in ``ITEMS_MAP``, when its
    metadata has no symmetry tag, or when the scale check fails.
    """
    meta = ITEMS_MAP.get(asset_type)
    if meta is None or meta.symmetry is None:
        return None
    sym = meta.symmetry
    if obj_scale is None:
        return sym
    scale = np.asarray(obj_scale, dtype=float).reshape(-1)
    if scale.shape[0] != 3:
        return sym
    tol = 1e-6
    if sym.kind == "full":
        if abs(scale[0] - scale[1]) > tol or abs(scale[0] - scale[2]) > tol:
            return None
        return sym
    if sym.kind == "continuous_axis":
        axis = sym.axis_local
        if axis is None:
            return None
        axis_vec = np.asarray(axis, dtype=float).reshape(-1)
        # Identify the two indices orthogonal to the symmetry axis.
        # For a canonical axis (one component ~1, others ~0) this is the
        # exact orthogonal plane; for a tilted axis we use the two axes
        # with the smallest components as a reasonable approximation
        # (first-pass assets all use canonical axes).
        abs_axis = np.abs(axis_vec)
        sym_idx = int(np.argmax(abs_axis))
        ortho = [i for i in (0, 1, 2) if i != sym_idx]
        if abs(scale[ortho[0]] - scale[ortho[1]]) > tol:
            return None
        return sym
    # Unknown kind: be conservative.
    return None


def scale_aabb(aabb, scale_factor):
    """
    Scales an AABB (6-value array) by a 3-dimensional scaling factor relative to its center.

    Args:
        aabb (np.array): [min_x, min_y, min_z, max_x, max_y, max_z]
        scale_factor (np.array): [scale_x, scale_y, scale_z]

    Returns:
        np.array: Scaled AABB
    """
    min_point = aabb[:3]
    max_point = aabb[3:]

    center = (min_point + max_point) / 2.0
    half_extents = (max_point - min_point) / 2.0

    scaled_half_extents = half_extents * scale_factor

    new_min = center - scaled_half_extents
    new_max = center + scaled_half_extents

    return np.concatenate([new_min, new_max])


_precomputed_geometry_cache: Optional[dict] = None


def _load_precomputed_geometry() -> dict:
    """Load asset_prim_geometry.json from the same directory as this module. Cached after first call."""
    global _precomputed_geometry_cache
    if _precomputed_geometry_cache is None:
        json_path = os.path.join(os.path.dirname(__file__), "asset_prim_geometry.json")
        try:
            with open(json_path, "r") as f:
                _precomputed_geometry_cache = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to load precomputed geometry from {json_path}: {e}")
            _precomputed_geometry_cache = {}
    return _precomputed_geometry_cache


def _quat_to_rotation_matrix(quat) -> np.ndarray:
    """Convert a [w, x, y, z] quaternion to a 3x3 rotation matrix."""
    w, x, y, z = quat
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def lookup_prim_geometry(asset_type: str, obj_scale=None, orientation=None) -> Optional[PrimGeometry]:
    """Look up precomputed geometry for an asset type, applying scale and orientation.

    The JSON contains geometry values at identity scale [1,1,1] and identity
    orientation. Scale is applied first (in local frame), then orientation
    rotates to world frame — matching the USD Xform order (Scale, Rotate, Translate).

    When orientation is non-identity, the local half-extents are rotated to
    produce the world-aligned AABB, and grasp_height / rest_height /
    top_surface_height are recomputed from the rotated bounding box.

    Args:
        asset_type: The asset type key (must match a key in asset_prim_geometry.json).
        obj_scale: Optional scale array [sx, sy, sz]. None or [1,1,1] means identity.
        orientation: Optional quaternion [w, x, y, z]. None or [1,0,0,0] means identity.

    Returns:
        PrimGeometry with transformed values, or None if asset_type is not found.
    """
    data = _load_precomputed_geometry()
    entry = data.get(asset_type)
    if entry is None:
        return None

    grasp_height = entry["grasp_height"]
    rest_height = entry["rest_height"]
    top_surface_height = entry["top_surface_height"]
    local_half_extents = np.array(entry["local_half_extents"])
    needs_correction = entry["needs_aabb_scale_correction"]

    # Asset-default grasp offset (object-local, pre-spawn-rotation, pre-scale).
    # When set, scale component-wise alongside the local half-extents below.
    meta_for_offset = ITEMS_MAP.get(asset_type)
    default_grasp_offset = np.zeros(3)
    if meta_for_offset is not None and meta_for_offset.default_grasp_offset is not None:
        default_grasp_offset = np.asarray(
            meta_for_offset.default_grasp_offset, dtype=float
        ).copy()

    # Step 1: Scale proportionally in local frame
    if obj_scale is not None:
        scale = np.asarray(obj_scale, dtype=float)
        if not np.allclose(scale, [1.0, 1.0, 1.0]):
            sz = scale[2]
            grasp_height *= sz
            rest_height *= sz
            top_surface_height *= sz
            local_half_extents = local_half_extents * scale
            default_grasp_offset = default_grasp_offset * scale

    # Step 2: Apply orientation to transform local AABB to world-aligned AABB.
    # Skip for spheres — a sphere's AABB and height geometry are rotation-
    # invariant, so np.abs(R) @ half_extents would incorrectly inflate them.
    if orientation is not None and asset_type != "ball":
        quat = np.asarray(orientation, dtype=float)
        if not np.allclose(quat, [1.0, 0.0, 0.0, 0.0]):
            R = _quat_to_rotation_matrix(quat)

            # Origin offset in local frame (assume origin is centered in X and Y;
            # Z offset derived from the asymmetry between top_surface and rest heights)
            origin_z_offset = top_surface_height - local_half_extents[2]
            local_origin_offset = np.array([0.0, 0.0, origin_z_offset])

            # Transform to world frame
            world_origin_offset = R @ local_origin_offset
            world_half_extents = np.abs(R) @ local_half_extents

            # Recompute heights from world-aligned AABB
            z_half = world_half_extents[2]
            center_z = world_origin_offset[2]
            grasp_height = z_half
            rest_height = z_half - center_z
            top_surface_height = z_half + center_z
            local_half_extents = world_half_extents

    # Apply ITEMS_MAP overrides (same logic as compute_prim_geometry)
    if asset_type in ITEMS_MAP:
        meta = ITEMS_MAP[asset_type]
        if meta.grasp_height is not None:
            grasp_height = meta.grasp_height
        if meta.rest_height is not None:
            rest_height = meta.rest_height
        if meta.top_surface_height is not None:
            top_surface_height = meta.top_surface_height

    # Reference orientation = the orientation argument we were called with.
    # When caller passes None (no orientation supplied), record identity so
    # downstream consumers can detect "this PrimGeometry was computed
    # relative to identity, jostle corrections can compare against that".
    if orientation is not None:
        reference_orientation = np.asarray(orientation, dtype=float).copy()
    else:
        reference_orientation = np.array([1.0, 0.0, 0.0, 0.0])

    return PrimGeometry(
        grasp_height=grasp_height,
        rest_height=rest_height,
        top_surface_height=top_surface_height,
        local_half_extents=local_half_extents,
        needs_aabb_scale_correction=needs_correction,
        reference_orientation=reference_orientation,
        symmetry=get_asset_symmetry(asset_type, obj_scale=obj_scale),
        default_grasp_offset=default_grasp_offset,
    )


simready_assets = [
    # https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/
      "/simready_content/common_assets/props/whitepackerbottle_a02/whitepackerbottle_a01.usd",
      "/simready_content/common_assets/props/whitepackerbottle_a02/whitepackerbottle_a02.usd",
      "/simready_content/common_assets/props/whitepackerbottle_a02/whitepackerbottle_a03.usd",
      "/simready_content/common_assets/props/whitepackerbottle_a02/whitepackerbottle_a04.usd",
      "/simready_content/common_assets/props/naturalbostonroundbottle_a01/naturalbostonroundbottle_a01.usd",
      "/simready_content/common_assets/props/naturalbostonroundbottle_a02/naturalbostonroundbottle_a02.usd",
      "/simready_content/common_assets/props/naturalbostonroundbottle_a03/naturalbostonroundbottle_a03.usd",
      "/simready_content/common_assets/props/utilitybucket_a01/utilitybucket_a01.usd",
      "/simready_content/common_assets/props/utilitybucket_a02/utilitybucket_a02.usd",
      "/simready_content/common_assets/props/utilityjug_a01/utilityjug_a01.usd",
      "/simready_content/common_assets/props/utilityjug_a02/utilityjug_a02.usd",
      "/simready_content/common_assets/props/utilityjug_a03/utilityjug_a03.usd",
      "/simready_content/common_assets/props/fstylejug_a01/fstylejug_a01.usd",
      "/simready_content/common_assets/props/fstylejug_a02/fstylejug_a02.usd",
      "/simready_content/common_assets/props/fstylejug_a03/fstylejug_a03.usd",
      "/simready_content/common_assets/props/fstylejug_a04/fstylejug_a04.usd",
      # several more (plastic) pails are available
      # a few vases
      # lots and lots of crates and boxes
      "/simready_content/common_assets/props/avocado01/avocado01.usd",
      "/simready_content/common_assets/props/lemon_01/lemon_01.usd",
      "/simready_content/common_assets/props/lemon_02/lemon_02.usd",
      "/simready_content/common_assets/props/lime01/lime01.usd",
      "/simready_content/common_assets/props/lychee01/lychee01.usd",
      "/simready_content/common_assets/props/orange_01/orange_01.usd",
      "/simready_content/common_assets/props/orange_02/orange_02.usd",
      "/simready_content/common_assets/props/pomegranate01/pomegranate01.usd",
      "/simready_content/common_assets/props/pumpkinlarge/pumpkinlarge.usd",
      "/simready_content/common_assets/props/pumpkinsmall/pumpkinsmall.usd",
      "/simready_content/common_assets/props/redonion/redonion.usd",
      #UTENSIL category has forks, spoons, a few plates and bowls, a cutting board, and many different spatulas
      "/simready_content/common_assets/props/plate_small/plate_small.usd",
      "/simready_content/common_assets/props/plate_large/plate_large.usd",
      "/simready_content/common_assets/props/serving_bowl/serving_bowl.usd",
      "/simready_content/common_assets/props/cutting_board_a/cutting_board_a.usd",
      "/simready_content/common_assets/props/blackandbrassbowl_large/blackandbrassbowl_large.usd",
      "/simready_content/common_assets/props/blackandbrassbowl_small/blackandbrassbowl_small.usd",
      #EQUIPMENT category has several good bins (called container_[ab]{nn} , also container_h{nn}
      "/simready_content/common_assets/props/coloredsporttrafficcone_a01/coloredsporttrafficcone_a01.usd",
      "/simready_content/common_assets/props/coloredtrafficcone_a01/coloredtrafficcone_a01.usd",

]
