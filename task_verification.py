"""Task verification module.

Consolidates spatial verification primitives (moved from asset_utils.py) and
provides a structured, extensible framework for task success checking with
detailed diagnostic logging.
"""

import dataclasses
import logging
import math
from typing import Callable, Optional

import numpy as np

from isaacsim.core.utils.bounds import create_bbox_cache, compute_aabb
from asset_utils import (
    get_asset_type,
    is_of_type,
    _get_prim_local_scale,
    needs_aabb_scale_correction,
)
from asset_data_utils import ITEMS_MAP, _quat_to_rotation_matrix, scale_aabb

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AABB correction utility
# ---------------------------------------------------------------------------

def get_corrected_aabb(prim, bb_cache, obj_scale=None):
    """Get AABB with shape-specific corrections applied.

    Corrections handled:
    - LightweightObj: computes AABB from stored position and half_extents.
    - Cuboids: applies inverse-scale to correct the double-scale AABB bug.
    - Spheres: recomputes AABB from center and scale to avoid inflation
      caused by USD BBoxCache rotating the extent *box* (a sphere's true
      AABB is rotation-invariant, but USD doesn't know that).

    Args:
        prim: The prim object (must have prim_path), or a LightweightObj.
        bb_cache: Bounding box cache for computation.
        obj_scale: Optional scale array. If provided and prim is a cuboid type,
            the inverse scale is applied to correct the double-scale bug.
            For spheres, used to compute the true rotation-invariant AABB.
            If None, the scale is queried from the prim directly.

    Returns:
        np.ndarray: Corrected AABB [min_x, min_y, min_z, max_x, max_y, max_z].
    """
    # LightweightObj with stored half extents — no USD prim to query
    half_extents = getattr(prim, '_local_half_extents', None)
    if half_extents is not None:
        pos, _ = prim.get_world_pose()
        return np.array([
            pos[0] - half_extents[0], pos[1] - half_extents[1], pos[2] - half_extents[2],
            pos[0] + half_extents[0], pos[1] + half_extents[1], pos[2] + half_extents[2],
        ])

    aabb = compute_aabb(bb_cache, prim_path=prim.prim_path, include_children=True)
    if needs_aabb_scale_correction(prim):
        if obj_scale is None:
            obj_scale = _get_prim_local_scale(prim)

        if obj_scale is not None:
            inv_scale = np.array([1.0 / obj_scale[0], 1.0 / obj_scale[1], 1.0 / obj_scale[2]])
            aabb = scale_aabb(aabb, inv_scale)
    elif is_of_type(prim, "ball"):
        # Sphere rotation correction: USD BBoxCache treats the sphere's extent
        # attribute as a box [-r,-r,-r] to [r,r,r]. When the sphere acquires
        # rotation during physics simulation, this box rotates and the AABB
        # inflates (e.g. by sqrt(2) for a 45° rotation). A sphere's true AABB
        # is rotation-invariant: center ± radius*scale.
        if obj_scale is None:
            obj_scale = _get_prim_local_scale(prim)
        if obj_scale is not None:
            center = (aabb[:3] + aabb[3:]) / 2
            # Sphere default radius=1.0; true half-extent = |scale| * radius
            half_ext = np.abs(np.asarray(obj_scale, dtype=float))
            aabb = np.array([
                center[0] - half_ext[0], center[1] - half_ext[1], center[2] - half_ext[2],
                center[0] + half_ext[0], center[1] + half_ext[1], center[2] + half_ext[2],
            ])
    return aabb


# ---------------------------------------------------------------------------
# Spatial verification primitives (moved from asset_utils.py)
# ---------------------------------------------------------------------------

def ranges_overlap(a_min, a_max, b_min, b_max):
    return not (a_max < b_min or b_max < a_min)


def is_resting_on(obj_a, aabb_a, obj_b, aabb_b, obj_scale=None, z_tol=0.02):
    # Check that bottom of obj_a is close to top of obj_b
    a_z_min = float(aabb_a[2])
    a_z_max = float(aabb_a[5])
    b_z_min = float(aabb_b[2])
    b_z_max = float(aabb_b[5])

    z_diff = a_z_min - b_z_max
    return abs(z_diff) <= z_tol


def is_on_top(obj_a, obj_b, bb_cache=None, obj_scale=None, z_tol=0.02,
              log_failure=False) -> bool:
    """
    Check if obj_a is on top of obj_b.

    Args:
        obj_a: The object that should be on top (e.g. pick object).
        obj_b: The object that should be strictly below obj_a.
        bb_cache: Bounding box cache for efficient computation.
        obj_scale: Optional scale of the objects, used for refining AABB calculations.
        log_failure: If True, log detailed AABB diagnostics on failure.

    Returns:
        bool: True if obj_a is considered to be on top of obj_b.
    """
    if bb_cache is None:
        bb_cache = create_bbox_cache()

    try:
        aabb_a = get_corrected_aabb(obj_a, bb_cache, obj_scale=obj_scale)
        aabb_b = get_corrected_aabb(obj_b, bb_cache, obj_scale=obj_scale)
    except Exception as ex:
        import traceback
        try:
            logger.warning(f"EXCEPTION attempting to get corrected aabb: {traceback.format_exc()}")
        except Exception:
            pass
        return False

    x_overlap = ranges_overlap(aabb_a[0], aabb_a[3], aabb_b[0], aabb_b[3])
    y_overlap = ranges_overlap(aabb_a[1], aabb_a[4], aabb_b[1], aabb_b[4])
    xy_overlap = x_overlap and y_overlap

    a_z_min = float(aabb_a[2])
    b_z_max = float(aabb_b[5])
    z_diff = a_z_min - b_z_max
    z_ok = abs(z_diff) <= z_tol

    if not xy_overlap or not z_ok:
        if log_failure:
            name_a = getattr(obj_a, 'name', '?')
            name_b = getattr(obj_b, 'name', '?')
            logger.info(
                "is_on_top FAIL: '%s' on '%s':"
                "\n\t%s aabb=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]"
                "\n\t%s aabb=[%.4f, %.4f, %.4f, %.4f, %.4f, %.4f]"
                "\n\txy_overlap=%s (x=%s, y=%s)  z_diff=%.4f (tol=%.4f, z_ok=%s)",
                name_a, name_b,
                name_a, *aabb_a,
                name_b, *aabb_b,
                xy_overlap, x_overlap, y_overlap, z_diff, z_tol, z_ok,
            )
        return False

    return True


def is_vertically_within(obj_a, aabb_a, obj_b, aabb_b, z_tol=0.05):
    # Check that bottom of obj_a is close to top surface of the floor of obj_b
    # assumes obj_b is a hollow, container object like a box or a bowl

    a_z_min = float(aabb_a[2])
    a_z_max = float(aabb_a[5])
    b_z_min = float(aabb_b[2])
    b_z_max = float(aabb_b[5])

    z_diff = a_z_min - b_z_min
    return z_diff >= z_tol


def is_within(obj_a, obj_b, bb_cache=None, obj_scale=None, z_tol=0.01) -> bool:
    """
    Check if obj_a is inside container obj_b.

    Args:
        obj_a: The object that should within the bounds of (e.g. pick object).
        obj_b: The object that should be below obj_a.
        bb_cache: Bounding box cache for efficient computation.
        obj_scale: Optional scale of the objects, used for refining AABB calculations.

    Returns:
        bool: True if obj_a is considered to be inside obj_b.
    """
    if bb_cache is None:
        bb_cache = create_bbox_cache()

    try:
        aabb_a = get_corrected_aabb(obj_a, bb_cache, obj_scale=obj_scale)
        aabb_b = get_corrected_aabb(obj_b, bb_cache, obj_scale=obj_scale)
    except Exception as ex:
        import traceback
        try:
            logger.warning(f"EXCEPTION attempting to get corrected aabb: {traceback.format_exc()}")
        except Exception:
            pass
        return False

    xy_overlap = (
        ranges_overlap(aabb_a[0], aabb_a[3], aabb_b[0], aabb_b[3]) and
        ranges_overlap(aabb_a[1], aabb_a[4], aabb_b[1], aabb_b[4])
    )
    if not xy_overlap:
        return False

    return is_vertically_within(obj_a, aabb_a, obj_b, aabb_b, z_tol=0.02)


def _world_up_axis(obj) -> Optional[np.ndarray]:
    """Return obj's natural local up-axis rotated into the world frame.

    Reads ``up_axis_local`` from the asset's ``AssetMetaData`` (looked up by
    ``get_asset_type(obj)``) and applies the obj's current world quaternion.
    Returns ``None`` when the asset type is unknown or has no metadata entry.
    """
    asset_type = get_asset_type(obj)
    if asset_type is None:
        return None
    meta = ITEMS_MAP.get(asset_type)
    if meta is None:
        return None
    try:
        _, quat = obj.get_world_pose()
    except Exception:
        import traceback
        try:
            logger.warning(
                "EXCEPTION in _world_up_axis getting world pose: %s",
                traceback.format_exc(),
            )
        except Exception:
            pass
        return None
    R = _quat_to_rotation_matrix(np.asarray(quat, dtype=float))
    return R @ np.asarray(meta.up_axis_local, dtype=float)


def is_vertical(obj, *, obj_scale=None, max_tilt_deg: float = 15.0,
                log_failure: bool = False) -> bool:
    """Check whether obj's natural up-axis is aligned with world +Z.

    Uses the asset's ``AssetMetaData.up_axis_local`` (looked up from the obj's
    semantic ``type`` label) rotated by the obj's current world quaternion.
    The tilt is the angle between that world-frame up direction and world +Z;
    the check passes when ``tilt <= max_tilt_deg``.

    This is strict on sign: an upside-down object (axis pointing toward -Z) has
    a tilt of 180° and always fails.

    Args:
        obj: The prim (real or LightweightObj) to check. Must expose
            ``get_world_pose()`` and carry a semantic ``type`` label.
        obj_scale: Reserved for future use (e.g. anisotropic-scale spheres).
            Currently ignored — pose-based tilt does not depend on scale.
        max_tilt_deg: Maximum allowed tilt from world +Z, in degrees. Default 15°.
        log_failure: If True, log detailed diagnostics on failure.

    Returns:
        True iff the world-frame up direction is within ``max_tilt_deg`` of +Z.
        False also returned when the asset type / metadata is missing.
    """
    up_world = _world_up_axis(obj)
    if up_world is None:
        if log_failure:
            obj_name = getattr(obj, 'name', '?')
            logger.warning(
                "is_vertical FAIL: no asset_type/metadata for '%s'", obj_name
            )
        return False

    cos_tilt = float(up_world[2])
    threshold = math.cos(math.radians(max_tilt_deg))
    passed = cos_tilt >= threshold

    if not passed and log_failure:
        obj_name = getattr(obj, 'name', '?')
        asset_type = get_asset_type(obj)
        actual_deg = math.degrees(math.acos(float(np.clip(cos_tilt, -1.0, 1.0))))
        logger.info(
            "is_vertical FAIL: '%s' (asset_type=%s):"
            "\n\tworld up_axis=[%.4f, %.4f, %.4f]"
            "\n\ttilt=%.1f°  limit=%.1f°",
            obj_name, asset_type,
            up_world[0], up_world[1], up_world[2],
            actual_deg, max_tilt_deg,
        )
    return passed


def build_box_verification_hooks(
    box_specs, pick_objs, *, is_pick_expected=None, extra_pick_check=None,
):
    """Build (box_targets, spatial_check_fn, valid_targets_fn) from box specs.

    Creates the three components needed to construct a PlacementChecker in
    containment mode, using per-box geometry and optional match_labels.

    Args:
        box_specs: List of box spec dicts, each with 'name', 'center_xy',
            'floor_z', 'inner_size', 'height', and optional 'match_labels'.
        pick_objs: List of pick object prims.
        is_pick_expected: Optional ``Callable[[str], bool]`` returning True
            when a pick (by name) is part of the task's intended placement
            set. Picks that return False — typically overflow picks —
            short-circuit to zero valid boxes. Defaults to "all picks
            expected".
        extra_pick_check: Optional callable
            ``fn(pick_obj, bb_cache=None, obj_scale=None) -> bool`` that is
            AND-ed with the box containment result. Use to add per-pick
            constraints (e.g. verticality) alongside containment.

    Returns:
        (box_targets, spatial_check_fn, valid_targets_fn) tuple.
    """
    from collections import namedtuple
    from asset_utils import has_label

    BoxTarget = namedtuple("BoxTarget", ["name"])
    box_targets = [BoxTarget(name=spec["name"]) for spec in box_specs]

    if is_pick_expected is None:
        is_pick_expected = lambda _name: True

    def box_containment_check(pick_obj, box_target, bb_cache=None, obj_scale=None,
                              log_failure=False):
        for spec in box_specs:
            if spec["name"] == box_target.name:
                kwargs = {}
                if "z_tol" in spec:
                    kwargs["z_tol"] = spec["z_tol"]
                inside = is_within_box_geometry(
                    pick_obj,
                    box_center_xy=spec["center_xy"],
                    box_inner_size=spec["inner_size"],
                    box_floor_z=spec["floor_z"],
                    box_height=spec["height"],
                    bb_cache=bb_cache,
                    obj_scale=obj_scale,
                    log_failure=log_failure,
                    **kwargs,
                )
                if not inside:
                    return False
                if extra_pick_check is None:
                    return True
                return bool(extra_pick_check(pick_obj, bb_cache=bb_cache, obj_scale=obj_scale))
        return False

    def valid_targets_for_pick(pick_index):
        pick_obj = pick_objs[pick_index]
        if not is_pick_expected(pick_obj.name):
            return []
        valid = []
        for j, spec in enumerate(box_specs):
            match_labels = spec.get("match_labels")
            if match_labels is None:
                valid.append(j)  # no matching criteria → always valid
            elif all(has_label(pick_obj, k, v) for k, v in match_labels.items()):
                valid.append(j)  # all labels match
        return valid

    return box_targets, box_containment_check, valid_targets_for_pick


def make_index_based_strategy_adapters(strategy, pick_objs, target_objs):
    """Adapt strategy's name-based verification hooks into index-based callables.

    ``PlacementChecker`` invokes ``valid_targets_fn(pick_index) -> list[int]`` and
    ``placement_constraints_fn(pick_idx, tgt_idx) -> (bool, str)`` — index-based
    by construction.  ``MultiPickStrategy`` exposes the same information through
    name-based methods so the policy never sees pair indices.  This helper
    bridges the two at the verifier construction boundary.
    """
    target_idx_by_name = {tgt.name: i for i, tgt in enumerate(target_objs)}

    def valid_targets_fn(pick_idx):
        try:
            pick_name = pick_objs[pick_idx].name
        except (IndexError, AttributeError):
            return []
        return [target_idx_by_name[n]
                for n in strategy.valid_targets_for_pick(pick_name)
                if n in target_idx_by_name]

    def placement_constraints_fn(pick_idx, tgt_idx):
        try:
            pick_name = pick_objs[pick_idx].name
            target_name = target_objs[tgt_idx].name
        except (IndexError, AttributeError):
            return (False, "")
        return strategy.placement_constraints_satisfied(pick_name, target_name)

    return valid_targets_fn, placement_constraints_fn


def is_within_box_geometry(obj, box_center_xy, box_inner_size, box_floor_z, box_height,
                           bb_cache=None, obj_scale=None, xy_tol=0.01,
                           z_tol=0.02, z_tol_top=0.01, log_failure=False):
    """Check if obj is within a box defined by geometric parameters.

    Uses the object's AABB center (XY) and bottom (Z) to test containment
    against known box dimensions, without requiring the box prim's AABB.

    The Z check verifies:
    - Lower bound: object bottom is not too far below the box floor
      (obj_z_min >= box_floor_z - z_tol).
    - Upper bound: object bottom is not above the top of the box walls
      (obj_z_min <= box_floor_z + box_height + z_tol_top).

    Args:
        obj: The object prim to check (must have prim_path).
        box_center_xy: [x, y] center of the box interior.
        box_inner_size: [width, depth] inner dimensions of the box.
        box_floor_z: Z coordinate of the box floor surface.
        box_height: Height of the box walls.
        bb_cache: Bounding box cache for AABB computation.
        obj_scale: Optional scale for AABB correction.
        xy_tol: Tolerance added to XY bounds (positive = more lenient).
        z_tol: Tolerance for lower-bound Z check (object bottom vs box floor).
        z_tol_top: Tolerance for upper-bound Z check (object bottom vs box
            wall top). Allows objects piled slightly above the walls.
        log_failure: If True, log diagnostics on failure (default False).

    Returns:
        bool: True if the object is considered inside the box.
    """
    if bb_cache is None:
        bb_cache = create_bbox_cache()

    try:
        aabb = get_corrected_aabb(obj, bb_cache, obj_scale=obj_scale)
    except Exception:
        import traceback
        try:
            logger.warning(f"EXCEPTION in is_within_box_geometry: {traceback.format_exc()}")
        except Exception:
            pass
        return False

    obj_center_x = (aabb[0] + aabb[3]) / 2
    obj_center_y = (aabb[1] + aabb[4]) / 2
    obj_z_min = aabb[2]

    half_w = box_inner_size[0] / 2 + xy_tol
    half_d = box_inner_size[1] / 2 + xy_tol

    in_x = abs(obj_center_x - box_center_xy[0]) <= half_w
    in_y = abs(obj_center_y - box_center_xy[1]) <= half_d
    box_top_z = box_floor_z + box_height
    in_z_lower = obj_z_min >= box_floor_z - z_tol
    in_z_upper = obj_z_min <= box_top_z + z_tol_top
    in_z = in_z_lower and in_z_upper

    result = in_x and in_y and in_z
    if not result and log_failure:
        obj_name = getattr(obj, 'name', '?')
        logger.info(
            f"is_within_box_geometry FAIL '{obj_name}':"
            f"\n\tbox=({box_inner_size[0]:.4f}, {box_inner_size[1]:.4f})"
            f" box_floor_z={box_floor_z:.4f} box_top_z={box_top_z:.4f}: "
            f"\n\tcenter=({obj_center_x:.4f}, {obj_center_y:.4f}), z_min={obj_z_min:.4f}, "
            f"\n\tin_x={in_x}, in_y={in_y}, in_z={in_z}"
            f" (in_z_lower={in_z_lower}, in_z_upper={in_z_upper})"
        )
    return result


def is_horizontal(obj, *, obj_scale=None, max_tilt_deg: float = 15.0,
                  log_failure: bool = False) -> bool:
    """Check whether obj's natural up-axis lies in the horizontal plane.

    Mirror of ``is_vertical``: passes when the obj's world-frame up direction
    is within ``max_tilt_deg`` of perpendicular to world +Z (i.e. the absolute
    Z-component is at most ``sin(max_tilt_deg)``).

    Args:
        obj: The prim (real or LightweightObj) to check.
        obj_scale: Reserved for future use; currently ignored.
        max_tilt_deg: Maximum allowed deviation from the horizontal plane, in
            degrees. Default 15°.
        log_failure: If True, log detailed diagnostics on failure.

    Returns:
        True iff the world-frame up direction is within ``max_tilt_deg`` of
        the horizontal plane. False also returned when the asset type /
        metadata is missing.
    """
    up_world = _world_up_axis(obj)
    if up_world is None:
        if log_failure:
            obj_name = getattr(obj, 'name', '?')
            logger.warning(
                "is_horizontal FAIL: no asset_type/metadata for '%s'", obj_name
            )
        return False

    abs_cos_tilt = abs(float(up_world[2]))
    threshold = math.sin(math.radians(max_tilt_deg))
    passed = abs_cos_tilt <= threshold

    if not passed and log_failure:
        obj_name = getattr(obj, 'name', '?')
        asset_type = get_asset_type(obj)
        # Angle from horizontal plane = 90° - angle to +Z (when up points
        # near +Z) or angle to +Z - 90° (when it points near -Z).
        angle_to_horiz_deg = abs(
            math.degrees(math.asin(float(np.clip(abs_cos_tilt, -1.0, 1.0))))
        )
        logger.info(
            "is_horizontal FAIL: '%s' (asset_type=%s):"
            "\n\tworld up_axis=[%.4f, %.4f, %.4f]"
            "\n\tdeviation=%.1f°  limit=%.1f°",
            obj_name, asset_type,
            up_world[0], up_world[1], up_world[2],
            angle_to_horiz_deg, max_tilt_deg,
        )
    return passed


# ---------------------------------------------------------------------------
# Structured verification framework
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class PlacementCheck:
    """Result of checking one pick object's placement."""
    pick_index: int
    pick_name: str
    target_index: Optional[int]   # None if not placed on any target
    target_name: Optional[str]
    passed: bool
    detail: str                   # human-readable explanation
    # Provenance tag: "live" for checks computed from the current scene state,
    # "snapshot@<t>s" for checks frozen when a moving target crossed the
    # conveyor fall-off threshold. Used by VerificationResult.summary().
    source: str = "live"


@dataclasses.dataclass
class VerificationResult:
    """Structured result from PlacementChecker.verify()."""
    success: bool
    checks: list
    failures: list                # backward-compat with existing (bool, list[str]) return
    # Optional informational lines appended to summary() — e.g. "Target '<name>'
    # was available but not filled in time (t=...s)". These do NOT affect
    # `success` and are not in `failures`.
    info_lines: list = dataclasses.field(default_factory=list)

    def summary(self) -> str:
        """Multi-line diagnostic summary for logging."""
        n_picks = len(self.checks)
        n_failures = len(self.failures)
        status = "PASSED" if self.success else "FAILED"

        if self.success:
            header = f"Task Verification: {status} (all {n_picks} picks properly placed)"
        else:
            header = f"Task Verification: {status} ({n_failures} of {n_picks} picks not properly placed)"

        lines = [header]
        for check in self.checks:
            src = getattr(check, "source", "live")
            if src.startswith("snapshot"):
                tag = f"[SNAPSHOT@{src.split('@', 1)[1]}]" if "@" in src else "[SNAPSHOT]"
            else:
                tag = "[LIVE]"
            if check.passed:
                if check.target_name is not None:
                    lines.append(f"  {tag} Pick '{check.pick_name}' -> target '{check.target_name}' (passed)")
                else:
                    lines.append(f"  {tag} Pick '{check.pick_name}' (no target needed)")
            else:
                lines.append(f"  {tag} FAIL Pick '{check.pick_name}': {check.detail}")

        for info in self.info_lines:
            lines.append(f"  [INFO] {info}")

        return "\n".join(lines)


def merge_verification_results(
    frozen_checks: list,
    live_result: "VerificationResult",
    pick_count: int,
    info_lines: Optional[list] = None,
) -> "VerificationResult":
    """Merge frozen snapshot checks with a live verification result.

    Frozen checks (one per pick whose paired target fell off the conveyor
    before task end) carry provenance ``source="snapshot@<t>s"`` and their
    verdicts are authoritative — live re-verification is not run for those
    picks. This helper combines both halves into a single result ordered by
    ``pick_index``.

    Args:
        frozen_checks: List of PlacementCheck with ``source`` starting with
            "snapshot". Must not overlap in pick_index with live_result.checks.
        live_result: VerificationResult from ``PlacementChecker.verify(pick_indices=...)``
            restricted to picks *not* in frozen_checks.
        pick_count: Total number of picks; used to size the ordering.
        info_lines: Optional informational lines appended to the merged summary.

    Returns:
        VerificationResult with:
          - success = all checks passed and no failure messages
          - checks ordered by pick_index
          - failures = frozen_failures + live failures (frozen failures are
            reconstructed from failed frozen checks using the same wording as
            PlacementChecker)
          - info_lines threaded through
    """
    by_idx = {}
    for check in frozen_checks:
        by_idx[check.pick_index] = check
    for check in live_result.checks:
        by_idx[check.pick_index] = check

    ordered = [by_idx[i] for i in sorted(by_idx.keys())]

    # Failures: frozen failures + live failures.
    failures = list(live_result.failures)
    for check in frozen_checks:
        if not check.passed:
            failures.append(f"Pick '{check.pick_name}': {check.detail}.")

    success = (len(failures) == 0)
    return VerificationResult(
        success=success,
        checks=ordered,
        failures=failures,
        info_lines=list(info_lines) if info_lines else [],
    )


class PlacementChecker:
    """Composable verifier for multi-pick-place task success.

    Accepts the existing task hook methods as callable parameters so that the
    override pattern in task subclasses is preserved exactly.

    Args:
        pick_objs: List of pick object prims.
        target_objs: List of target object prims.
        obj_scale: Optional object scale for AABB refinement.
        spatial_check_fn: Callable(pick, target, bb_cache, obj_scale) -> bool.
            Default: is_on_top.
        valid_targets_fn: Callable(pick_index) -> list[int].
            Default: all targets valid.
        placement_constraints_fn: Callable(pick_index, target_index) -> (bool, str).
            Returns (passed, reason). reason is empty on success, contains
            a failure description on failure. Default: always (True, "").
        bb_cache_factory: Callable() -> bbox_cache.
            Default: create_bbox_cache.
    """

    def __init__(
        self,
        pick_objs,
        target_objs,
        obj_scale=None,
        spatial_check_fn=None,
        valid_targets_fn=None,
        placement_constraints_fn=None,
        bb_cache_factory=None,
        allow_multi_occupancy=False,
        containment_mode=False,
    ):
        self._pick_objs = pick_objs or []
        self._target_objs = target_objs or []
        self._obj_scale = obj_scale
        self._spatial_check_fn = spatial_check_fn or (
            lambda pick, tgt, bb_cache=None, obj_scale=None: is_on_top(
                pick, tgt, bb_cache=bb_cache, obj_scale=obj_scale
            )
        )
        self._valid_targets_fn = valid_targets_fn or (
            lambda pick_index: list(range(len(self._target_objs)))
        )
        self._placement_constraints_fn = placement_constraints_fn or (
            lambda pick_index, target_index: (True, "")
        )
        self._bb_cache_factory = bb_cache_factory or create_bbox_cache
        self._containment_mode = containment_mode
        # containment_mode implies multi-occupancy (one container matches all picks)
        self._allow_multi_occupancy = allow_multi_occupancy or containment_mode

    def check_occupancy(self) -> dict:
        """Return occupancy mapping.

        Only checks targets that are valid for each pick (via valid_targets_fn),
        preventing a pick from claiming a target that belongs to another pick.

        When allow_multi_occupancy is False (default):
            Returns {target_index: pick_index} — first match wins per target.
        When allow_multi_occupancy is True:
            Returns {target_index: [pick_index, ...]} — all matching picks per target.
        """
        cache = self._bb_cache_factory()
        target_occupied_by = {}
        for i, p in enumerate(self._pick_objs):
            valid_targets = set(self._valid_targets_fn(i))
            for j, t in enumerate(self._target_objs):
                if j not in valid_targets:
                    continue
                if self._spatial_check_fn(p, t, bb_cache=cache, obj_scale=self._obj_scale):
                    if self._allow_multi_occupancy:
                        target_occupied_by.setdefault(j, []).append(i)
                    else:
                        target_occupied_by.setdefault(j, i)
        return target_occupied_by

    def verify(self, pick_indices=None) -> VerificationResult:
        """Run full verification, return structured result.

        Args:
            pick_indices: Optional list of pick indices to check. When provided,
                only those picks are verified. When None (default), all picks
                are checked.
        """
        checks = []
        failures = []

        if not self._pick_objs or not self._target_objs:
            return VerificationResult(success=True, checks=checks, failures=failures)

        indices_to_check = pick_indices if pick_indices is not None else range(len(self._pick_objs))

        if self._allow_multi_occupancy:
            self._verify_multi_occupancy(indices_to_check, checks, failures)
        else:
            self._verify_exclusive(indices_to_check, checks, failures)

        success = len(failures) == 0
        return VerificationResult(success=success, checks=checks, failures=failures)

    def _verify_multi_occupancy(self, indices_to_check, checks, failures):
        """Verify picks in multi-occupancy / containment mode.

        Calls spatial_check_fn directly per pick per valid target — no occupancy
        map needed since targets are never "full."
        """
        cache = self._bb_cache_factory()
        for i in indices_to_check:
            p = self._pick_objs[i]
            valid_targets = self._valid_targets_fn(i)
            if valid_targets is None:
                valid_targets = []

            if not valid_targets:
                # No valid targets (e.g. unpaired overflow pick) — not a failure
                checks.append(PlacementCheck(
                    pick_index=i, pick_name=p.name,
                    target_index=None, target_name=None,
                    passed=True,
                    detail="no valid targets (unpaired overflow pick)",
                ))
                continue

            # Check each valid target: spatial check, then placement constraints.
            # Stop on first match.
            placed_target_idx = None
            placed_target_name = None
            last_constraint_reason = ""
            for t in valid_targets:
                if self._spatial_check_fn(p, self._target_objs[t],
                                          bb_cache=cache, obj_scale=self._obj_scale):
                    constraint_ok, constraint_reason = self._placement_constraints_fn(i, t)
                    if constraint_ok:
                        placed_target_idx = t
                        placed_target_name = self._target_objs[t].name
                        break
                    else:
                        last_constraint_reason = constraint_reason

            if placed_target_idx is not None:
                checks.append(PlacementCheck(
                    pick_index=i, pick_name=p.name,
                    target_index=placed_target_idx,
                    target_name=placed_target_name,
                    passed=True, detail="placed on valid target",
                ))
            else:
                # Re-check with log_failure=True for diagnostic output
                diag_cache = self._bb_cache_factory()
                for t in valid_targets:
                    try:
                        self._spatial_check_fn(
                            p, self._target_objs[t],
                            bb_cache=diag_cache, obj_scale=self._obj_scale,
                            log_failure=True,
                        )
                    except TypeError:
                        pass  # spatial_check_fn doesn't accept log_failure

                if last_constraint_reason:
                    detail = last_constraint_reason
                    failure_msg = f"Pick '{p.name}': {last_constraint_reason}."
                elif self._containment_mode:
                    detail = "not placed inside target container"
                    failure_msg = f"Pick '{p.name}' is not inside a valid container."
                else:
                    target_names = [self._target_objs[t].name for t in valid_targets]
                    detail = (
                        f"not placed on any valid target "
                        f"({len(valid_targets)} valid targets available: "
                        f"{', '.join(target_names)})"
                    )
                    failure_msg = (
                        f"Pick '{p.name}' is not on a valid target while "
                        f"{len(valid_targets)} valid target(s) remain available."
                    )
                checks.append(PlacementCheck(
                    pick_index=i, pick_name=p.name,
                    target_index=None, target_name=None,
                    passed=False, detail=detail,
                ))
                failures.append(failure_msg)

    def _verify_exclusive(self, indices_to_check, checks, failures):
        """Verify picks in exclusive-occupancy mode (1:1 target assignment).

        Builds an occupancy map first, then checks each pick against it.
        Uses a diagnostic re-check with log_failure=True for failed picks.
        """
        target_occupied_by = self.check_occupancy()

        for i in indices_to_check:
            p = self._pick_objs[i]
            valid_targets = self._valid_targets_fn(i)
            if valid_targets is None:
                valid_targets = []

            # Find which target this pick is placed on (if any)
            placed_target_idx = None
            placed_target_name = None
            last_constraint_reason = ""
            for t in valid_targets:
                if target_occupied_by.get(t) == i:
                    constraint_ok, constraint_reason = self._placement_constraints_fn(i, t)
                    if constraint_ok:
                        placed_target_idx = t
                        placed_target_name = self._target_objs[t].name
                        break
                    else:
                        last_constraint_reason = constraint_reason

            if placed_target_idx is not None:
                checks.append(PlacementCheck(
                    pick_index=i, pick_name=p.name,
                    target_index=placed_target_idx,
                    target_name=placed_target_name,
                    passed=True, detail="placed on valid target",
                ))
                continue

            # Not placed on a valid target — check if there are available ones
            available_valid = [t for t in valid_targets if t not in target_occupied_by]
            if len(available_valid) > 0:
                # Re-check with log_failure=True for diagnostic output
                diag_cache = self._bb_cache_factory()
                for t in valid_targets:
                    try:
                        self._spatial_check_fn(
                            p, self._target_objs[t],
                            bb_cache=diag_cache, obj_scale=self._obj_scale,
                            log_failure=True,
                        )
                    except TypeError:
                        pass  # spatial_check_fn doesn't accept log_failure
                target_names = [self._target_objs[t].name for t in available_valid]
                if last_constraint_reason:
                    detail = last_constraint_reason
                    failure_msg = f"Pick '{p.name}': {last_constraint_reason}."
                else:
                    detail = (
                        f"not placed on any valid target "
                        f"({len(available_valid)} valid targets available: "
                        f"{', '.join(target_names)})"
                    )
                    failure_msg = (
                        f"Pick '{p.name}' is not on a valid target while "
                        f"{len(available_valid)} valid target(s) remain available."
                    )
                checks.append(PlacementCheck(
                    pick_index=i, pick_name=p.name,
                    target_index=None, target_name=None,
                    passed=False, detail=detail,
                ))
                failures.append(failure_msg)
            else:
                # All valid targets occupied — acceptable
                checks.append(PlacementCheck(
                    pick_index=i, pick_name=p.name,
                    target_index=None, target_name=None,
                    passed=True,
                    detail="no valid targets available (all occupied)",
                ))
