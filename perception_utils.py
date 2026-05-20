"""Perception-side helpers for grasp-affordance / place-pose reasoning.

This module is the home for per-item grasp-pose and per-(pick, target)
place-pose computations that the cortex-style behaviour tree consults at
the start of each pick and place sub-sequence.  The goal is to localise
the geometry + orientation-resolution logic, so the context
can shrink to a thin adapter that forwards cached affordances to the
motion-command builders.

This file is intentionally free of Isaac-Sim runtime imports so it is safe to use from mock tasks,
unit tests, and any path that does not initialise ``SimulationApp``.
The only external dependencies are
``isaacsim.cortex.framework.math_util`` (shadowed by ``extsMock`` on test
paths) and ``asset_data_utils.PrimGeometry`` (pure-data).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

import numpy as np

import isaacsim.cortex.framework.math_util as math_util

if TYPE_CHECKING:
    from asset_data_utils import PrimGeometry


# Default pick-side approach funnel parameters.  Callers (including
# ``compute_grasp_pose``) may override per-item; these reproduce the values
# previously hardcoded in ``task_context_base.PICK_APPROACH_PARAMS``.
DEFAULT_PICK_APPROACH_DIRECTION = np.array([0.0, 0.0, -1.0])
DEFAULT_PICK_APPROACH_DISTANCE = 0.2
# Lateral std-dev of the descent funnel.  Tight (5 mm) values worked for
# stationary picks but, for moving picks (e.g. items on a conveyor),
# RMPFlow gets stuck in a small-amplitude oscillation a centimetre or so
# above the grasp pose: the sub-cm-wide funnel makes lateral correction
# stiff enough to compete with the descent attractor and the EE bounces
# in Z without ever reaching p_thresh.  15 mm is still tight enough for
# precision grasping (CortexExecuteApproach.p_thresh stays at 5 mm) but
# loose enough that the descent dominates.  Place side already runs
# wider (DEFAULT_PLACE_APPROACH_STD_DEV = 0.02) for the same reason.
DEFAULT_PICK_APPROACH_STD_DEV = 0.015

# Fallback grasp height when no ``PrimGeometry`` is cached for a pick.
# Effectively unreachable in normal operation; the caller is expected to
# log a warning when it is used.
DEFAULT_GRASP_HEIGHT_FALLBACK = 0.02

# Default place-side approach funnel parameters.  Callers may override
# per-item; these reproduce the values previously hardcoded in
# ``task_context_base.PLACE_APPROACH_PARAMS``.
DEFAULT_PLACE_APPROACH_DIRECTION = np.array([0.0, 0.0, -1.0])
DEFAULT_PLACE_APPROACH_DISTANCE = 0.20
DEFAULT_PLACE_APPROACH_STD_DEV = 0.02
# Default relative Z hover above the place target during the cortex tree's
# CortexMoveToPlace transport phase.  Per-task overrides flow through
# TaskSpec.place_hover_above_z and TaskContextBase.get_place_hover_above_z;
# the context clamps the effective hover to be at least
# DEFAULT_PLACE_APPROACH_DISTANCE so the descent funnel handoff stays coherent.
DEFAULT_PLACE_HOVER_ABOVE_Z = 0.13


# ---------------------------------------------------------------------------
# Low-level helpers shared by the three compute entry points.
# Stand-alone so unit tests can exercise them without building a dataclass.
# ---------------------------------------------------------------------------


def _quaternions_equivalent(
    q1: Optional[np.ndarray], q2: Optional[np.ndarray], tol: float = 1e-6,
) -> bool:
    """Return True if ``q1`` and ``q2`` represent the same rotation.

    Treats ``q`` and ``-q`` as equivalent (SO(3) double cover).  Two
    ``None`` inputs are considered equivalent; one ``None`` is not.
    """
    if q1 is None or q2 is None:
        return q1 is None and q2 is None
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    if a.shape != b.shape:
        return False
    return abs(abs(float(np.dot(a, b))) - 1.0) < tol


def _rotate_offset_by_rel_quat(
    offset_world: np.ndarray, q_from: np.ndarray, q_to: np.ndarray,
) -> np.ndarray:
    """Rotate a world-frame offset by the relative rotation ``q_from → q_to``.

    The vector from a held item's centre to the EE flange is rigid in
    the EE's local frame; when the EE rotates from ``q_from`` to
    ``q_to`` the world-frame offset at the new orientation is

        offset_to_world = R_to · R_fromᵀ · offset_world

    This is the core of the drop-side EE-offset adjustment for strategies
    whose drop orientation differs from their pick orientation (e.g.
    picking a bottle from above and placing it on its side).
    """
    R_from = math_util.quat_to_rot_matrix(q_from)
    R_to = math_util.quat_to_rot_matrix(q_to)
    R_rel = R_to @ R_from.T
    return R_rel @ np.asarray(offset_world, dtype=float)


def _quat_angle(q: np.ndarray) -> float:
    """Return the absolute rotation magnitude encoded by a unit quaternion, in radians."""
    q = np.asarray(q, dtype=float)
    w = max(-1.0, min(1.0, abs(float(q[0]))))
    return 2.0 * float(np.arccos(w))


def _quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for [w, x, y, z] quaternions."""
    w1, x1, y1, z1 = np.asarray(q1, dtype=float)
    w2, x2, y2, z2 = np.asarray(q2, dtype=float)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _swing_twist_decomp(
    q: np.ndarray, axis_unit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decompose ``q = q_swing * q_twist`` about ``axis_unit``.

    ``q_twist`` is the rotation about ``axis_unit`` whose axis-component
    matches ``q``; ``q_swing`` absorbs the residual (tilt of the axis
    away from itself).  For a symmetry axis this splits a measured
    jostle into:
      - ``q_swing``: observable tilt — must be corrected.
      - ``q_twist``: rotation about the symmetry axis — physically
        unobservable for symmetric items, so absorbed.

    Standard projection decomposition (Dobrowolski):
        proj     = (v · n) * n    where v = q[1:4]
        q_twist  = normalize([q[0], proj])
        q_swing  = q * q_twist.conj()

    Edge case: when ``q`` is a 180° rotation whose axis is orthogonal
    to ``axis_unit`` the projection collapses to zero; we fall back to
    ``q_twist = identity`` (``q_swing`` then equals ``q`` — the tilt
    IS the entire rotation).
    """
    q = np.asarray(q, dtype=float)
    axis = np.asarray(axis_unit, dtype=float)
    v = q[1:4]
    proj = float(np.dot(v, axis)) * axis
    q_twist = np.array([q[0], proj[0], proj[1], proj[2]])
    n = float(np.linalg.norm(q_twist))
    if n < 1e-10:
        q_twist = np.array([1.0, 0.0, 0.0, 0.0])
    else:
        q_twist = q_twist / n
    q_swing = _quat_multiply(q, _quat_conjugate(q_twist))
    return q_swing, q_twist


# ---------------------------------------------------------------------------
# Dataclasses: the three artefacts the BT behaviours ferry through a cycle.
# ---------------------------------------------------------------------------


@dataclass
class GraspPose:
    """Per-pick grasp affordance: target EE pose + approach plan.

    Produced by ``compute_grasp_pose`` (step 2 of the refactor) at the
    start of each pick attempt and cached on the context so the pre-grasp
    and approach behaviours share a single source of truth.

    Fields:
        ee_position: Target EE position at grasp time, world frame.
        ee_orientation: Target EE orientation at grasp time (quaternion,
            [w, x, y, z]).
        approach_direction: Unit vector along which the final translation
            into the grasp pose is performed.  ``pre_grasp_position``
            sits at ``ee_position - approach_direction * approach_distance``.
            Typically ``[0, 0, -1]`` (descend from above) but the data
            model supports object-local approach axes when a later step
            begins using them.
        approach_distance: Length (metres) of the final approach segment.
            Combined with ``approach_direction`` defines where the
            pre-grasp move ends.
        approach_std_dev: Gaussian std-dev for the RMPFlow approach funnel.
        ee_offset_world_at_grasp: World-frame vector from item centre to
            EE flange *at grasp time*.  Treated as rigid in the EE's
            local frame when computing the rotated world-frame offset at
            a different drop orientation (``R_drop · R_pickᵀ · offset``).
            Also used as the expected relative pose in the post-lift
            ``ItemInEEPose`` comparison.
        item_position_at_grasp: Item centre in world frame at the moment
            ``PrepareGrasp`` ran.  Optional — when ``None`` the post-lift
            verification can still compute a position deviation against
            ``ee_offset_world_at_grasp`` but cannot compute a truthful
            orientation error.
        item_orientation_at_grasp: Item orientation (quaternion) in world
            frame at the moment ``PrepareGrasp`` ran.  Consumed by
            ``compute_item_in_ee_pose`` to populate
            ``ItemInEEPose.orientation_error_rad`` — when ``None`` that
            field stays ``0.0`` for back-compat.
    """

    ee_position: np.ndarray
    ee_orientation: np.ndarray
    approach_direction: np.ndarray
    approach_distance: float
    approach_std_dev: float
    ee_offset_world_at_grasp: np.ndarray
    item_position_at_grasp: Optional[np.ndarray] = None
    item_orientation_at_grasp: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        # Defensive copies + normalisation so mutation of upstream arrays
        # does not silently change a cached pose.
        self.ee_position = np.asarray(self.ee_position, dtype=float).copy()
        self.ee_orientation = np.asarray(self.ee_orientation, dtype=float).copy()
        direction = np.asarray(self.approach_direction, dtype=float).copy()
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("approach_direction must be non-zero")
        self.approach_direction = direction / norm
        self.approach_distance = float(self.approach_distance)
        self.approach_std_dev = float(self.approach_std_dev)
        self.ee_offset_world_at_grasp = np.asarray(
            self.ee_offset_world_at_grasp, dtype=float,
        ).copy()
        if self.item_position_at_grasp is not None:
            self.item_position_at_grasp = np.asarray(
                self.item_position_at_grasp, dtype=float,
            ).copy()
        if self.item_orientation_at_grasp is not None:
            self.item_orientation_at_grasp = np.asarray(
                self.item_orientation_at_grasp, dtype=float,
            ).copy()

    @property
    def pre_grasp_position(self) -> np.ndarray:
        """World-frame position at which the approach segment begins."""
        return self.ee_position - self.approach_direction * self.approach_distance


@dataclass
class ItemInEEPose:
    """Measured relative pose of a held item w.r.t. the EE, after lift.

    Produced by ``compute_item_in_ee_pose`` (step 7) after the post-pick
    lift completes.  Carries both the measured EE-local pose and the
    deviation from what ``GraspPose`` predicted — the deviation is what
    the ``VerifyGrasp`` behaviour turns into a SUCCESS / FAILURE.

    Fields:
        position_in_ee: Item centre expressed in the EE's local frame.
        orientation_in_ee: Item orientation relative to the EE's frame
            (quaternion).
        position_error: Euclidean distance (metres) between
            ``position_in_ee`` and the expected
            ``GraspPose.ee_offset_world_at_grasp``-derived vector.
        orientation_error_rad: Absolute rotation magnitude between the
            measured orientation and the expected pick-time relative
            orientation (radians).
    """

    position_in_ee: np.ndarray
    orientation_in_ee: np.ndarray
    position_error: float
    orientation_error_rad: float

    def __post_init__(self) -> None:
        self.position_in_ee = np.asarray(self.position_in_ee, dtype=float).copy()
        self.orientation_in_ee = np.asarray(self.orientation_in_ee, dtype=float).copy()
        self.position_error = float(self.position_error)
        self.orientation_error_rad = float(self.orientation_error_rad)


@dataclass
class PlacePose:
    """Per-(pick, target) place affordance: hover EE pose + descent plan.

    Produced by ``compute_place_pose`` (step 3) after a successful pick.
    Consumes the measured ``ItemInEEPose`` so the insert Z lands the
    held item's bottom on the target's top surface even when the grasp
    offset differs from the pre-grasp expectation.

    Fields:
        ee_position: World-frame hover position (the pose
            ``CortexMoveToPlace`` converges on; i.e. ``insert_z + above``
            along world Z by default).
        ee_orientation: Target drop orientation.
        approach_direction: Unit vector along which the descent into the
            place pose is performed (typically ``[0, 0, -1]``).
        approach_distance: Length (metres) of the descent funnel.
        approach_std_dev: Gaussian std-dev for the RMPFlow descent funnel.
        insert_z: Final EE Z (world frame) at which the gripper opens.
            Equal to ``target_top_z + carried_rest_height`` with any
            ``item_in_ee`` correction already applied.
    """

    ee_position: np.ndarray
    ee_orientation: np.ndarray
    approach_direction: np.ndarray
    approach_distance: float
    approach_std_dev: float
    insert_z: float

    def __post_init__(self) -> None:
        self.ee_position = np.asarray(self.ee_position, dtype=float).copy()
        self.ee_orientation = np.asarray(self.ee_orientation, dtype=float).copy()
        direction = np.asarray(self.approach_direction, dtype=float).copy()
        norm = float(np.linalg.norm(direction))
        if norm <= 0.0:
            raise ValueError("approach_direction must be non-zero")
        self.approach_direction = direction / norm
        self.approach_distance = float(self.approach_distance)
        self.approach_std_dev = float(self.approach_std_dev)
        self.insert_z = float(self.insert_z)

    @property
    def pre_place_position(self) -> np.ndarray:
        """World-frame position at which the descent segment begins."""
        return self.ee_position - self.approach_direction * self.approach_distance


# ---------------------------------------------------------------------------
# Entry points (pure functions)
# ---------------------------------------------------------------------------


def compute_grasp_pose(
    pick_name: str,
    *,
    pick_position: np.ndarray,
    pick_geometry: Optional["PrimGeometry"],
    pick_orientation_preference: np.ndarray,
    item_position_at_grasp: Optional[np.ndarray] = None,
    item_orientation_at_grasp: Optional[np.ndarray] = None,
    approach_direction: Optional[np.ndarray] = None,
    approach_distance: float = DEFAULT_PICK_APPROACH_DISTANCE,
    approach_std_dev: float = DEFAULT_PICK_APPROACH_STD_DEV,
    grasp_height_fallback: float = DEFAULT_GRASP_HEIGHT_FALLBACK,
    grasp_offset_world: Optional[np.ndarray] = None,
) -> GraspPose:
    """Compute a ``GraspPose`` for a single pick attempt.

    EE flange sits at ``pick_position + [0, 0, grasp_height] +
    grasp_offset_world`` with the strategy-provided pick orientation.
    Grasp height comes from the cached ``PrimGeometry``; when no
    geometry is available the caller is expected to have already
    logged a warning and we fall back to ``grasp_height_fallback`` (a
    silent 2 cm offset, same as the legacy ``_EE_OFFSET_FALLBACK``).

    Args:
        pick_name: Item name (only used in diagnostics).
        pick_position: World-frame centre of the item to grasp.
        pick_geometry: Cached ``PrimGeometry`` for this item, or ``None``
            when no geometry is available (see fallback above).
        pick_orientation_preference: Desired EE orientation at grasp
            time (quaternion [w, x, y, z]).  Typically provided by
            ``MultiPickStrategy.get_end_effector_orientation``.
        item_position_at_grasp: Optional item centre in world frame at
            grasp time (as observed by ``PrepareGrasp``).  Stamped onto
            the returned ``GraspPose`` for downstream use by
            ``compute_item_in_ee_pose``.
        item_orientation_at_grasp: Optional item orientation (quaternion)
            in world frame at grasp time.  When present,
            ``compute_item_in_ee_pose`` can compute a real
            ``orientation_error_rad``.
        approach_direction: Unit vector for the approach segment.
            Defaults to ``[0, 0, -1]`` (descend from above).
        approach_distance: Length of the approach segment.
        approach_std_dev: RMPFlow funnel std-dev for the approach.
        grasp_height_fallback: Grasp height to use when
            ``pick_geometry is None``.
        grasp_offset_world: Optional 3-vector added to the canonical
            ``[0, 0, grasp_height]`` flange offset.  Caller supplies it
            in world frame (already rotated by the item's reference
            orientation) — typically via
            ``TaskContextBase._grasp_offset_world``.  Default ``None``
            preserves the legacy "grasp at center" formula.

    Returns:
        A fully-populated ``GraspPose``.
    """
    if pick_geometry is not None:
        grasp_height = float(pick_geometry.grasp_height)
    else:
        grasp_height = float(grasp_height_fallback)
    ee_offset_world = np.array([0.0, 0.0, grasp_height])
    if grasp_offset_world is not None:
        offset_arr = np.asarray(grasp_offset_world, dtype=float).reshape(-1)
        if offset_arr.shape[0] == 3:
            ee_offset_world = ee_offset_world + offset_arr
    ee_position = np.asarray(pick_position, dtype=float) + ee_offset_world
    direction = (
        approach_direction if approach_direction is not None
        else DEFAULT_PICK_APPROACH_DIRECTION
    )
    return GraspPose(
        ee_position=ee_position,
        ee_orientation=pick_orientation_preference,
        approach_direction=direction,
        approach_distance=approach_distance,
        approach_std_dev=approach_std_dev,
        ee_offset_world_at_grasp=ee_offset_world,
        item_position_at_grasp=item_position_at_grasp,
        item_orientation_at_grasp=item_orientation_at_grasp,
    )


def compute_item_in_ee_pose(
    *,
    pick_obj,
    ee_pose,
    expected_grasp_pose: "GraspPose",
) -> ItemInEEPose:
    """Measure the held item's relative pose w.r.t. the EE, post-lift.

    Computes the current item-to-EE offset in world frame and compares
    it to the expected ``ee_offset_world_at_grasp`` recorded on the
    ``GraspPose``.  A large ``position_error`` indicates grasp slippage
    or a failed grasp (e.g. the gripper closed but did not attach to
    the object, and the item has remained on the bin while the EE
    lifted away).

    Note on EE orientation: this step assumes the EE orientation has
    not changed between grasp and lift (the default post-pick lift is
    a pure Z translation via ``CortexMoveRelative`` with
    ``use_world_frame=True``).  If that assumption is relaxed in the
    future, rotate the expected offset by
    ``R_ee_current · R_ee_graspᵀ`` before comparing.

    Args:
        pick_obj: Scene object (``LightweightObj``, sim prim, …) with
            a ``get_world_pose()`` method returning ``(p, q)``.
        ee_pose: Current end-effector pose (``PosePq``-like — anything
            with ``.p`` and ``.q``).
        expected_grasp_pose: The ``GraspPose`` cached at grasp time.

    Returns:
        A fully-populated ``ItemInEEPose``.  ``orientation_error_rad``
        is ``0.0`` until we start recording the grasp-time item
        orientation (follow-up).
    """
    item_world_p, item_world_q = pick_obj.get_world_pose()
    item_world_p = np.asarray(item_world_p, dtype=float)
    item_world_q = np.asarray(item_world_q, dtype=float)
    ee_p = np.asarray(ee_pose.p, dtype=float)
    ee_q = np.asarray(ee_pose.q, dtype=float)

    # Measured world-frame offset from item centre to EE flange.
    measured_offset_world = ee_p - item_world_p
    expected_offset = expected_grasp_pose.ee_offset_world_at_grasp
    position_error = float(np.linalg.norm(measured_offset_world - expected_offset))

    # Express measured item position in EE-local frame.
    R_ee = math_util.quat_to_rot_matrix(ee_q)
    item_in_ee_position = R_ee.T @ (item_world_p - ee_p)

    # Item orientation in EE-local frame: q_ee⁻¹ · q_item.
    item_in_ee_q = _quat_multiply(_quat_conjugate(ee_q), item_world_q)

    # Orientation error: magnitude of the rotation between the expected
    # item-in-EE orientation (captured at grasp time) and the measured
    # item-in-EE orientation (now).  When the grasp-time item orientation
    # was not recorded on the GraspPose, we cannot compute this — keep
    # the back-compat 0.0 so callers that never populated that field
    # still work.
    if expected_grasp_pose.item_orientation_at_grasp is not None:
        q_ee_grasp = expected_grasp_pose.ee_orientation
        q_item_world_at_grasp = expected_grasp_pose.item_orientation_at_grasp
        expected_item_in_ee_q = _quat_multiply(
            _quat_conjugate(q_ee_grasp), q_item_world_at_grasp,
        )
        rel_q = _quat_multiply(
            _quat_conjugate(expected_item_in_ee_q), item_in_ee_q,
        )
        orientation_error_rad = _quat_angle(rel_q)
    else:
        orientation_error_rad = 0.0

    return ItemInEEPose(
        position_in_ee=item_in_ee_position,
        orientation_in_ee=item_in_ee_q,
        position_error=position_error,
        orientation_error_rad=orientation_error_rad,
    )


def compute_place_pose(
    pick_name: str,
    target_name: str,
    *,
    target_position: np.ndarray,
    pick_geometry: Optional["PrimGeometry"],
    pick_orientation: np.ndarray,
    drop_orientation: Optional[np.ndarray],
    item_in_ee: Optional["ItemInEEPose"] = None,
    above: float = 0.0,
    approach_direction: Optional[np.ndarray] = None,
    approach_distance: float = DEFAULT_PLACE_APPROACH_DISTANCE,
    approach_std_dev: float = DEFAULT_PLACE_APPROACH_STD_DEV,
    grasp_height_fallback: float = DEFAULT_GRASP_HEIGHT_FALLBACK,
    grasp_offset_world: Optional[np.ndarray] = None,
) -> PlacePose:
    """Compute a ``PlacePose`` for the place phase of a single cycle.

    Preserves the pre-refactor drop-side EE-offset math exactly.  The
    caller is expected to have already computed the drop-Z-adjusted
    ``target_position[2]`` (= ``target_z + target.top_surface_height +
    pick.rest_height``) via the geometry cache — typically through
    ``TaskContextBase.get_placing_info``.  In a later refactor pass the
    insert-Z math will move here, but for the step-3 migration keeping
    it on the caller preserves byte-for-byte output equivalence.

    Drop-side EE offset handling:

    - When ``drop_orientation is None`` the drop inherits the pick
      orientation and the offset equals ``[0, 0, grasp_height]``.
    - When ``drop_orientation == pick_orientation`` (double-cover aware)
      the rotation short-circuits to the same offset.
    - Otherwise the offset is rotated by ``R_drop · R_pickᵀ`` so the
      flange lands at the correct world-frame position for the rotated
      held item (e.g. ``BottlePickStrategy`` flipping 90° about X).
    - When ``pick_geometry is None`` we fall back to
      ``[0, 0, grasp_height_fallback]`` (silent 2 cm offset, matching
      the legacy ``_EE_OFFSET_FALLBACK``).

    Args:
        pick_name, target_name: Diagnostics only.
        target_position: World-frame target placement position whose Z
            is the insert-Z base assuming an upright drop — i.e.
            ``target_top_z + pick.rest_height`` (typically provided by
            ``TaskContextBase.get_placing_info``).  XY should already
            include any live-target XY-lead correction the caller wanted.
        pick_geometry: Cached ``PrimGeometry`` for the held item, or
            ``None`` → fallback grasp height.
        pick_orientation: EE orientation used at grasp time (quaternion).
        drop_orientation: Strategy's drop-orientation preference.  When
            ``None`` the drop inherits the pick orientation.
        item_in_ee: Optional measured item-in-EE pose from
            ``VerifyGrasp``.  When supplied, the EE drop pose is
            derived from the measured flange-to-item vector so the
            item centre lands on ``target_position`` at drop.  For a
            perfectly-aligned rigid grasp this coincides with the
            nominal-geometry result; the benefit shows up when
            ``position_in_ee`` deviates from nominal (item slipped
            laterally in the gripper).  If ``pick_geometry`` carries
            a non-None ``symmetry``, an orientation-restoring EE
            correction is also applied — the inverse of the
            observable (non-symmetric) portion of the grasp-time
            jostle — so the item lands at its reference orientation
            (modulo symmetry).  When ``None`` the nominal-geometry
            fallback runs — preserves teleport mode and any call
            path that has not yet populated a measurement.
        above: Extra Z offset (metres) added to ``ee_position[2]``.
            ``CortexMoveToPlace`` passes ``PLACE_HOVER_ABOVE_Z`` (0.2);
            ``CortexDownToInsert`` passes 0.
        approach_direction, approach_distance, approach_std_dev: RMPFlow
            descent funnel parameters.
        grasp_height_fallback: Used when ``pick_geometry is None``.

    Returns:
        A fully-populated ``PlacePose``.  ``insert_z`` is the Z at which
        the gripper opens (= ``ee_position[2]`` when ``above == 0``).
    """
    final_orient = (
        drop_orientation if drop_orientation is not None else pick_orientation
    )

    use_measured = item_in_ee is not None
    if use_measured:
        # Defensive: a zero-magnitude ``position_in_ee`` would collapse
        # the EE onto the target centre.  Pre-Step-4 teleport wrote
        # exactly this placeholder; Step 4 switched to ``None``, so
        # hitting this branch now indicates a surprise.  Fall back to
        # nominal and warn once.
        pos_mag = float(np.linalg.norm(item_in_ee.position_in_ee))
        if pos_mag < 1e-4:
            import logging  # local import — perception_utils stays lean
            logging.getLogger(__name__).warning(
                "compute_place_pose: zero-magnitude position_in_ee for "
                "pick=%r target=%r — falling back to nominal geometry",
                pick_name, target_name,
            )
            use_measured = False

    if use_measured:
        # **Orientation-restoring EE correction — symmetry-aware.**
        #
        # When the item was jostled off its reference orientation at
        # grasp time (e.g. bottle settled tilted in the bin), a rigid
        # grasp carries that tilt through the lift.  Without
        # correction the item lands tilted.  The naive fix (pre-rotate
        # the EE by the full inverse-delta) works for asymmetric items
        # but overcorrects for items with rotational symmetry (e.g.
        # bottles rolling about their long axis), driving the wrist
        # toward unreachable poses.  Per-asset symmetry metadata
        # (``pick_geometry.symmetry``) lets us project out the
        # unobservable part of the jostle before applying the inverse.
        #
        # Untagged assets (``symmetry is None``) hit no branch —
        # ``final_orient`` is left unchanged, matching the prior
        # deferred-correction behaviour.
        if (
            pick_geometry is not None
            and pick_geometry.symmetry is not None
            and pick_geometry.reference_orientation is not None
        ):
            symmetry = pick_geometry.symmetry
            if symmetry.kind == "full":
                # Every rotation is a symmetry — no correction ever needed.
                pass
            elif symmetry.kind == "continuous_axis" and symmetry.axis_local is not None:
                q_ref = np.asarray(pick_geometry.reference_orientation, dtype=float)
                # Nominal item-in-EE if the item were exactly at reference:
                q_ref_in_ee = _quat_multiply(
                    _quat_conjugate(pick_orientation), q_ref,
                )
                # Full jostle rotation in EE frame (measured vs nominal):
                q_jostle_ee = _quat_multiply(
                    item_in_ee.orientation_in_ee, _quat_conjugate(q_ref_in_ee),
                )
                # Symmetry axis expressed in EE frame at reference attitude:
                R_ref_in_ee = math_util.quat_to_rot_matrix(q_ref_in_ee)
                axis_ee = R_ref_in_ee @ np.asarray(
                    symmetry.axis_local, dtype=float,
                )
                axis_norm = float(np.linalg.norm(axis_ee))
                if axis_norm > 1e-9:
                    axis_ee = axis_ee / axis_norm
                    q_swing, _ = _swing_twist_decomp(q_jostle_ee, axis_ee)
                    # Correct the drop orientation by the inverse swing
                    # (observable tilt).  The twist component (rotation
                    # about the symmetry axis) is absorbed by the symmetry
                    # and left unapplied — keeping the commanded EE pose
                    # reachable.
                    final_orient = _quat_multiply(
                        final_orient, _quat_conjugate(q_swing),
                    )
            # Other kinds: no-op in this first pass (correction stays disabled).

        R_drop = math_util.quat_to_rot_matrix(final_orient)

        # Position correction: place the item centre at ``target_position``
        # (which already bakes in ``target_top_z + pick.rest_height``),
        # given the measured flange-to-item vector in EE-local frame.
        # For a rigid grasp held from lift to drop,
        # ``item_world_p = ee_p + R_ee · position_in_ee``.  Inverting:
        ee_position = (
            np.asarray(target_position, dtype=float)
            - R_drop @ item_in_ee.position_in_ee
        )
    else:
        # Nominal fallback: drop-side EE offset derived from geometry.
        # Identity short-circuit when ``drop_orientation`` is None or
        # matches the pick orientation.
        if pick_geometry is not None:
            grasp_height = float(pick_geometry.grasp_height)
        else:
            grasp_height = float(grasp_height_fallback)
        pick_offset_world = np.array([0.0, 0.0, grasp_height])
        if grasp_offset_world is not None:
            offset_arr = np.asarray(grasp_offset_world, dtype=float).reshape(-1)
            if offset_arr.shape[0] == 3:
                pick_offset_world = pick_offset_world + offset_arr
        if (
            drop_orientation is None
            or _quaternions_equivalent(drop_orientation, pick_orientation)
        ):
            drop_offset = pick_offset_world
        else:
            drop_offset = _rotate_offset_by_rel_quat(
                pick_offset_world, pick_orientation, drop_orientation,
            )
        ee_position = np.asarray(target_position, dtype=float).copy()
        ee_position += drop_offset

    insert_z = float(ee_position[2])
    if above > 0.0:
        ee_position = ee_position.copy()
        ee_position[2] += float(above)

    direction = (
        approach_direction if approach_direction is not None
        else DEFAULT_PLACE_APPROACH_DIRECTION
    )
    return PlacePose(
        ee_position=ee_position,
        ee_orientation=final_orient,
        approach_direction=direction,
        approach_distance=approach_distance,
        approach_std_dev=approach_std_dev,
        insert_z=insert_z,
    )
