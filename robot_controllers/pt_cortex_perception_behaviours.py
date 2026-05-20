"""Perception-side py_trees behaviours for the cortex-style tree.

These behaviours compute grasp / place affordances via ``perception_utils``
and cache the result on the ``TaskContext`` so downstream motion
behaviours (``CortexMoveToPick`` / ``CortexMoveToPlace`` / ``CortexDownToInsert``)
can consume the resolved pose without re-deriving it every tick.

Step 7 of the refactor adds ``PrepareGrasp``, ``VerifyGrasp``,
``DeferPickAndRelease``, and ``HaveItemInGripper``; ``PreparePlacement``
lands in step 9 alongside its motion-behaviour rewire.

All behaviours are task-level (no motion): they manipulate state and
return SUCCESS / FAILURE / RUNNING without calling ``send_motion_command``.
"""
import logging
import math
import time
from typing import Dict, Tuple

import numpy as np
import py_trees

import perception_utils

logger = logging.getLogger(__name__)

# Default grace window (seconds of sim time) for both reachability gates
# (``CheckPickReachable`` radial branch, ``CheckGraspPoseReachable`` 3D
# sphere branch).  At the default conveyor speed (0.015 m/s) this lets a
# moving item drift ~0.15 m of XY — enough to recover a marginal arrival
# at the cylinder edge.  Static items consume the window once and are
# promoted to permanently-unreachable.
DEFAULT_REACH_GRACE_S: float = 10.0


class _ReachGraceTracker:
    """Per-pick grace-window state shared by the reachability gates.

    Both ``CheckPickReachable`` (2D radial cylinder) and
    ``CheckGraspPoseReachable`` (3D pre-grasp sphere) idle RUNNING for
    ``grace_s`` seconds of sim time before promoting a pick to
    permanently-unreachable.  This tracker owns the per-pick first-out-
    of-reach timestamps and the elapsed-time arithmetic; each gate
    supplies its own reachability predicate and diagnostic strings.

    The internal map is intentionally NOT cleared in any behaviour's
    ``initialise()`` — that would reset the grace timer on every Retry
    restart, multiplying the wait by ``PICK_RETRY_BUDGET`` for free.
    Per-pick entries are popped automatically on ``IN_REACH`` and
    ``EXPIRED`` transitions inside ``evaluate``; on the ``EXPIRED``
    transition the caller is also responsible for calling
    ``_mark_pick_permanent`` and logging the gate-specific diagnostic.

    Supports dict-style ``__contains__`` / ``__getitem__`` /
    ``__setitem__`` so tests can back-date the seed timestamp to force
    grace expiry without constructing additional helpers.
    """

    # Phase return values from evaluate().
    IN_REACH = "in_reach"
    FIRST_OUT = "first_out"
    WAITING = "waiting"
    EXPIRED = "expired"

    def __init__(self, grace_s: float = DEFAULT_REACH_GRACE_S):
        self.grace_s = float(grace_s)
        self._since: Dict[str, float] = {}

    def __contains__(self, pick_name: str) -> bool:
        return pick_name in self._since

    def __getitem__(self, pick_name: str) -> float:
        return self._since[pick_name]

    def __setitem__(self, pick_name: str, t: float) -> None:
        self._since[pick_name] = float(t)

    def clear(self, pick_name: str) -> None:
        """Drop any pending grace-window entry for ``pick_name``.

        Use when a terminal disposition is decided by some path *other*
        than the grace window itself (e.g. z-floor promotion in
        ``CheckPickReachable``).  No-op when the pick is not tracked.
        """
        self._since.pop(pick_name, None)

    def evaluate(
        self, pick_name: str, in_reach: bool, now_sim: float,
    ) -> Tuple[str, float]:
        """Classify the current observation and update internal state.

        Returns ``(phase, elapsed)`` where ``phase`` is one of
        ``IN_REACH | FIRST_OUT | WAITING | EXPIRED``.  ``elapsed`` is
        seconds since the first out-of-reach observation (0.0 for
        ``IN_REACH`` and ``FIRST_OUT``).  Side-effects: pops on
        ``IN_REACH`` and ``EXPIRED``; seeds on ``FIRST_OUT``.  Caller is
        responsible for logging diagnostics and (on ``EXPIRED``) calling
        ``_mark_pick_permanent``.
        """
        if in_reach:
            self._since.pop(pick_name, None)
            return self.IN_REACH, 0.0
        first_t = self._since.get(pick_name)
        if first_t is None:
            self._since[pick_name] = now_sim
            return self.FIRST_OUT, 0.0
        elapsed = now_sim - first_t
        if elapsed < self.grace_s:
            return self.WAITING, elapsed
        self._since.pop(pick_name, None)
        return self.EXPIRED, elapsed


def _mark_pick_permanent(ctx, pick_name: str, behaviour_name: str) -> None:
    """Promote pick to permanently-unreachable, swallowing strategy errors.

    Shared terminal handling for the three permanent-promotion branches
    in this module: ``CheckPickReachable`` z-floor, ``CheckPickReachable``
    radial grace-expiry, ``CheckGraspPoseReachable`` sphere grace-expiry.
    Caller logs the branch-specific diagnostic before invoking this
    helper (so the warning here is reserved for the unexpected strategy
    failure case).
    """
    try:
        ctx.strategy.mark_pick_permanently_unreachable(pick_name)
    except Exception as exc:
        logger.warning(
            f"{behaviour_name}: strategy.mark_pick_permanently_unreachable("
            f"{pick_name!r}) raised {exc!r}"
        )


class _PerceptionBehaviour(py_trees.behaviour.Behaviour):
    """Base class — receives ``context`` and optionally ``gripper_commander`` via setup."""

    def __init__(self, name: str):
        super().__init__(name=name)
        self._context = None
        self._gripper_commander = None

    def setup(self, **kwargs) -> None:
        if "context" in kwargs:
            self._context = kwargs["context"]
        if "gripper_commander" in kwargs and kwargs["gripper_commander"] is not None:
            self._gripper_commander = kwargs["gripper_commander"]
        elif self._context is not None:
            self._gripper_commander = getattr(self._context, "gripper_commander", None)


class CheckPickReachable(_PerceptionBehaviour):
    """Gate the pick attempt on item reachability.

    Inserted at the head of ``pick_attempt`` (before ``PrepareGrasp``) so
    the BT never starts a grasp approach against an item that is either
    currently outside the UR10's working radius or has dropped below the
    pick-side Z floor.

    Returns:
        SUCCESS — current pick is within the working radius and (if
            configured) above the Z floor; the rest of the
            pick_attempt sequence may proceed.  Per-pick grace state is
            cleared.
        RUNNING — pick is currently outside the XY working radius but
            still above the Z floor, AND the per-pick grace window has
            not yet expired.  The Sequence has memory=True, so the BT
            parks here and re-evaluates each tick; the EE keeps its
            last commanded pose (typically the previous lift altitude)
            until the item drifts into reach.
        FAILURE — pick is below the Z floor (dropped off the conveyor
            or otherwise permanently out of reach), OR the radial grace
            window has expired without the item drifting into reach
            (static conveyor / item spawned just outside the cylinder).
            In both terminal branches the gate calls
            ``strategy.mark_pick_permanently_unreachable(pick_name)`` so
            ``IsPickReachableGuard`` short-circuits subsequent Retry
            attempts and the cycle advances to the next pick.

    Grace-window state is owned by ``_unreachable_since`` (a
    ``_ReachGraceTracker``); see that class for the don't-clear-in-
    initialise rationale.
    """

    # Throttled "still waiting" log so a long out-of-reach hold doesn't
    # spam one DEBUG line per BT tick.
    _WAIT_LOG_INTERVAL_S = 2.0

    # Class attribute kept for backwards compatibility with tests / external
    # references.  Authoritative value lives on ``_ReachGraceTracker.grace_s``.
    UNREACHABLE_GRACE_S: float = DEFAULT_REACH_GRACE_S

    def __init__(self, name: str = "CheckPickReachable"):
        super().__init__(name=name)
        self._last_wait_log_t = 0.0
        self._unreachable_since = _ReachGraceTracker(self.UNREACHABLE_GRACE_S)

    def update(self) -> py_trees.common.Status:
        ctx = self._context
        if ctx is None:
            return py_trees.common.Status.FAILURE
        # Mock harness: physics aren't simulated, so a moving conveyor
        # never advances and the gate would block forever on items
        # spawned out of reach.  Skip the check entirely.
        if getattr(ctx, "mock_mode", False):
            return py_trees.common.Status.SUCCESS
        pick_name = ctx.get_current_pick_name()
        if pick_name is None:
            return py_trees.common.Status.FAILURE

        # Short-circuit: the pick was already flagged permanently
        # unreachable on a prior tick (e.g. earlier z-floor failure inside
        # the Retry).  Fail fast without re-querying position or
        # re-logging — the cursor will advance past this pick on the next
        # SelectNextPick.
        if ctx.strategy.is_pick_permanently_unreachable(pick_name):
            return py_trees.common.Status.FAILURE

        pick_pos = ctx.get_picking_position(pick_name)
        if pick_pos is None:
            return py_trees.common.Status.FAILURE

        # Z-floor (permanent): item has dropped below the conveyor / table
        # surface and is not coming back — physics will not return it.
        # Mark permanently unreachable (which also defers it for in-pass
        # cursor advancement); FAILURE propagates up to the Retry, and
        # subsequent ticks short-circuit via the check above.
        min_z = ctx.get_pick_min_reachable_z()
        if min_z is not None and float(pick_pos[2]) < float(min_z):
            logger.info(
                f"{self.name}: pick '{pick_name}' below z-floor "
                f"(z={float(pick_pos[2]):.3f} < {float(min_z):.3f}); marking permanently unreachable"
            )
            _mark_pick_permanent(ctx, pick_name, self.name)
            # Clear any pending radial-grace state for this pick — its
            # terminal disposition is now decided by the z-floor branch.
            self._unreachable_since.clear(pick_name)
            return py_trees.common.Status.FAILURE

        # Radial reach: item is on the conveyor (or otherwise in the
        # workspace) but currently outside the EE's working radius.  Idle
        # the BT here for a grace window so a moving conveyor can drift
        # the item into reach; if the window expires without recovery
        # (static conveyor / item spawned just outside the cylinder),
        # promote to permanently-unreachable so the cycle advances rather
        # than stalling forever.
        base_xy = ctx.get_robot_base_xy()
        radial = float(np.linalg.norm(
            np.asarray(pick_pos[:2], dtype=float) - base_xy
        ))
        max_radius = ctx.get_pick_max_reachable_radius_xy()
        in_reach = radial <= max_radius
        phase, elapsed = self._unreachable_since.evaluate(
            pick_name,
            in_reach=in_reach,
            now_sim=float(ctx.get_current_sim_time()),
        )
        if phase == _ReachGraceTracker.IN_REACH:
            return py_trees.common.Status.SUCCESS
        if phase == _ReachGraceTracker.FIRST_OUT:
            logger.debug(
                f"{self.name}: pick '{pick_name}' out of reach "
                f"(radial={radial:.3f} > {max_radius:.3f}); waiting up to "
                f"{self._unreachable_since.grace_s:.1f}s for drift"
            )
            self._last_wait_log_t = time.time()
            return py_trees.common.Status.RUNNING
        if phase == _ReachGraceTracker.WAITING:
            now_wall = time.time()
            if now_wall - self._last_wait_log_t > self._WAIT_LOG_INTERVAL_S:
                logger.debug(
                    f"{self.name}: pick '{pick_name}' out of reach "
                    f"(radial={radial:.3f} > {max_radius:.3f}); "
                    f"waiting ({elapsed:.1f}/{self._unreachable_since.grace_s:.1f}s)"
                )
                self._last_wait_log_t = now_wall
            return py_trees.common.Status.RUNNING
        # EXPIRED.
        logger.info(
            f"{self.name}: pick '{pick_name}' out of reach for {elapsed:.1f}s "
            f"(radial={radial:.3f} > {max_radius:.3f}); marking permanently unreachable"
        )
        _mark_pick_permanent(ctx, pick_name, self.name)
        return py_trees.common.Status.FAILURE


class PrepareGrasp(_PerceptionBehaviour):
    """Compute the ``GraspPose`` for the current pick and cache it on the context.

    Called at the start of the pick sub-sequence.  Uses the current
    pick's world position + ``PrimGeometry`` + strategy-preferred pick
    orientation.  Stores the result via
    ``TaskContextBase.set_current_grasp_pose`` so ``CortexMoveToPreGrasp``
    and ``CortexExecuteApproach`` (step 8) can consume it.

    Returns:
        FAILURE if no current pick or no pick position is available
        (no amount of retrying will fix that); otherwise SUCCESS.  A
        missing ``PrimGeometry`` falls through to the perception
        fallback (silent 2 cm offset) and logs via the context's
        existing warn-once path — this preserves legacy behaviour.
    """

    def __init__(self, name: str = "PrepareGrasp"):
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        ctx = self._context
        if ctx is None:
            logger.warning(f"{self.name}: no context, returning FAILURE")
            return py_trees.common.Status.FAILURE
        pick_name = ctx.get_current_pick_name()
        if pick_name is None:
            logger.debug(f"{self.name}: no current pick")
            return py_trees.common.Status.FAILURE
        # Drop the prior attempt's drift-cache snapshot so the first
        # refresh_grasp_pose_position call after this PrepareGrasp doesn't
        # finite-difference across the lift/verify gap (which would yield a
        # wildly inflated drift velocity → runaway pick lead).  The dt cap
        # in _estimate_target_drift_xy already gates the worst case, but
        # clearing here makes the post-retry baseline explicit.
        ctx._target_drift_cache.clear()
        pick_pos = ctx.get_picking_position(pick_name)
        if pick_pos is None:
            logger.warning(f"{self.name}: no pick position for '{pick_name}'")
            return py_trees.common.Status.FAILURE
        pick_geom = ctx._prim_geometry.get(pick_name)
        if pick_geom is None:
            # Fire the warn-once log the legacy offset path does.
            ctx.get_end_effector_offset(pick_name)

        # Capture the item's world pose at grasp time so VerifyGrasp /
        # compute_place_pose can compare against — and correct for —
        # the measured-vs-expected deviation.  Read from the scene
        # object via the duck-typed get_world_pose() interface shared by
        # real prims and LightweightObj.  Skip gracefully when the pick
        # object cannot be resolved; downstream consumers already guard
        # against None.
        pick_obj = ctx.strategy.pick_objs_by_name.get(pick_name)
        item_pos_at_grasp = None
        item_q_at_grasp = None
        if pick_obj is not None:
            try:
                item_pos_at_grasp, item_q_at_grasp = pick_obj.get_world_pose()
            except Exception as exc:
                logger.warning(
                    f"{self.name}: get_world_pose() raised {exc!r} for '{pick_name}'"
                )

        grasp_pose = perception_utils.compute_grasp_pose(
            pick_name,
            pick_position=pick_pos,
            pick_geometry=pick_geom,
            pick_orientation_preference=ctx.get_end_effector_orientation(pick_name),
            item_position_at_grasp=item_pos_at_grasp,
            item_orientation_at_grasp=item_q_at_grasp,
            # Per-task pick-approach funnel std-dev (TaskSpec.pick_approach_std_dev).
            # Default delegates to perception_utils.DEFAULT_PICK_APPROACH_STD_DEV.
            # The cached grasp pose carries this value through to
            # compute_pick_command_for_active_item / compute_pregrasp_command_for_active_item.
            approach_std_dev=ctx.get_pick_approach_std_dev(),
            # Per-task / per-asset grasp-point shift in world frame.  Composed
            # with the canonical [0, 0, grasp_height] flange offset so the EE
            # lands at the intended offset from the bottle's centre (and so
            # ``ee_offset_world_at_grasp`` on the cached GraspPose reflects
            # the expected post-lift item-in-EE vector for VerifyGrasp's
            # deviation check).  Mirrors the wiring on
            # ``TaskContextBase._resolve_grasp_pose``; PrepareGrasp computes
            # the GraspPose directly (it's the cortex-tree primary path) so
            # we duplicate the kwarg here rather than route through
            # ``_resolve_grasp_pose``.
            grasp_offset_world=ctx._grasp_offset_world(pick_name),
        )
        ctx.set_current_grasp_pose(grasp_pose)
        logger.debug(
            f"{self.name}: grasp pose cached for '{pick_name}' "
            f"at {grasp_pose.ee_position.tolist()}"
        )
        return py_trees.common.Status.SUCCESS


class CheckGraspPoseReachable(_PerceptionBehaviour):
    """Gate the pick attempt on 3D reachability of the *commanded* pre-grasp pose.

    Inserted after ``PrepareGrasp`` (which produces the ``GraspPose``)
    and before ``CortexMoveToPreGrasp``.  Tests whether the pose the
    robot's IK must actually reach — the pre-grasp position, ~0.20 m
    above the grasp itself — is within the UR10's 3D reachable envelope,
    approximated as a sphere of radius
    ``context.get_pick_max_reachable_radius_xy()`` anchored at the mount
    point ``(base_xy, base_z)``.

    The complementary ``CheckPickReachable`` (run earlier in the same
    ``pick_attempt`` Sequence) tests the *item's* XY radius against the
    same value as a 2D cylinder; this gate fills the high-Z gap where a
    pose at the cylinder edge climbs out the top of the kinematic shell
    when the EE has to lift to the pre-grasp altitude.

    Conveyor-wait grace period: yields RUNNING (idle) for up to
    ``UNREACHABLE_GRACE_S`` seconds of sim time, mirroring
    ``CheckPickReachable``'s wait-for-drift semantics so a moving
    conveyor item that arrives at the cylinder edge with a momentarily-
    out-of-sphere pre-grasp gets a chance to drift deeper before its
    first attempt is charged against the retry budget.  After grace
    expires the gate calls ``strategy.mark_pick_permanently_unreachable``
    directly — the same terminal path used by ``CheckPickReachable``'s
    z-floor branch.  This skips both the wrapping ``Retry`` budget *and*
    the defer-counter fallback, so a static tipped item is dispatched in
    a single 10 s grace window instead of N×PICK_RETRY_BUDGET cycles.
    Promotion-after-grace is safe because the grace window itself
    already encodes the "is this item drifting in?" question — items
    that move fast enough to recover do so within 10 s; items that
    don't aren't going to.

    Live pose, not cached: the gate computes the pre-grasp position
    each tick from the LIVE ``picking_position``, not from the
    ``GraspPose.pre_grasp_position`` cached at ``PrepareGrasp`` time.
    On a moving conveyor an item drifts during the grace window; reading
    the cached snapshot would freeze ``dist3d`` at its entry-tick value
    and incorrectly promote a still-drifting-in item to permanent.  The
    cached ``ee_offset_world_at_grasp`` and approach geometry are reused
    (they depend on item geometry / grasp orientation, not position).

    Grace-window state is owned by ``_unreachable_since`` (a
    ``_ReachGraceTracker``); see that class for the don't-clear-in-
    initialise rationale.

    Returns:
        SUCCESS — pre-grasp pose is inside the 3D sphere.  Per-pick
            grace state is cleared.
        RUNNING — pre-grasp pose is outside the sphere but the per-pick
            grace window has not yet expired.  Caller idles at this
            gate (Sequence has memory=True).
        FAILURE — pre-grasp pose is outside the sphere and the grace
            window has expired (or no grasp pose is cached).  When the
            window has expired, the strategy is also notified
            (``mark_pick_permanently_unreachable``) so the pick won't
            be reselected.
    """

    # Class attribute kept for backwards compatibility with tests / external
    # references.  Authoritative value lives on ``_ReachGraceTracker.grace_s``.
    UNREACHABLE_GRACE_S: float = DEFAULT_REACH_GRACE_S

    def __init__(self, name: str = "CheckGraspPoseReachable"):
        super().__init__(name=name)
        self._unreachable_since = _ReachGraceTracker(self.UNREACHABLE_GRACE_S)

    @staticmethod
    def _compute_live_pre_grasp_position(ctx, grasp_pose, pick_name: str) -> np.ndarray:
        """Return the pre-grasp world position computed against the LIVE item pose.

        Mirrors ``perception_utils.compute_grasp_pose`` followed by
        ``GraspPose.pre_grasp_position`` but with the up-to-date
        ``picking_position`` instead of the snapshot captured at
        ``PrepareGrasp`` time.  Reuses the cached ``ee_offset_world_at_grasp``
        and ``approach_direction``/``approach_distance`` since those are
        functions of item geometry and grasp orientation (not item position).

        Falls back to the cached ``pre_grasp_position`` if the live pick
        position is unavailable (e.g. transient lookup glitch).
        """
        live_pick_pos = None
        if pick_name:
            try:
                live_pick_pos = ctx.get_picking_position(pick_name)
            except Exception:
                live_pick_pos = None
        if live_pick_pos is None:
            return np.asarray(grasp_pose.pre_grasp_position, dtype=float)
        ee_offset = np.asarray(grasp_pose.ee_offset_world_at_grasp, dtype=float)
        approach_dir = np.asarray(grasp_pose.approach_direction, dtype=float)
        approach_dist = float(grasp_pose.approach_distance)
        ee_position_now = np.asarray(live_pick_pos, dtype=float) + ee_offset
        return ee_position_now - approach_dir * approach_dist

    def update(self) -> py_trees.common.Status:
        ctx = self._context
        if ctx is None:
            return py_trees.common.Status.FAILURE
        # Mock harness: the mock arm doesn't simulate IK, so a sphere
        # check is meaningless — and a strict gate would reject test
        # poses contrived for other behaviours.  Match CheckPickReachable.
        if getattr(ctx, "mock_mode", False):
            return py_trees.common.Status.SUCCESS

        grasp_pose = ctx.get_current_grasp_pose()
        if grasp_pose is None:
            # PrepareGrasp didn't produce one; that path is itself a
            # FAILURE the upstream behaviour already logged.  Don't
            # double-log here.
            return py_trees.common.Status.FAILURE

        pick_name = ctx.get_current_pick_name() or ""

        # Refresh the pre-grasp position against the LIVE item world pose
        # rather than reading the cached ``GraspPose.pre_grasp_position``.
        # ``PrepareGrasp`` only runs once at the start of the wrapping
        # ``Sequence(memory=True)``, so the cached pose is frozen at item-
        # entry time.  On a moving conveyor the item drifts during the
        # grace window; without this refresh the gate sees the same stale
        # dist3d for the full window and incorrectly promotes a still-
        # drifting-in pick to permanent.  We reuse the cached ee_offset
        # and approach-direction/distance (which depend on item geometry
        # and grasp orientation, not item position) and rebuild the
        # pre-grasp position from the live picking_position.
        try:
            base_xy = np.asarray(ctx.get_robot_base_xy(), dtype=float)
            mount_z = float(ctx.get_robot_base_z())
            target = self._compute_live_pre_grasp_position(ctx, grasp_pose, pick_name)
        except Exception as exc:
            logger.warning(f"{self.name}: pose/base lookup raised {exc!r}; passing through")
            return py_trees.common.Status.SUCCESS

        dx = float(target[0]) - float(base_xy[0])
        dy = float(target[1]) - float(base_xy[1])
        dz = float(target[2]) - mount_z
        dist_3d = math.sqrt(dx * dx + dy * dy + dz * dz)
        r_max = float(ctx.get_pick_max_reachable_radius_xy())

        phase, elapsed = self._unreachable_since.evaluate(
            pick_name,
            in_reach=(dist_3d <= r_max),
            now_sim=float(ctx.get_current_sim_time()),
        )
        if phase == _ReachGraceTracker.IN_REACH:
            return py_trees.common.Status.SUCCESS
        if phase == _ReachGraceTracker.FIRST_OUT:
            # Idle for the grace window so a moving conveyor can drift
            # the item deeper into the workspace before charging the
            # retry budget.
            logger.debug(
                f"{self.name}: pre-grasp pose for '{pick_name}' out of sphere "
                f"(dist3d={dist_3d:.3f} > {r_max:.3f}); waiting up to "
                f"{self._unreachable_since.grace_s:.1f}s for drift"
            )
            return py_trees.common.Status.RUNNING
        if phase == _ReachGraceTracker.WAITING:
            return py_trees.common.Status.RUNNING
        # EXPIRED — promote to permanently-unreachable.  Skips the
        # wrapping Retry budget AND the defer counter, so a single grace
        # window bounds the cost; IsPickReachableGuard short-circuits
        # subsequent ticks before any motion is dispatched.
        logger.info(
            f"{self.name}: pre-grasp pose for '{pick_name}' unreachable for {elapsed:.1f}s "
            f"(dist3d={dist_3d:.3f} > {r_max:.3f}); marking permanently unreachable"
        )
        _mark_pick_permanent(ctx, pick_name, self.name)
        return py_trees.common.Status.FAILURE


class VerifyGrasp(_PerceptionBehaviour):
    """Verify that the held item actually came along with the lift.

    Called after the post-pick lift completes.  Measures the current
    item-to-EE offset via ``perception_utils.compute_item_in_ee_pose``
    and compares it to the ``GraspPose`` cached by ``PrepareGrasp``.
    If the gripper explicitly reports ``"empty"`` (no contact), or the
    measured deviation exceeds ``slip_threshold``, returns FAILURE so
    the enclosing ``Retry`` decorator can re-attempt (or defer) the pick.

    A gripper that returns ``"unknown"`` (the common case for adapters
    without contact feedback) does not veto SUCCESS; the pose-based
    check stands on its own.

    Persistent-failure guard: tracks per-pick consecutive FAILURE count
    across ``Retry`` restarts (the count survives ``initialise()``).
    Once the count reaches ``MAX_CONSECUTIVE_FAILURES``, the pick is
    promoted to permanently-unreachable.  This bounds the wasted-time
    cost on items whose grasp keeps failing the same way (e.g. tipped
    boxes where the gripper can't get a flush close): one transient
    retry is allowed; a second consecutive failure flips the pick
    permanent so the cycle ends quickly instead of burning the full
    ``PICK_RETRY_BUDGET × MAX_DEFERS_BEFORE_PERMANENT`` allotment.
    Counter resets on SUCCESS.

    Args:
        slip_threshold: Max allowed Euclidean position error (metres)
            before the grasp is considered slipped.  Default 0.03 m —
            generous enough to tolerate normal RMPFlow tracking error
            during the lift, tight enough to flag a failed grasp where
            the item was left behind.
    """

    # After this many consecutive FAILUREs for the same pick (across
    # Retry restarts), VerifyGrasp promotes the pick to permanent.
    # Threshold of 2 covers one transient blip (e.g. pose noise during
    # the lift settling) while still bounding the cost on stably-failing
    # grasps to ~2 × full-pick-attempt time.
    MAX_CONSECUTIVE_FAILURES: int = 2

    def __init__(
        self, name: str = "VerifyGrasp", slip_threshold: float = 0.03,
    ):
        super().__init__(name=name)
        self._slip_threshold = float(slip_threshold)
        # pick_name -> consecutive FAILURE count.  NOT cleared in
        # initialise() — must survive Retry restarts so the count
        # accumulates across attempts within a single defer cycle.
        # Cleared per-pick on SUCCESS or on grace-expiry promotion.
        self._fail_counts: Dict[str, int] = {}

    def _on_failure(self, ctx, pick_name: str) -> py_trees.common.Status:
        """Bump the consecutive-failure counter and return FAILURE.

        When the counter reaches ``MAX_CONSECUTIVE_FAILURES`` the pick
        is promoted to permanently-unreachable so subsequent ticks of
        the wrapping ``guarded_retry`` Sequence short-circuit at
        ``IsPickReachableGuard`` rather than burning more motion time.
        """
        if not pick_name:
            return py_trees.common.Status.FAILURE
        new_count = self._fail_counts.get(pick_name, 0) + 1
        self._fail_counts[pick_name] = new_count
        if new_count >= self.MAX_CONSECUTIVE_FAILURES:
            logger.info(
                f"{self.name}: '{pick_name}' failed {new_count} consecutive grasps "
                f"(threshold={self.MAX_CONSECUTIVE_FAILURES}); marking permanently unreachable"
            )
            _mark_pick_permanent(ctx, pick_name, self.name)
            self._fail_counts.pop(pick_name, None)
        return py_trees.common.Status.FAILURE

    def update(self) -> py_trees.common.Status:
        ctx = self._context
        if ctx is None:
            return py_trees.common.Status.FAILURE
        pick_name = ctx.get_current_pick_name()
        if pick_name is None:
            logger.warning(f"{self.name}: no current pick")
            return py_trees.common.Status.FAILURE

        grasp_pose = ctx.get_current_grasp_pose()
        if grasp_pose is None:
            logger.warning(
                f"{self.name}: no cached grasp_pose for '{pick_name}'; "
                f"PrepareGrasp must run before VerifyGrasp"
            )
            return py_trees.common.Status.FAILURE

        pick_obj = ctx.strategy.pick_objs_by_name.get(pick_name)
        if pick_obj is None:
            logger.warning(f"{self.name}: no pick object for '{pick_name}'")
            return py_trees.common.Status.FAILURE

        # Short-circuit in teleport mode (no-op arm/gripper commanders) and
        # in mock mode (pure-Python tick-countdown arm whose timing does
        # not align with the wall-clock py_trees.timers.Timer that gates
        # grip/release waits).  In both cases the FK + item-world pose
        # carry no meaningful information about grasp success —
        # mark_pick_complete will snap the item onto its target in
        # teleport mode, and in mock mode the runner separately updates
        # the item pose on completion.  Just flip the grasp-succeeded
        # flag so HaveItemInGripper gates the place subtree correctly.
        # Leave _current_item_in_ee as None so compute_place_pose falls
        # back to its nominal branch — writing a zero placeholder would
        # collapse the EE onto the target centre.
        if ctx.teleport_mode or ctx.mock_mode:
            ctx.set_holding_item(True)
            logger.debug(
                f"{self.name}: {'teleport' if ctx.teleport_mode else 'mock'}_mode — "
                f"skipping pose check for '{pick_name}'"
            )
            return py_trees.common.Status.SUCCESS

        # Gripper feedback gate (best-effort; "unknown" → fall through).
        if self._gripper_commander is not None:
            try:
                gstate = self._gripper_commander.grasp_state()
            except Exception:
                gstate = "unknown"
            if gstate == "empty":
                logger.info(
                    f"{self.name}: gripper reports 'empty' for '{pick_name}' — FAILURE"
                )
                return self._on_failure(ctx, pick_name)

        # Pose-based check.
        ee_pose = ctx.arm_commander.get_fk_pq()
        item_in_ee = perception_utils.compute_item_in_ee_pose(
            pick_obj=pick_obj,
            ee_pose=ee_pose,
            expected_grasp_pose=grasp_pose,
        )
        ctx.set_current_item_in_ee(item_in_ee)

        if item_in_ee.position_error > self._slip_threshold:
            logger.info(
                f"{self.name}: position_error={item_in_ee.position_error:.4f} > "
                f"threshold={self._slip_threshold:.4f} for '{pick_name}' — FAILURE"
            )
            return self._on_failure(ctx, pick_name)

        # Clear the consecutive-failure count on a clean grasp.
        self._fail_counts.pop(pick_name, None)
        ctx.set_holding_item(True)
        logger.debug(
            f"{self.name}: verified grasp of '{pick_name}' "
            f"(position_error={item_in_ee.position_error:.4f})"
        )
        return py_trees.common.Status.SUCCESS


class DeferPickAndRelease(_PerceptionBehaviour):
    """Handle a retry-exhausted pick: release the gripper and defer the pick.

    Wired as the fallback branch of the ``pick_or_defer`` Selector in
    the cortex tree (see step 10).  Opens the gripper (in case a
    partial grasp is still active), then calls
    ``strategy.defer_pick(pick_name)`` so the item is skipped for the
    rest of the current pass; ``mark_pick_complete`` on any subsequent
    successful pick clears the deferral.

    Also resets the cycle cache so stale ``GraspPose`` /
    ``ItemInEEPose`` entries don't contaminate the next attempt.

    Always returns SUCCESS (wrapped in ``FailureIsSuccess`` is not
    needed — this is the explicit recovery path).
    """

    def __init__(self, name: str = "DeferPickAndRelease"):
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        ctx = self._context
        if ctx is None:
            return py_trees.common.Status.SUCCESS
        pick_name = ctx.get_current_pick_name()
        if self._gripper_commander is not None:
            try:
                self._gripper_commander.open()
            except Exception as exc:
                logger.warning(f"{self.name}: gripper.open() raised {exc!r}")
        if pick_name:
            ctx.strategy.defer_pick(pick_name)
            logger.info(f"{self.name}: deferred pick '{pick_name}' after retries exhausted")
        ctx.reset_cycle_cache()
        return py_trees.common.Status.SUCCESS


class PreparePlacement(_PerceptionBehaviour):
    """Compute the initial ``PlacePose`` for the current (pick, target) pair.

    Runs once at the start of the place sub-sequence (before
    ``LatchPlacementTarget``).  Calls
    ``context.compute_dynamic_place_command_for_active_item(above=0)``
    for its side-effect of caching a ``PlacePose`` on the context — so
    downstream behaviours can assume ``get_current_place_pose() is not
    None`` and inspect the resolved orientation / insert-Z / approach
    without re-deriving the math themselves.

    The motion behaviours (``CortexMoveToPlace``, ``CortexDownToInsert``)
    continue to recompute each tick so that XY-lead for moving targets
    (e.g. items on a conveyor) stays responsive; the per-tick
    recomputation refreshes the cache too.  ``PreparePlacement`` exists
    to guarantee *initial* cache population, particularly as a hook
    point for future VerifyPlacement / logging / retry logic.

    Returns FAILURE only when no current pick / no reachable target
    exists — conditions the subsequent motion behaviours would also
    fail on.
    """

    def __init__(self, name: str = "PreparePlacement"):
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        ctx = self._context
        if ctx is None:
            return py_trees.common.Status.FAILURE
        pick_name = ctx.get_current_pick_name()
        if pick_name is None:
            logger.debug(f"{self.name}: no current pick")
            return py_trees.common.Status.FAILURE
        # Fire the command builder for its side-effect cache (above=0 is
        # the canonical insert-Z form).  The actual MotionCommand is
        # discarded — the motion behaviours will refresh it themselves.
        cmd = ctx.compute_dynamic_place_command_for_active_item(above=0.0)
        if cmd is None:
            logger.debug(f"{self.name}: no place command for '{pick_name}'")
            return py_trees.common.Status.FAILURE
        place_pose = ctx.get_current_place_pose()
        if place_pose is None:
            # Defensive: the command builder always populates the cache.
            return py_trees.common.Status.FAILURE
        logger.debug(
            f"{self.name}: place pose cached for '{pick_name}' "
            f"insert_z={place_pose.insert_z:.4f}"
        )
        return py_trees.common.Status.SUCCESS


class HaveItemInGripper(_PerceptionBehaviour):
    """Gate behaviour used as the first child of ``place_item``.

    Returns SUCCESS when the context's grasp-succeeded flag
    (``is_holding_item``) is set by ``VerifyGrasp``, else FAILURE.  The
    FAILURE causes the enclosing ``place_or_recover`` Selector to fall
    through to ``release_and_skip``, which is a no-op when the gripper
    is already empty but leaves the rest of the tree consistent.

    Decoupled from ``get_current_item_in_ee()`` so teleport mode (which
    has no measured pose) can still pass the gate while leaving the
    place-pose cache in its nominal-fallback state.

    Used to avoid running the place sub-sequence after a failed pick
    has been deferred — without this gate the tree would try to place
    a non-existent held item.
    """

    def __init__(self, name: str = "HaveItemInGripper"):
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        ctx = self._context
        if ctx is None:
            return py_trees.common.Status.FAILURE
        if ctx.is_holding_item():
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE
