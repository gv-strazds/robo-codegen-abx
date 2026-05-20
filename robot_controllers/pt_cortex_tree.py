"""Factory function for building a cortex-style py_trees behaviour tree.

Alternative to pt_task_tree.make_task_controller_tree() that uses
MotionCommand-based behaviours (threshold-checked completion) instead of
the 9-phase time-interpolated pick-place sequence.

Based on the tree structure from pt_experiments/pt_pickplace.py, adapted to
use the existing task orchestration behaviours (SelectNextPick, etc.).
"""
import numpy as np
import py_trees

from robot_controllers.pt_cortex_behaviours import (
    CortexCloseGripper,
    CortexDownToInsert,
    CortexExecuteApproach,
    CortexMoveRelative,
    CortexMoveToPlace,
    CortexMoveToPreGrasp,
    CortexOpenGripper,
)
from robot_controllers.pt_cortex_perception_behaviours import (
    CheckGraspPoseReachable,
    CheckPickReachable,
    DeferPickAndRelease,
    HaveItemInGripper,
    PrepareGrasp,
    PreparePlacement,
    VerifyGrasp,
)
from robot_controllers.pt_context_monitor import ContextMonitorBehaviour
from robot_controllers.pt_sim_time_decorators import (
    SimTimer,
    sim_timeout_to_success,
)
from robot_controllers.pt_task_behaviours import (
    CheckAllDone,
    CheckCycleProgress,
    CheckTargetAvailable,
    IsPickReachableGuard,
    LatchCurrentPick,
    LatchPlacementTarget,
    MarkPickComplete,
    SelectNextPick,
    SetTaskFinished,
    SnagDetectionGuard,
    WaitForCycleTime,
)
from task_context_base import (
    DEFAULT_APPROACH_TIMEOUT_S,
    DEFAULT_INSERT_TIMEOUT_S,
    DEFAULT_MOVE_TIMEOUT_S,
)

# Default relative lift distances consumed by this tree.  Named
# separately so per-phase tuning is a one-line change.
#
# The two transport lifts (after pick, after place) are wrapped with
# ``cap_to_ee_height_for_move=True`` below so the absolute Z floor of the
# lift is at least ``context.get_ee_height_for_move()`` — tasks setting a
# higher ``ee_height_for_move`` (for obstacle clearance) get the lift
# *extended*; tasks at the default get only the smaller relative lift.
#
# (The place-side hover above the target is governed by the context, not
# a constant here — see ``perception_utils.DEFAULT_PLACE_HOVER_ABOVE_Z``
# and ``TaskContextBase.get_place_hover_above_z`` /
# ``TaskSpec.place_hover_above_z``.)
POST_PICK_LIFT_Z = 0.13
POST_PLACE_LIFT_Z = 0.13
RECOVERY_LIFT_Z = 0.13

# Maximum number of pick-attempt retries before deferring the item.  Each
# fresh selection of a pick (after a successful neighbour completes, or
# on the second-chance pass) starts a new 5-attempt budget.
PICK_RETRY_BUDGET = 5

# Default no-progress safety-net threshold.  After this many consecutive
# do_one_pick_place cycles with no successful mark_pick_complete,
# CheckCycleProgress aborts the task.  Comfortably above any legitimate
# retry pattern (e.g. waiting for several conveyor items to drift into
# reach) but well below an overnight runaway.
DEFAULT_CYCLE_PROGRESS_THRESHOLD = 50

# Slip threshold for the post-lift VerifyGrasp check (metres).  Larger
# than the approach funnel's std_dev to tolerate normal tracking error
# during the lift; tight enough to flag an item left behind on the bin.
VERIFY_GRASP_SLIP_THRESHOLD = 0.03


def make_cortex_task_controller_tree(fake_fast: bool = False, loose_fit: bool = True,
                                     down_to_insert_above: float = 0.0,
                                     down_to_insert_z_thresh: float = None,
                                     down_to_insert_approach_std_dev: float = None,
                                     down_to_insert_approach_distance: float = None,
                                     down_to_insert_use_approach_funnel: bool = True,
                                     down_to_insert_gap_log_interval_s: float = 1.0,
                                     cycle_progress_threshold: int = DEFAULT_CYCLE_PROGRESS_THRESHOLD,
                                     snag_z_threshold: float = 0.1):
    """Build a cortex-style task behaviour tree.

    Uses MotionCommand-based movement behaviours that check robot_at_target()
    for completion, rather than the 9-phase time-interpolated sequence.

    Structure:
        Parallel("TaskRoot", SuccessOnOne)
        |-- ContextMonitorBehaviour        [always RUNNING, for diagnostics]
        |-- Selector("task_orchestration", memory=False)
            |-- CheckAllDone               [SUCCESS if done]
            |-- Sequence("finish_or_fail", memory=True)
                |-- FailureIsSuccess(Repeat(num_success=-1))
                |   |-- Sequence("do_one_pick_place", memory=True)
                |       |-- WaitForCycleTime    [no-op when min_cycle_time_s=0]
                |       |-- SelectNextPick
                |       |-- CheckTargetAvailable
                |       |-- Selector("pick_or_defer", memory=True)
                |       |   |-- Retry(num_failures=PICK_RETRY_BUDGET)
                |       |   |   |-- Sequence("pick_attempt", memory=True)
                |       |   |       |-- CheckPickReachable
                |       |   |       |-- PrepareGrasp
                |       |   |       |-- CheckGraspPoseReachable
                |       |   |       |-- CortexMoveToPreGrasp
                |       |   |       |-- CortexExecuteApproach
                |       |   |       |-- CortexCloseGripper
                |       |   |       |-- Timer(0.5s)
                |       |   |       |-- LatchCurrentPick
                |       |   |       |-- CortexMoveRelative (lift 0.2m)
                |       |   |       |-- VerifyGrasp
                |       |   |-- FailureIsSuccess(DeferPickAndRelease)
                |       |-- Selector("place_or_recover")
                |       |   |-- guarded_place (Sequence, memory=False)
                |       |   |   |-- SnagDetectionGuard    [fails on target-Z runaway]
                |       |   |   |-- place_item (Sequence)
                |       |   |   |   |-- HaveItemInGripper     [gates the normal branch]
                |       |   |   |   |-- PreparePlacement
                |       |   |   |   |-- LatchPlacementTarget
                |       |   |   |   |-- CortexMoveToPlace (above from context.get_place_hover_above_z())
                |       |   |   |   |-- CortexDownToInsert
                |       |   |   |   |-- CortexOpenGripper
                |       |   |   |   |-- Timer(0.3s)
                |       |   |   |   |-- CortexMoveRelative (lift 0.2m)
                |       |   |-- release_and_skip (Sequence)
                |       |       |-- CortexOpenGripper
                |       |       |-- Timer(0.3s)
                |       |       |-- CortexMoveRelative (lift 0.2m)
                |       |-- MarkPickComplete              [skips deferred picks]
                |-- SetTaskFinished

    Args:
        fake_fast: If True, movement behaviours return SUCCESS immediately
            (skip robot_at_target check) and timer durations are near-zero.
        loose_fit: If True, use looser insertion threshold (0.02 vs 0.01).
        down_to_insert_above: Extra Z offset (metres) added to the commanded
            drop position during the ``CortexDownToInsert`` phase.  The
            default 0.0 places the held item's bottom exactly on the
            target's top surface.  Increase this for tasks that should
            release the item a small distance above the target (e.g. when
            reach or contact forces prevent fully descending to 0).
        down_to_insert_z_thresh: Optional tighter Z-axis tolerance for the
            ``CortexDownToInsert`` SUCCESS check.  When ``None`` it tracks
            ``p_thresh`` (i.e. legacy behaviour: the 3D ``p_thresh`` is the
            only criterion).  Setting a small value (e.g. 0.005 m) prevents
            SUCCESS from firing while the EE is still hovering above the
            commanded descent Z — the held item is then released closer to
            the target surface, avoiding bounce or rolling on release.
            Useful for tasks placing items onto a thin pad/rectangle where
            the loose 3D threshold would otherwise allow ~1-2 cm of hover
            from steady-state RmpFlow Z error.
        down_to_insert_approach_std_dev: Override the RMPFlow approach-funnel
            std_dev (Gaussian width) for the descent.  ``None`` keeps the
            perception default (0.02 m).  Widen to 0.05–0.10 m when the
            cone appears to reject the descent on a moving target
            (symptom: EE descends into contact, then backs up repeatedly
            before the force-timeout fires).
        down_to_insert_approach_distance: Override the approach-funnel
            length.  ``None`` keeps the default (0.20 m).  Shrink to make
            the guided descent segment shorter.
        down_to_insert_use_approach_funnel: When ``False``, the descent
            is a pure target-attraction motion with no approach funnel —
            the most permissive setting.  Useful as a posture-vs-cone
            discriminator experiment.
        down_to_insert_gap_log_interval_s: Wall-clock seconds between
            per-tick ``(fk_z, cmd_z, z_gap, xy_gap)`` diagnostic logs
            during the descent.  ``1.0`` logs once per second; set high
            to silence.  Helps distinguish posture-bias drag (z_gap
            trends down) from cone rejection (z_gap oscillates near a
            fixed value).
        cycle_progress_threshold: After this many consecutive
            ``do_one_pick_place`` cycles without a successful
            ``mark_pick_complete``, ``CheckCycleProgress`` aborts the
            task with FAILURE.  Default 50 — well above any legitimate
            retry pattern, well below an overnight runaway.
        snag_z_threshold: Metres of unexpected upward target motion
            (live world Z vs the snapshot taken on the first guard
            tick) that classifies a placement attempt as snagged.  When
            exceeded, ``SnagDetectionGuard`` aborts the place phase,
            defers the pick (so ``MarkPickComplete`` short-circuits),
            and the cycle falls through to ``release_and_skip``.
            Default 0.1 m — deliberately conservative because triggering
            this all but guarantees a failed pick-place; the threshold
            should be well above any legitimate target motion during
            placement (sub-mm physics noise on stationary / conveyor
            targets, plus contact wiggle during insertion).

    Null-space posture configs for pick and place are supplied by the
    ``TaskContext`` (``get_posture_config("pick"|"place")``) — the
    per-task override mechanism lives there (or on ``TaskSpec``) rather
    than on this factory.

    Returns:
        The root behaviour (Parallel).
    """
    # Context monitor (kept for diagnostics and blackboard compatibility)
    context_monitor = ContextMonitorBehaviour(name="ContextMonitor")

    # Task-level behaviours (reused from existing framework)
    check_all_done = CheckAllDone(name="check_all_done")
    wait_cycle = WaitForCycleTime(name="WaitForCycleTime")
    select_next_pick = SelectNextPick(name="SelectNextPick")
    check_target = CheckTargetAvailable(name="CheckTargetAvailable")
    mark_complete = MarkPickComplete(name="MarkPickComplete")
    set_finished = SetTaskFinished(name="SetTaskFinished")

    # Timer durations
    grip_wait = 0.001 if fake_fast else 0.5
    release_wait = 0.001 if fake_fast else 0.3

    # Sim-time watchdog wrappers for the cortex-move phases.  Duration
    # resolvers read from the live ``TaskContext`` at initialise() time,
    # so per-task ``TaskSpec.{move,approach,insert}_timeout_s`` overrides
    # take effect without any factory-time wiring.  When no context is
    # available (rare; e.g. some unit tests) the decorators fall back to
    # the module-level DEFAULT_*_TIMEOUT_S values.
    def _move_timeout(ctx):
        return ctx.get_move_timeout_s() if ctx is not None else DEFAULT_MOVE_TIMEOUT_S

    def _approach_timeout(ctx):
        return ctx.get_approach_timeout_s() if ctx is not None else DEFAULT_APPROACH_TIMEOUT_S

    def _insert_timeout(ctx):
        return ctx.get_insert_timeout_s() if ctx is not None else DEFAULT_INSERT_TIMEOUT_S

    def _watchdog(child, duration_resolver):
        """Wrap *child* in a sim-time watchdog that yields SUCCESS on expiry.

        The ``on_timeout`` callback surfaces the wrapped behaviour's
        ``_timeout_diagnostic()`` payload (target_p, fk_p, distance,
        thresholds) so log lines on a real timeout retain the same
        diagnostic richness the old behaviour-internal force-success
        emitted.
        """
        return sim_timeout_to_success(
            name=child.name,
            child=child,
            duration=duration_resolver,
            on_timeout=child._timeout_diagnostic,
        )

    # --- Pick attempt (wrapped in Retry(PICK_RETRY_BUDGET)) ---
    # PrepareGrasp runs first to compute + cache the GraspPose from the
    # current pick's geometry; both CortexMoveToPreGrasp (freespace
    # setup move) and CortexExecuteApproach (tight funnel descent) then
    # consume that single cached pose.  After the lift, VerifyGrasp
    # checks that the held item actually came along — any FAILURE
    # within the sequence propagates to the Retry decorator.
    pick_attempt = py_trees.composites.Sequence(
        name="pick_attempt", memory=True,
        children=[
            # Reachability gate first: returns RUNNING when the item is
            # still outside the working radius (BT idles at the previous
            # commanded pose) and FAILURE+defer when the item has dropped
            # below the configured Z floor.  Prevents the rest of the
            # pick attempt from chasing items that are physically too far
            # away or have fallen off the conveyor.
            CheckPickReachable(),
            PrepareGrasp(),
            # 3D pre-grasp reachability gate.  CheckPickReachable above
            # tested the *item's* XY against a 2D cylinder; this tests
            # the *commanded EE pre-grasp pose* against a 3D mount-anchored
            # sphere of the same radius.  Closes the high-Z gap where a
            # pose at the cylinder edge climbs out the top of the
            # kinematic shell when the EE has to lift to the pre-grasp
            # altitude (e.g. items tipped near the workspace edge).
            # Idles RUNNING for a grace window so moving-conveyor items
            # can drift into reach before charging the retry budget.
            CheckGraspPoseReachable(),
            _watchdog(CortexMoveToPreGrasp(fake_fast=fake_fast), _move_timeout),
            _watchdog(CortexExecuteApproach(fake_fast=fake_fast), _approach_timeout),
            CortexCloseGripper(name="CortexCloseGripper"),
            SimTimer(name="wait_for_grip", duration=grip_wait),
            LatchCurrentPick(),
            _watchdog(
                CortexMoveRelative(
                    name="lift_after_pick",
                    offset=np.array([0.0, 0.0, POST_PICK_LIFT_Z]),
                    fake_fast=fake_fast,
                    cap_to_ee_height_for_move=True,
                ),
                _move_timeout,
            ),
            VerifyGrasp(slip_threshold=VERIFY_GRASP_SLIP_THRESHOLD),
        ],
    )
    pick_with_retry = py_trees.decorators.Retry(
        name="pick_with_retry",
        child=pick_attempt,
        num_failures=PICK_RETRY_BUDGET,
    )
    # Wrap Retry in a guarded Sequence so a pick that goes permanently
    # unreachable mid-Retry (e.g. CheckPickReachable detects it dropped
    # below the z-floor on attempt 1) aborts the remaining attempts
    # immediately.  ``Sequence(memory=False)`` re-evaluates the guard
    # every tick — standard py_trees pattern for guarding a long-
    # running child (cf. EternalGuard).  When the guard fails, the
    # Sequence fails, the outer Selector falls through to defer_fallback,
    # and the cycle continues without burning the recovery_lift watchdog
    # 5× per fallen item.
    guarded_retry = py_trees.composites.Sequence(
        name="guard_then_retry", memory=False,
        children=[
            IsPickReachableGuard(name="not_permanent"),
            pick_with_retry,
        ],
    )
    # pick_or_defer: guarded retry branch first, deferred fallback second.
    # DeferPickAndRelease returns SUCCESS, and FailureIsSuccess ensures
    # the Selector propagates that as SUCCESS so the outer Sequence
    # continues to place_or_recover (which falls through to the no-op
    # release_and_skip when no item is held).
    defer_fallback = py_trees.decorators.FailureIsSuccess(
        name="defer_on_exhausted",
        child=DeferPickAndRelease(),
    )
    pick_or_defer = py_trees.composites.Selector(
        name="pick_or_defer", memory=True,
        children=[guarded_retry, defer_fallback],
    )

    # --- Place sequence ---
    # HaveItemInGripper gates the normal branch: if a prior deferral
    # left the gripper empty, place_item fails immediately and the
    # Selector falls through to release_and_skip (cheap no-op).
    # PreparePlacement primes the PlacePose cache for downstream reads.
    # LatchPlacementTarget runs before motion so JIT strategies (e.g.
    # ConveyorProximityStrategy) see a stable target throughout the
    # descent.
    place_item = py_trees.composites.Sequence(
        name="place_item", memory=True,
        children=[
            HaveItemInGripper(),
            PreparePlacement(),
            LatchPlacementTarget(),
            _watchdog(CortexMoveToPlace(fake_fast=fake_fast), _move_timeout),
            # Insert wrap is non-optional — CortexDownToInsert always demotes
            # SUCCESS→RUNNING when Z is above commanded, so without the
            # decorator the descent could RUNNING indefinitely if RMPFlow
            # cannot close the gap.
            _watchdog(
                CortexDownToInsert(
                    above=down_to_insert_above,
                    loose_fit=loose_fit, fake_fast=fake_fast,
                    z_thresh=down_to_insert_z_thresh,
                    approach_std_dev=down_to_insert_approach_std_dev,
                    approach_distance=down_to_insert_approach_distance,
                    use_approach_funnel=down_to_insert_use_approach_funnel,
                    gap_log_interval_s=down_to_insert_gap_log_interval_s,
                ),
                _insert_timeout,
            ),
            CortexOpenGripper(name="CortexOpenGripper"),
            SimTimer(name="wait_for_release", duration=release_wait),
            _watchdog(
                CortexMoveRelative(
                    name="lift_after_place",
                    offset=np.array([0.0, 0.0, POST_PLACE_LIFT_Z]),
                    fake_fast=fake_fast,
                    # Floor + ceiling: the post-place lift target is exactly
                    # ee_height_for_move (matching the default tree's transport
                    # semantic).  Floor handles low places (raise to transport
                    # altitude); ceiling handles tall stacks (don't push the
                    # wrist beyond reach when natural = place_z + grasp_height
                    # + relative_offset would exceed ee_height_for_move).
                    cap_to_ee_height_for_move=True,
                    cap_max_to_ee_height_for_move=True,
                ),
                _move_timeout,
            ),
        ],
    )

    # Recovery branch: if place_item fails (e.g. target became unreachable
    # mid-placement and no re-target was possible), release the held item
    # before continuing to the next pick cycle.
    release_and_skip = py_trees.composites.Sequence(
        name="release_and_skip", memory=True,
        children=[
            CortexOpenGripper(name="recovery_open_gripper"),
            SimTimer(name="recovery_wait", duration=release_wait),
            _watchdog(
                CortexMoveRelative(
                    name="recovery_lift",
                    offset=np.array([0.0, 0.0, RECOVERY_LIFT_Z]),
                    fake_fast=fake_fast,
                ),
                _move_timeout,
            ),
        ],
    )

    # Wrap place_item in a memory=False Sequence with a SnagDetectionGuard
    # first child.  The guard ticks before place_item every iteration and
    # returns FAILURE if the live target Z has risen by more than
    # snag_z_threshold above its first-tick snapshot — the signature of the
    # held item dragging the target upward.  On FAILURE the Sequence halts
    # the running CortexMoveToPlace / CortexDownToInsert (status=INVALID),
    # and the outer Selector falls through to release_and_skip.  Same
    # guarded-Sequence pattern used by guard_then_retry above.
    guarded_place = py_trees.composites.Sequence(
        name="guarded_place", memory=False,
        children=[
            SnagDetectionGuard(snag_z_threshold=snag_z_threshold),
            place_item,
        ],
    )

    # Try normal placement; if it fails (target lost mid-cycle, or snag
    # detected by the guard above), release the held item so the gripper
    # is free for the next pick.  The Selector succeeds if either branch
    # succeeds, keeping the Repeat alive.
    place_or_recover = py_trees.composites.Selector(
        name="place_or_recover", memory=True,
        children=[guarded_place, release_and_skip],
    )

    # do_one_pick_place: progress-check -> wait -> select -> check ->
    # pick_or_defer -> place -> mark.  A deferred pick flows through
    # place_or_recover's recovery branch (HaveItemInGripper fails →
    # release_and_skip) and MarkPickComplete short-circuits so the pick
    # remains in the deferred set until a neighbour completes (clearing
    # the deferral) or the second-chance pass in _scan_for_available_pick
    # fires.  CheckCycleProgress at the head bumps the no-progress
    # counter and aborts the Repeat if the threshold is exceeded — the
    # last-line livelock guard for failure modes the per-pick z-floor
    # permanent flag does not catch.
    cycle_progress = CheckCycleProgress(
        name="CheckCycleProgress", threshold=cycle_progress_threshold,
    )
    do_one_pick_place = py_trees.composites.Sequence(
        name="do_one_pick_place", memory=True,
        children=[cycle_progress, wait_cycle, select_next_pick, check_target,
                  pick_or_defer, place_or_recover, mark_complete],
    )

    # Repeat indefinitely (exits via FAILURE from SelectNextPick or CheckTargetAvailable)
    repeat_picks = py_trees.decorators.Repeat(
        name="repeat_picks",
        child=do_one_pick_place,
        num_success=-1,
    )

    # Convert terminal FAILURE to SUCCESS so Sequence proceeds to SetTaskFinished
    repeat_until_exhausted = py_trees.decorators.FailureIsSuccess(
        name="repeat_until_exhausted",
        child=repeat_picks,
    )

    finish_or_fail = py_trees.composites.Sequence(
        name="finish_or_fail", memory=True,
        children=[repeat_until_exhausted, set_finished],
    )

    task_orchestration = py_trees.composites.Selector(
        name="task_orchestration", memory=False,
        children=[check_all_done, finish_or_fail],
    )

    root = py_trees.composites.Parallel(
        name="TaskRoot",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
        children=[context_monitor, task_orchestration],
    )

    return root
