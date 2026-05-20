"""Task-level py_trees behaviours for multi-pick orchestration.

These behaviours operate at the task level (selecting picks, checking completion,
etc.) and query/mutate the TaskContext directly rather than via the blackboard.
"""
import logging
from typing import Dict, Optional

import py_trees

logger = logging.getLogger(__name__)


class TaskBehaviour(py_trees.behaviour.Behaviour):
    """Base class for task-level behaviours that receive a TaskContext via setup()."""

    def __init__(self, name: str):
        super().__init__(name=name)
        self._context = None

    def setup(self, **kwargs) -> None:
        if "context" in kwargs:
            self._context = kwargs["context"]


class CheckAllDone(TaskBehaviour):
    """Returns SUCCESS if context.task_finished is True, else FAILURE.

    Used as the first child of the task orchestration Selector so that
    once the task is marked done the Selector short-circuits immediately.
    """

    def update(self) -> py_trees.common.Status:
        if self._context is not None and self._context.task_finished:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE


class SelectNextPick(TaskBehaviour):
    """Advance the context to the next pick item.

    On the first tick of a do_one_pick_place cycle this is called.
    - If the current pick index is at the beginning (first pick), just returns SUCCESS
      (the initial index already points to pick 0).
    - Otherwise, advances the pick index.
    - Returns SUCCESS if a pick is available, FAILURE if all picks are exhausted.
    """

    def __init__(self, name: str = "SelectNextPick"):
        super().__init__(name=name)
        self._first_call = True

    def initialise(self) -> None:
        # Each time this node transitions from non-RUNNING to RUNNING,
        # it represents a new pick cycle entry.
        pass

    def setup(self, **kwargs) -> None:
        super().setup(**kwargs)
        self._first_call = True

    def update(self) -> py_trees.common.Status:
        if self._context is None:
            return py_trees.common.Status.FAILURE

        if self._first_call:
            # First cycle: don't advance, just check current
            self._first_call = False
            pick_name = self._context.get_current_pick_name()
        else:
            pick_name = self._context.advance_pick_index()

        if pick_name is None:
            if self._context.strategy.more_items_expected:
                logger.debug("SelectNextPick: waiting for more items from incremental generation")
                return py_trees.common.Status.RUNNING
            logger.debug("SelectNextPick: no more picks available")
            return py_trees.common.Status.FAILURE

        logger.debug(f"SelectNextPick: selected '{pick_name}'")
        return py_trees.common.Status.SUCCESS


class CheckTargetAvailable(TaskBehaviour):
    """Returns SUCCESS if the current pick has a valid, reachable target.

    Returns RUNNING when a target exists but is not yet reachable (e.g.
    still approaching on a conveyor).  When a target is permanently
    unreachable (e.g. fell off the conveyor), advances to the next pick
    and re-ticks.  Returns FAILURE when all picks are exhausted.
    """

    def __init__(self, name: str = "CheckTargetAvailable"):
        super().__init__(name=name)
        # Track the pick name we most recently logged a "waiting" message
        # for, so the log fires once per wait-start rather than every tick.
        self._waiting_logged_for: Optional[str] = None

    def update(self) -> py_trees.common.Status:
        if self._context is None:
            return py_trees.common.Status.FAILURE

        pick_name = self._context.get_current_pick_name()
        if pick_name is None:
            return py_trees.common.Status.FAILURE

        target_name = self._context.get_placing_target_name(pick_name)
        if target_name is not None:
            self._waiting_logged_for = None
            return py_trees.common.Status.SUCCESS

        # Target unavailable — determine why
        strategy = self._context.strategy
        paired_target_name = strategy.pairings_by_pick_name.get(pick_name)

        # Permanently unreachable or occupied by another pick → skip
        if paired_target_name is not None and (
            strategy.is_target_permanently_unreachable(paired_target_name)
            or strategy._is_target_occupied(paired_target_name, exclude_pick=pick_name)
        ):
            reason = (
                "permanently unreachable"
                if strategy.is_target_permanently_unreachable(paired_target_name)
                else "occupied by another pick"
            )
            logger.info(
                "CheckTargetAvailable: target for '%s' %s, "
                "advancing to next pick", pick_name, reason,
            )
            next_name = self._context.advance_pick_index()
            if next_name is not None:
                return py_trees.common.Status.RUNNING
            if strategy.more_items_expected:
                return py_trees.common.Status.RUNNING
            logger.info("CheckTargetAvailable: no available targets remaining")
            self._context.targets_exhausted = True
            return py_trees.common.Status.FAILURE

        # Target exists but not yet reachable (still arriving on conveyor)
        if paired_target_name is not None:
            logger.debug(
                "CheckTargetAvailable: target for '%s' not yet reachable, waiting",
                pick_name,
            )
            return py_trees.common.Status.RUNNING

        # No pairing exists — wait for incremental targets or fail
        if strategy.more_targets_expected:
            if self._waiting_logged_for != pick_name:
                logger.debug(
                    "CheckTargetAvailable: waiting for more targets for '%s'",
                    pick_name,
                )
                self._waiting_logged_for = pick_name
            return py_trees.common.Status.RUNNING
        logger.info(f"CheckTargetAvailable: no target for '{pick_name}'")
        self._context.targets_exhausted = True
        return py_trees.common.Status.FAILURE


class ResetPickPlaceTree(TaskBehaviour):
    """Resets the pick_then_place subtree to INVALID so it re-initialises for the next cycle.

    Holds a reference to the subtree root, set via set_subtree().
    """

    def __init__(self, name: str = "ResetPickPlaceTree"):
        super().__init__(name=name)
        self._subtree = None

    def set_subtree(self, subtree: py_trees.behaviour.Behaviour) -> None:
        self._subtree = subtree

    def update(self) -> py_trees.common.Status:
        if self._subtree is not None and self._subtree.status != py_trees.common.Status.INVALID:
            self._subtree.stop(py_trees.common.Status.INVALID)
        return py_trees.common.Status.SUCCESS


class MarkPickComplete(TaskBehaviour):
    """Marks the current pick as complete in the context.

    Skips deferred picks: when ``DeferPickAndRelease`` has moved the
    current pick into the deferred set (retries exhausted), we must NOT
    mark it completed — otherwise it leaves ``_deferred_picks`` via the
    "any completion clears deferrals" rule without ever being grasped,
    and lands in ``_completed_picks`` where it is permanently skipped.
    Same applies to permanently-unreachable picks (z-floor failures): a
    pick that fell off the conveyor was never grasped, so marking it
    completed would corrupt ``_completed_picks``.  Returning SUCCESS
    without mutating state lets the outer Repeat loop back around to
    ``SelectNextPick``, which skips the deferred / permanent pick for
    the current pass.
    """

    def update(self) -> py_trees.common.Status:
        if self._context is None:
            return py_trees.common.Status.SUCCESS

        pick_name = self._context.get_current_pick_name()
        if pick_name is None:
            return py_trees.common.Status.SUCCESS

        strategy = self._context.strategy
        if strategy.is_pick_permanently_unreachable(pick_name):
            logger.debug(
                f"MarkPickComplete: skipping '{pick_name}' (permanently unreachable)"
            )
            return py_trees.common.Status.SUCCESS

        if strategy.is_pick_deferred(pick_name):
            logger.debug(
                f"MarkPickComplete: skipping '{pick_name}' (deferred this pass)"
            )
            return py_trees.common.Status.SUCCESS

        self._context.mark_pick_complete(pick_name)
        logger.debug(f"MarkPickComplete: completed '{pick_name}'")
        return py_trees.common.Status.SUCCESS


class IsPickReachableGuard(TaskBehaviour):
    """Gate that aborts the per-pick Retry once the pick goes permanent.

    Used by the cortex tree as the head of a ``Sequence(memory=False)``
    wrapping ``Retry(pick_attempt)``: re-evaluated on every tick, so a
    pick that gets flagged permanently unreachable mid-Retry causes the
    wrapping Sequence to fail immediately, the outer ``pick_or_defer``
    Selector falls through to ``DeferPickAndRelease``, and the cycle
    moves on without burning the remaining retry attempts.

    Returns:
        FAILURE — current pick is permanently unreachable (abort retries).
        SUCCESS — current pick is still a candidate (or there is no
            current pick, in which case the wrapping Retry will discover
            that on its own and fail the normal way).
    """

    def __init__(self, name: str = "IsPickReachableGuard"):
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        if self._context is None:
            return py_trees.common.Status.SUCCESS
        pick_name = self._context.get_current_pick_name()
        if pick_name is None:
            return py_trees.common.Status.SUCCESS
        if self._context.strategy.is_pick_permanently_unreachable(pick_name):
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class CheckCycleProgress(TaskBehaviour):
    """Livelock safety net at the head of ``do_one_pick_place``.

    Bumps the strategy's no-progress counter on every cycle entry; when
    the counter exceeds *threshold* without any successful
    ``mark_pick_complete``, returns FAILURE so the outer Repeat exits
    cleanly into ``SetTaskFinished``.  Catches livelock causes that are
    not covered by the per-pick z-floor permanent flag — for example,
    an item that perpetually slips in the gripper but never falls below
    the floor.

    Threshold default is 50 cycles, which is comfortably above any
    legitimate retry pattern but well below an overnight-runaway run.
    Override via ``cycle_progress_threshold`` on the cortex-tree factory.
    """

    def __init__(self, name: str = "CheckCycleProgress", threshold: int = 50):
        super().__init__(name=name)
        self._threshold = int(threshold)

    def update(self) -> py_trees.common.Status:
        if self._context is None:
            return py_trees.common.Status.SUCCESS
        strategy = self._context.strategy
        cycles = strategy.increment_cycle_count()
        if cycles > self._threshold:
            logger.warning(
                "%s: %d cycles without progress (threshold=%d); aborting task",
                self.name, cycles, self._threshold,
            )
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class WaitForCycleTime(TaskBehaviour):
    """Cycle-time floor inserted at the start of each pick-place cycle.

    Holds the cycle in RUNNING until at least ``context.min_cycle_time_s``
    seconds have elapsed since the *start* of the previous cycle (not the
    end), so the gate caps cycle frequency without adding a fixed gap when
    the cycle itself already takes longer than the floor:

      - cycle takes X < min_cycle_time_s → next cycle waits
        ``min_cycle_time_s - X`` extra seconds.
      - cycle takes X ≥ min_cycle_time_s → next cycle starts immediately.

    Always SUCCESS on the first cycle (no previous start to measure from)
    and when the gate is inactive (``min_cycle_time_s == 0``).

    The actual time source is the host-loop-supplied
    ``context.simulation_time`` — mock mode populates it from ``mock_time``
    (see ``MOCK_TICK_HZ``), real sim populates it from
    ``CortexWorld.step()`` simulation time.  ``last_cycle_start_time``
    lives on the context so a single shared clock drives the gate.
    """

    def __init__(self, name: str = "WaitForCycleTime"):
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        if self._context is None:
            return py_trees.common.Status.SUCCESS
        now = float(getattr(self._context, "simulation_time", 0.0) or 0.0)
        min_cycle = float(getattr(self._context, "min_cycle_time_s", 0.0) or 0.0)
        # Inactive gate: still record cycle start so that toggling the gate
        # mid-run (real sim only — mock sets it once) measures from a sane
        # baseline rather than zero.
        if min_cycle <= 0.0:
            self._context.last_cycle_start_time = now
            return py_trees.common.Status.SUCCESS
        last_start = getattr(self._context, "last_cycle_start_time", None)
        if last_start is None:
            # First cycle — release immediately, baseline the clock.
            self._context.last_cycle_start_time = now
            return py_trees.common.Status.SUCCESS
        elapsed = now - float(last_start)
        if elapsed >= min_cycle:
            self._context.last_cycle_start_time = now
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.RUNNING


class SetTaskFinished(TaskBehaviour):
    """Sets context.task_finished = True. Terminal node after repeat loop completes."""

    def update(self) -> py_trees.common.Status:
        if self._context is not None:
            self._context.task_finished = True
            logger.info("SetTaskFinished: task marked as finished")
        return py_trees.common.Status.SUCCESS


class LatchPlacementTarget(TaskBehaviour):
    """Pin the current pick's target for the duration of the place phase.

    Inserted as the first child of the place sequence (after grasp).
    Calls ``strategy.latch_current_target(pick_name)`` so that subsequent
    place behaviours (``MoveToPlace``, ``DownToInsert``, ``OpenGripper``)
    track a stable target rather than being pulled off-course by a
    just-in-time re-selection when a lower-urgency target appears.

    For strategies that don't override ``latch_current_target`` the call
    is a no-op and this behaviour just returns SUCCESS.
    """

    def __init__(self, name: str = "LatchPlacementTarget"):
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        if self._context is None:
            return py_trees.common.Status.SUCCESS
        pick_name = self._context.get_current_pick_name()
        if pick_name is None:
            return py_trees.common.Status.SUCCESS
        self._context.strategy.latch_current_target(pick_name)
        return py_trees.common.Status.SUCCESS


class SnagDetectionGuard(TaskBehaviour):
    """Abort the place phase when the latched target rises with the held item.

    CortexMoveToPlace / CortexDownToInsert command Z = live ``target_z +
    above``, so a held-item-to-target jam creates a positive-feedback runaway:
    the wrist chases the rising target upward.  This guard snapshots the
    target's world Z on its first tick for a given pick; on subsequent ticks
    it fails the surrounding ``Sequence(memory=False)`` when live Z exceeds
    the snapshot by more than ``snag_z_threshold``.  ``defer_pick`` is called
    so the downstream ``MarkPickComplete`` short-circuits and a later cycle
    re-attempts the pick (with a fresh snapshot, because the dict entry is
    popped on detection).

    State lives on the instance (``_latched_z_by_pick``) keyed by pick name.
    ``TaskBehaviour`` does not override ``initialise``, and py_trees' default
    ``initialise`` is a no-op, so this state survives across ticks even
    though the guard returns SUCCESS each tick (which causes py_trees to
    re-enter ``initialise`` on the next tick).  Normally-completed picks go
    into ``_completed_picks`` and are never re-selected, so leftover entries
    are harmless.
    """

    def __init__(self, snag_z_threshold: float = 0.1, name: str = "SnagDetectionGuard"):
        super().__init__(name=name)
        self._snag_z_threshold = snag_z_threshold
        self._latched_z_by_pick: Dict[str, float] = {}

    def update(self) -> py_trees.common.Status:
        if self._context is None:
            return py_trees.common.Status.SUCCESS
        pick_name = self._context.get_current_pick_name()
        if pick_name is None:
            return py_trees.common.Status.SUCCESS
        _, target_obj = self._context.get_placement_target()
        if target_obj is None:
            return py_trees.common.Status.SUCCESS
        try:
            raw_p, _ = target_obj.get_world_pose()
            live_z = float(raw_p[2])
        except Exception:
            return py_trees.common.Status.SUCCESS

        latched_z = self._latched_z_by_pick.get(pick_name)
        if latched_z is None:
            self._latched_z_by_pick[pick_name] = live_z
            return py_trees.common.Status.SUCCESS

        delta = live_z - latched_z
        if delta > self._snag_z_threshold:
            logger.warning(
                f"{self.name}: target for pick '{pick_name}' Z rose from "
                f"{latched_z:.4f} to {live_z:.4f} (delta={delta:+.4f} > "
                f"threshold {self._snag_z_threshold:.4f}); treating as snag, "
                f"deferring pick"
            )
            self._context.strategy.defer_pick(pick_name)
            self._latched_z_by_pick.pop(pick_name, None)
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


class LatchCurrentPick(TaskBehaviour):
    """Pin the current pick for the rest of the pick-then-place cycle.

    Inserted in the pick sequence after ``CortexCloseGripper`` + the
    grip-wait Timer and before ``lift_after_pick``.  By this point the
    gripper has physically closed around the committed bottle, so the
    pick choice is fixed — any newer higher-Z bottle that arrives later
    must NOT redirect the arm during the lift or place phases.

    Prefers ``strategy.committed_pick_name`` (the name the arm was
    actually approaching when the gripper closed, stashed by
    ``CortexMoveToPick`` every tick) over a fresh
    ``get_current_pick_name()`` query, which mitigates the race where a
    new higher bottle settles exactly at latch time and would otherwise
    redirect the latch to the wrong physical item.

    For strategies that don't override ``latch_current_pick`` the call
    is a no-op and this behaviour just returns SUCCESS.
    """

    def __init__(self, name: str = "LatchCurrentPick"):
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        if self._context is None:
            return py_trees.common.Status.SUCCESS
        strategy = self._context.strategy
        pick_name = strategy.committed_pick_name
        if pick_name is None:
            pick_name = self._context.get_current_pick_name()
        if pick_name is None:
            return py_trees.common.Status.SUCCESS
        strategy.latch_current_pick(pick_name)
        return py_trees.common.Status.SUCCESS
