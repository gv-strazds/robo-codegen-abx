"""py_trees visitor that fires callbacks on BT phase transitions.

Used by `run_task.py --snapshots` to trigger snapshot captures when the
behaviour tree advances through pick-and-place phases.  The visitor is
configured with two `{behaviour_name -> event_name}` maps and calls
`on_event(event_name)` once per SUCCESS edge (default map) or FAILURE
edge (failure map) of each named behaviour.

The same visitor works on both the default 9-phase tree and the cortex
tree because behaviour `name=` strings are globally unique
(`CloseGripper` vs `CortexCloseGripper`, `LiftPicked` vs `lift_after_pick`,
etc.).

A first-tick `task_started` event is emitted once per visitor lifetime,
since no behaviour name maps to it.

Watchdog timeouts (the cortex tree's ``sim_timeout_to_success`` wrappers)
don't surface as a tree-visible FAILURE edge — the ``FailureIsSuccess``
outer decorator masks them — so this module also exposes
:func:`install_timeout_event_hooks`, which walks a built tree and patches
each :class:`SimTimeout`'s ``on_timeout`` callback to fire a configured
event in addition to its existing diagnostic.

Pure-Python: this module has no Isaac Sim dependency and can be imported
pre-SimulationApp.
"""

import logging
from typing import Callable, Dict, Optional

import py_trees

logger = logging.getLogger(__name__)


# --- Event name constants -------------------------------------------------

# SUCCESS-edge events — phase advancement on the happy path.
EV_TASK_STARTED  = "task_started"
EV_PICK_STARTED  = "pick_started"
EV_ITEM_GRASPED  = "item_grasped"
EV_ITEM_LIFTED   = "item_lifted"
EV_ITEM_AT_PLACE = "item_at_place"
EV_ITEM_RELEASED = "item_released"
EV_PICK_COMPLETE = "pick_complete"
EV_TASK_FINISHED = "task_finished"

# SUCCESS-edge events that signal a recovery branch ran (something went
# wrong upstream and the tree took a fallback path).
EV_PICK_DEFERRED   = "pick_deferred"     # DeferPickAndRelease ran (retry budget exhausted)
EV_PLACE_RECOVERED = "place_recovered"   # release_and_skip ran (place_item failed)

# FAILURE-edge events — leaf behaviours returning FAILURE inside the
# cortex tree's pick-attempt sub-sequence.  Each fires per attempt
# (so up to ``PICK_RETRY_BUDGET`` times per deferred item).
EV_PICK_UNREACHABLE   = "pick_unreachable"     # CheckPickReachable FAILURE
EV_GRASP_PREP_FAILED  = "grasp_prep_failed"    # PrepareGrasp FAILURE
EV_GRASP_SLIPPED      = "grasp_slipped"        # VerifyGrasp FAILURE (post-lift slip)

# Watchdog-timeout events — fired via the SimTimeout patcher when a
# wrapped motion phase exceeds its timeout budget.  The ``FailureIsSuccess``
# outer decorator means these won't show up as tree FAILURE edges, hence
# the separate hook mechanism.
EV_TIMEOUT_PRE_GRASP     = "timeout_pre_grasp"
EV_TIMEOUT_APPROACH      = "timeout_approach"
EV_TIMEOUT_MOVE_TO_PLACE = "timeout_move_to_place"
EV_TIMEOUT_DESCENT       = "timeout_descent"
EV_TIMEOUT_LIFT          = "timeout_lift"   # all three lifts share; child name disambiguates in metadata


# --- Non-BT snapshot events ----------------------------------------------
#
# Events fired from outside the behaviour tree.  The naming convention
# matches the BT events (snake_case), but they reach the snapshot
# pipeline via direct callbacks rather than the BTEventVisitor.

# Incremental verification check failed for a completed pick.  Fired from
# ``multi_pickplace_task._check_incremental`` — BT-agnostic, covers both
# the default 9-phase tree and the cortex tree.
EV_VERIFY_FAIL = "verify_fail"

# Final ground-truth verification completed at task end.  Fired once per
# task run from ``run_task.py`` after ``check_groundtruth_task_success``
# returns — captures the post-settle end-state of the scene with the
# verification verdict (success bool + failure list) in the JSON sidecar.
# Fires under both --snapshots and --snapshot-errors regardless of
# whether verification reported failures.
EV_TASK_VERIFIED = "task_verified"


# --- Failure-event registry ----------------------------------------------
#
# Single source of truth for the set of events that signal something went
# wrong (as opposed to normal phase advancement).  Used by
# ``run_task.py --snapshot-errors`` to filter out happy-path events while
# still capturing diagnostic snapshots on failures.
#
# Includes:
#   * FAILURE-edge BT events (pick_attempt leaf failures).
#   * Recovery-branch SUCCESS events whose firing itself signals an
#     upstream failure (DeferPickAndRelease, recovery_open_gripper).
#   * Watchdog timeout events (motion stalls).
#   * Verification check failures (non-BT).
FAILURE_EVENT_NAMES = frozenset({
    # FAILURE-edge BT events
    EV_PICK_UNREACHABLE,
    EV_GRASP_PREP_FAILED,
    EV_GRASP_SLIPPED,
    # Recovery-branch SUCCESS events (firing means upstream failed)
    EV_PICK_DEFERRED,
    EV_PLACE_RECOVERED,
    # Watchdog timeouts
    EV_TIMEOUT_PRE_GRASP,
    EV_TIMEOUT_APPROACH,
    EV_TIMEOUT_MOVE_TO_PLACE,
    EV_TIMEOUT_DESCENT,
    EV_TIMEOUT_LIFT,
    # Non-BT verification failures
    EV_VERIFY_FAIL,
})


# --- Default behaviour-name to event-name mappings ------------------------
#
# Combined mapping covering both the default 9-phase tree
# (pt_task_tree.py + pt_pick_place_behaviours.py) and the cortex tree
# (pt_cortex_tree.py).  Names are globally unique across the two trees,
# so the dicts can be used as-is for either.  Behaviours not present in
# the active tree are simply never seen by the visitor.

DEFAULT_BEHAVIOUR_TO_EVENT: Dict[str, str] = {
    # Pick selection (both trees)
    "SelectNextPick":      EV_PICK_STARTED,
    # Gripper close (item grasped)
    "CloseGripper":        EV_ITEM_GRASPED,   # default tree
    "CortexCloseGripper":  EV_ITEM_GRASPED,   # cortex tree
    # Lift after grasp
    "LiftPicked":          EV_ITEM_LIFTED,    # default tree
    "lift_after_pick":     EV_ITEM_LIFTED,    # cortex tree
    # Move to place position
    "MoveToPlaceXY":       EV_ITEM_AT_PLACE,  # default tree
    "CortexMoveToPlace":   EV_ITEM_AT_PLACE,  # cortex tree
    # Gripper open (item released)
    "OpenGripper":         EV_ITEM_RELEASED,  # default tree
    "CortexOpenGripper":   EV_ITEM_RELEASED,  # cortex tree
    # Cycle / task lifecycle
    "MarkPickComplete":    EV_PICK_COMPLETE,
    "SetTaskFinished":     EV_TASK_FINISHED,
    # Cortex recovery branches — these leaves run only when something
    # upstream failed, so a SUCCESS edge here is itself a failure signal.
    "DeferPickAndRelease":  EV_PICK_DEFERRED,
    "recovery_open_gripper": EV_PLACE_RECOVERED,
}

# FAILURE-edge mapping.  Only leaves whose FAILURE reflects an actionable
# problem (not normal control flow like CheckAllDone or end-of-picks
# selectors) are listed.  These are pick-attempt errors inside the
# cortex tree's ``pick_attempt`` Sequence, each counted by the
# ``pick_with_retry`` decorator.
DEFAULT_BEHAVIOUR_TO_FAIL_EVENT: Dict[str, str] = {
    "CheckPickReachable": EV_PICK_UNREACHABLE,
    "PrepareGrasp":       EV_GRASP_PREP_FAILED,
    "VerifyGrasp":        EV_GRASP_SLIPPED,
}

# SimTimeout watchdog → event mapping.  Keyed by the *wrapped child's*
# name (the SimTimeout decorator itself is named ``{child.name}/timeout``
# by ``sim_timeout_to_success`` — see ``pt_sim_time_decorators.py``).
DEFAULT_TIMEOUT_NAME_TO_EVENT: Dict[str, str] = {
    "CortexMoveToPreGrasp":  EV_TIMEOUT_PRE_GRASP,
    "CortexExecuteApproach": EV_TIMEOUT_APPROACH,
    "CortexMoveToPlace":     EV_TIMEOUT_MOVE_TO_PLACE,
    "CortexDownToInsert":    EV_TIMEOUT_DESCENT,
    "lift_after_pick":       EV_TIMEOUT_LIFT,
    "lift_after_place":      EV_TIMEOUT_LIFT,
    "recovery_lift":         EV_TIMEOUT_LIFT,
}


class BTEventVisitor(py_trees.visitors.VisitorBase):
    """Detects SUCCESS / FAILURE-edge transitions on named behaviours and
    fires ``on_event(event_name)`` once per edge.

    Mirrors the structure of ``RunningTransitionVisitor`` (tracking
    previous/current status keyed by ``behaviour.id``).  Uses
    ``full=False``: the tip path is sufficient because a leaf returning
    SUCCESS or FAILURE is on the tip the tick it transitions.

    Two mappings are consulted independently per tick, so the same
    behaviour can register both a SUCCESS and a FAILURE event if needed
    (no current behaviour does so, but the design allows it).
    """

    def __init__(
        self,
        on_event: Callable[[str], None],
        mapping: Optional[Dict[str, str]] = None,
        fail_mapping: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__(full=False)
        self._on_event = on_event
        self._mapping = mapping if mapping is not None else DEFAULT_BEHAVIOUR_TO_EVENT
        self._fail_mapping = (
            fail_mapping if fail_mapping is not None else DEFAULT_BEHAVIOUR_TO_FAIL_EVENT
        )
        self._prev_status: Dict[object, py_trees.common.Status] = {}
        self._curr_status: Dict[object, py_trees.common.Status] = {}
        self._emitted_task_started = False

    def initialise(self) -> None:
        self._prev_status = self._curr_status
        self._curr_status = {}

    def _fire(self, event: str) -> None:
        try:
            self._on_event(event)
        except Exception as e:
            logger.warning(f"BTEventVisitor: on_event({event}) raised: {e}")

    def run(self, behaviour) -> None:
        if not self._emitted_task_started:
            self._emitted_task_started = True
            self._fire(EV_TASK_STARTED)

        self._curr_status[behaviour.id] = behaviour.status
        prev = self._prev_status.get(behaviour.id)

        # SUCCESS edge
        if behaviour.status == py_trees.common.Status.SUCCESS and prev != py_trees.common.Status.SUCCESS:
            event = self._mapping.get(behaviour.name)
            if event is not None:
                self._fire(event)

        # FAILURE edge
        if behaviour.status == py_trees.common.Status.FAILURE and prev != py_trees.common.Status.FAILURE:
            event = self._fail_mapping.get(behaviour.name)
            if event is not None:
                self._fire(event)


def install_timeout_event_hooks(
    root: py_trees.behaviour.Behaviour,
    on_event: Callable[[str], None],
    mapping: Optional[Dict[str, str]] = None,
) -> int:
    """Walk *root* and patch every ``SimTimeout`` decorator to fire a
    snapshot event in addition to its existing diagnostic when the
    watchdog deadline expires.

    The ``FailureIsSuccess`` outer wrapper applied by
    ``sim_timeout_to_success`` masks the underlying FAILURE edge from
    the visitor, so we hook ``SimTimeout._on_timeout`` directly.  The
    original callback (typically ``child._timeout_diagnostic``) is
    preserved — its return value (the diag string) still flows back into
    the SimTimeout's feedback message.

    Args:
        root: Root behaviour of an already-constructed tree.
        on_event: Callable invoked with the event name string when a
            wrapped SimTimeout's deadline expires.
        mapping: Optional ``{wrapped_child_name -> event_name}`` map.
            Defaults to :data:`DEFAULT_TIMEOUT_NAME_TO_EVENT`.

    Returns:
        Number of SimTimeout decorators that were patched (useful for
        sanity checks in tests).
    """
    if mapping is None:
        mapping = DEFAULT_TIMEOUT_NAME_TO_EVENT

    # Local import: pt_sim_time_decorators must not depend on this
    # module (kept event-system-agnostic), so the import is here.
    from robot_controllers.pt_sim_time_decorators import SimTimeout

    patched = 0

    def _walk(node):
        nonlocal patched
        if isinstance(node, SimTimeout):
            child = node.decorated
            child_name = child.name if child is not None else node.name
            event = mapping.get(child_name)
            if event is not None:
                original = node._on_timeout

                def _wrapped(_orig=original, _ev=event):
                    diag = ""
                    if _orig is not None:
                        try:
                            diag = _orig() or ""
                        except Exception as e:
                            logger.warning(
                                f"SimTimeout({child_name}): original on_timeout raised: {e}"
                            )
                    try:
                        on_event(_ev)
                    except Exception as e:
                        logger.warning(
                            f"SimTimeout({child_name}): event fire on_event({_ev}) raised: {e}"
                        )
                    return diag

                node._on_timeout = _wrapped
                patched += 1

        # py_trees Decorators expose ``children = [decorated]`` *and*
        # ``decorated``; walking both would double-visit, so descend only
        # via ``children`` (which Composites also expose).
        for c in getattr(node, "children", ()):
            _walk(c)

    _walk(root)
    return patched
