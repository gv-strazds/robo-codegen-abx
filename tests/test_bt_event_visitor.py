"""Unit tests for BTEventVisitor and install_timeout_event_hooks.

Covers:
  * SUCCESS-edge detection fires once per edge (not on subsequent
    SUCCESS ticks).
  * FAILURE-edge detection fires once per edge.
  * ``task_started`` synthetic event fires exactly once per visitor
    lifetime.
  * Unmapped behaviour names produce no events.
  * ``install_timeout_event_hooks`` walks a tree, patches each
    ``SimTimeout``, and the patched callback fires the configured event
    when the watchdog deadline expires (without losing the original
    diagnostic).
"""
import py_trees
import pytest

from bt_event_visitor import (
    BTEventVisitor,
    DEFAULT_BEHAVIOUR_TO_EVENT,
    DEFAULT_BEHAVIOUR_TO_FAIL_EVENT,
    DEFAULT_TIMEOUT_NAME_TO_EVENT,
    EV_GRASP_SLIPPED,
    EV_PICK_DEFERRED,
    EV_PICK_UNREACHABLE,
    EV_PLACE_RECOVERED,
    EV_TASK_FINISHED,
    EV_TASK_STARTED,
    EV_TIMEOUT_DESCENT,
    install_timeout_event_hooks,
)
from robot_controllers.pt_sim_time_decorators import (
    SimTimeout,
    sim_timeout_to_success,
)


class _FakeContext:
    """Minimal stand-in exposing ``simulation_time`` for the sim-time fallback."""

    def __init__(self, simulation_time: float = 0.0):
        self.simulation_time = simulation_time


class _Scripted(py_trees.behaviour.Behaviour):
    """Returns a scripted sequence of statuses, one per ``update`` call."""

    def __init__(self, name: str, statuses):
        super().__init__(name=name)
        self._statuses = list(statuses)
        self._idx = 0

    def update(self) -> py_trees.common.Status:
        s = self._statuses[min(self._idx, len(self._statuses) - 1)]
        self._idx += 1
        return s


class _Recorder:
    """Captures ``on_event`` calls in order."""

    def __init__(self):
        self.events = []

    def __call__(self, event_name: str) -> None:
        self.events.append(event_name)


# ---------------------------------------------------------------------------
# Visitor: SUCCESS-edge detection
# ---------------------------------------------------------------------------

def _tick_with_visitor(root, visitor, n: int) -> None:
    """Tick a tree ``n`` times with the visitor attached."""
    tree = py_trees.trees.BehaviourTree(root=root)
    tree.add_visitor(visitor)
    for _ in range(n):
        tree.tick()


def test_task_started_fires_once_on_first_tick():
    rec = _Recorder()
    leaf = _Scripted("noop", [py_trees.common.Status.RUNNING] * 3)
    visitor = BTEventVisitor(on_event=rec, mapping={}, fail_mapping={})
    _tick_with_visitor(leaf, visitor, n=3)
    assert rec.events == [EV_TASK_STARTED]


def test_success_edge_fires_once_per_transition():
    rec = _Recorder()
    # Use SetTaskFinished's name so it maps to EV_TASK_FINISHED via the default map.
    leaf = _Scripted(
        "SetTaskFinished",
        [py_trees.common.Status.RUNNING,
         py_trees.common.Status.SUCCESS,
         py_trees.common.Status.SUCCESS,
         py_trees.common.Status.SUCCESS],
    )
    visitor = BTEventVisitor(on_event=rec)
    _tick_with_visitor(leaf, visitor, n=4)
    assert rec.events == [EV_TASK_STARTED, EV_TASK_FINISHED]


def test_unmapped_behaviour_emits_no_event_on_success():
    rec = _Recorder()
    leaf = _Scripted("SomethingNotInMap", [py_trees.common.Status.SUCCESS])
    visitor = BTEventVisitor(on_event=rec)
    _tick_with_visitor(leaf, visitor, n=1)
    assert rec.events == [EV_TASK_STARTED]  # only synthetic


# ---------------------------------------------------------------------------
# Visitor: FAILURE-edge detection
# ---------------------------------------------------------------------------

def test_failure_edge_fires_for_mapped_behaviour():
    rec = _Recorder()
    leaf = _Scripted(
        "CheckPickReachable",
        [py_trees.common.Status.RUNNING,
         py_trees.common.Status.FAILURE],
    )
    visitor = BTEventVisitor(on_event=rec)
    _tick_with_visitor(leaf, visitor, n=2)
    assert rec.events == [EV_TASK_STARTED, EV_PICK_UNREACHABLE]


def test_failure_edge_fires_once_then_silent_on_subsequent_failure():
    rec = _Recorder()
    leaf = _Scripted(
        "VerifyGrasp",
        [py_trees.common.Status.FAILURE] * 3,
    )
    visitor = BTEventVisitor(on_event=rec)
    _tick_with_visitor(leaf, visitor, n=3)
    # First FAILURE: edge transition (prev was None).  Subsequent ticks
    # all FAILURE → no edge.
    assert rec.events == [EV_TASK_STARTED, EV_GRASP_SLIPPED]


def test_failure_edge_fires_again_after_intervening_running():
    rec = _Recorder()
    leaf = _Scripted(
        "PrepareGrasp",
        [py_trees.common.Status.FAILURE,
         py_trees.common.Status.RUNNING,
         py_trees.common.Status.FAILURE],
    )
    visitor = BTEventVisitor(on_event=rec)
    _tick_with_visitor(leaf, visitor, n=3)
    # FAILURE → RUNNING → FAILURE: two edges into FAILURE.
    failure_events = [e for e in rec.events if e != EV_TASK_STARTED]
    assert len(failure_events) == 2


def test_unmapped_behaviour_emits_no_event_on_failure():
    rec = _Recorder()
    leaf = _Scripted("SomethingNotInMap", [py_trees.common.Status.FAILURE])
    visitor = BTEventVisitor(on_event=rec)
    _tick_with_visitor(leaf, visitor, n=1)
    assert rec.events == [EV_TASK_STARTED]


# ---------------------------------------------------------------------------
# Recovery-branch SUCCESS mappings
# ---------------------------------------------------------------------------

def test_defer_pick_and_release_success_fires_pick_deferred():
    rec = _Recorder()
    leaf = _Scripted(
        "DeferPickAndRelease",
        [py_trees.common.Status.SUCCESS],
    )
    visitor = BTEventVisitor(on_event=rec)
    _tick_with_visitor(leaf, visitor, n=1)
    assert EV_PICK_DEFERRED in rec.events


def test_recovery_open_gripper_success_fires_place_recovered():
    rec = _Recorder()
    leaf = _Scripted(
        "recovery_open_gripper",
        [py_trees.common.Status.SUCCESS],
    )
    visitor = BTEventVisitor(on_event=rec)
    _tick_with_visitor(leaf, visitor, n=1)
    assert EV_PLACE_RECOVERED in rec.events


# ---------------------------------------------------------------------------
# Default mappings sanity (catch typos / accidental removals)
# ---------------------------------------------------------------------------

def test_default_success_mapping_includes_recovery_branches():
    assert DEFAULT_BEHAVIOUR_TO_EVENT["DeferPickAndRelease"] == EV_PICK_DEFERRED
    assert DEFAULT_BEHAVIOUR_TO_EVENT["recovery_open_gripper"] == EV_PLACE_RECOVERED


def test_default_fail_mapping_covers_pick_attempt_leaves():
    assert DEFAULT_BEHAVIOUR_TO_FAIL_EVENT["CheckPickReachable"] == EV_PICK_UNREACHABLE
    assert DEFAULT_BEHAVIOUR_TO_FAIL_EVENT["VerifyGrasp"] == EV_GRASP_SLIPPED


def test_default_timeout_mapping_covers_descent():
    assert DEFAULT_TIMEOUT_NAME_TO_EVENT["CortexDownToInsert"] == EV_TIMEOUT_DESCENT


# ---------------------------------------------------------------------------
# install_timeout_event_hooks
# ---------------------------------------------------------------------------

def test_install_timeout_event_hooks_patches_matching_simtimeout():
    """A SimTimeout wrapping a child whose name is in the mapping should
    have its _on_timeout patched; firing the timeout invokes both the
    original diagnostic and the snapshot event callback."""
    rec = _Recorder()
    diag_calls = {"n": 0}

    def _diagnostic():
        diag_calls["n"] += 1
        return "DIAG"

    ctx = _FakeContext(simulation_time=0.0)
    child = _Scripted("CortexDownToInsert", [py_trees.common.Status.RUNNING] * 100)
    wrapped = sim_timeout_to_success(
        name=child.name,
        child=child,
        duration=0.5,
        context=ctx,
        on_timeout=_diagnostic,
    )

    n_patched = install_timeout_event_hooks(wrapped, rec)
    assert n_patched == 1

    # Tick once at t=0 to kick off (initialise sets deadline).
    wrapped.tick_once()

    # Advance past the deadline.
    ctx.simulation_time = 1.0
    wrapped.tick_once()

    assert EV_TIMEOUT_DESCENT in rec.events
    assert diag_calls["n"] == 1, "Original diagnostic must still be invoked."


def test_install_timeout_event_hooks_skips_unmapped_simtimeout():
    rec = _Recorder()
    ctx = _FakeContext(simulation_time=0.0)
    child = _Scripted("UnmappedChild", [py_trees.common.Status.RUNNING] * 100)
    wrapped = sim_timeout_to_success(
        name=child.name,
        child=child,
        duration=0.1,
        context=ctx,
    )

    n_patched = install_timeout_event_hooks(wrapped, rec)
    assert n_patched == 0


def test_install_timeout_event_hooks_walks_composite_children():
    """Patcher must descend into Sequence/Selector children and Decorator wrappers."""
    rec = _Recorder()
    ctx = _FakeContext(simulation_time=0.0)

    # Build: Sequence(SimTimeout(CortexDownToInsert), SimTimeout(lift_after_pick))
    c1 = _Scripted("CortexDownToInsert", [py_trees.common.Status.RUNNING] * 100)
    c2 = _Scripted("lift_after_pick", [py_trees.common.Status.RUNNING] * 100)
    w1 = sim_timeout_to_success(name=c1.name, child=c1, duration=0.5, context=ctx)
    w2 = sim_timeout_to_success(name=c2.name, child=c2, duration=0.5, context=ctx)

    seq = py_trees.composites.Sequence(name="seq", memory=True, children=[w1, w2])

    n_patched = install_timeout_event_hooks(seq, rec)
    assert n_patched == 2


def test_install_timeout_event_hooks_returns_zero_on_tree_without_simtimeout():
    rec = _Recorder()
    leaf = _Scripted("plain_leaf", [py_trees.common.Status.SUCCESS])
    seq = py_trees.composites.Sequence(name="seq", memory=True, children=[leaf])
    assert install_timeout_event_hooks(seq, rec) == 0


def test_simtimeout_walked_directly_without_outer_wrapper():
    """Patcher must work when given a SimTimeout as the root, not just
    when wrapped in FailureIsSuccess."""
    rec = _Recorder()
    ctx = _FakeContext(simulation_time=0.0)
    child = _Scripted("CortexMoveToPlace", [py_trees.common.Status.RUNNING] * 100)
    bare_simtimeout = SimTimeout(
        name=f"{child.name}/timeout", child=child, duration=0.1, context=ctx,
    )
    n_patched = install_timeout_event_hooks(bare_simtimeout, rec)
    assert n_patched == 1
