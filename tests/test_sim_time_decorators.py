"""Unit tests for SimTimeout, SimTimer, and sim_timeout_to_success.

Drives time via a fake context with a mutable ``simulation_time``
attribute (the same surface ``WaitForCycleTime`` and the cortex
behaviours read in mock mode).  Avoids any reliance on a live World
or wall-clock so the tests are deterministic.
"""
import py_trees
import pytest

from robot_controllers.pt_sim_time_decorators import (
    SimTimeout,
    SimTimer,
    _get_sim_time,
    sim_timeout_to_success,
)


class _FakeContext:
    """Minimal stand-in for TaskContextBase.  Only ``simulation_time`` is read."""

    def __init__(self, simulation_time: float = 0.0):
        self.simulation_time = simulation_time


class _Counter(py_trees.behaviour.Behaviour):
    """Returns RUNNING for ``running_ticks`` ticks, then ``terminal``.

    Tracks the number of times ``initialise`` and ``terminate`` are
    called so tests can assert the decorator drives the child's
    lifecycle correctly.
    """

    def __init__(self, name: str = "Counter", running_ticks: int = 100,
                 terminal: py_trees.common.Status = py_trees.common.Status.SUCCESS):
        super().__init__(name=name)
        self._running_ticks = running_ticks
        self._terminal = terminal
        self.update_calls = 0
        self.initialise_calls = 0
        self.terminate_calls = 0
        self.last_terminate_status = None

    def initialise(self) -> None:
        self.initialise_calls += 1
        self.update_calls = 0

    def update(self) -> py_trees.common.Status:
        self.update_calls += 1
        if self.update_calls > self._running_ticks:
            return self._terminal
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        self.terminate_calls += 1
        self.last_terminate_status = new_status


# ---------------------------------------------------------------------------
# _get_sim_time fallback chain
# ---------------------------------------------------------------------------

def test_get_sim_time_reads_context():
    ctx = _FakeContext(simulation_time=12.5)
    assert _get_sim_time(ctx) == pytest.approx(12.5)


def test_get_sim_time_no_context_falls_back_to_monotonic():
    # When no context is supplied (and no live World exists in the
    # test env), the helper uses time.monotonic.  We just assert it
    # returns a float and doesn't blow up — exact value depends on
    # wall-clock state.
    t = _get_sim_time(None)
    assert isinstance(t, float)


# ---------------------------------------------------------------------------
# SimTimeout — core semantics
# ---------------------------------------------------------------------------

def test_sim_timeout_passes_child_running_status_through_when_under_deadline():
    ctx = _FakeContext(simulation_time=0.0)
    child = _Counter(running_ticks=100)
    deco = SimTimeout(name="t", child=child, duration=10.0, context=ctx)
    deco.tick_once()
    assert deco.status == py_trees.common.Status.RUNNING
    # Advance sim time, still well under deadline.
    ctx.simulation_time = 5.0
    deco.tick_once()
    assert deco.status == py_trees.common.Status.RUNNING
    assert child.terminate_calls == 0


def test_sim_timeout_returns_failure_and_invalidates_child_on_expiry():
    ctx = _FakeContext(simulation_time=0.0)
    child = _Counter(running_ticks=1000)
    deco = SimTimeout(name="t", child=child, duration=2.0, context=ctx)
    deco.tick_once()
    assert deco.status == py_trees.common.Status.RUNNING
    # Jump past the deadline.
    ctx.simulation_time = 3.0
    deco.tick_once()
    assert deco.status == py_trees.common.Status.FAILURE
    assert child.terminate_calls == 1
    # base Timeout calls child.stop(INVALID) on expiry
    assert child.last_terminate_status == py_trees.common.Status.INVALID


def test_sim_timeout_passes_child_success_through_before_expiry():
    ctx = _FakeContext(simulation_time=0.0)
    # SUCCESS on the second update (running_ticks=1).
    child = _Counter(running_ticks=1, terminal=py_trees.common.Status.SUCCESS)
    deco = SimTimeout(name="t", child=child, duration=10.0, context=ctx)
    deco.tick_once()  # RUNNING (update_calls=1)
    deco.tick_once()  # SUCCESS (update_calls=2)
    assert deco.status == py_trees.common.Status.SUCCESS
    # Deadline never fired — child.terminate gets called by the framework
    # on the SUCCESS transition, but with SUCCESS, not INVALID.
    assert child.last_terminate_status == py_trees.common.Status.SUCCESS


def test_sim_timeout_resets_deadline_on_re_entry():
    """After SUCCESS, re-ticking re-initialises and resets the timer.

    Mirrors the ``Repeat(SimTimeout(child))`` pattern used in the
    cortex tree.
    """
    ctx = _FakeContext(simulation_time=0.0)
    child = _Counter(running_ticks=1, terminal=py_trees.common.Status.SUCCESS)
    deco = SimTimeout(name="t", child=child, duration=2.0, context=ctx)
    # First cycle.
    deco.tick_once()
    deco.tick_once()
    assert deco.status == py_trees.common.Status.SUCCESS
    first_finish = deco.finish_time
    # Advance sim time well past the first deadline.
    ctx.simulation_time = 100.0
    # Reset the child for the next cycle.
    child._running_ticks = 1
    deco.tick_once()  # status was SUCCESS → not RUNNING → initialise() runs
    second_finish = deco.finish_time
    # New finish_time should be 100.0 + 2.0 = 102.0, not the original 2.0.
    assert second_finish == pytest.approx(102.0)
    assert second_finish > first_finish


def test_sim_timeout_on_timeout_callback_fires_with_diagnostic():
    ctx = _FakeContext(simulation_time=0.0)
    child = _Counter(running_ticks=1000)
    fired = {"count": 0, "diag": None}

    def diag():
        fired["count"] += 1
        return "fk_p=[1,2,3] dist=0.42"

    deco = SimTimeout(
        name="t", child=child, duration=2.0, context=ctx, on_timeout=diag,
    )
    deco.tick_once()
    ctx.simulation_time = 3.0
    deco.tick_once()
    assert deco.status == py_trees.common.Status.FAILURE
    assert fired["count"] == 1
    assert "fk_p=[1,2,3]" in deco.feedback_message
    assert "dist=0.42" in deco.feedback_message


def test_sim_timeout_on_timeout_exception_does_not_break_tree():
    ctx = _FakeContext(simulation_time=0.0)
    child = _Counter(running_ticks=1000)

    def boom():
        raise RuntimeError("diagnostic computation failed")

    deco = SimTimeout(
        name="t", child=child, duration=2.0, context=ctx, on_timeout=boom,
    )
    deco.tick_once()
    ctx.simulation_time = 3.0
    deco.tick_once()
    # Exception in on_timeout must be swallowed; timeout still fires.
    assert deco.status == py_trees.common.Status.FAILURE


# ---------------------------------------------------------------------------
# SimTimeout — duration as a callable resolver (per-task tunability)
# ---------------------------------------------------------------------------

class _ContextWithGetter:
    def __init__(self, simulation_time: float = 0.0, timeout_s: float = 5.0):
        self.simulation_time = simulation_time
        self._timeout_s = timeout_s

    def get_move_timeout_s(self) -> float:
        return self._timeout_s


def test_sim_timeout_duration_callable_resolves_per_initialise():
    ctx = _ContextWithGetter(simulation_time=0.0, timeout_s=4.0)
    child = _Counter(running_ticks=1000)
    deco = SimTimeout(
        name="t", child=child,
        duration=lambda c: c.get_move_timeout_s(),
        context=ctx,
    )
    deco.tick_once()
    assert deco.duration == pytest.approx(4.0)
    # Caller bumps the per-task timeout between iterations — re-initialise
    # picks up the new value.
    ctx._timeout_s = 9.0
    # Force re-initialise by stopping the decorator (status → INVALID
    # triggers initialise on next tick).
    deco.stop(py_trees.common.Status.INVALID)
    child.update_calls = 0  # reset child too so it doesn't return SUCCESS
    deco.tick_once()
    assert deco.duration == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# SimTimer
# ---------------------------------------------------------------------------

def test_sim_timer_running_then_success():
    ctx = _FakeContext(simulation_time=0.0)
    timer = SimTimer(name="settle", duration=1.5, context=ctx)
    timer.tick_once()
    assert timer.status == py_trees.common.Status.RUNNING
    ctx.simulation_time = 1.0
    timer.tick_once()
    assert timer.status == py_trees.common.Status.RUNNING
    ctx.simulation_time = 2.0
    timer.tick_once()
    assert timer.status == py_trees.common.Status.SUCCESS


def test_sim_timer_resets_on_re_entry():
    ctx = _FakeContext(simulation_time=0.0)
    timer = SimTimer(name="settle", duration=1.0, context=ctx)
    timer.tick_once()
    ctx.simulation_time = 2.0
    timer.tick_once()
    assert timer.status == py_trees.common.Status.SUCCESS
    # Re-enter at sim_time=10 — new deadline should be 11, not 1.
    ctx.simulation_time = 10.0
    timer.tick_once()
    assert timer.status == py_trees.common.Status.RUNNING
    assert timer.finish_time == pytest.approx(11.0)


# ---------------------------------------------------------------------------
# sim_timeout_to_success — the "force SUCCESS on timeout" wrapper
# ---------------------------------------------------------------------------

def test_sim_timeout_to_success_maps_failure_to_success_on_expiry():
    ctx = _FakeContext(simulation_time=0.0)
    child = _Counter(running_ticks=1000)
    deco = sim_timeout_to_success(
        name="bounded_move", child=child, duration=2.0, context=ctx,
    )
    deco.tick_once()
    ctx.simulation_time = 3.0
    deco.tick_once()
    # FailureIsSuccess wraps the inner SimTimeout; outer status should be
    # SUCCESS even though the inner SimTimeout returned FAILURE.
    assert deco.status == py_trees.common.Status.SUCCESS


def test_sim_timeout_to_success_passes_child_success_through():
    ctx = _FakeContext(simulation_time=0.0)
    child = _Counter(running_ticks=1, terminal=py_trees.common.Status.SUCCESS)
    deco = sim_timeout_to_success(
        name="bounded_move", child=child, duration=10.0, context=ctx,
    )
    deco.tick_once()  # RUNNING
    deco.tick_once()  # SUCCESS
    assert deco.status == py_trees.common.Status.SUCCESS


def test_sim_timeout_to_success_setup_propagates_context_kwarg():
    """When context isn't passed at construction, setup() captures it."""
    child = _Counter(running_ticks=1000)
    deco = sim_timeout_to_success(
        name="bounded_move", child=child, duration=2.0,
    )
    ctx = _FakeContext(simulation_time=0.0)
    # Drive the same setup path the controller uses.
    py_trees.trees.setup(root=deco, context=ctx)
    deco.tick_once()
    ctx.simulation_time = 3.0
    deco.tick_once()
    assert deco.status == py_trees.common.Status.SUCCESS  # FailureIsSuccess on timeout
