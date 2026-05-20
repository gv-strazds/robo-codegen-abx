"""Sim-time-aware py_trees decorators and timing primitives.

Stock ``py_trees.decorators.Timeout`` reads ``time.monotonic()`` and
``py_trees.behaviours.Timer`` reads ``time.time()`` — both wall-clock.
The cortex BT runs against a sim that may step at arbitrary
physics_dt and may pause/throttle relative to wall-clock; tying
watchdog timeouts to wall-clock means timeouts fire at the wrong sim
moment as soon as physics_dt or render cadence changes.

This module provides drop-in sim-time variants:

* :class:`SimTimeout` — same semantics as
  ``py_trees.decorators.Timeout`` (FAILURE + child INVALID on expiry,
  child status passthrough otherwise) but the clock is sim time.
  Adds an optional ``on_timeout`` callback so the wrapped behaviour
  can surface a rich diagnostic (e.g. EE FK vs target distance) when
  the timeout fires, instead of the generic "timed out" feedback
  message.
* :class:`SimTimer` — same semantics as
  ``py_trees.behaviours.Timer`` (RUNNING then SUCCESS) but on sim
  time.
* :func:`sim_timeout_to_success` — factory that composes
  ``FailureIsSuccess(SimTimeout(child))``.  Encodes the "force SUCCESS
  on timeout" semantic that ``CortexMove`` previously had baked in.

Sim-time resolution chain (lazy, per call) — see :func:`_get_sim_time`:
  1. ``World.instance().current_time`` (real Isaac Sim path)
  2. ``context.simulation_time`` when a context is supplied (mock path
     — ``tasks_mock/mock_task_utils.py`` advances this each tick)
  3. ``time.monotonic()`` (last-resort fallback)
"""
import logging
import time
from typing import Any, Callable, Optional, Union

import py_trees


logger = logging.getLogger(__name__)


# Duration may be a fixed value or a resolver that gets the live context.
# The latter lets callers pin a duration to ``context.get_*_timeout_s()``
# so per-task TaskSpec overrides flow through without factory-time wiring.
DurationSpec = Union[float, Callable[[Optional[Any]], float]]


def _resolve_duration(spec: DurationSpec, context: Optional[Any]) -> float:
    if callable(spec):
        return float(spec(context))
    return float(spec)


def _get_sim_time(context: Optional[Any] = None) -> float:
    """Return the current sim time in seconds, with the standard fallback chain.

    Resolution is lazy (per call): the World may not exist at factory
    time but does once the loop is running, and the mock context may
    swap in/out between test cases.
    """
    try:
        from isaacsim.core.api.world import World  # type: ignore[import-not-found]
        world = World.instance()
        if world is not None:
            return float(world.current_time)
    except Exception:
        pass
    if context is not None:
        sim_time = getattr(context, "simulation_time", None)
        if sim_time is not None:
            return float(sim_time)
    return time.monotonic()


class SimTimeout(py_trees.decorators.Timeout):
    """Sim-time variant of :class:`py_trees.decorators.Timeout`.

    Semantics match the base class:
      * On entry (``initialise``), the finish time is recomputed from
        the current sim time + ``duration``.  Re-entry under
        ``Repeat``/``Selector`` resets the clock cleanly because the
        base ``Decorator.tick`` calls ``initialise`` whenever the
        decorator's status is not RUNNING.
      * On each ``update``, if the child is RUNNING and the deadline
        has passed: invoke ``on_timeout`` (if provided), stop the
        child with INVALID, and return FAILURE.
      * Otherwise, pass the child status through unchanged.
    """

    def __init__(
        self,
        name: str,
        child: py_trees.behaviour.Behaviour,
        duration: DurationSpec = 5.0,
        context: Optional[Any] = None,
        on_timeout: Optional[Callable[[], str]] = None,
    ):
        # Stash the spec separately from the float on the base — the base
        # ``Timeout`` constructor expects a float for ``self.duration``.
        # We override initialise/update so the float on the base is never
        # actually consumed; we still set a sensible value for any
        # diagnostics that read it.
        initial = _resolve_duration(duration, context)
        super().__init__(name=name, child=child, duration=initial)
        self._duration_spec: DurationSpec = duration
        self._context = context
        self._on_timeout = on_timeout

    def setup(self, **kwargs: Any) -> None:
        """Capture the task context propagated through ``BehaviourTree.setup``.

        Mirrors the cortex behaviours' setup convention: the
        controller calls ``tree.setup(context=task_context, ...)`` and
        that kwarg propagates to every node.  We grab it here so the
        sim-time fallback chain can read ``context.simulation_time``
        in the mock path (no live World).  Constructor-time context
        wins when both are supplied (lets callers pin a specific
        context for unit tests).
        """
        super().setup(**kwargs)
        if self._context is None and "context" in kwargs:
            self._context = kwargs["context"]

    def initialise(self) -> None:
        """Recompute deadline from sim time on (re-)entry.

        Re-resolves ``duration`` against the live context so per-task
        ``TaskSpec.move_timeout_s`` (etc.) overrides take effect without
        any factory-time wiring.
        """
        self.duration = _resolve_duration(self._duration_spec, self._context)
        self.finish_time = _get_sim_time(self._context) + self.duration
        self.feedback_message = ""

    def update(self) -> py_trees.common.Status:
        current_time = _get_sim_time(self._context)
        if (
            self.decorated.status == py_trees.common.Status.RUNNING
            and current_time > self.finish_time
        ):
            elapsed = current_time - (self.finish_time - self.duration)
            diag = ""
            if self._on_timeout is not None:
                try:
                    diag = self._on_timeout() or ""
                except Exception:  # noqa: BLE001 — diagnostic must never break the tree
                    logger.exception(
                        f"{self.name}: on_timeout callback raised; "
                        f"continuing with timeout."
                    )
            self.feedback_message = (
                f"sim-timed out after {elapsed:.2f}s "
                f"(duration={self.duration}s){'; ' + diag if diag else ''}"
            )
            logger.warning(f"{self.name}: {self.feedback_message}")
            self.decorated.stop(py_trees.common.Status.INVALID)
            return py_trees.common.Status.FAILURE
        if self.decorated.status == py_trees.common.Status.RUNNING:
            self.feedback_message = (
                f"sim-time still ticking ... "
                f"[remaining: {self.finish_time - current_time:.2f}s]"
            )
        else:
            self.feedback_message = "child finished before sim-timeout triggered"
        return self.decorated.status


class SimTimer(py_trees.timers.Timer):
    """Sim-time variant of :class:`py_trees.behaviours.Timer`.

    RUNNING until ``duration`` sim-seconds have elapsed since
    ``initialise``, then SUCCESS on the next tick.
    """

    def __init__(
        self,
        name: str = "SimTimer",
        duration: DurationSpec = 5.0,
        context: Optional[Any] = None,
    ):
        initial = _resolve_duration(duration, context)
        super().__init__(name=name, duration=initial)
        self._duration_spec: DurationSpec = duration
        self._context = context

    def setup(self, **kwargs: Any) -> None:
        """Capture the task context propagated through ``BehaviourTree.setup``."""
        super().setup(**kwargs)
        if self._context is None and "context" in kwargs:
            self._context = kwargs["context"]

    def initialise(self) -> None:
        self.duration = _resolve_duration(self._duration_spec, self._context)
        self.finish_time = _get_sim_time(self._context) + self.duration
        self.feedback_message = (
            f"sim-time configured to fire in '{self.duration}'s"
        )

    def update(self) -> py_trees.common.Status:
        current_time = _get_sim_time(self._context)
        if current_time > self.finish_time:
            self.feedback_message = f"sim-timer ran out [{self.duration}]"
            return py_trees.common.Status.SUCCESS
        self.feedback_message = "still running"
        return py_trees.common.Status.RUNNING


def sim_timeout_to_success(
    name: str,
    child: py_trees.behaviour.Behaviour,
    duration: DurationSpec,
    context: Optional[Any] = None,
    on_timeout: Optional[Callable[[], str]] = None,
) -> py_trees.decorators.FailureIsSuccess:
    """Wrap *child* in a sim-time timeout that yields SUCCESS on expiry.

    Composition: ``FailureIsSuccess(SimTimeout(child))``.  Encodes the
    "give up and let the parent advance" semantic that ``CortexMove``
    previously had baked in (returning SUCCESS internally on the
    force-success deadline).  The wrapped child returns SUCCESS only
    when it actually succeeds; on timeout the decorator surfaces the
    rich ``on_timeout`` diagnostic, stops the child, and the
    ``FailureIsSuccess`` wrapper maps the resulting FAILURE to SUCCESS
    so the parent Sequence advances.
    """
    return py_trees.decorators.FailureIsSuccess(
        name=f"{name}/ok_on_timeout",
        child=SimTimeout(
            name=f"{name}/timeout",
            child=child,
            duration=duration,
            context=context,
            on_timeout=on_timeout,
        ),
    )
