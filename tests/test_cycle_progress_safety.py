"""Tests for the no-progress livelock safety net.

Covers ``CheckCycleProgress`` (the BT behaviour at the head of
``do_one_pick_place``) and the strategy-level counter it drives.  The
counter is bumped on every tick of the behaviour and reset on
``mark_pick_complete``; once the counter exceeds the configured
threshold the behaviour returns FAILURE so the outer Repeat exits into
``SetTaskFinished``.
"""
import os
import sys

import py_trees
import pytest

_current_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
_mock_path = os.path.join(_repo_root, "extsMock")
for _p in (_mock_path, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robot_controllers.pt_task_behaviours import (
    CheckCycleProgress,
    IsPickReachableGuard,
)
from task_context_mock import MockTaskContext


@pytest.fixture(autouse=True)
def clear_blackboard():
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


class TestCheckCycleProgress:
    def test_returns_success_below_threshold(self):
        ctx = MockTaskContext()
        b = CheckCycleProgress(threshold=3)
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        assert ctx.strategy.cycles_since_last_completion == 1

    def test_returns_failure_when_threshold_exceeded(self):
        ctx = MockTaskContext()
        b = CheckCycleProgress(threshold=2)
        b.setup(context=ctx)
        # 1, 2 → SUCCESS; 3 > 2 → FAILURE
        b.tick_once()
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE

    def test_completion_resets_counter(self):
        ctx = MockTaskContext()
        b = CheckCycleProgress(threshold=2)
        b.setup(context=ctx)
        b.tick_once()
        b.tick_once()
        assert ctx.strategy.cycles_since_last_completion == 2
        ctx.mark_pick_complete(ctx.get_current_pick_name())
        assert ctx.strategy.cycles_since_last_completion == 0
        # The counter is reset, so the next tick is back to 1 → SUCCESS.
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        assert ctx.strategy.cycles_since_last_completion == 1

    def test_no_context_is_success(self):
        b = CheckCycleProgress(threshold=2)
        # No setup → context is None.
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS

    def test_default_threshold_is_50(self):
        b = CheckCycleProgress()
        assert b._threshold == 50


class TestIsPickReachableGuard:
    def test_success_when_pick_not_permanent(self):
        ctx = MockTaskContext()
        b = IsPickReachableGuard()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS

    def test_failure_when_pick_permanent(self):
        ctx = MockTaskContext()
        ctx.strategy.mark_pick_permanently_unreachable(
            ctx.get_current_pick_name()
        )
        b = IsPickReachableGuard()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE

    def test_success_when_no_current_pick(self):
        """Empty / exhausted pick set: the wrapping Retry will discover
        that itself; the guard does not preempt."""
        ctx = MockTaskContext(pick_names=["p0"], target_names=["t0"])
        ctx.mark_pick_complete("p0")
        ctx.advance_pick_index()  # cursor walks off the end
        b = IsPickReachableGuard()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS

    def test_no_context_is_success(self):
        b = IsPickReachableGuard()
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
