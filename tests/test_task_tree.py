"""Tests for the full task controller tree and task-level behaviours."""

import os
import sys

import numpy as np
import pytest

# Add extsMock and repo root to path
current_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
mock_path = os.path.join(repo_root, "extsMock")
sys.path.insert(0, mock_path)
sys.path.insert(0, repo_root)

import py_trees
from robot_controllers.mock_robot import (
    MockEndEffectorController, MockGripper,
    MockArmCommander, MockGripperCommander,
)
from robot_controllers.pt_context_monitor import ContextMonitorBehaviour
from robot_controllers.pt_task_tree import make_task_controller_tree
from robot_controllers.pt_task_behaviours import (
    CheckAllDone,
    CheckTargetAvailable,
    MarkPickComplete,
    ResetPickPlaceTree,
    SelectNextPick,
    SetTaskFinished as SetTaskFinishedBehaviour,
)
from task_context_mock import MockTaskContext


@pytest.fixture(autouse=True)
def clear_blackboard():
    """Clear py_trees blackboard before and after each test."""
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


# ---------------------------------------------------------------------------
# Individual behaviour tests
# ---------------------------------------------------------------------------


class TestCheckAllDone:
    def test_failure_when_not_done(self):
        ctx = MockTaskContext()
        b = CheckAllDone(name="test")
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE

    def test_success_when_done(self):
        ctx = MockTaskContext()
        ctx.task_finished = True
        b = CheckAllDone(name="test")
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS


class TestSelectNextPick:
    def test_first_pick_success(self):
        ctx = MockTaskContext()
        b = SelectNextPick(name="test")
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        assert ctx.get_current_pick_name() == "pick_0"

    def test_advance_and_exhaust(self):
        ctx = MockTaskContext(pick_names=["p0"])
        b = SelectNextPick(name="test")
        b.setup(context=ctx)
        # First call: success (current is p0)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        # Second call: advance past p0 -> None -> FAILURE
        b.status = py_trees.common.Status.INVALID  # reset for re-tick
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE


class TestCheckTargetAvailable:
    def test_success_with_target(self):
        ctx = MockTaskContext()
        b = CheckTargetAvailable(name="test")
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS

    def test_failure_without_target(self):
        ctx = MockTaskContext(pick_names=["p0"], target_names=[])
        ctx.strategy._pairings_by_pick_name = {"p0": None}
        b = CheckTargetAvailable(name="test")
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE
        assert ctx.targets_exhausted


class TestMarkPickComplete:
    def test_marks_current_pick(self):
        ctx = MockTaskContext()
        b = MarkPickComplete(name="test")
        b.setup(context=ctx)
        b.tick_once()
        assert "pick_0" in ctx._completed_picks


class TestSetTaskFinished:
    def test_sets_finished(self):
        ctx = MockTaskContext()
        b = SetTaskFinishedBehaviour(name="test")
        b.setup(context=ctx)
        b.tick_once()
        assert ctx.task_finished


class TestContextMonitorBehaviour:
    def test_writes_to_blackboard(self):
        ctx = MockTaskContext()
        monitor = ContextMonitorBehaviour(name="test_monitor")
        monitor.setup(context=ctx)
        monitor.tick_once()

        bb = py_trees.blackboard.Client(name="reader", namespace="/pickplace")
        bb.register_key(key="picking_position", access=py_trees.common.Access.READ)
        bb.register_key(key="placing_position", access=py_trees.common.Access.READ)
        bb.register_key(key="ee_height_for_move", access=py_trees.common.Access.READ)

        assert bb.picking_position is not None
        assert bb.placing_position is not None
        assert bb.ee_height_for_move == pytest.approx(0.3)

    def test_writes_task_finished(self):
        ctx = MockTaskContext()
        monitor = ContextMonitorBehaviour(name="test_monitor")
        monitor.setup(context=ctx)
        monitor.tick_once()

        task_bb = py_trees.blackboard.Client(name="task_reader", namespace="/task")
        task_bb.register_key(key="task_finished", access=py_trees.common.Access.READ)
        assert not task_bb.task_finished

    def test_returns_running(self):
        ctx = MockTaskContext()
        monitor = ContextMonitorBehaviour(name="test_monitor")
        monitor.setup(context=ctx)
        monitor.tick_once()
        assert monitor.status == py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# Full tree integration tests
# ---------------------------------------------------------------------------


class TestFullTaskTree:
    def _run_tree(self, ctx, max_ticks=2000):
        """Helper: build tree, run to completion, return tick count."""
        root = make_task_controller_tree(fake_fast=True)
        tree = py_trees.trees.BehaviourTree(root=root)
        tree.setup(
            timeout=15,
            context=ctx,
            arm_commander=ctx.arm_commander,
            gripper_commander=ctx.gripper_commander,
        )
        for i in range(max_ticks):
            tree.tick()
            if tree.root.status != py_trees.common.Status.RUNNING:
                return i + 1, tree.root.status
        return max_ticks, tree.root.status

    def test_three_picks_three_targets(self):
        ctx = MockTaskContext(
            pick_names=["a", "b", "c"],
            target_names=["t0", "t1", "t2"],
        )
        ticks, status = self._run_tree(ctx)
        assert status == py_trees.common.Status.SUCCESS
        assert ctx.task_finished
        assert ctx._completed_picks == {"a", "b", "c"}
        assert ticks < 200

    def test_more_picks_than_targets(self):
        ctx = MockTaskContext(
            pick_names=["a", "b", "c"],
            target_names=["t0", "t1"],
        )
        ticks, status = self._run_tree(ctx)
        assert status == py_trees.common.Status.SUCCESS
        assert ctx.task_finished
        assert ctx._completed_picks == {"a", "b"}
        assert ctx.targets_exhausted

    def test_single_pick(self):
        ctx = MockTaskContext(
            pick_names=["only"],
            target_names=["t0"],
        )
        ticks, status = self._run_tree(ctx)
        assert status == py_trees.common.Status.SUCCESS
        assert ctx.task_finished
        assert ctx._completed_picks == {"only"}

    def test_no_targets_immediate_finish(self):
        ctx = MockTaskContext(
            pick_names=["a"],
            target_names=[],
        )
        ctx._pairings_by_pick_name = {"a": None}
        ticks, status = self._run_tree(ctx)
        assert status == py_trees.common.Status.SUCCESS
        assert ctx.task_finished
        assert ctx.targets_exhausted
        assert ticks < 10


# ---------------------------------------------------------------------------
# UR10MultiPickPlaceController integration tests
# ---------------------------------------------------------------------------


class TestUR10MultiPickPlaceControllerIntegration:
    def test_controller_completes(self):
        from robot_controllers import UR10MultiPickPlaceController

        ctx = MockTaskContext(
            pick_names=["p0", "p1"],
            target_names=["t0", "t1"],
        )
        ctrl = UR10MultiPickPlaceController(
            name="test",
            task_context=ctx,
            fake_fast=True,
        )
        assert not ctrl.is_done()
        for i in range(2000):
            ctrl.forward()
            if ctrl.is_done():
                break
        assert ctrl.is_done()
        assert ctx.task_finished
        assert ctx._completed_picks == {"p0", "p1"}

    def test_controller_reset(self):
        from robot_controllers import UR10MultiPickPlaceController

        ctx = MockTaskContext()
        ctrl = UR10MultiPickPlaceController(
            name="test",
            task_context=ctx,
            fake_fast=True,
        )
        # Run a few ticks
        for _ in range(10):
            ctrl.forward()

        ctrl.reset()
        assert not ctrl.is_done()
        assert ctrl.get_current_pick_name() == "pick_0"
        assert not ctx.task_finished

    def test_context_state_accessible(self):
        from robot_controllers import UR10MultiPickPlaceController

        ctx = MockTaskContext()
        ctrl = UR10MultiPickPlaceController(
            name="test",
            task_context=ctx,
            fake_fast=True,
        )
        assert ctx.picking_order_item_names == ["pick_0", "pick_1", "pick_2"]
        assert ctx.strategy._current_pick_index == 0
        assert not ctx.targets_exhausted


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
