"""Integration smoke tests for the cortex tree wiring added in step 10.

The retry/defer flow is covered at the unit level in
``tests/test_pick_deferral.py`` (strategy API) and
``tests/test_cortex_perception_behaviours.py`` (individual behaviours).
Here we check that:

1. The new cortex tree builds and sets up without raising.
2. Direct ``strategy.defer_pick`` + ``mark_pick_complete`` interaction
   clears deferrals as advertised.
3. ``MarkPickComplete`` skips deferred picks rather than marking them.
"""
import os
import sys

import numpy as np
import py_trees
import pytest

_current_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
_mock_path = os.path.join(_repo_root, "extsMock")
for _p in (_mock_path, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from robot_controllers.pt_cortex_tree import (
    PICK_RETRY_BUDGET,
    make_cortex_task_controller_tree,
)
from robot_controllers.pt_task_behaviours import MarkPickComplete
from task_context_mock import MockTaskContext


@pytest.fixture(autouse=True)
def clear_blackboard():
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


class TestCortexTreeBuild:
    def test_tree_builds_with_defaults(self):
        root = make_cortex_task_controller_tree(fake_fast=True)
        assert root is not None
        assert root.name == "TaskRoot"

    def test_tree_setup_with_mock_context_succeeds(self):
        ctx = MockTaskContext()
        root = make_cortex_task_controller_tree(fake_fast=True)
        tree = py_trees.trees.BehaviourTree(root=root)
        # Should not raise.
        tree.setup(
            timeout=15,
            context=ctx,
            arm_commander=ctx.arm_commander,
            gripper_commander=ctx.gripper_commander,
        )

    def test_tree_contains_retry_decorator_with_budget(self):
        """Confirm the pick_with_retry Retry decorator has the configured budget."""
        root = make_cortex_task_controller_tree(fake_fast=True)
        found = []
        for node in root.iterate():
            if isinstance(node, py_trees.decorators.Retry):
                found.append(node)
        assert len(found) == 1
        assert found[0].num_failures == PICK_RETRY_BUDGET

    def test_tree_contains_prepare_and_verify(self):
        """Sanity check: PrepareGrasp and VerifyGrasp are wired into the tree."""
        from robot_controllers.pt_cortex_perception_behaviours import (
            PrepareGrasp, VerifyGrasp, DeferPickAndRelease, HaveItemInGripper,
            PreparePlacement,
        )
        root = make_cortex_task_controller_tree(fake_fast=True)
        class_names = {type(n).__name__ for n in root.iterate()}
        assert "PrepareGrasp" in class_names
        assert "VerifyGrasp" in class_names
        assert "DeferPickAndRelease" in class_names
        assert "HaveItemInGripper" in class_names
        assert "PreparePlacement" in class_names


class TestDeferralInteractionWithCompletion:
    def test_mark_pick_complete_clears_deferrals(self):
        """Direct strategy-level test: any completion clears _deferred_picks."""
        ctx = MockTaskContext(
            pick_names=["pick_0", "pick_1"],
            target_names=["target_0", "target_1"],
        )
        ctx.strategy.defer_pick("pick_0")
        assert ctx.strategy.is_pick_deferred("pick_0")
        ctx.mark_pick_complete("pick_1")
        assert not ctx.strategy.is_pick_deferred("pick_0")
        assert "pick_1" in ctx.strategy.completed_picks

    def test_mark_pick_complete_behaviour_skips_deferred_pick(self):
        """MarkPickComplete must NOT complete a pick that's currently deferred."""
        ctx = MockTaskContext(
            pick_names=["pick_0", "pick_1"],
            target_names=["target_0", "target_1"],
        )
        # Defer pick_0 (the current pick) and run MarkPickComplete.
        ctx.strategy.defer_pick("pick_0")
        b = MarkPickComplete(name="test")
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        # The deferred pick should NOT have been marked complete.
        assert "pick_0" not in ctx.strategy.completed_picks
        # And it remains deferred.
        assert ctx.strategy.is_pick_deferred("pick_0")

    def test_mark_pick_complete_behaviour_completes_non_deferred(self):
        """Non-deferred picks complete normally."""
        ctx = MockTaskContext(
            pick_names=["pick_0", "pick_1"],
            target_names=["target_0", "target_1"],
        )
        b = MarkPickComplete(name="test")
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        assert "pick_0" in ctx.strategy.completed_picks


class TestPermanentSkip:
    """MarkPickComplete must skip permanent picks the same as deferred ones."""

    def test_mark_pick_complete_behaviour_skips_permanent(self):
        ctx = MockTaskContext(
            pick_names=["pick_0", "pick_1"],
            target_names=["target_0", "target_1"],
        )
        ctx.strategy.mark_pick_permanently_unreachable("pick_0")
        b = MarkPickComplete(name="test")
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        assert "pick_0" not in ctx.strategy.completed_picks
        assert ctx.strategy.is_pick_permanently_unreachable("pick_0")


class TestGuardWiring:
    """Verify IsPickReachableGuard is wired to short-circuit Retry."""

    def test_tree_contains_guard_then_retry_sequence(self):
        root = make_cortex_task_controller_tree(fake_fast=True)
        names = [n.name for n in root.iterate()]
        assert "guard_then_retry" in names
        assert "not_permanent" in names

    def test_tree_contains_is_pick_reachable_guard(self):
        root = make_cortex_task_controller_tree(fake_fast=True)
        class_names = {type(n).__name__ for n in root.iterate()}
        assert "IsPickReachableGuard" in class_names


class TestCycleProgressWiring:
    """Verify CheckCycleProgress is wired at the head of do_one_pick_place."""

    def test_tree_contains_check_cycle_progress(self):
        root = make_cortex_task_controller_tree(fake_fast=True)
        class_names = {type(n).__name__ for n in root.iterate()}
        assert "CheckCycleProgress" in class_names

    def test_cycle_progress_threshold_kwarg_propagates(self):
        """The factory kwarg must reach the CheckCycleProgress instance."""
        from robot_controllers.pt_task_behaviours import CheckCycleProgress
        root = make_cortex_task_controller_tree(
            fake_fast=True, cycle_progress_threshold=7,
        )
        cp_nodes = [n for n in root.iterate() if isinstance(n, CheckCycleProgress)]
        assert len(cp_nodes) == 1
        assert cp_nodes[0]._threshold == 7


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
