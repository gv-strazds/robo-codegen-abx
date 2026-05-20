"""Tests for UR10MultiPickPlaceController.

Verifies that the py_trees-based controller can be instantiated with
MockTaskContext, ticks through a full task to completion, resets properly,
and reports is_done() correctly.

Adapted from concepts in unused/test_pick_place_refactor.py which tested
the old PickPlaceController.
"""
import sys
import os
import pytest
import numpy as np

# Add extsMock and repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'extsMock'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import py_trees

from robot_controllers.ur10_multi_pick_place_controller import UR10MultiPickPlaceController
from task_context_mock import MockTaskContext


@pytest.fixture(autouse=True)
def clear_blackboard():
    """Clear py_trees blackboard before and after each test."""
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


def _make_controller(num_picks=3, num_targets=3, fake_fast=True):
    """Helper to create a UR10MultiPickPlaceController with mock context."""
    pick_names = [f"pick_{i}" for i in range(num_picks)]
    target_names = [f"target_{i}" for i in range(num_targets)]

    pick_positions = {
        f"pick_{i}": np.array([0.5, 0.1 * i, 0.05]) for i in range(num_picks)
    }
    target_positions = {
        f"target_{i}": np.array([-0.5, 0.1 * i, 0.05]) for i in range(num_targets)
    }

    ctx = MockTaskContext(
        pick_names=pick_names,
        target_names=target_names,
        pick_positions=pick_positions,
        target_positions=target_positions,
    )

    controller = UR10MultiPickPlaceController(
        name="test_controller",
        task_context=ctx,
        fake_fast=fake_fast,
    )
    return controller, ctx


class TestUR10ControllerLifecycle:
    """Tests for controller initialization, completion, and reset."""

    def test_init_not_done(self):
        """Freshly created controller should not be done."""
        controller, _ctx = _make_controller()
        assert not controller.is_done()

    def test_completes(self):
        """Controller should complete all picks within a bounded number of ticks."""
        controller, _ctx = _make_controller(num_picks=2, num_targets=2)

        max_ticks = 2000
        for i in range(max_ticks):
            controller.forward()
            if controller.is_done():
                break

        assert controller.is_done(), (
            f"Controller should complete after enough ticks (ran {i + 1})"
        )
        assert i > 0, "Should take at least a few ticks to complete"

    def test_reset(self):
        """After reset, controller should not be done and can run again."""
        controller, ctx = _make_controller(num_picks=1, num_targets=1)

        # Run to completion
        for _ in range(2000):
            controller.forward()
            if controller.is_done():
                break
        assert controller.is_done(), "Should complete the first run"

        # Reset and verify
        controller.reset()
        assert not controller.is_done(), "Should not be done after reset"

    def test_single_pick_completes(self):
        """Controller with a single pick-target pair should complete."""
        controller, _ctx = _make_controller(num_picks=1, num_targets=1)

        for i in range(2000):
            controller.forward()
            if controller.is_done():
                break

        assert controller.is_done(), "Single pick should complete"


class TestUR10ControllerState:
    """Tests for controller state queries."""

    def test_get_current_pick_name(self):
        """Should return the current pick name before any ticks."""
        controller, _ctx = _make_controller()
        name = controller.get_current_pick_name()
        assert name is not None
        assert name.startswith("pick_")

    def test_is_done_reflects_task_finished(self):
        """is_done() should reflect task_context.task_finished."""
        controller, ctx = _make_controller()
        assert not controller.is_done()

        ctx.task_finished = True
        assert controller.is_done()

    def test_is_done_reflects_all_picks_done(self):
        """is_done() should reflect task_context.all_picks_done."""
        controller, ctx = _make_controller()
        assert not controller.is_done()

        # Advance past all picks in the strategy
        strategy = ctx._strategy
        while not strategy.all_picks_done:
            strategy.advance_pick_index()
        assert controller.is_done()

    def test_is_done_reflects_targets_exhausted(self):
        """is_done() should reflect task_context.targets_exhausted."""
        controller, ctx = _make_controller()
        assert not controller.is_done()

        ctx.targets_exhausted = True
        assert controller.is_done()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
