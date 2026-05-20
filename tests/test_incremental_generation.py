"""Tests for incremental/time-based item generation.

Tests:
- IncrementalItemScheduler: batching, timing, bt_should_start
- MultiPickStrategy.add_incremental_picks: pairing updates
- SelectNextPick RUNNING state when more_items_expected
- Integration: mock BT with incremental picks
"""

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

from item_generation import (
    IncrementalGenerationConfig,
    IncrementalItemScheduler,
    ItemGenerator,
    FixedValue,
    GridPositionGenerator,
)
from multi_pick_strategy import MultiPickStrategy
from task_context_base import LightweightObj


@pytest.fixture(autouse=True)
def clear_blackboard():
    """Clear py_trees blackboard before and after each test."""
    import py_trees
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


def _make_generator(count=6):
    """Create a simple ItemGenerator producing *count* cubes on a grid."""
    return ItemGenerator(
        position_generator=GridPositionGenerator(
            center=np.array([0.5, 0.0, 0.05]),
            rows=count, cols=1,
            spacing_x=0.1, spacing_y=0.1,
            randomize=False,
        ),
        asset_type_strategy=FixedValue("cube"),
    )


def _make_lightweight_objs(items, prefix="pick", start_idx=0):
    """Convert ItemSpec list to LightweightObj list."""
    objs = []
    for i, item in enumerate(items):
        name = item.name or f"{prefix}_{start_idx + i}"
        obj = LightweightObj(name=name, position=item.position)
        objs.append(obj)
    return objs


# ---------------------------------------------------------------------------
# IncrementalGenerationConfig
# ---------------------------------------------------------------------------


class TestIncrementalGenerationConfig:
    def test_defaults(self):
        cfg = IncrementalGenerationConfig()
        assert cfg.items_per_batch == 1
        assert cfg.batch_interval == 0.5
        assert cfg.bt_start_threshold is None
        assert cfg.bt_start_delay is None

    def test_custom_values(self):
        cfg = IncrementalGenerationConfig(
            items_per_batch=3,
            batch_interval=1.0,
            bt_start_threshold=5,
            bt_start_delay=2.0,
        )
        assert cfg.items_per_batch == 3
        assert cfg.batch_interval == 1.0
        assert cfg.bt_start_threshold == 5
        assert cfg.bt_start_delay == 2.0


# ---------------------------------------------------------------------------
# IncrementalItemScheduler
# ---------------------------------------------------------------------------


class TestIncrementalItemScheduler:
    def test_initial_batch(self):
        gen = _make_generator(count=6)
        cfg = IncrementalGenerationConfig(items_per_batch=2, batch_interval=0.5)
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=6, seed=42)

        initial = scheduler.get_initial_batch()
        assert len(initial) == 2
        assert scheduler.released_count == 2
        assert scheduler.total_count == 6
        assert scheduler.pending_count == 4
        assert not scheduler.all_released

    def test_initial_batch_larger_than_total(self):
        gen = _make_generator(count=3)
        cfg = IncrementalGenerationConfig(items_per_batch=10, batch_interval=0.5)
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=3, seed=42)

        initial = scheduler.get_initial_batch()
        assert len(initial) == 3
        assert scheduler.all_released

    def test_tick_releases_batches(self):
        gen = _make_generator(count=6)
        cfg = IncrementalGenerationConfig(items_per_batch=2, batch_interval=0.5)
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=6, seed=42)
        scheduler.get_initial_batch()

        # First tick: sets start time, no new items
        items = scheduler.tick(0.0)
        assert len(items) == 0

        # Before interval: no items
        items = scheduler.tick(0.3)
        assert len(items) == 0

        # At interval: batch 2
        items = scheduler.tick(0.5)
        assert len(items) == 2
        assert scheduler.released_count == 4

        # At next interval: batch 3 (last 2 items)
        items = scheduler.tick(1.0)
        assert len(items) == 2
        assert scheduler.released_count == 6
        assert scheduler.all_released

        # After all released: empty
        items = scheduler.tick(1.5)
        assert len(items) == 0

    def test_tick_partial_last_batch(self):
        gen = _make_generator(count=5)
        cfg = IncrementalGenerationConfig(items_per_batch=2, batch_interval=0.5)
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=5, seed=42)
        scheduler.get_initial_batch()  # 2 items
        scheduler.tick(0.0)  # start time
        scheduler.tick(0.5)  # batch 2: 2 items (total 4)
        items = scheduler.tick(1.0)  # batch 3: only 1 item left
        assert len(items) == 1
        assert scheduler.released_count == 5
        assert scheduler.all_released

    def test_reset(self):
        gen = _make_generator(count=4)
        cfg = IncrementalGenerationConfig(items_per_batch=2, batch_interval=0.5)
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=4, seed=42)
        scheduler.get_initial_batch()
        scheduler.tick(0.0)
        scheduler.tick(0.5)
        assert scheduler.all_released

        scheduler.reset()
        assert scheduler.released_count == 2
        assert not scheduler.all_released

    # -- bt_should_start --

    def test_bt_should_start_default_waits_for_all(self):
        """Default config (both None): waits for all items."""
        gen = _make_generator(count=4)
        cfg = IncrementalGenerationConfig(items_per_batch=2, batch_interval=0.5)
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=4, seed=42)
        scheduler.get_initial_batch()

        assert not scheduler.bt_should_start(0.0)
        scheduler.tick(0.0)
        scheduler.tick(0.5)
        assert scheduler.all_released
        assert scheduler.bt_should_start(0.5)

    def test_bt_should_start_threshold_zero(self):
        """bt_start_threshold=0: start immediately."""
        gen = _make_generator(count=4)
        cfg = IncrementalGenerationConfig(
            items_per_batch=1, batch_interval=0.5,
            bt_start_threshold=0,
        )
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=4, seed=42)
        scheduler.get_initial_batch()
        assert scheduler.bt_should_start(0.0)

    def test_bt_should_start_threshold_n(self):
        """bt_start_threshold=3: start after 3 items released."""
        gen = _make_generator(count=6)
        cfg = IncrementalGenerationConfig(
            items_per_batch=2, batch_interval=0.5,
            bt_start_threshold=3,
        )
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=6, seed=42)
        scheduler.get_initial_batch()  # 2 released

        assert not scheduler.bt_should_start(0.0)  # only 2, need 3
        scheduler.tick(0.0)
        scheduler.tick(0.5)  # 4 released
        assert scheduler.bt_should_start(0.5)  # 4 >= 3

    def test_bt_should_start_delay(self):
        """bt_start_delay=1.0: start after 1 second."""
        gen = _make_generator(count=6)
        cfg = IncrementalGenerationConfig(
            items_per_batch=1, batch_interval=0.5,
            bt_start_delay=1.0,
        )
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=6, seed=42)
        scheduler.get_initial_batch()

        # Before first tick, no generation_start_time set yet
        assert not scheduler.bt_should_start(0.0)

        scheduler.tick(0.1)  # sets generation_start_time=0.1
        assert not scheduler.bt_should_start(0.5)   # 0.4s < 1.0s
        assert scheduler.bt_should_start(1.1)        # 1.0s >= 1.0s

    def test_bt_should_start_either_condition(self):
        """Both threshold and delay set: either triggers start."""
        gen = _make_generator(count=6)
        cfg = IncrementalGenerationConfig(
            items_per_batch=2, batch_interval=0.5,
            bt_start_threshold=5,  # won't be met until 3rd batch
            bt_start_delay=0.3,    # will be met after 0.3s
        )
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=6, seed=42)
        scheduler.get_initial_batch()  # 2 released

        scheduler.tick(0.0)  # start time = 0.0
        assert not scheduler.bt_should_start(0.2)  # 2<5 and 0.2<0.3
        assert scheduler.bt_should_start(0.3)       # delay met (0.3>=0.3)


# ---------------------------------------------------------------------------
# MultiPickStrategy.add_incremental_picks
# ---------------------------------------------------------------------------


class TestStrategyIncrementalPicks:
    def _make_objs(self, prefix, count, start_idx=0):
        return [
            LightweightObj(f"{prefix}_{start_idx + i}",
                           position=np.array([0.5, 0.1 * i, 0.05]))
            for i in range(count)
        ]

    def test_add_incremental_picks_extends_pairings(self):
        picks = self._make_objs("pick", 2)
        targets = self._make_objs("target", 4)
        strategy = MultiPickStrategy(pick_objs=picks, target_objs=targets)
        strategy.initialize_pairings()

        assert len(strategy.picking_order_item_names) == 2

        # Add 2 more picks
        new_picks = self._make_objs("pick", 2, start_idx=2)
        strategy.add_incremental_picks(new_picks)

        assert len(strategy.picking_order_item_names) == 4
        assert "pick_2" in strategy.picking_order_item_names
        assert "pick_3" in strategy.picking_order_item_names

    def test_add_incremental_picks_preserves_completed(self):
        picks = self._make_objs("pick", 2)
        targets = self._make_objs("target", 4)
        strategy = MultiPickStrategy(pick_objs=picks, target_objs=targets)
        strategy.initialize_pairings()

        # Complete pick_0
        strategy.mark_pick_complete("pick_0")
        strategy.advance_pick_index()

        # Add more picks
        new_picks = self._make_objs("pick", 2, start_idx=2)
        strategy.add_incremental_picks(new_picks)

        # pick_0 should still be completed
        assert "pick_0" in strategy.completed_picks
        # All 4 picks should be in the order
        assert len(strategy.picking_order_item_names) == 4

    def test_more_items_expected_flag(self):
        picks = self._make_objs("pick", 2)
        targets = self._make_objs("target", 2)
        strategy = MultiPickStrategy(pick_objs=picks, target_objs=targets)

        assert not strategy.more_items_expected
        strategy.more_items_expected = True
        assert strategy.more_items_expected
        strategy.more_items_expected = False
        assert not strategy.more_items_expected


# ---------------------------------------------------------------------------
# SelectNextPick RUNNING state
# ---------------------------------------------------------------------------


class TestSelectNextPickRunning:
    def test_returns_running_when_more_expected(self):
        import py_trees
        from robot_controllers.pt_task_behaviours import SelectNextPick
        from task_context_mock import MockTaskContext

        ctx = MockTaskContext(
            pick_names=["pick_0"],
            target_names=["target_0"],
        )
        ctx.strategy.more_items_expected = True

        node = SelectNextPick(name="test")
        node.setup(context=ctx)

        # First tick: pick_0 available → SUCCESS
        node.tick_once()
        assert node.status == py_trees.common.Status.SUCCESS

        # Complete pick_0 and advance
        ctx.mark_pick_complete("pick_0")
        ctx.advance_pick_index()

        # Reset node for next cycle
        node.stop(py_trees.common.Status.INVALID)
        node.tick_once()
        # Second tick: no picks, but more_items_expected → RUNNING
        assert node.status == py_trees.common.Status.RUNNING

    def test_returns_failure_when_no_more_expected(self):
        import py_trees
        from robot_controllers.pt_task_behaviours import SelectNextPick
        from task_context_mock import MockTaskContext

        ctx = MockTaskContext(
            pick_names=["pick_0"],
            target_names=["target_0"],
        )
        # more_items_expected is False by default

        node = SelectNextPick(name="test")
        node.setup(context=ctx)

        # First tick: pick_0 available → SUCCESS
        node.tick_once()
        assert node.status == py_trees.common.Status.SUCCESS

        # Complete pick_0 and advance
        ctx.mark_pick_complete("pick_0")
        ctx.advance_pick_index()

        # Reset node for next cycle
        node.stop(py_trees.common.Status.INVALID)
        node.tick_once()
        # Second tick: no picks, no more expected → FAILURE
        assert node.status == py_trees.common.Status.FAILURE


# ---------------------------------------------------------------------------
# Integration: mock BT with incremental picks
# ---------------------------------------------------------------------------


class TestIncrementalMockIntegration:
    """Integration test: full BT with incremental item generation."""

    def _run_incremental_bt(self, bt_start_threshold, total_picks=4,
                            items_per_batch=1, batch_interval=0.5):
        """Helper to run a mock BT with incremental generation."""
        import py_trees
        from robot_controllers.pt_task_tree import make_task_controller_tree
        from task_context_mock import MockTaskContext

        # Create initial picks and all targets
        gen = _make_generator(count=total_picks)
        cfg = IncrementalGenerationConfig(
            items_per_batch=items_per_batch,
            batch_interval=batch_interval,
            bt_start_threshold=bt_start_threshold,
        )
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=total_picks, seed=42)
        initial_items = scheduler.get_initial_batch()
        initial_objs = _make_lightweight_objs(initial_items)

        target_objs = [
            LightweightObj(f"target_{i}", position=np.array([-0.5, 0.1 * i, 0.05]))
            for i in range(total_picks)
        ]

        strategy = MultiPickStrategy(pick_objs=initial_objs, target_objs=target_objs)
        strategy.initialize_pairings()
        if not scheduler.all_released:
            strategy.more_items_expected = True

        ctx = MockTaskContext(
            pick_names=[o.name for o in initial_objs],
            target_names=[o.name for o in target_objs],
            strategy=strategy,
        )

        root = make_task_controller_tree(fake_fast=True)
        tree = py_trees.trees.BehaviourTree(root=root)
        tree.setup(
            timeout=15, context=ctx,
            arm_commander=ctx.arm_commander,
            gripper_commander=ctx.gripper_commander,
        )

        mock_time = 0.0
        mock_dt = 1.0 / 60.0
        bt_started = scheduler.all_released or (bt_start_threshold == 0)
        max_ticks = 5000
        ticks_run = 0

        for i in range(1, max_ticks + 1):
            mock_time += mock_dt

            # Tick scheduler
            if not scheduler.all_released:
                new_items = scheduler.tick(mock_time)
                if new_items:
                    # Use released_count minus batch size as start index
                    # to produce unique names (pick_1, pick_2, etc.)
                    start = scheduler.released_count - len(new_items)
                    new_objs = _make_lightweight_objs(
                        new_items, prefix="pick", start_idx=start,
                    )
                    strategy.add_incremental_picks(new_objs)
                if scheduler.all_released:
                    strategy.more_items_expected = False

            # BT gate
            if not bt_started:
                if scheduler.bt_should_start(mock_time):
                    bt_started = True
                else:
                    continue

            tree.tick()
            if hasattr(ctx.arm_commander, 'tick'):
                ctx.arm_commander.tick()
            ticks_run += 1

            if tree.root.status != py_trees.common.Status.RUNNING:
                break

        return ctx, strategy, ticks_run, tree.root.status

    def test_immediate_start_completes(self):
        """bt_start_threshold=0: BT starts immediately, items arrive during."""
        ctx, strategy, ticks, status = self._run_incremental_bt(
            bt_start_threshold=0, total_picks=4,
            items_per_batch=1, batch_interval=0.5,
        )
        assert ctx.task_finished
        assert len(strategy.completed_picks) == 4

    def test_wait_for_all_completes(self):
        """bt_start_threshold=None: wait for all items then start."""
        ctx, strategy, ticks, status = self._run_incremental_bt(
            bt_start_threshold=None, total_picks=4,
            items_per_batch=2, batch_interval=0.5,
        )
        assert ctx.task_finished
        assert len(strategy.completed_picks) == 4

    def test_partial_threshold_completes(self):
        """bt_start_threshold=2: start after 2 items, rest arrive later."""
        ctx, strategy, ticks, status = self._run_incremental_bt(
            bt_start_threshold=2, total_picks=4,
            items_per_batch=1, batch_interval=0.5,
        )
        assert ctx.task_finished
        assert len(strategy.completed_picks) == 4
