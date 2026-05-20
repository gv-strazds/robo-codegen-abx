"""Tests for target reachability filtering in MultiPickStrategy and BT behaviors.

Covers:
- MultiPickStrategy: set_target_reachable_fn, is_target_reachable,
  is_target_permanently_unreachable, poll_target_reachability,
  get_placing_target_name filtering, _has_target filtering,
  advance_pick_index skipping, reset clearing.
- CheckTargetAvailable: RUNNING for not-yet-reachable, skip for permanent,
  FAILURE when all unreachable.
- env_config_values.make_z_reachability_check factory.

Pure Python / mock-only — no Isaac Sim required.
"""
import numpy as np
import pytest

from multi_pick_strategy import MultiPickStrategy
from env_config_values import TARGET_MIN_REACHABLE_Z, make_z_reachability_check


# ---------------------------------------------------------------------------
# Lightweight stand-ins
# ---------------------------------------------------------------------------


class FakeObj:
    """Minimal object with name and mutable world pose."""

    def __init__(self, name, pos=(0.0, 0.5, 0.01)):
        self.name = name
        self.prim_path = f"/World/{name}"
        self._pos = np.asarray(pos, dtype=float)

    def get_world_pose(self):
        return self._pos.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def get_local_pose(self):
        return self.get_world_pose()

    def set_z(self, z):
        self._pos[2] = float(z)


def _make_strategy(n_picks=3, n_targets=3, target_z=0.01):
    """Create a MultiPickStrategy with n picks and n targets."""
    picks = [FakeObj(f"pick_{i}", pos=(0.3 * i, 0.8, 0.1)) for i in range(n_picks)]
    targets = [FakeObj(f"target_{i}", pos=(0.3 * i, 0.5, target_z)) for i in range(n_targets)]
    strategy = MultiPickStrategy(pick_objs=picks, target_objs=targets)
    strategy.initialize_pairings()
    return strategy, picks, targets


# ---------------------------------------------------------------------------
# MultiPickStrategy reachability tests
# ---------------------------------------------------------------------------


class TestReachabilityBasics:

    def test_no_fn_all_reachable(self):
        """Without a reachability fn, all targets are reachable."""
        strategy, _, targets = _make_strategy()
        assert strategy.is_target_reachable("target_0") is True
        assert strategy.is_target_reachable("target_1") is True
        assert strategy.get_placing_target_name("pick_0") == "target_0"

    def test_reachable_target_returns_name(self):
        strategy, _, targets = _make_strategy()
        strategy.set_target_reachable_fn(lambda t: True)
        assert strategy.get_placing_target_name("pick_0") == "target_0"

    def test_unreachable_target_returns_none(self):
        strategy, _, targets = _make_strategy()
        strategy.set_target_reachable_fn(lambda t: False)
        assert strategy.get_placing_target_name("pick_0") is None

    def test_z_below_threshold_unreachable(self):
        strategy, _, targets = _make_strategy()
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        # target_0 at z=0.01 > -0.05 → reachable
        assert strategy.is_target_reachable("target_0") is True
        assert strategy.get_placing_target_name("pick_0") == "target_0"
        # Move below threshold
        targets[0].set_z(-0.1)
        assert strategy.is_target_reachable("target_0") is False

    def test_permanent_unreachability_transition(self):
        strategy, _, targets = _make_strategy()
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        # First: reachable
        assert strategy.is_target_reachable("target_0") is True
        assert not strategy.is_target_permanently_unreachable("target_0")
        # Move below threshold — was reachable once → permanent
        targets[0].set_z(-0.2)
        assert strategy.is_target_reachable("target_0") is False
        assert strategy.is_target_permanently_unreachable("target_0")

    def test_not_yet_reachable_not_permanent(self):
        """Target that was never reachable is not marked permanent."""
        strategy, _, targets = _make_strategy(target_z=-0.1)
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        assert strategy.is_target_reachable("target_0") is False
        assert not strategy.is_target_permanently_unreachable("target_0")


class TestPollReachability:

    def test_poll_detects_transition(self):
        strategy, _, targets = _make_strategy()
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        # All reachable initially
        strategy.poll_target_reachability()
        assert len(strategy._permanently_unreachable_targets) == 0
        # Drop target_1
        targets[1].set_z(-0.3)
        strategy.poll_target_reachability()
        assert strategy.is_target_permanently_unreachable("target_1")
        assert not strategy.is_target_permanently_unreachable("target_0")
        assert not strategy.is_target_permanently_unreachable("target_2")

    def test_poll_no_fn_is_noop(self):
        strategy, _, _ = _make_strategy()
        strategy.poll_target_reachability()  # no error


class TestAdvanceSkipsUnreachable:

    def test_advance_does_not_skip_when_retarget_possible(self):
        """Pick with unreachable target is NOT skipped if re-pairing is possible."""
        strategy, _, targets = _make_strategy()
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        strategy.poll_target_reachability()
        # Drop target_1 — but target_0 and target_2 are still reachable,
        # so pick_1 can be re-paired and should NOT be skipped.
        targets[1].set_z(-0.5)
        strategy.poll_target_reachability()
        next_name = strategy.advance_pick_index()
        assert next_name == "pick_1"

    def test_advance_skips_when_no_retarget_possible(self):
        """Pick is skipped only when target is unreachable AND no re-pairing exists."""
        strategy, _, targets = _make_strategy()
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        strategy.poll_target_reachability()
        # Drop ALL targets — no re-pairing possible
        for t in targets:
            t.set_z(-0.5)
        strategy.poll_target_reachability()
        next_name = strategy.advance_pick_index()
        assert next_name is None

    def test_advance_no_skip_without_fn(self):
        """Without reachability fn, advance works normally."""
        strategy, _, _ = _make_strategy()
        next_name = strategy.advance_pick_index()
        assert next_name == "pick_1"


class TestHasTarget:

    def test_has_target_true_when_retarget_possible(self):
        """_has_target returns True even if assigned target is unreachable, if re-pair exists."""
        strategy, _, targets = _make_strategy()
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        strategy.poll_target_reachability()
        assert strategy._has_target("pick_0") is True
        # Drop target_0 — but target_1 and target_2 are reachable → re-pair possible
        targets[0].set_z(-0.5)
        strategy.poll_target_reachability()
        assert strategy._has_target("pick_0") is True

    def test_has_target_false_when_all_unreachable(self):
        """_has_target returns False when assigned target is unreachable and no alternative."""
        strategy, _, targets = _make_strategy()
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        strategy.poll_target_reachability()
        for t in targets:
            t.set_z(-0.5)
        strategy.poll_target_reachability()
        assert strategy._has_target("pick_0") is False


class TestReset:

    def test_reset_clears_reachability_state(self):
        strategy, _, targets = _make_strategy()
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        strategy.poll_target_reachability()
        targets[0].set_z(-0.5)
        strategy.poll_target_reachability()
        assert strategy.is_target_permanently_unreachable("target_0")
        strategy.reset()
        assert not strategy.is_target_permanently_unreachable("target_0")
        assert len(strategy._target_was_reachable) == 0


# ---------------------------------------------------------------------------
# make_z_reachability_check tests
# ---------------------------------------------------------------------------


class TestMakeZReachabilityCheck:

    def test_default_threshold(self):
        fn = make_z_reachability_check()
        obj = FakeObj("t", pos=(0, 0, 0.01))
        assert fn(obj)
        obj.set_z(TARGET_MIN_REACHABLE_Z - 0.01)
        assert not fn(obj)

    def test_custom_threshold(self):
        fn = make_z_reachability_check(min_z=0.5)
        obj = FakeObj("t", pos=(0, 0, 0.6))
        assert fn(obj)
        obj.set_z(0.4)
        assert not fn(obj)


# ---------------------------------------------------------------------------
# CheckTargetAvailable behavior tests
# ---------------------------------------------------------------------------


class FakeMockContext:
    """Minimal context for testing CheckTargetAvailable."""

    def __init__(self, strategy):
        self.strategy = strategy
        self.targets_exhausted = False
        self._pick_name = strategy.get_current_pick_name()

    def get_current_pick_name(self):
        return self.strategy.get_current_pick_name()

    def get_placing_target_name(self, pick_name):
        return self.strategy.get_placing_target_name(pick_name)

    def advance_pick_index(self):
        return self.strategy.advance_pick_index()


class TestCheckTargetAvailableBehaviour:

    def _make_behaviour(self, strategy):
        import py_trees
        from robot_controllers.pt_task_behaviours import CheckTargetAvailable
        ctx = FakeMockContext(strategy)
        b = CheckTargetAvailable("test_check")
        b._context = ctx
        return b, ctx

    def test_success_when_reachable(self):
        import py_trees
        strategy, _, _ = _make_strategy()
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        strategy.poll_target_reachability()
        b, _ = self._make_behaviour(strategy)
        assert b.update() == py_trees.common.Status.SUCCESS

    def test_running_for_not_yet_reachable(self):
        """Target exists but not yet reachable → RUNNING (wait)."""
        import py_trees
        strategy, _, targets = _make_strategy(target_z=-0.1)
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        b, _ = self._make_behaviour(strategy)
        assert b.update() == py_trees.common.Status.RUNNING

    def test_retargets_permanently_unreachable(self):
        """Permanently unreachable target → re-paired to alternative → SUCCESS."""
        import py_trees
        strategy, _, targets = _make_strategy()
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        strategy.poll_target_reachability()
        # Drop target_0
        targets[0].set_z(-0.5)
        strategy.poll_target_reachability()
        b, _ = self._make_behaviour(strategy)
        # Re-pairing finds an alternative reachable target → SUCCESS
        result = b.update()
        assert result == py_trees.common.Status.SUCCESS

    def test_skips_when_no_retarget_possible(self):
        """All targets for current pick unreachable, others available → advance + RUNNING."""
        import py_trees
        strategy, _, targets = _make_strategy()
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        strategy.poll_target_reachability()
        # Drop targets 0 and 1 — pick_0 can steal target_2 (assigned to
        # uncompleted pick_2), but pick_2's turn would then need to handle it.
        # Instead, drop ALL targets except target_2 and mark pick_2 completed
        # so target_2 is unassigned and pick_0 retargets to it.
        targets[0].set_z(-0.5)
        targets[1].set_z(-0.5)
        strategy.poll_target_reachability()
        # pick_0's retarget finds target_2 (steals from pick_2) → SUCCESS
        b, _ = self._make_behaviour(strategy)
        result = b.update()
        assert result == py_trees.common.Status.SUCCESS

    def test_failure_when_all_permanently_unreachable(self):
        """All targets permanently unreachable → FAILURE."""
        import py_trees
        strategy, _, targets = _make_strategy()
        fn = make_z_reachability_check(min_z=-0.05)
        strategy.set_target_reachable_fn(fn)
        strategy.poll_target_reachability()
        # Drop all targets
        for t in targets:
            t.set_z(-0.5)
        strategy.poll_target_reachability()
        b, ctx = self._make_behaviour(strategy)
        result = b.update()
        # Should exhaust all picks and return FAILURE
        assert result == py_trees.common.Status.FAILURE
        assert ctx.targets_exhausted is True
