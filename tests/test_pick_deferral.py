"""Tests for MultiPickStrategy.defer_pick + related deferral machinery.

Step 6 of the grasp-affordance refactor: temporary pick exclusion that
clears on any other pick's completion, with a second-chance pass to
avoid livelock when all remaining picks are deferred.
"""
import os
import sys

# Ensure extsMock shadows real isaacsim and repo root is importable.
_current_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
_mock_path = os.path.join(_repo_root, "extsMock")
for _p in (_mock_path, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pytest

from multi_pick_strategy import MultiPickStrategy


class _FakeObj:
    def __init__(self, name, pos=(0.0, 0.0, 0.0)):
        self.name = name
        self._position = np.array(pos, dtype=float)
        self._orientation = np.array([1.0, 0.0, 0.0, 0.0])

    def get_world_pose(self):
        return self._position.copy(), self._orientation.copy()


def _make_strategy(n=3):
    picks = [_FakeObj(f"pick_{i}", pos=(0.3 * i, 0.8, 0.1)) for i in range(n)]
    targets = [_FakeObj(f"target_{i}", pos=(0.3 * i, 0.5, 0.01)) for i in range(n)]
    strategy = MultiPickStrategy(pick_objs=picks, target_objs=targets)
    strategy.initialize_pairings()
    return strategy


class TestDeferralBasics:
    def test_empty_by_default(self):
        strategy = _make_strategy()
        assert strategy.deferred_picks == set()
        assert not strategy.is_pick_deferred("pick_0")

    def test_defer_pick_adds_to_set(self):
        strategy = _make_strategy()
        strategy.defer_pick("pick_1")
        assert strategy.is_pick_deferred("pick_1")
        assert strategy.deferred_picks == {"pick_1"}

    def test_defer_ignores_empty_name(self):
        strategy = _make_strategy()
        strategy.defer_pick("")
        assert strategy.deferred_picks == set()

    def test_clear_deferred_picks(self):
        strategy = _make_strategy()
        strategy.defer_pick("pick_0")
        strategy.defer_pick("pick_2")
        strategy.clear_deferred_picks()
        assert strategy.deferred_picks == set()

    def test_deferred_picks_returns_copy(self):
        strategy = _make_strategy()
        strategy.defer_pick("pick_1")
        snap = strategy.deferred_picks
        snap.add("injected")
        assert "injected" not in strategy.deferred_picks

    def test_reset_clears_deferred(self):
        strategy = _make_strategy()
        strategy.defer_pick("pick_0")
        strategy.reset()
        assert strategy.deferred_picks == set()


class TestDeferralSkipping:
    def test_advance_pick_index_skips_deferred(self):
        strategy = _make_strategy()
        strategy.defer_pick("pick_1")
        # Advance from pick_0 should skip pick_1 and land on pick_2.
        next_name = strategy.advance_pick_index()
        assert next_name == "pick_2"

    def test_get_current_pick_is_unaffected_by_other_deferrals(self):
        """Deferring a *different* pick should not affect the current one."""
        strategy = _make_strategy()
        strategy.defer_pick("pick_2")
        assert strategy.get_current_pick_name() == "pick_0"

    def test_scan_skips_deferred_all_middle_chunk(self):
        strategy = _make_strategy(n=5)
        # Defer the middle three; scan should jump past them.
        for name in ("pick_1", "pick_2", "pick_3"):
            strategy.defer_pick(name)
        next_name = strategy.advance_pick_index()
        assert next_name == "pick_4"


class TestDeferralClearOnCompletion:
    def test_mark_pick_complete_clears_deferred(self):
        """Key semantics: after any pick completes, deferred picks get another chance."""
        strategy = _make_strategy()
        strategy.defer_pick("pick_0")
        assert strategy.is_pick_deferred("pick_0")
        strategy.mark_pick_complete("pick_2")
        assert strategy.deferred_picks == set()

    def test_previously_deferred_pick_reselectable_after_completion(self):
        """Full deferral cycle: pick_0 deferred → pick_1 completes → pick_0 eligible."""
        strategy = _make_strategy()
        strategy.defer_pick("pick_0")
        # Advance past pick_0 → pick_1 (since pick_0 is deferred).
        next_name = strategy.advance_pick_index()
        assert next_name == "pick_1"
        # Complete pick_1 — clears deferrals.
        strategy.mark_pick_complete("pick_1")
        assert not strategy.is_pick_deferred("pick_0")
        # Re-scan from current position wraps around and finds pick_0.
        strategy._current_pick_index = len(strategy.picking_order_item_names)
        next_name = strategy._scan_for_available_pick()
        assert next_name == "pick_0"


class TestSecondChancePass:
    def test_all_deferred_triggers_second_chance(self):
        """When every non-completed pick is deferred the scan clears them and retries once."""
        strategy = _make_strategy()
        for name in ("pick_0", "pick_1", "pick_2"):
            strategy.defer_pick(name)
        assert len(strategy.deferred_picks) == 3
        # scan should trigger second-chance, return pick_0 (fresh cursor), and clear the set.
        next_name = strategy._scan_for_available_pick()
        assert next_name == "pick_0"
        assert strategy.deferred_picks == set()

    def test_second_chance_returns_none_when_no_candidates(self):
        """If all picks are completed AND the deferred set is empty, scan returns None."""
        strategy = _make_strategy()
        for i in range(3):
            strategy.mark_pick_complete(f"pick_{i}")
        strategy._current_pick_index = 3
        assert strategy._scan_for_available_pick() is None
        assert strategy.deferred_picks == set()

    def test_second_chance_fires_only_once_in_a_row(self):
        """After second-chance clears the set, a subsequent advance with no successful
        completion should NOT resurrect the cleared deferrals — they're gone."""
        strategy = _make_strategy()
        for name in ("pick_0", "pick_1", "pick_2"):
            strategy.defer_pick(name)
        # First scan: second-chance fires, returns pick_0.
        assert strategy._scan_for_available_pick() == "pick_0"
        # Defer them all again (simulating each failing again).
        for name in ("pick_0", "pick_1", "pick_2"):
            strategy.defer_pick(name)
        # Second scan: second-chance fires again (each iteration is independent).
        assert strategy._scan_for_available_pick() == "pick_0"


class TestAdvancePickIndexFallthrough:
    def test_tail_all_deferred_falls_through_to_scan(self):
        """Advancing past a tail of deferred picks wraps around to find a non-deferred one."""
        strategy = _make_strategy(n=3)
        strategy.defer_pick("pick_1")
        strategy.defer_pick("pick_2")
        # From pick_0 (index 0), advance skips pick_1 and pick_2, then
        # falls through to _scan_for_available_pick which wraps around
        # to find pick_0 (non-deferred).  Second-chance is NOT triggered
        # because a non-deferred candidate was found.
        next_name = strategy.advance_pick_index()
        assert next_name == "pick_0"
        # Deferrals for pick_1 / pick_2 are preserved since the
        # second-chance pass did not fire.
        assert strategy.deferred_picks == {"pick_1", "pick_2"}

    def test_all_deferred_from_tail_triggers_second_chance(self):
        """When even wrap-around can't find a non-deferred candidate, second-chance fires."""
        strategy = _make_strategy(n=3)
        strategy.defer_pick("pick_0")
        strategy.defer_pick("pick_1")
        strategy.defer_pick("pick_2")
        # Advance from index 0: forward-skip all three, tail exhausted,
        # wrap-around also finds nothing (all deferred) → second-chance
        # clears the deferred set and finds pick_0.
        next_name = strategy.advance_pick_index()
        assert next_name == "pick_0"
        assert strategy.deferred_picks == set()


class TestPermanentlyUnreachablePicks:
    """The run-lifetime exclusion: items that fell below the z-floor."""

    def test_empty_by_default(self):
        strategy = _make_strategy()
        assert strategy.permanently_unreachable_picks == set()
        assert not strategy.is_pick_permanently_unreachable("pick_0")

    def test_mark_adds_and_logs(self):
        strategy = _make_strategy()
        strategy.mark_pick_permanently_unreachable("pick_1")
        assert strategy.is_pick_permanently_unreachable("pick_1")
        assert strategy.permanently_unreachable_picks == {"pick_1"}

    def test_mark_also_defers(self):
        """Permanent flag also enters _deferred_picks for in-pass cursor advance."""
        strategy = _make_strategy()
        strategy.mark_pick_permanently_unreachable("pick_1")
        assert strategy.is_pick_deferred("pick_1")

    def test_mark_idempotent(self):
        strategy = _make_strategy()
        strategy.mark_pick_permanently_unreachable("pick_1")
        strategy.mark_pick_permanently_unreachable("pick_1")
        assert strategy.permanently_unreachable_picks == {"pick_1"}

    def test_mark_ignores_empty_name(self):
        strategy = _make_strategy()
        strategy.mark_pick_permanently_unreachable("")
        assert strategy.permanently_unreachable_picks == set()

    def test_returns_copy(self):
        strategy = _make_strategy()
        strategy.mark_pick_permanently_unreachable("pick_1")
        snap = strategy.permanently_unreachable_picks
        snap.add("injected")
        assert "injected" not in strategy.permanently_unreachable_picks

    def test_mark_pick_complete_does_not_clear_permanent(self):
        """Crucial difference from _deferred_picks: completion does not resurrect."""
        strategy = _make_strategy()
        strategy.mark_pick_permanently_unreachable("pick_0")
        strategy.mark_pick_complete("pick_2")
        # Deferred set was cleared by completion, but permanent set survives.
        assert strategy.is_pick_permanently_unreachable("pick_0")
        assert strategy.permanently_unreachable_picks == {"pick_0"}

    def test_advance_pick_index_skips_permanent(self):
        strategy = _make_strategy()
        strategy.mark_pick_permanently_unreachable("pick_1")
        next_name = strategy.advance_pick_index()
        assert next_name == "pick_2"

    def test_scan_skips_permanent(self):
        strategy = _make_strategy(n=5)
        for name in ("pick_1", "pick_2", "pick_3"):
            strategy.mark_pick_permanently_unreachable(name)
        next_name = strategy.advance_pick_index()
        assert next_name == "pick_4"

    def test_second_chance_pass_does_not_resurrect_permanent(self):
        """When all candidates are permanent + deferred, second-chance must NOT pick a permanent one."""
        strategy = _make_strategy(n=3)
        # pick_0 is the regular livelock-causing pick (deferred).
        strategy.defer_pick("pick_0")
        # pick_1 and pick_2 fell off — permanent.
        strategy.mark_pick_permanently_unreachable("pick_1")
        strategy.mark_pick_permanently_unreachable("pick_2")
        # advance from 0: skip pick_1, skip pick_2, tail exhausted; wrap
        # finds nothing (pick_0 deferred, pick_1/pick_2 permanent);
        # second-chance clears _deferred_picks but pick_1/pick_2 stay permanent.
        next_name = strategy.advance_pick_index()
        assert next_name == "pick_0"
        # Deferred cleared, permanent intact.
        assert strategy.deferred_picks == set()
        assert strategy.permanently_unreachable_picks == {"pick_1", "pick_2"}

    def test_all_permanent_returns_none(self):
        strategy = _make_strategy(n=3)
        for i in range(3):
            strategy.mark_pick_permanently_unreachable(f"pick_{i}")
        # advance_pick_index from pick_0: increments to 1, skips all, tail
        # exhausted, falls through to scan because deferred set non-empty.
        next_name = strategy.advance_pick_index()
        # Scan + second-chance both filter out permanent picks, so None.
        assert next_name is None
        assert strategy.permanently_unreachable_picks == {"pick_0", "pick_1", "pick_2"}

    def test_reset_clears_permanent(self):
        strategy = _make_strategy()
        strategy.mark_pick_permanently_unreachable("pick_0")
        strategy.reset()
        assert strategy.permanently_unreachable_picks == set()


class TestCycleProgressCounter:
    """The no-progress safety-net counter."""

    def test_starts_at_zero(self):
        strategy = _make_strategy()
        assert strategy.cycles_since_last_completion == 0

    def test_increment_returns_new_value(self):
        strategy = _make_strategy()
        assert strategy.increment_cycle_count() == 1
        assert strategy.increment_cycle_count() == 2
        assert strategy.cycles_since_last_completion == 2

    def test_mark_pick_complete_resets_counter(self):
        strategy = _make_strategy()
        strategy.increment_cycle_count()
        strategy.increment_cycle_count()
        strategy.mark_pick_complete("pick_0")
        assert strategy.cycles_since_last_completion == 0

    def test_reset_clears_counter(self):
        strategy = _make_strategy()
        strategy.increment_cycle_count()
        strategy.increment_cycle_count()
        strategy.reset()
        assert strategy.cycles_since_last_completion == 0


class TestDeferCountPromotion:
    """Per-pick consecutive-defer counter promotes to permanently-unreachable."""

    def test_first_defer_does_not_promote(self):
        strategy = _make_strategy()
        strategy.defer_pick("pick_0")
        assert strategy.is_pick_deferred("pick_0")
        assert not strategy.is_pick_permanently_unreachable("pick_0")

    def test_threshold_defers_promote_to_permanent(self):
        """N consecutive defers (no completion in between) → permanent."""
        strategy = _make_strategy()
        threshold = strategy.MAX_DEFERS_BEFORE_PERMANENT
        for _ in range(threshold - 1):
            strategy.defer_pick("pick_0")
            assert not strategy.is_pick_permanently_unreachable("pick_0")
        # Nth defer triggers promotion.
        strategy.defer_pick("pick_0")
        assert strategy.is_pick_permanently_unreachable("pick_0")
        # Other picks unaffected.
        assert not strategy.is_pick_permanently_unreachable("pick_1")

    def test_mark_pick_complete_resets_defer_counter(self):
        """A successful completion clears the per-pick defer counter for *all* picks."""
        strategy = _make_strategy()
        threshold = strategy.MAX_DEFERS_BEFORE_PERMANENT
        # Defer pick_0 threshold-1 times (one short of permanent).
        for _ in range(threshold - 1):
            strategy.defer_pick("pick_0")
        # A sibling completion resets the counter.
        strategy.mark_pick_complete("pick_1")
        # pick_0 can now be deferred threshold-1 more times before permanent.
        for _ in range(threshold - 1):
            strategy.defer_pick("pick_0")
            assert not strategy.is_pick_permanently_unreachable("pick_0")
        strategy.defer_pick("pick_0")
        assert strategy.is_pick_permanently_unreachable("pick_0")

    def test_second_chance_pass_does_not_reset_defer_counter(self):
        """The livelock-clearing second-chance pass clears _deferred_picks but
        not _defer_counts — otherwise two unreachable picks could reset each
        other's counters forever."""
        strategy = _make_strategy(n=2)
        threshold = strategy.MAX_DEFERS_BEFORE_PERMANENT
        # Defer both picks once.
        strategy.defer_pick("pick_0")
        strategy.defer_pick("pick_1")
        # Second-chance pass clears _deferred_picks.
        strategy._scan_for_available_pick()
        assert strategy.deferred_picks == set()
        # But defer counters survive — defer pick_0 (threshold-1) more times
        # and it should promote.
        for _ in range(threshold - 2):
            strategy.defer_pick("pick_0")
            assert not strategy.is_pick_permanently_unreachable("pick_0")
        strategy.defer_pick("pick_0")
        assert strategy.is_pick_permanently_unreachable("pick_0")

    def test_promotion_clears_defer_count_entry(self):
        """Once promoted to permanent, the pick's count entry is removed —
        this is bookkeeping; promoted picks are no longer eligible for
        defer_pick anyway since IsPickReachableGuard short-circuits earlier."""
        strategy = _make_strategy()
        threshold = strategy.MAX_DEFERS_BEFORE_PERMANENT
        for _ in range(threshold):
            strategy.defer_pick("pick_0")
        assert strategy.is_pick_permanently_unreachable("pick_0")
        assert "pick_0" not in strategy._defer_counts

    def test_explicit_mark_permanent_clears_count(self):
        """Direct call to mark_pick_permanently_unreachable also tidies the counter."""
        strategy = _make_strategy()
        strategy.defer_pick("pick_0")
        assert strategy._defer_counts.get("pick_0", 0) == 1
        strategy.mark_pick_permanently_unreachable("pick_0")
        assert "pick_0" not in strategy._defer_counts

    def test_reset_clears_defer_counts(self):
        strategy = _make_strategy()
        strategy.defer_pick("pick_0")
        strategy.defer_pick("pick_1")
        assert strategy._defer_counts
        strategy.reset()
        assert strategy._defer_counts == {}

    def test_independent_picks_have_independent_counters(self):
        strategy = _make_strategy(n=3)
        threshold = strategy.MAX_DEFERS_BEFORE_PERMANENT
        # Hammer pick_0 alone — pick_1, pick_2 must remain unaffected.
        for _ in range(threshold):
            strategy.defer_pick("pick_0")
        assert strategy.is_pick_permanently_unreachable("pick_0")
        assert not strategy.is_pick_permanently_unreachable("pick_1")
        assert not strategy.is_pick_permanently_unreachable("pick_2")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
