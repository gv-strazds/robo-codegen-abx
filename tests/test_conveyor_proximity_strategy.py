"""Tests for ConveyorProximityStrategy — JIT, urgency-aware target selection.

Covers:
- JIT selection returns the lowest-Y reachable unoccupied target.
- Latch pins the target during the place phase even when a more urgent
  target arrives.
- Latched target becoming permanently unreachable → latch is cleared and
  re-latched to the new lowest-Y survivor (carried item is redirected,
  not dropped).
- Occupied targets (completed picks, frozen passing snapshots) are
  excluded from candidates.
- `mark_pick_complete` clears the latch.
- No reachable candidates → returns None and `_pairings_by_pick_name`
  is cleared so `CheckTargetAvailable` takes the correct path.

Pure Python / mock-only — no Isaac Sim required.
"""
import numpy as np
import pytest

from conveyor_proximity_strategy import ConveyorProximityStrategy


class FakeObj:
    def __init__(self, name, pos=(0.0, 0.5, 0.01)):
        self.name = name
        self.prim_path = f"/World/{name}"
        self._pos = np.asarray(pos, dtype=float)

    def get_world_pose(self):
        return self._pos.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def get_local_pose(self):
        return self.get_world_pose()

    def set_y(self, y):
        self._pos[1] = float(y)

    def set_z(self, z):
        self._pos[2] = float(z)


def _make_strategy(pick_ys=(0.8, 0.8, 0.8), target_ys=(0.2, -0.1, -0.5, -0.3, 0.0),
                   target_z=0.01):
    picks = [FakeObj(f"pick_{i}", pos=(0.3 * i, pick_ys[i], 0.1))
             for i in range(len(pick_ys))]
    targets = [FakeObj(f"target_{i}", pos=(0.1 * i, y, target_z))
               for i, y in enumerate(target_ys)]
    s = ConveyorProximityStrategy(picks, targets, conveyor_axis="y",
                                  conveyor_sign=-1)
    s.initialize_pairings()
    return s, picks, targets


# ---------------------------------------------------------------------------
# 1. Basic JIT selection
# ---------------------------------------------------------------------------


class TestUrgencySelection:

    def test_returns_lowest_y_target(self):
        s, _, targets = _make_strategy(
            target_ys=(0.2, -0.1, -0.5, -0.3, 0.0),
        )
        # target_2 at y=-0.5 is lowest → most urgent under conveyor_sign=-1
        assert s.get_placing_target_name("pick_0") == "target_2"

    def test_conveyor_sign_positive_picks_highest_y(self):
        picks = [FakeObj("pick_0", pos=(0, 0.8, 0.1))]
        targets = [FakeObj(f"t_{i}", pos=(0, y, 0.01))
                   for i, y in enumerate([0.2, -0.1, -0.5])]
        s = ConveyorProximityStrategy(picks, targets, conveyor_axis="y",
                                      conveyor_sign=+1)
        s.initialize_pairings()
        # With sign=+1, edge is in +y direction → highest y = most urgent
        assert s.get_placing_target_name("pick_0") == "t_0"  # y=0.2

    def test_no_reachable_returns_none(self):
        s, _, targets = _make_strategy(target_ys=(0.2, -0.1))
        from env_config_values import make_z_reachability_check
        s.set_target_reachable_fn(make_z_reachability_check(min_z=-0.05))
        s.poll_target_reachability()
        for t in targets:
            t.set_z(-0.5)
        s.poll_target_reachability()
        assert s.get_placing_target_name("pick_0") is None
        assert s.pairings_by_pick_name.get("pick_0") is None


# ---------------------------------------------------------------------------
# 2. Latching: stability during place phase
# ---------------------------------------------------------------------------


class TestLatchStability:

    def test_latch_survives_new_lower_y_arrival(self):
        """Once latched, selection sticks even if a more urgent target appears."""
        s, _, targets = _make_strategy(
            target_ys=(0.2, -0.1, 0.0),
        )
        # Initial selection: y=-0.1 (target_1)
        assert s.get_placing_target_name("pick_0") == "target_1"
        s.latch_current_target("pick_0")
        # Simulate a newly arrived, more-urgent target
        targets.append(FakeObj("target_3", pos=(0.0, -0.4, 0.01)))
        # Base class exposes _target_objs/_target_objs_by_name — mirror
        # add_incremental_targets's bookkeeping for test purposes.
        s._target_objs.append(targets[-1])
        s._target_objs_by_name[targets[-1].name] = targets[-1]
        # Still returns latched target_1
        assert s.get_placing_target_name("pick_0") == "target_1"

    def test_latch_clears_on_mark_complete(self):
        s, _, _ = _make_strategy(target_ys=(0.2, -0.1, 0.0))
        s.get_placing_target_name("pick_0")
        s.latch_current_target("pick_0")
        assert "pick_0" in s.latched_target_by_pick
        s.mark_pick_complete("pick_0")
        assert "pick_0" not in s.latched_target_by_pick


# ---------------------------------------------------------------------------
# 3. Latched target dies → redirect, don't drop
# ---------------------------------------------------------------------------


class TestLatchedTargetLost:

    def test_latched_unreachable_redirects_not_drops(self):
        """Latched target becoming permanently unreachable triggers re-latch."""
        from env_config_values import make_z_reachability_check
        s, _, targets = _make_strategy(target_ys=(0.2, -0.1, -0.3, 0.0))
        s.set_target_reachable_fn(make_z_reachability_check(min_z=-0.05))
        s.poll_target_reachability()
        # Initial lowest = target_2 (y=-0.3); latch it
        assert s.get_placing_target_name("pick_0") == "target_2"
        s.latch_current_target("pick_0")
        assert s.latched_target_by_pick["pick_0"] == "target_2"
        # Latched target falls off
        targets[2].set_z(-0.5)
        s.poll_target_reachability()
        # Re-select: remaining lowest reachable is target_1 (y=-0.1)
        new_name = s.get_placing_target_name("pick_0")
        assert new_name == "target_1"
        # And re-latched to the new winner
        assert s.latched_target_by_pick["pick_0"] == "target_1"

    def test_latched_unreachable_no_alternative_returns_none(self):
        """All targets lost while latched → None (place behaviours will wait/drop)."""
        from env_config_values import make_z_reachability_check
        s, _, targets = _make_strategy(target_ys=(0.2, -0.1))
        s.set_target_reachable_fn(make_z_reachability_check(min_z=-0.05))
        s.poll_target_reachability()
        s.get_placing_target_name("pick_0")
        s.latch_current_target("pick_0")
        # All targets fall off
        for t in targets:
            t.set_z(-0.5)
        s.poll_target_reachability()
        assert s.get_placing_target_name("pick_0") is None
        # Latch cleared
        assert "pick_0" not in s.latched_target_by_pick


# ---------------------------------------------------------------------------
# 4. Occupancy exclusion
# ---------------------------------------------------------------------------


class TestOccupancyExclusion:

    def test_completed_picks_exclude_their_targets(self):
        s, _, targets = _make_strategy(
            pick_ys=(0.8, 0.8),
            target_ys=(0.2, -0.3, -0.5),
        )
        # pick_0 takes target_2 (y=-0.5)
        assert s.get_placing_target_name("pick_0") == "target_2"
        s.mark_pick_complete("pick_0")
        # pick_1 must skip target_2 → take target_1 (y=-0.3)
        assert s.get_placing_target_name("pick_1") == "target_1"

    def test_frozen_passing_snapshot_excludes_target(self):
        """Frozen passing check on a target blocks future selections."""
        s, _, _ = _make_strategy(
            pick_ys=(0.8, 0.8),
            target_ys=(0.2, -0.3, -0.5),
        )
        # pick_0 would normally take target_2 — simulate the task verifier
        # reporting target_2 as already claimed by a passing frozen snapshot.
        s.set_frozen_target_names_fn(lambda: {"target_2"})
        # pick_1's next selection must skip target_2
        name = s.get_placing_target_name("pick_1")
        assert name == "target_1"

    def test_latched_by_other_pick_excluded(self):
        """A target latched to another in-flight pick is not selectable."""
        s, _, _ = _make_strategy(
            pick_ys=(0.8, 0.8),
            target_ys=(0.2, -0.3, -0.5),
        )
        # pick_0 selects + latches target_2
        s.get_placing_target_name("pick_0")
        s.latch_current_target("pick_0")
        # pick_1 must select target_1 (can't steal latched target_2)
        assert s.get_placing_target_name("pick_1") == "target_1"


# ---------------------------------------------------------------------------
# 5. Post-completion selection
# ---------------------------------------------------------------------------


class TestPostCompletion:

    def test_fresh_jit_selection_after_complete(self):
        s, _, _ = _make_strategy(
            pick_ys=(0.8, 0.8),
            target_ys=(0.2, -0.3, -0.5),
        )
        assert s.get_placing_target_name("pick_0") == "target_2"
        s.latch_current_target("pick_0")
        s.mark_pick_complete("pick_0")
        # pick_1 gets fresh JIT selection — lowest remaining is target_1
        assert s.get_placing_target_name("pick_1") == "target_1"

    def test_add_incremental_targets_preserves_completed_pairings(self):
        """Incremental target arrivals must not clobber completed picks' pairings.

        In TableTaskSoupCans2, conveyor batches arrive via
        ``add_incremental_targets``.  The base class calls
        ``recompute_pairings`` there, which rebuilds the pairing map
        from the sequential default — that reverts a completed pick's
        pairing from "where it was actually placed" back to its index.
        The ``occupied`` set is then computed from those stale pairings,
        letting the next pick's JIT lock onto a physically-occupied
        target.
        """
        s, _, _ = _make_strategy(
            pick_ys=(0.8, 0.8, 0.8, 0.8),
            target_ys=(0.2, -0.1, 0.0, -0.3),
        )
        # pick_0 grabs the lowest-y (target_3, y=-0.3)
        assert s.get_placing_target_name("pick_0") == "target_3"
        s.latch_current_target("pick_0")
        s.mark_pick_complete("pick_0")
        assert s.pairings_by_pick_name["pick_0"] == "target_3"

        # New conveyor batch arrives
        new_targets = [FakeObj(f"target_{i}", pos=(0.1 * i, y, 0.01))
                       for i, y in [(4, 0.4), (5, 0.5)]]
        s.add_incremental_targets(new_targets)

        # Completed pick's pairing must still point at target_3
        assert s.pairings_by_pick_name["pick_0"] == "target_3"
        assert s.get_placing_target_name("pick_0") == "target_3"

        # pick_1 must see target_3 as occupied and not re-pair to it
        chosen = s.get_placing_target_name("pick_1")
        assert chosen != "target_3"
        assert chosen == "target_1"  # next lowest-y (y=-0.1)

    def test_completed_pick_pairing_is_stable(self):
        """get_placing_target_name must not rewrite completed picks' pairings.

        task_controller._build_pick_observations() queries every pick
        (including completed ones) every tick.  If JIT selection ran for
        completed picks, their pairings would flip around, destabilising
        the ``occupied`` set used by the next pick's selection — causing
        the controller to pair the current pick with a target already
        carrying a completed pick.
        """
        s, _, _ = _make_strategy(
            pick_ys=(0.8, 0.8, 0.8),
            target_ys=(0.2, -0.3, -0.5, 0.0),
        )
        # pick_0 takes its latched target
        assert s.get_placing_target_name("pick_0") == "target_2"
        s.latch_current_target("pick_0")
        s.mark_pick_complete("pick_0")
        pairing_after = s.pairings_by_pick_name["pick_0"]

        # Simulate many observation ticks over the completed pick
        for _ in range(20):
            assert s.get_placing_target_name("pick_0") == "target_2"
            assert s.pairings_by_pick_name["pick_0"] == pairing_after

        # The next pick still sees target_2 as occupied and picks the
        # next-most-urgent target (target_1 at y=-0.3)
        assert s.get_placing_target_name("pick_1") == "target_1"
        s.latch_current_target("pick_1")
        s.mark_pick_complete("pick_1")

        # Now pick_2 must not be allowed to re-pair to target_1 or target_2
        chosen = s.get_placing_target_name("pick_2")
        assert chosen not in ("target_1", "target_2")
        # Lowest remaining among unoccupied (target_3 y=0.0, target_0 y=0.2)
        assert chosen == "target_3"


# ---------------------------------------------------------------------------
# 5b. Stale-pairing cleanup invariant
# ---------------------------------------------------------------------------


class TestStalePairingCleanup:
    """JIT candidature must not leave multiple uncompleted picks pointing at the same target.

    Every ``get_placing_target_name`` call for an uncompleted pick writes
    its claimed target idx into ``_pairings_by_pick_name``.  Without the
    cleanup helper, a JIT pick strategy that cycled through several
    candidates before latching would leave each earlier candidate's
    pairing pointing at the eventual winner's target.  Downstream
    readers that don't filter by completion (e.g. the monitor's
    ``_find_pick_for_target``) would then return the wrong pick for
    that target.
    """

    def test_second_pick_claim_clears_first_picks_pairing(self):
        """Simulates JIT pick churn: pick_A claims target, then pick_B does."""
        s, _, _ = _make_strategy(
            pick_ys=(0.8, 0.8),
            target_ys=(0.2, -0.1, -0.5),
        )
        # First candidate claims the most-urgent target (target_2, y=-0.5)
        assert s.get_placing_target_name("pick_0") == "target_2"
        assert s.pairings_by_pick_name["pick_0"] == "target_2"
        # JIT switches to pick_1; it claims the same target
        assert s.get_placing_target_name("pick_1") == "target_2"
        assert s.pairings_by_pick_name["pick_1"] == "target_2"
        # pick_0's stale pairing must be cleared — the invariant is that
        # at most one uncompleted pick owns each target.
        assert s.pairings_by_pick_name["pick_0"] is None

    def test_completed_pick_pairing_never_cleared_by_cleanup(self):
        """Completed picks' pairings are authoritative (where the pick was placed)."""
        s, _, _ = _make_strategy(
            pick_ys=(0.8, 0.8),
            target_ys=(0.2, -0.1, -0.5),
        )
        assert s.get_placing_target_name("pick_0") == "target_2"
        s.latch_current_target("pick_0")
        s.mark_pick_complete("pick_0")
        assert s.pairings_by_pick_name["pick_0"] == "target_2"
        # A later pick scanning for a target that COINCIDES with an
        # already-completed pick's pairing can't happen in practice
        # (the completed target is in `occupied`, so _jit_select skips
        # it), but the cleanup helper must not clobber the completed
        # pairing even if called.  Drive it directly to cover the
        # invariant.
        s._clear_stale_uncompleted_pairings_to("target_2", except_pick="pick_1")
        assert s.pairings_by_pick_name["pick_0"] == "target_2"

    def test_three_way_jit_churn_only_last_claimant_owns_idx(self):
        """Three uncompleted picks cycle through claiming idx=0 → only the latest survives."""
        picks = [FakeObj(f"p_{i}", pos=(0, 0.8, 0.1)) for i in range(3)]
        targets = [FakeObj("t_0", pos=(0, -0.5, 0.01))]
        s = ConveyorProximityStrategy(picks, targets, conveyor_axis="y",
                                      conveyor_sign=-1)
        s.initialize_pairings()

        s.get_placing_target_name("p_0")
        s.get_placing_target_name("p_1")
        s.get_placing_target_name("p_2")

        claimants = [n for n, t in s.pairings_by_pick_name.items() if t == "t_0"]
        assert claimants == ["p_2"]


# ---------------------------------------------------------------------------
# 6. Cortex "wait vs drop" behaviour
# ---------------------------------------------------------------------------


class _FakeContext:
    def __init__(self, strategy, target_p=None, pick_name=None):
        self.strategy = strategy
        self._target_p = target_p
        self._pick_name = pick_name

    def get_placement_target(self):
        return self._target_p, None

    def get_current_pick_name(self):
        return self._pick_name


class TestCortexWaitVsDrop:
    """Verify CortexMoveToPlace returns RUNNING when more_targets_expected.

    Uses a handwritten fake context to exercise the new branch directly
    (the cortex behaviour expects an arm_commander, so we only drive the
    target-None branch).
    """

    def _make_behaviour_and_context(self, more_expected: bool, strategy):
        import py_trees
        from robot_controllers.pt_cortex_behaviours import CortexMoveToPlace
        strategy._more_targets_expected = more_expected
        ctx = _FakeContext(strategy, target_p=None, pick_name="pick_0")
        b = CortexMoveToPlace(fake_fast=True)
        # Attach our fake context directly (bypass the Behaviour setup path)
        b.context = ctx
        return b, py_trees

    def test_running_when_more_targets_expected(self):
        s, _, _ = _make_strategy()
        b, py_trees = self._make_behaviour_and_context(True, s)
        assert b.update() == py_trees.common.Status.RUNNING

    def test_failure_when_no_more_targets_expected(self):
        s, _, _ = _make_strategy()
        b, py_trees = self._make_behaviour_and_context(False, s)
        assert b.update() == py_trees.common.Status.FAILURE
