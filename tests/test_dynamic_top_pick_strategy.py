"""Tests for DynamicTopPickStrategy — JIT, height-aware pick selection.

Covers:
- JIT selection returns the highest-Z settled uncompleted pick.
- A pick is not selected until its position history has stabilised.
- Mid-flight redirect: a newly-arriving higher bottle that is still
  unsettled does NOT redirect the selection; once it settles, it does.
- Latch pins the selection after grasp: subsequent higher arrivals are
  ignored until mark_pick_complete clears the latch.
- get_placing_target_name assigns targets first-unoccupied and never
  reuses targets bound to completed picks.
- add_incremental_picks does not rebuild pairings; new picks become
  selectable once settled.
- more_items_expected stays True while any uncompleted pick is
  unsettled, even after incremental spawning is done.

Pure Python / mock-only — no Isaac Sim required.
"""
import numpy as np
import pytest

from dynamic_top_pick_strategy import DynamicTopPickStrategy


class FakeObj:
    def __init__(self, name, pos=(0.0, 0.0, 0.1)):
        self.name = name
        self.prim_path = f"/World/{name}"
        self._pos = np.asarray(pos, dtype=float)

    def get_world_pose(self):
        return self._pos.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def get_local_pose(self):
        return self.get_world_pose()

    def set_pos(self, pos):
        self._pos = np.asarray(pos, dtype=float)

    def set_z(self, z):
        self._pos[2] = float(z)


SETTLE_WINDOW = 3  # small window for faster tests
FORCE_SETTLED_TICKS = 20  # small watchdog for faster tests (default is 60)


def _make_strategy(
    pick_positions,
    target_positions=None,
    **kwargs,
):
    picks = [FakeObj(f"pick_{i}", pos=p) for i, p in enumerate(pick_positions)]
    if target_positions is None:
        target_positions = [(0.1 * i, 0.0, 0.01) for i in range(max(4, len(picks)))]
    targets = [FakeObj(f"target_{i}", pos=p) for i, p in enumerate(target_positions)]
    kwargs.setdefault("settle_window", SETTLE_WINDOW)
    kwargs.setdefault("force_settled_after_ticks", FORCE_SETTLED_TICKS)
    s = DynamicTopPickStrategy(picks, targets, **kwargs)
    s.initialize_pairings()
    return s, picks, targets


def _settle(strategy, n_polls=None):
    """Poll enough times that all static picks register as settled."""
    if n_polls is None:
        n_polls = strategy._settle_window + 2
    for _ in range(n_polls):
        strategy.poll_pick_positions()


# ---------------------------------------------------------------------------
# 1. Basic JIT selection
# ---------------------------------------------------------------------------


class TestJITSelection:

    def test_selects_highest_z_settled(self):
        s, _, _ = _make_strategy(
            pick_positions=[
                (0.0, 0.0, 0.10),
                (0.1, 0.0, 0.25),  # highest
                (0.2, 0.0, 0.18),
            ],
        )
        _settle(s)
        assert s.get_current_pick_name() == "pick_1"

    def test_returns_none_before_settle_window_fills(self):
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.25)],
        )
        # Only one poll — history too short.
        s.poll_pick_positions()
        assert s.get_current_pick_name() is None

    def test_unsettled_pick_not_selected(self):
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.25)],
        )
        # pick_1 drifts in Z on every poll (still falling)
        for i in range(SETTLE_WINDOW + 2):
            picks[1].set_z(0.25 + 0.01 * i)
            s.poll_pick_positions()
        # pick_1 is unsettled even though it's higher → pick_0 wins.
        assert s.get_current_pick_name() == "pick_0"

    def test_jitter_around_mean_is_settled(self):
        """Regression: random ±1-2mm sample noise (typical sim contact
        chatter) must not trap bottles in an 'unsettled' state forever.
        Net displacement is near zero → settled."""
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10)],
        )
        # Each poll, nudge position by a tiny alternating offset ±0.0015m
        # (1.5 mm) in x, y, z — bigger than the old range-based tol but
        # with no net drift across the window.
        import math
        for i in range(SETTLE_WINDOW + 3):
            dx = 0.0015 * math.cos(i * 2.0)
            dy = 0.0015 * math.sin(i * 1.7)
            dz = 0.0015 * math.sin(i * 1.3)
            picks[0].set_pos((0.0 + dx, 0.0 + dy, 0.10 + dz))
            s.poll_pick_positions()
        # End position back near start → net ≈ 0 → settled → selectable.
        picks[0].set_pos((0.0, 0.0, 0.10))
        s.poll_pick_positions()
        assert s.get_current_pick_name() == "pick_0"

    def test_steady_drift_is_unsettled(self):
        """A bottle drifting steadily in one direction accumulates net
        displacement and remains unsettled."""
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10)],
        )
        # 2 mm per sample Z drift → over window net >> tolerance.
        for i in range(SETTLE_WINDOW + 2):
            picks[0].set_z(0.10 - 0.002 * i)
            s.poll_pick_positions()
        assert s.get_current_pick_name() is None

    def test_min_pick_z_excludes_items_below_floor(self):
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, -0.05), (0.1, 0.0, 0.20)],
            min_pick_z=0.0,
        )
        _settle(s)
        # pick_0 is below the floor, pick_1 is above.
        assert s.get_current_pick_name() == "pick_1"

    def test_deterministic_tie_break_within_top_z_margin(self):
        """Two bottles at the same layer with sub-mm Z noise must resolve
        to a single stable pick, not ping-pong by instantaneous Z order.

        Regression guard for the JIT ping-pong bug (BottlesToConveyor2):
        when two picks sit within ``top_z_margin`` of each other, physics
        jitter flipping which is numerically highest must not flip the
        returned name.  Alphabetical tie-break among tied candidates
        produces the stable answer.
        """
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.100), (0.1, 0.0, 0.101)],
            top_z_margin=0.010,
        )
        _settle(s)
        # pick_0 and pick_1 are within 1 cm → pick_0 wins by name.
        assert s.get_current_pick_name() == "pick_0"
        # Now flip the instantaneous Z ordering — pick_1 nominally higher.
        picks[0].set_z(0.099)
        picks[1].set_z(0.102)
        s.poll_pick_positions()
        # Still pick_0: alphabetical tie-break overrides raw-Z order so
        # long as both picks remain within top_z_margin of the max.
        assert s.get_current_pick_name() == "pick_0"
        # Flip again — should stay stable.
        picks[0].set_z(0.103)
        picks[1].set_z(0.100)
        s.poll_pick_positions()
        assert s.get_current_pick_name() == "pick_0"


# ---------------------------------------------------------------------------
# 2. Mid-flight redirection
# ---------------------------------------------------------------------------


class TestMidFlightRedirect:

    def test_redirects_once_new_bottle_settles(self):
        """A new higher bottle is ignored until it settles, then selected."""
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.15)],
        )
        _settle(s)
        # Initial choice: pick_1 (z=0.15).
        assert s.get_current_pick_name() == "pick_1"

        # A new higher bottle arrives, still dropping.
        new_bottle = FakeObj("pick_2", pos=(0.2, 0.0, 0.30))
        s.add_incremental_picks([new_bottle])

        # First poll: new bottle has 1 sample — unsettled.
        s.poll_pick_positions()
        assert s.get_current_pick_name() == "pick_1"

        # Keep it drifting in Z across the window — remains unsettled.
        for i in range(SETTLE_WINDOW):
            new_bottle.set_z(0.30 - 0.01 * i)  # wobbling
            s.poll_pick_positions()
        assert s.get_current_pick_name() == "pick_1"

        # Freeze the new bottle's position; poll enough times to settle.
        new_bottle.set_pos((0.2, 0.0, 0.28))
        _settle(s)
        assert s.get_current_pick_name() == "pick_2"

    def test_committed_pick_name_round_trip(self):
        """committed_pick_name stashing is a plain attribute round trip."""
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10)],
        )
        _settle(s)
        s.committed_pick_name = "pick_0"
        assert s.committed_pick_name == "pick_0"
        s.mark_pick_complete("pick_0")
        # mark_pick_complete clears committed name when it matches.
        assert s.committed_pick_name is None

    def test_committed_pick_sticky_within_margin(self):
        """Once committed to an in-flight pick, a freshly-settled
        neighbour that lands within ``top_z_margin`` of the committed
        pick must NOT redirect the selection.  Prevents mid-approach
        yanks driven by a neighbour crossing its settle threshold.

        Regression for BottlesToConveyor2: before the sticky fix, the
        cascade of nearby-layer bottles becoming settled caused the
        pre-grasp target to jump 80-150 mm per tick, so RMPFlow never
        converged.  With stickiness, only a *genuinely* higher arrival
        (> top_z_margin above the committed pick) redirects.
        """
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.100), (0.1, 0.0, 0.102)],
            top_z_margin=0.010,
        )
        _settle(s)
        # Tie-break by name → pick_0 initially selected.
        assert s.get_current_pick_name() == "pick_0"
        # Arm commits to pick_0 (CortexMoveToPreGrasp stashes every tick).
        s.committed_pick_name = "pick_0"

        # A neighbour settles into the tied band (pick_1 at 0.105 is
        # 5 mm above pick_0 — inside the 1 cm margin).
        picks[1].set_z(0.105)
        s.poll_pick_positions()
        assert s.get_current_pick_name() == "pick_0"

        # A genuinely-higher arrival (pick_2 at 0.150, 5 cm above the
        # committed pick — well outside margin) must redirect.
        high = FakeObj("pick_2", pos=(0.2, 0.0, 0.150))
        s.add_incremental_picks([high])
        _settle(s)
        assert s.get_current_pick_name() == "pick_2"

    def test_committed_pick_falls_out_of_tied_set_redirects(self):
        """If the committed pick becomes unsettled or leaves the tied
        set (e.g. kicked by a contact so it drops out of the margin
        band), sticky-preservation must NOT pin the arm to it —
        selection falls back to the alphabetical tie-break on the
        remaining tied candidates.
        """
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.100), (0.1, 0.0, 0.100)],
            top_z_margin=0.010,
        )
        _settle(s)
        assert s.get_current_pick_name() == "pick_0"
        s.committed_pick_name = "pick_0"

        # pick_0 gets knocked well below the margin band.
        picks[0].set_z(0.050)
        s.poll_pick_positions()
        # pick_0 is no longer in the tied set → alphabetical fallback
        # returns pick_1 (the only remaining tied candidate).
        assert s.get_current_pick_name() == "pick_1"


# ---------------------------------------------------------------------------
# 3. Pick latching: stability after grasp
# ---------------------------------------------------------------------------


class TestPickLatch:

    def test_latch_survives_new_higher_arrival(self):
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.20)],
        )
        _settle(s)
        assert s.get_current_pick_name() == "pick_1"

        # Gripper closed around pick_1 — latch.
        s.latch_current_pick("pick_1")

        # A new higher settled bottle arrives.
        high = FakeObj("pick_2", pos=(0.2, 0.0, 0.40))
        s.add_incremental_picks([high])
        _settle(s)

        # Latched pick wins.
        assert s.get_current_pick_name() == "pick_1"

    def test_latch_clears_on_mark_complete(self):
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.20)],
        )
        _settle(s)
        s.latch_current_pick("pick_1")
        assert s.latched_pick_name == "pick_1"
        s.mark_pick_complete("pick_1")
        assert s.latched_pick_name is None
        # Next query returns the other uncompleted pick.
        assert s.get_current_pick_name() == "pick_0"

    def test_advance_pick_index_clears_latch(self):
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.20)],
        )
        _settle(s)
        s.latch_current_pick("pick_1")
        s.advance_pick_index()
        assert s.latched_pick_name is None

    def test_stale_latch_cleared_defensively(self):
        """If a latched name leaves the pick list, latch clears on next query."""
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.20)],
        )
        _settle(s)
        s.latch_current_pick("ghost_pick")
        # "ghost_pick" isn't in _pick_objs_by_name → latch clears on query
        # and JIT returns the real top.
        assert s.get_current_pick_name() == "pick_1"
        assert s.latched_pick_name is None


# ---------------------------------------------------------------------------
# 4. JIT target pairing
# ---------------------------------------------------------------------------


class TestTargetPairing:

    def test_assigns_first_unoccupied_target(self):
        """Sequential picks each get distinct targets once the prior pick's
        target is latched (which is what LatchPlacementTarget does in the
        cortex tree between pick and place).  Without a latch, both picks
        would see the same lowest-Y target."""
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.20)],
            target_positions=[(0.0, 0.0, 0.01), (0.1, 0.0, 0.01),
                               (0.2, 0.0, 0.01)],
        )
        _settle(s)
        assert s.get_placing_target_name("pick_0") == "target_0"
        s.latch_current_target("pick_0")
        assert s.get_placing_target_name("pick_1") == "target_1"

    def test_memoises_assignment(self):
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10)],
            target_positions=[(0.0, 0.0, 0.01), (0.1, 0.0, 0.01)],
        )
        _settle(s)
        first = s.get_placing_target_name("pick_0")
        second = s.get_placing_target_name("pick_0")
        assert first == second == "target_0"

    def test_completed_target_not_reused(self):
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.20)],
            target_positions=[(0.0, 0.0, 0.01), (0.1, 0.0, 0.01)],
        )
        _settle(s)
        s.get_placing_target_name("pick_0")  # assigns target_0
        s.mark_pick_complete("pick_0")
        # pick_1 must get a different target
        assert s.get_placing_target_name("pick_1") == "target_1"

    def test_no_targets_returns_none(self):
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10)],
            target_positions=[(0.0, 0.0, 0.01)],
        )
        _settle(s)
        s.get_placing_target_name("pick_0")
        s.mark_pick_complete("pick_0")
        # No targets left, and the pick is completed — pairing returned.
        assert s.get_placing_target_name("pick_0") == "target_0"


# ---------------------------------------------------------------------------
# 5. Incremental picks
# ---------------------------------------------------------------------------


class TestIncrementalPicks:

    def test_add_does_not_clobber_existing_pairings(self):
        """Incremental picks don't rebuild target pairings; latches held by
        in-flight picks are still respected when a new pick queries."""
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.20)],
            target_positions=[(0.0, 0.0, 0.01), (0.1, 0.0, 0.01),
                               (0.2, 0.0, 0.01)],
        )
        _settle(s)
        # Simulate pick_0 and pick_1 mid-cycle: each has latched its target.
        s.get_placing_target_name("pick_0")
        s.latch_current_target("pick_0")
        s.get_placing_target_name("pick_1")
        s.latch_current_target("pick_1")

        # Add a new bottle incrementally.
        s.add_incremental_picks([FakeObj("pick_2", pos=(0.2, 0.0, 0.05))])

        # Existing latches preserved.
        assert s.latched_target_by_pick.get("pick_0") == "target_0"
        assert s.latched_target_by_pick.get("pick_1") == "target_1"
        # New pick gets a target that is neither latched nor occupied.
        _settle(s)
        assert s.get_placing_target_name("pick_2") == "target_2"

    def test_new_pick_selectable_after_settling(self):
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10)],
        )
        _settle(s)
        assert s.get_current_pick_name() == "pick_0"

        # New higher bottle arrives and needs to settle.
        new_bottle = FakeObj("pick_99", pos=(0.5, 0.0, 0.35))
        s.add_incremental_picks([new_bottle])
        _settle(s)
        assert s.get_current_pick_name() == "pick_99"


# ---------------------------------------------------------------------------
# 6. more_items_expected semantics
# ---------------------------------------------------------------------------


class TestMoreItemsExpected:

    def test_true_while_spawn_pending(self):
        s, _, _ = _make_strategy(pick_positions=[(0.0, 0.0, 0.10)])
        s.more_items_expected = True
        # Always True when spawn pending, regardless of settled state.
        assert s.more_items_expected is True

    def test_true_while_unsettled(self):
        s, _, _ = _make_strategy(pick_positions=[(0.0, 0.0, 0.10)])
        s.more_items_expected = False
        # History is empty → uncompleted pick counts as unsettled.
        assert s.more_items_expected is True

    def test_false_when_all_completed(self):
        s, _, _ = _make_strategy(pick_positions=[(0.0, 0.0, 0.10)])
        s.more_items_expected = False
        _settle(s)
        s.mark_pick_complete("pick_0")
        assert s.more_items_expected is False

    def test_true_when_one_pick_unsettled_others_done(self):
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.20)],
        )
        s.more_items_expected = False
        _settle(s)
        s.mark_pick_complete("pick_0")
        # Keep pick_1 perpetually drifting — should stay "expected".
        for i in range(SETTLE_WINDOW + 2):
            picks[1].set_z(0.20 + 0.01 * i)
            s.poll_pick_positions()
        assert s.more_items_expected is True

    def test_false_when_no_target_available_even_if_picks_unsettled(self):
        """Regression: an unsettled leftover pick must not trap SelectNextPick
        in RUNNING when every target is used or permanently unreachable."""
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.20)],
            target_positions=[(0.0, 0.0, 0.01)],
        )
        s.more_items_expected = False
        # Complete pick_0 on target_0 — target is now occupied.
        _settle(s)
        assert s.get_placing_target_name("pick_0") == "target_0"
        s.mark_pick_complete("pick_0")
        # pick_1 is the only uncompleted pick; make it perpetually unsettled.
        for i in range(SETTLE_WINDOW + 2):
            picks[1].set_z(0.20 + 0.01 * i)
            s.poll_pick_positions()
        # No target remains and no more targets expected → False.
        assert s.more_items_expected is False

    def test_true_when_no_target_now_but_more_expected(self):
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.1, 0.0, 0.20)],
            target_positions=[(0.0, 0.0, 0.01)],
        )
        s.more_items_expected = False
        s.more_targets_expected = True  # scheduler still running
        _settle(s)
        assert s.get_placing_target_name("pick_0") == "target_0"
        s.mark_pick_complete("pick_0")
        for i in range(SETTLE_WINDOW + 2):
            picks[1].set_z(0.20 + 0.01 * i)
            s.poll_pick_positions()
        # No target right now, but more coming → keep waiting.
        assert s.more_items_expected is True


# ---------------------------------------------------------------------------
# 7. all_picks_done semantics
# ---------------------------------------------------------------------------


class TestForceSettledWatchdog:

    def test_perpetually_drifting_pick_becomes_selectable(self):
        """Regression: a bottle that never fully settles (sim noise,
        slow contact shuffle) must eventually become selectable so
        SelectNextPick doesn't spin forever."""
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10)],
        )
        # Drift Z by 1mm per poll — exceeds settle_z_tol but is tiny.
        for i in range(FORCE_SETTLED_TICKS + 2):
            picks[0].set_z(0.10 + 0.003 * i)
            s.poll_pick_positions()
        # Normal settle check would refuse this bottle; watchdog kicks
        # in once the pick has been resident long enough.
        assert s.get_current_pick_name() == "pick_0"

    def test_watchdog_disabled_by_none(self):
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10)],
            force_settled_after_ticks=None,
        )
        for i in range(FORCE_SETTLED_TICKS + 2):
            picks[0].set_z(0.10 + 0.003 * i)
            s.poll_pick_positions()
        # Without watchdog, drifting bottle is never selectable.
        assert s.get_current_pick_name() is None

    def test_watchdog_respects_min_pick_z(self):
        """A bottle still falling below the min_pick_z floor must not
        be force-settled — the watchdog should not pick an item that
        is genuinely still airborne."""
        s, picks, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.50)],
            min_pick_z=0.05,
        )
        # Simulate continuous fall below min_pick_z.
        for i in range(FORCE_SETTLED_TICKS + 5):
            picks[0].set_z(-0.10 - 0.001 * i)
            s.poll_pick_positions()
        assert s.get_current_pick_name() is None


class _Region:
    """Minimal duck-typed stand-in for env_config_values.Region2D."""
    def __init__(self, min_x, max_x, min_y, max_y):
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y


class TestPickRegion:

    def test_in_region_pick_selected(self):
        region = _Region(-0.5, 0.5, -0.5, 0.5)
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10)],
            pick_region=region,
        )
        _settle(s)
        assert s.get_current_pick_name() == "pick_0"

    def test_out_of_region_pick_excluded(self):
        """A bottle outside pick_region is never selected, even if highest-Z."""
        region = _Region(-0.5, 0.5, -0.5, 0.5)
        s, _, _ = _make_strategy(
            pick_positions=[
                (0.0, 0.0, 0.10),     # in region
                (5.0, 5.0, 0.50),     # higher Z but far outside region
            ],
            pick_region=region,
        )
        _settle(s)
        assert s.get_current_pick_name() == "pick_0"

    def test_out_of_region_pick_does_not_block_termination(self):
        """Regression: a displaced uncompleted bottle that happens to keep
        wobbling must not trap more_items_expected at True forever."""
        region = _Region(-0.5, 0.5, -0.5, 0.5)
        s, picks, _ = _make_strategy(
            pick_positions=[
                (0.0, 0.0, 0.10),     # in region
                (5.0, 5.0, 0.30),     # displaced — on the conveyor, wobbling
            ],
            target_positions=[(0.0, 0.0, 0.01)],
            pick_region=region,
        )
        s.more_items_expected = False
        _settle(s)
        # Complete pick_0 onto the only target.
        assert s.get_placing_target_name("pick_0") == "target_0"
        s.mark_pick_complete("pick_0")
        # pick_1 is perpetually wobbling outside the region.
        for i in range(SETTLE_WINDOW + 2):
            picks[1].set_pos((5.0, 5.0, 0.30 + 0.01 * i))
            s.poll_pick_positions()
        # Displaced pick does not count toward "waiting for settle".
        # No in-region uncompleted picks AND no targets → False.
        assert s.more_items_expected is False
        assert s.get_current_pick_name() is None

    def test_region_none_disables_filter(self):
        """pick_region=None (default) accepts bottles anywhere."""
        s, _, _ = _make_strategy(
            pick_positions=[(100.0, 100.0, 0.10)],
        )
        _settle(s)
        assert s.get_current_pick_name() == "pick_0"


class TestConveyorProximityPairing:
    """Target-side behaviour inherited from ConveyorProximityStrategy."""

    def test_lowest_y_target_wins(self):
        """With conveyor_sign=-1, the unoccupied target with the smallest Y
        (closest to the -Y fall-off edge) is selected."""
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10)],
            target_positions=[(0.0, 0.5, 0.01),    # upstream
                               (0.0, 0.2, 0.01),   # mid
                               (0.0, -0.1, 0.01)], # most urgent
        )
        _settle(s)
        assert s.get_placing_target_name("pick_0") == "target_2"

    def test_target_latched_on_place_blocks_redirect(self):
        """Once latch_current_target pins the target, a newly-arriving
        lower-Y pad does NOT steal the latch away."""
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10)],
            target_positions=[(0.0, 0.5, 0.01),
                               (0.0, 0.2, 0.01),
                               (0.0, -0.1, 0.01)],
        )
        _settle(s)
        # Initial JIT pick: target_2 at Y=-0.1
        assert s.get_placing_target_name("pick_0") == "target_2"
        s.latch_current_target("pick_0")
        # A new, more-urgent pad arrives.
        new_pad = FakeObj("target_urgent", pos=(0.0, -0.3, 0.01))
        s.add_incremental_targets([new_pad])
        # Latch held: still target_2.
        assert s.get_placing_target_name("pick_0") == "target_2"

    def test_latched_target_falls_off_redirects(self):
        """If the latched target becomes permanently unreachable during
        the place phase, the strategy re-latches to the next-most-urgent
        survivor so the carried bottle is redirected, not dropped."""
        s, _, targets = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10)],
            target_positions=[(0.0, 0.5, 0.01),
                               (0.0, 0.2, 0.01),
                               (0.0, -0.1, 0.01)],
        )
        _settle(s)
        assert s.get_placing_target_name("pick_0") == "target_2"
        s.latch_current_target("pick_0")
        # The latched target falls off the belt.
        s._permanently_unreachable_targets.add("target_2")
        # Next query re-latches to the next-most-urgent available target
        # (target_1 at Y=0.2).
        assert s.get_placing_target_name("pick_0") == "target_1"
        assert s.latched_target_by_pick.get("pick_0") == "target_1"

    def test_add_incremental_targets_does_not_rebuild(self):
        """Adding new targets during the run must not reset the pairings
        map (which would clobber already-latched in-flight picks)."""
        s, _, _ = _make_strategy(
            pick_positions=[(0.0, 0.0, 0.10), (0.5, 0.0, 0.10)],
            target_positions=[(0.0, 0.2, 0.01)],
        )
        _settle(s)
        # pick_0 latches target_0.
        assert s.get_placing_target_name("pick_0") == "target_0"
        s.latch_current_target("pick_0")
        # A new pad arrives with a lower Y.
        s.add_incremental_targets([FakeObj("target_new", pos=(0.0, -0.3, 0.01))])
        # pick_0's latch is still on the original target.
        assert s.latched_target_by_pick.get("pick_0") == "target_0"
        # pick_1 sees the new urgent target (not stolen from pick_0).
        assert s.get_placing_target_name("pick_1") == "target_new"


class TestAllPicksDone:

    def test_false_before_completion(self):
        s, _, _ = _make_strategy(pick_positions=[(0.0, 0.0, 0.10)])
        s.more_items_expected = False
        assert s.all_picks_done is False

    def test_false_while_spawn_pending(self):
        s, _, _ = _make_strategy(pick_positions=[(0.0, 0.0, 0.10)])
        s.more_items_expected = True
        _settle(s)
        s.mark_pick_complete("pick_0")
        assert s.all_picks_done is False

    def test_true_when_done(self):
        s, _, _ = _make_strategy(pick_positions=[(0.0, 0.0, 0.10)])
        s.more_items_expected = False
        _settle(s)
        s.mark_pick_complete("pick_0")
        assert s.all_picks_done is True
