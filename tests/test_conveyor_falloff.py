"""Tests for conveyor fall-off snapshot verification.

Covers:
- ConveyorFalloffMonitor poll() logic (snapshot trigger, available-lost,
  hide-after, object-identity matching across incremental spawn).
- MultiPickStrategy state (freeze/retrieve, lost-available list, reset).
- merge_verification_results() helper.
- TaskSpec.falloff_is_enabled() and the four new fields round-tripping.

Pure Python / mock-only — no Isaac Sim required. Uses LightweightObj-style
stand-ins for targets so the monitor can be stepped with synthetic Y-positions
independent of any physics engine (mock mode does not currently animate the
conveyor).
"""
import numpy as np
import pytest

from conveyor_falloff_monitor import ConveyorFalloffMonitor
from task_spec import TaskSpec


# ---------------------------------------------------------------------------
# Lightweight stand-ins
# ---------------------------------------------------------------------------


class FakeTarget:
    """Minimal target stand-in with name + mutable world pose + visibility."""

    def __init__(self, name, pos, half_extents=(0.05, 0.05, 0.005)):
        self.name = name
        self._pos = np.asarray(pos, dtype=float)
        self.visible = True
        self._local_half_extents = np.asarray(half_extents, dtype=float)

    def get_world_pose(self):
        return self._pos.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def set_position(self, y):
        self._pos[1] = float(y)

    def set_visibility(self, visible):
        self.visible = bool(visible)


class FakePick:
    """Pick stand-in. Picks tracked by the monitor for hide/leading-edge checks also
    need a pose + half-extents; older tests use the name-only variant."""

    def __init__(self, name, pos=(0.0, 0.0, 0.0), half_extents=(0.03, 0.05, 0.03)):
        self.name = name
        self._pos = np.asarray(pos, dtype=float)
        self.visible = True
        self._local_half_extents = np.asarray(half_extents, dtype=float)

    def get_world_pose(self):
        return self._pos.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    def set_position(self, y):
        self._pos[1] = float(y)

    def set_visibility(self, visible):
        self.visible = bool(visible)


class FakeStrategy:
    """Minimal MultiPickStrategy-like stand-in for monitor tests.

    Provides the attributes the monitor reads directly:
      - completed_picks (property: set of pick names)
      - _pairings (list of (pick_idx, tgt_idx))
      - _pick_objs (indexed by pairings)
    """

    def __init__(
        self,
        pick_objs,
        pairings,
        completed_names=None,
        pairings_by_pick_name=None,
        target_objs=None,
    ):
        self._pick_objs = list(pick_objs)
        self._pick_objs_by_name = {obj.name: obj for obj in self._pick_objs}
        self._completed = set(completed_names or [])
        # Mirror MultiPickStrategy's name-keyed pairing dict. Defaults to
        # ``pairings`` when not overridden — tests needing to simulate stale
        # JIT-style entries pass an explicit dict with ordered duplicates.
        if pairings_by_pick_name is None:
            self._pairings_by_pick_name = {
                self._pick_objs[p_idx].name: t_idx
                for p_idx, t_idx in pairings
            }
        else:
            self._pairings_by_pick_name = dict(pairings_by_pick_name)
        # The monitor's accessor resolves a target *name* → pick name by
        # walking the strategy's target list.  Tests may pass a target_objs
        # ref (the live monitor target list); fall back to an empty list.
        self._target_objs = list(target_objs) if target_objs is not None else []

    @property
    def completed_picks(self):
        return self._completed

    def get_pick_name_for_target(self, target_name):
        fallback = None
        for pick_name, paired_tgt_idx in self._pairings_by_pick_name.items():
            if paired_tgt_idx is None:
                continue
            try:
                paired_name = self._target_objs[paired_tgt_idx].name
            except (IndexError, AttributeError):
                continue
            if paired_name != target_name:
                continue
            if pick_name in self._completed:
                return pick_name
            if fallback is None:
                fallback = pick_name
        return fallback


# ---------------------------------------------------------------------------
# ConveyorFalloffMonitor tests
# ---------------------------------------------------------------------------


class TestMonitorSnapshot:
    def _build(self, targets, strategy, margin=0.1, end_y=0.0, hide_after=False):
        # The monitor reverse-looks up pairings by target name via
        # ``strategy.get_pick_name_for_target`` — give the fake strategy
        # the same live target list the monitor uses.
        strategy._target_objs = targets
        calls = {"snapshot": [], "lost": [], "hide": []}

        def on_snap(pname, tname, t):
            calls["snapshot"].append((pname, tname, t))

        def on_lost(tname, t):
            calls["lost"].append((tname, t))

        def on_hide(obj):
            calls["hide"].append(obj.name)
            obj.set_visibility(False)

        mon = ConveyorFalloffMonitor(
            strategy=strategy,
            target_objs_ref=targets,
            conveyor_end_y=end_y,
            snapshot_margin=margin,
            hide_after=hide_after,
            on_snapshot=on_snap,
            on_available_lost=on_lost,
            on_hide=on_hide,
        )
        return mon, calls

    def test_snapshot_fires_when_pick_completed(self):
        # Target half-extent Y = 0.05. With margin=0.1, end_y=0.0, snapshot
        # fires when leading_y < 0.1. leading_y = min(target_leading, pick_leading).
        # Position picks co-located with their targets (matches real simulation
        # behavior where pick tracks target after placement).
        picks = [
            FakePick("p0", pos=[0.0, 0.5, 0.0]),   # leading 0.45, no trigger
            FakePick("p1", pos=[0.0, 0.14, 0.0]),  # leading 0.09, triggers
        ]
        targets = [
            FakeTarget("t0", [0.0, 0.5, 0.0]),
            FakeTarget("t1", [0.0, 0.14, 0.0]),
        ]
        strategy = FakeStrategy(picks, [(0, 0), (1, 1)], completed_names={"p1"})
        mon, calls = self._build(targets, strategy)

        mon.poll(simulation_time=10.0)
        assert calls["snapshot"] == [("p1", "t1", 10.0)]
        assert calls["lost"] == []

        # Second poll does NOT re-fire.
        mon.poll(simulation_time=11.0)
        assert calls["snapshot"] == [("p1", "t1", 10.0)]

    def test_leading_edge_of_placed_pick_triggers_earlier(self):
        """A tall placed pick whose leading edge sticks out past the target triggers first."""
        # Pick's leading edge is further ahead (smaller Y) than the target's.
        # Target center 0.3, half_y 0.05 → target leading 0.25
        # Pick   center 0.3, half_y 0.15 → pick   leading 0.15 (further ahead)
        # With end_y=0.0, margin=0.2: threshold = 0.2. Trigger when leading < 0.2.
        # Without the pick, target leading 0.25 wouldn't trigger. With the
        # placed pick, pick leading 0.15 < 0.2 → trigger.
        pick = FakePick("p0", pos=[0.0, 0.3, 0.0], half_extents=[0.03, 0.15, 0.03])
        target = FakeTarget("t0", [0.0, 0.3, 0.0], half_extents=[0.05, 0.05, 0.005])
        strategy = FakeStrategy([pick], [(0, 0)], completed_names={"p0"})
        strategy._target_objs = [target]

        snapshotted = []
        mon = ConveyorFalloffMonitor(
            strategy=strategy,
            target_objs_ref=[target],
            conveyor_end_y=0.0,
            snapshot_margin=0.2,
            hide_after=False,
            on_snapshot=lambda p, t, s: snapshotted.append((p, t, s)),
            on_available_lost=lambda *a, **k: None,
        )

        mon.poll(simulation_time=5.0)
        assert snapshotted == [("p0", "t0", 5.0)]

    def test_uncompleted_pick_position_ignored_in_trigger(self):
        """Uncompleted pick (in bin or carried by robot) must not trigger target snapshot.

        Target is far from the edge. Paired pick is near the edge (simulating
        a can being transported toward its drop point). The monitor must NOT
        treat the pick's leading edge as a trigger, because the pick isn't on
        the target yet.
        """
        pick = FakePick("p0", pos=[0.0, 0.05, 0.0])  # near edge (being carried)
        target = FakeTarget("t0", [0.0, 0.8, 0.0])   # far from edge
        # Not in completed_picks: pick not yet placed.
        strategy = FakeStrategy([pick], [(0, 0)], completed_names=set())
        strategy._target_objs = [target]

        snapshotted = []
        lost = []
        mon = ConveyorFalloffMonitor(
            strategy=strategy,
            target_objs_ref=[target],
            conveyor_end_y=0.0,
            snapshot_margin=0.1,
            hide_after=False,
            on_snapshot=lambda p, t, s: snapshotted.append((p, t, s)),
            on_available_lost=lambda n, s: lost.append((n, s)),
        )
        mon.poll(simulation_time=2.0)
        # Target is far from edge → no trigger; pick's position is ignored
        # because the pick is not completed.
        assert snapshotted == []
        assert lost == []

    def test_snapshot_skipped_for_far_targets(self):
        picks = [FakePick("p0", pos=[0.0, 5.0, 0.0])]
        targets = [FakeTarget("t0", [0.0, 5.0, 0.0])]
        strategy = FakeStrategy(picks, [(0, 0)], completed_names={"p0"})
        mon, calls = self._build(targets, strategy)

        mon.poll(simulation_time=1.0)
        assert calls["snapshot"] == []
        assert calls["lost"] == []

    def test_available_lost_when_no_completed_pick(self):
        """Target crosses edge but its paired pick isn't in completed_picks."""
        picks = [FakePick("p0")]
        # target half_y = 0.05, so center 0.02 → leading -0.03 → < 0.1 threshold
        targets = [FakeTarget("t0", [0.0, 0.02, 0.0])]
        strategy = FakeStrategy(picks, [(0, 0)], completed_names=set())
        mon, calls = self._build(targets, strategy)

        mon.poll(simulation_time=5.0)
        assert calls["snapshot"] == []
        assert calls["lost"] == [("t0", 5.0)]
        # Target is now considered "snapshotted" (one-shot) so no re-fire:
        mon.poll(simulation_time=6.0)
        assert calls["lost"] == [("t0", 5.0)]

    def test_available_lost_when_target_has_no_pairing(self):
        """Extra target with no pairing triggers available_lost, not snapshot."""
        picks = [FakePick("p0", pos=[0.0, 0.5, 0.0])]
        targets = [
            FakeTarget("t0", [0.0, 0.5, 0.0]),
            FakeTarget("t1_extra", [0.0, 0.02, 0.0]),  # crosses edge
        ]
        # Only pairing is for pick p0 → target 0
        strategy = FakeStrategy(picks, [(0, 0)], completed_names={"p0"})
        mon, calls = self._build(targets, strategy)

        mon.poll(simulation_time=3.0)
        # t1_extra has no pairing → counted as lost-available
        assert calls["snapshot"] == []
        assert calls["lost"] == [("t1_extra", 3.0)]

    def test_snapshot_prefers_completed_pick_over_stale_pairing(self):
        """Stale uncompleted entry paired to same target must not mask the completed pick.

        Regression for a JIT-strategy race: ``get_placing_target_name`` can
        write an entry to ``_pairings_by_pick_name`` for every candidate pick
        it considers before latching; those entries persist after the
        strategy moves on. If the stale uncompleted entry appears before the
        completed one in dict insertion order, the monitor would otherwise
        mis-classify the filled target as "available but not filled".
        """
        picks = [
            FakePick("p_stale", pos=[0.0, 2.0, 0.0]),      # still in the bin
            FakePick("p_placed", pos=[0.0, 0.02, 0.0]),    # riding the target
        ]
        # Target crosses threshold; half_y=0.05, margin=0.1, end_y=0.0.
        targets = [FakeTarget("t0", [0.0, 0.02, 0.0])]
        # Both picks carry a stale pairing to tgt 0; p_stale is iterated
        # first (insertion order), p_placed second. Only p_placed is
        # completed — that's the one physically on the target.
        strategy = FakeStrategy(
            picks,
            pairings=[(1, 0)],
            completed_names={"p_placed"},
            pairings_by_pick_name={"p_stale": 0, "p_placed": 0},
        )
        mon, calls = self._build(targets, strategy)

        mon.poll(simulation_time=7.0)
        # Must fire snapshot for the completed pick, NOT available_lost.
        assert calls["snapshot"] == [("p_placed", "t0", 7.0)]
        assert calls["lost"] == []

    def test_snapshot_prefers_completed_even_when_stale_listed_last(self):
        """Symmetry check: completed-pick preference is independent of dict order."""
        picks = [
            FakePick("p_placed", pos=[0.0, 0.02, 0.0]),
            FakePick("p_stale", pos=[0.0, 2.0, 0.0]),
        ]
        targets = [FakeTarget("t0", [0.0, 0.02, 0.0])]
        strategy = FakeStrategy(
            picks,
            pairings=[(0, 0)],
            completed_names={"p_placed"},
            pairings_by_pick_name={"p_placed": 0, "p_stale": 0},
        )
        mon, calls = self._build(targets, strategy)

        mon.poll(simulation_time=7.0)
        assert calls["snapshot"] == [("p_placed", "t0", 7.0)]
        assert calls["lost"] == []


class TestMonitorHide:
    def test_hide_after_edge(self):
        # Leading-edge hide: target center needs to be < end_y + half_y for
        # leading edge < end_y. With end_y=0.0 and half_y=0.05, position.y <
        # 0.05 is required. Put target at pos.y=-0.05 so leading edge is -0.10.
        pick = FakePick("p0", pos=[0.0, -0.1, 0.0])
        targets = [FakeTarget("t0", [0.0, -0.05, 0.0])]
        strategy = FakeStrategy([pick], [(0, 0)], completed_names={"p0"})
        strategy._target_objs = targets

        hidden = []

        def on_hide(obj):
            hidden.append(obj.name)
            obj.set_visibility(False)

        mon = ConveyorFalloffMonitor(
            strategy=strategy,
            target_objs_ref=targets,
            conveyor_end_y=0.0,
            snapshot_margin=0.1,
            hide_after=True,
            on_snapshot=lambda *a, **k: None,
            on_available_lost=lambda *a, **k: None,
            on_hide=on_hide,
        )

        mon.poll(simulation_time=1.0)
        # Target AND paired pick both hidden (paired pick hide follows target).
        assert set(hidden) == {"t0", "p0"}
        assert targets[0].visible is False
        assert pick.visible is False
        # Second poll should not re-hide.
        mon.poll(simulation_time=2.0)
        assert set(hidden) == {"t0", "p0"}

    def test_hide_skipped_when_hide_after_false(self):
        pick = FakePick("p0", pos=[0.0, -0.1, 0.0])
        targets = [FakeTarget("t0", [0.0, -0.05, 0.0])]
        strategy = FakeStrategy([pick], [(0, 0)], completed_names={"p0"})
        strategy._target_objs = targets

        hidden = []
        mon = ConveyorFalloffMonitor(
            strategy=strategy,
            target_objs_ref=targets,
            conveyor_end_y=0.0,
            snapshot_margin=0.1,
            hide_after=False,
            on_snapshot=lambda *a, **k: None,
            on_available_lost=lambda *a, **k: None,
            on_hide=lambda obj: hidden.append(obj.name),
        )
        mon.poll(simulation_time=1.0)
        assert hidden == []
        assert targets[0].visible is True
        assert pick.visible is True

    def test_hide_target_without_paired_pick(self):
        """A target with no pairing gets hidden; no pick hide fires."""
        targets = [FakeTarget("t_extra", [0.0, -0.05, 0.0])]
        strategy = FakeStrategy(pick_objs=[], pairings=[], completed_names=set())
        strategy._target_objs = targets

        hidden = []
        mon = ConveyorFalloffMonitor(
            strategy=strategy,
            target_objs_ref=targets,
            conveyor_end_y=0.0,
            snapshot_margin=0.1,
            hide_after=True,
            on_snapshot=lambda *a, **k: None,
            on_available_lost=lambda *a, **k: None,
            on_hide=lambda obj: hidden.append(obj.name),
        )
        mon.poll(simulation_time=1.0)
        assert hidden == ["t_extra"]

    def test_uncompleted_pick_not_hidden_with_target(self):
        """Target crosses edge, but its paired pick is still in the bin — don't hide the pick."""
        # Pick sitting far from the edge (e.g. in the bin or being carried).
        pick = FakePick("p0", pos=[0.0, 1.0, 0.0])
        target = FakeTarget("t0", [0.0, -0.05, 0.0])  # past edge
        # Not in completed_picks.
        strategy = FakeStrategy([pick], [(0, 0)], completed_names=set())
        strategy._target_objs = [target]

        hidden = []
        mon = ConveyorFalloffMonitor(
            strategy=strategy,
            target_objs_ref=[target],
            conveyor_end_y=0.0,
            snapshot_margin=0.1,
            hide_after=True,
            on_snapshot=lambda *a, **k: None,
            on_available_lost=lambda *a, **k: None,
            on_hide=lambda obj: hidden.append(obj.name),
        )
        mon.poll(simulation_time=1.0)
        # Only the target is hidden — the pick stays visible because it
        # hasn't been placed on the target yet.
        assert hidden == ["t0"]
        assert pick.visible is True


class TestMonitorIdentityMatching:
    """Incremental target spawn can shift indices; monitor must match by identity."""

    def test_index_shift_preserved_by_identity(self):
        """Append a new target at list[0] to shift the moving target's index."""
        # With target half_y=0.05, margin=0.2, end_y=0.0: threshold 0.2.
        # Need target leading edge < 0.2 → center < 0.25.
        pick = FakePick("p0", pos=[0.0, 0.1, 0.0])
        moving = FakeTarget("t_moving", [0.0, 0.5, 0.0])
        targets = [moving]

        # Pairing says pick_0 -> target index 0 (at strategy creation time).
        strategy = FakeStrategy([pick], [(0, 0)], completed_names={"p0"})
        strategy._target_objs = targets

        calls = []

        def on_snap(pidx, tidx, t):
            calls.append((pidx, tidx, t))

        mon = ConveyorFalloffMonitor(
            strategy=strategy,
            target_objs_ref=targets,
            conveyor_end_y=0.0,
            snapshot_margin=0.2,
            hide_after=False,
            on_snapshot=on_snap,
            on_available_lost=lambda *a, **k: None,
        )

        # Simulate incremental spawn that inserts a new target at index 0,
        # shifting `moving` to index 1. The monitor should still recognize it
        # as *the same target* via object identity.
        targets.insert(0, FakeTarget("t_new", [0.0, 0.8, 0.0]))
        moving.set_position(0.1)  # leading edge 0.05 < threshold 0.2

        # Update the strategy's pairing to reflect the new index of `moving`.
        # FakeStrategy is intentionally index-internal (see __init__);
        # the value is treated as an index into _target_objs.
        strategy._pairings_by_pick_name = {"p0": 1}

        mon.poll(simulation_time=4.2)
        assert calls == [("p0", "t_moving", 4.2)]


# ---------------------------------------------------------------------------
# TaskVerifier snapshot-state tests
# ---------------------------------------------------------------------------

from task_verification import PlacementCheck, VerificationResult, merge_verification_results  # noqa: E402
from multi_pick_strategy import MultiPickStrategy  # noqa: E402
from task_verifier import TaskVerifier  # noqa: E402


def _make_strategy(n_picks=2, n_targets=2):
    picks = [FakePick(f"p{i}") for i in range(n_picks)]
    targets = [FakeTarget(f"t{i}", [0.0, 0.0, 0.0]) for i in range(n_targets)]
    return MultiPickStrategy(picks, targets)


def _make_verifier(n_picks=2, n_targets=2):
    strategy = _make_strategy(n_picks=n_picks, n_targets=n_targets)
    return TaskVerifier(
        pick_objs=strategy._pick_objs,
        strategy=strategy,
        bb_cache_factory=lambda: None,
    )


class TestTaskVerifierFrozenState:
    def test_freeze_and_retrieve(self):
        verifier = _make_verifier()
        assert verifier.frozen_pick_indices() == set()
        assert verifier.frozen_checks_ordered() == []

        c0 = PlacementCheck(pick_index=0, pick_name="p0",
                            target_index=0, target_name="t0",
                            passed=True, detail="ok", source="snapshot@1.0s")
        c1 = PlacementCheck(pick_index=1, pick_name="p1",
                            target_index=1, target_name="t1",
                            passed=False, detail="bad", source="snapshot@2.0s")
        verifier.freeze_check(c0)
        verifier.freeze_check(c1)

        assert verifier.frozen_pick_indices() == {0, 1}
        ordered = verifier.frozen_checks_ordered()
        assert [c.pick_index for c in ordered] == [0, 1]

    def test_refreeze_is_noop(self, caplog):
        verifier = _make_verifier()
        c0_v1 = PlacementCheck(pick_index=0, pick_name="p0",
                               target_index=0, target_name="t0",
                               passed=True, detail="first", source="snapshot@1s")
        c0_v2 = PlacementCheck(pick_index=0, pick_name="p0",
                               target_index=0, target_name="t0",
                               passed=False, detail="second", source="snapshot@2s")
        verifier.freeze_check(c0_v1)
        verifier.freeze_check(c0_v2)

        ordered = verifier.frozen_checks_ordered()
        assert len(ordered) == 1
        assert ordered[0].detail == "first"  # re-freeze ignored

    def test_record_available_target_lost(self):
        verifier = _make_verifier()
        verifier.record_available_target_lost("t1", 3.5)
        verifier.record_available_target_lost("t3", 5.2)
        assert verifier.lost_available_targets() == [
            ("t1", 3.5), ("t3", 5.2),
        ]

    def test_frozen_target_names(self):
        verifier = _make_verifier()
        verifier.freeze_check(PlacementCheck(
            pick_index=0, pick_name="p0",
            target_index=0, target_name="t0",
            passed=True, detail="ok", source="snapshot@1.0s",
        ))
        verifier.freeze_check(PlacementCheck(
            pick_index=1, pick_name="p1",
            target_index=1, target_name="t1",
            passed=False, detail="bad", source="snapshot@2.0s",
        ))
        # Only passing checks claim a target name.
        assert verifier.frozen_target_names() == {"t0"}


# ---------------------------------------------------------------------------
# merge_verification_results tests
# ---------------------------------------------------------------------------


class TestMergeVerificationResults:
    @staticmethod
    def _check(i, passed, source="live", target_i=None, detail="ok"):
        return PlacementCheck(
            pick_index=i, pick_name=f"p{i}",
            target_index=target_i,
            target_name=f"t{target_i}" if target_i is not None else None,
            passed=passed, detail=detail, source=source,
        )

    def test_interleave_by_pick_index(self):
        frozen = [
            self._check(0, True, source="snapshot@1.0s", target_i=0),
            self._check(2, True, source="snapshot@3.5s", target_i=2),
        ]
        live = VerificationResult(
            success=True,
            checks=[self._check(1, True, target_i=1), self._check(3, True, target_i=3)],
            failures=[],
        )
        merged = merge_verification_results(frozen, live, pick_count=4)
        assert [c.pick_index for c in merged.checks] == [0, 1, 2, 3]
        assert merged.success is True
        assert merged.failures == []

    def test_frozen_fail_shows_in_failures(self):
        frozen = [
            self._check(0, False, source="snapshot@2.0s",
                        target_i=0, detail="missed target"),
        ]
        live = VerificationResult(
            success=True,
            checks=[self._check(1, True, target_i=1)],
            failures=[],
        )
        merged = merge_verification_results(frozen, live, pick_count=2)
        assert merged.success is False
        assert any("missed target" in f for f in merged.failures)

    def test_summary_tags_snapshot_and_live(self):
        frozen = [self._check(0, True, source="snapshot@12.3s", target_i=0)]
        live = VerificationResult(
            success=True,
            checks=[self._check(1, True, target_i=1)],
            failures=[],
        )
        merged = merge_verification_results(
            frozen, live, pick_count=2,
            info_lines=["Target 'tX' was available but not filled in time (t=9.0s)"],
        )
        text = merged.summary()
        assert "[SNAPSHOT@12.3s]" in text
        assert "[LIVE]" in text
        assert "[INFO] Target 'tX' was available but not filled in time" in text

    def test_info_lines_do_not_fail_task(self):
        frozen = []
        live = VerificationResult(
            success=True,
            checks=[self._check(0, True, target_i=0)],
            failures=[],
        )
        merged = merge_verification_results(
            frozen, live, pick_count=1,
            info_lines=["target was available but not filled"],
        )
        assert merged.success is True
        assert merged.failures == []


class TestLostAvailableFiltering:
    """The task-side filter that drops stale lost-available entries.

    When a target crosses the fall-off threshold just before its paired pick
    completes (race condition), the monitor records it as "lost available".
    But if the pick later lands on it and passes live verification, the info
    line is misleading and should be suppressed. This test exercises the
    filter logic directly against the verifier's recorded state + the merged
    result the task would produce.
    """

    def test_filter_drops_stale_entries(self):
        verifier = _make_verifier(n_picks=2, n_targets=2)
        # Simulate: monitor recorded target 1 as lost-available at t=66.1s,
        # but shortly after soup_can_1 was placed on it and live-passed.
        verifier.record_available_target_lost("t1", 66.1)

        live = VerificationResult(
            success=True,
            checks=[
                PlacementCheck(pick_index=0, pick_name="p0",
                               target_index=0, target_name="t0",
                               passed=True, detail="ok", source="live"),
                PlacementCheck(pick_index=1, pick_name="p1",
                               target_index=1, target_name="t1",
                               passed=True, detail="ok", source="live"),
            ],
            failures=[],
        )

        # Replicate the filter logic from TaskVerifier.verify_final.
        passed_target_names = verifier.frozen_target_names()
        for c in live.checks:
            if c.passed and c.target_name is not None:
                passed_target_names.add(c.target_name)

        info_lines = [
            f"Target '{name}' was available but not filled in time (t={t:.1f}s)"
            for (name, t) in verifier.lost_available_targets()
            if name not in passed_target_names
        ]
        assert info_lines == []  # t1 ended up filled → entry suppressed

    def test_filter_keeps_genuinely_lost_targets(self):
        verifier = _make_verifier(n_picks=1, n_targets=2)
        # target_1 never got a pick on it
        verifier.record_available_target_lost("t1", 40.0)

        live = VerificationResult(
            success=True,
            checks=[
                PlacementCheck(pick_index=0, pick_name="p0",
                               target_index=0, target_name="t0",
                               passed=True, detail="ok", source="live"),
            ],
            failures=[],
        )
        passed_target_names = {
            c.target_name for c in live.checks
            if c.passed and c.target_name is not None
        }
        info_lines = [
            f"Target '{name}' was available but not filled in time (t={t:.1f}s)"
            for (name, t) in verifier.lost_available_targets()
            if name not in passed_target_names
        ]
        assert info_lines == [
            "Target 't1' was available but not filled in time (t=40.0s)"
        ]


# ---------------------------------------------------------------------------
# TaskSpec falloff field tests
# ---------------------------------------------------------------------------


class TestTaskSpecFalloff:
    def test_defaults(self):
        spec = TaskSpec(task_name="t", task_description="d")
        assert spec.conveyor_falloff_enabled is None
        assert spec.conveyor_falloff_snapshot_margin == 0.0
        assert spec.conveyor_falloff_hide_after is True
        assert spec.conveyor_end_y is None

    def test_auto_enable_when_belt_moves(self):
        spec = TaskSpec(
            task_name="t", task_description="d", conveyor_speed=-0.015,
        )
        assert spec.falloff_is_enabled() is True

    def test_auto_disable_when_belt_stationary(self):
        spec = TaskSpec(task_name="t", task_description="d")
        assert spec.falloff_is_enabled() is False
        spec2 = TaskSpec(task_name="t", task_description="d", conveyor_speed=0.0)
        assert spec2.falloff_is_enabled() is False

    def test_explicit_override_wins(self):
        spec_on = TaskSpec(
            task_name="t", task_description="d",
            conveyor_speed=0.0, conveyor_falloff_enabled=True,
        )
        assert spec_on.falloff_is_enabled() is True

        spec_off = TaskSpec(
            task_name="t", task_description="d",
            conveyor_speed=-0.015, conveyor_falloff_enabled=False,
        )
        assert spec_off.falloff_is_enabled() is False

    def test_roundtrip_falloff_fields(self):
        spec = TaskSpec(
            task_name="t", task_description="d",
            conveyor_speed=-0.02,
            conveyor_falloff_enabled=True,
            conveyor_falloff_snapshot_margin=0.15,
            conveyor_falloff_hide_after=False,
            conveyor_end_y=-0.25,
        )
        d = spec.to_dict()
        restored = TaskSpec.from_dict(d)
        assert restored.conveyor_falloff_enabled is True
        assert restored.conveyor_falloff_snapshot_margin == 0.15
        assert restored.conveyor_falloff_hide_after is False
        assert restored.conveyor_end_y == -0.25
        assert restored.falloff_is_enabled() is True

    def test_roundtrip_default_falloff_fields(self):
        spec = TaskSpec(task_name="t", task_description="d")
        d = spec.to_dict()
        restored = TaskSpec.from_dict(d)
        assert restored.conveyor_falloff_enabled is None
        assert restored.conveyor_falloff_snapshot_margin == 0.0
        assert restored.conveyor_falloff_hide_after is True
        assert restored.conveyor_end_y is None
