"""Tests for the strategy's name-based pairing surface.

Covers:
- ``MultiPickStrategy.get_pick_name_for_target`` reverse lookup with
  completed-pick preference.
- ``_pairings_by_pick_name`` is the only pairings store and stores
  ``Optional[str]`` values.
- ``TaskVerifier.record_available_target_lost`` and the
  name→index translation at the PlacementChecker boundary.
"""

import os
import sys

import numpy as np
import pytest

current_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
mock_path = os.path.join(repo_root, "extsMock")
sys.path.insert(0, mock_path)
sys.path.insert(0, repo_root)

from multi_pick_strategy import MultiPickStrategy
from task_context_base import LightweightObj


def _make_strategy(num_picks: int = 3, num_targets: int = 3) -> MultiPickStrategy:
    picks = [
        LightweightObj(name=f"pick_{i}", position=np.array([0.5, 0.1 * i, 0.05]))
        for i in range(num_picks)
    ]
    targets = [
        LightweightObj(name=f"target_{i}", position=np.array([0.5, 0.3 + 0.1 * i, 0.05]))
        for i in range(num_targets)
    ]
    s = MultiPickStrategy(pick_objs=picks, target_objs=targets)
    s.initialize_pairings()
    return s


class TestPairingsByNameValues:
    def test_pairings_values_are_target_names(self):
        s = _make_strategy(num_picks=3, num_targets=3)
        assert s.pairings_by_pick_name == {
            "pick_0": "target_0",
            "pick_1": "target_1",
            "pick_2": "target_2",
        }

    def test_pairings_handles_more_picks_than_targets(self):
        s = _make_strategy(num_picks=3, num_targets=2)
        assert s.pairings_by_pick_name["pick_0"] == "target_0"
        assert s.pairings_by_pick_name["pick_1"] == "target_1"
        assert s.pairings_by_pick_name["pick_2"] is None

    def test_pairings_attribute_removed(self):
        s = _make_strategy(num_picks=2, num_targets=2)
        assert not hasattr(s, "_pairings")
        assert not hasattr(s, "_pick_name_to_index")


class TestGetPickNameForTarget:
    def test_no_completed_pick_returns_first_match(self):
        s = _make_strategy(num_picks=3, num_targets=3)
        assert s.get_pick_name_for_target("target_1") == "pick_1"

    def test_completed_pick_wins_over_stale_uncompleted_entry(self):
        s = _make_strategy(num_picks=3, num_targets=3)
        # Simulate a stale JIT entry: pick_0 has a stale pairing to target_2;
        # pick_2 is the completed one actually placed on it.
        s._pairings_by_pick_name["pick_0"] = "target_2"
        s._pairings_by_pick_name["pick_2"] = "target_2"
        s._completed_picks.add("pick_2")
        assert s.get_pick_name_for_target("target_2") == "pick_2"

    def test_unknown_target_returns_none(self):
        s = _make_strategy(num_picks=2, num_targets=2)
        assert s.get_pick_name_for_target("nonexistent") is None

    def test_target_with_no_pairing_returns_none(self):
        s = _make_strategy(num_picks=2, num_targets=3)
        # target_2 is not paired with any pick.
        assert s.get_pick_name_for_target("target_2") is None


class TestVerifierNameBasedAPI:
    """Verify that TaskVerifier accepts pick / target names."""

    def _make_verifier(self, strategy):
        from task_verifier import TaskVerifier
        from isaacsim.core.utils.bounds import create_bbox_cache
        return TaskVerifier(
            pick_objs=strategy.pick_objs,
            strategy=strategy,
            bb_cache_factory=create_bbox_cache,
            spatial_check_fn=None,
            placement_constraints_fn=None,
            containment_check=False,
            box_verification_info=None,
            adjust_box_specs_fn=None,
            on_incremental_check_fail=None,
        )

    def test_record_available_target_lost_takes_target_name(self):
        s = _make_strategy(num_picks=2, num_targets=2)
        v = self._make_verifier(s)
        v.record_available_target_lost("target_1", simulation_time=42.0)
        assert v.lost_available_targets() == [("target_1", 42.0)]

    def test_names_to_indices_translates_at_boundary(self):
        """Internal name→index translation maps names to pick_objs positions."""
        s = _make_strategy(num_picks=3, num_targets=3)
        v = self._make_verifier(s)
        assert v._names_to_indices(["pick_0", "pick_2"]) == [0, 2]
        assert v._names_to_indices([]) == []
        # Unknown names are silently dropped.
        assert v._names_to_indices(["pick_0", "nonexistent"]) == [0]


class TestStackingSubclassesNoIndexInternals:
    """Stacking subclass internal collections store names, not indices."""

    def test_single_stack_strategy_stacking_order_is_names(self):
        from multi_pick_strategy import SingleStackStrategy
        picks = [
            LightweightObj(name=f"c_{i}", position=np.array([0.5, 0.5, 0.05]))
            for i in range(3)
        ]
        targets = [
            LightweightObj(name="base", position=np.array([0.5, 0.5, 0.0])),
        ]
        s = SingleStackStrategy(pick_objs=picks, target_objs=targets)
        s.initialize_pairings()
        assert all(isinstance(n, str) for n in s._stacking_order)
        assert set(s._stacking_order) == {"c_0", "c_1", "c_2"}
