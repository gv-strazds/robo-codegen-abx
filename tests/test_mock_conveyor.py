"""Tests for mock-mode conveyor motion.

Covers:
- MockConveyor.is_on_belt: geometric Z + X/Y bounds.
- MockConveyor.advance: per-tick -Y drift, instant Z drop past CONVEYOR_END_Y,
  held-item exemption.
- End-to-end: spatial-trigger replenishment fires in mock when the belt is
  moving (TableTaskBottlesToConveyor2) and stays suppressed when it is not.
"""
import importlib

import numpy as np
import pytest

from tasks_mock.mock_task_utils import setup_mock_modules
setup_mock_modules()

from env_config_values import (
    CONVEYOR_SURFACE_CENTER,
    CONVEYOR_END_Y,
    CONVEYOR_SURFACE_TOP_Z,
    CONVEYOR_SURFACE_X_HALF_EXTENT,
    CONVEYOR_SURFACE_Y_HALF_EXTENT,
    TARGET_MIN_REACHABLE_Z,
)
from task_context_base import LightweightObj
from tasks_mock.mock_task_utils import MOCK_TICK_DT_S, MockConveyor


SPEED = -0.015
DT = MOCK_TICK_DT_S


def _on_belt_pos(y_offset_from_end=0.5):
    """A position resting on the belt surface, ``y_offset_from_end`` above the
    -Y fall-off edge."""
    return np.array([
        float(CONVEYOR_SURFACE_CENTER[0]),
        CONVEYOR_END_Y + y_offset_from_end,
        CONVEYOR_SURFACE_TOP_Z + 0.005,    # 5 mm above belt top (typical pad)
    ])


# ---------------------------------------------------------------------------
# is_on_belt
# ---------------------------------------------------------------------------


class TestIsOnBelt:
    def setup_method(self):
        self.conv = MockConveyor(SPEED)

    def test_position_on_belt_passes(self):
        assert self.conv.is_on_belt(_on_belt_pos())

    def test_z_too_high_fails(self):
        pos = _on_belt_pos()
        pos[2] = CONVEYOR_SURFACE_TOP_Z + 0.5   # well above any tolerance
        assert not self.conv.is_on_belt(pos)

    def test_z_well_above_in_bin_fails(self):
        # Bin items sit ~0.2 m above the conveyor; should never be misidentified.
        assert not self.conv.is_on_belt(np.array([-0.37, 0.6, 0.21]))

    def test_y_past_edge_fails(self):
        pos = _on_belt_pos()
        pos[1] = CONVEYOR_END_Y - 0.01
        assert not self.conv.is_on_belt(pos)

    def test_y_past_far_end_fails(self):
        pos = _on_belt_pos()
        pos[1] = CONVEYOR_END_Y + 2.0 * CONVEYOR_SURFACE_Y_HALF_EXTENT + 0.01
        assert not self.conv.is_on_belt(pos)

    def test_x_outside_belt_fails(self):
        pos = _on_belt_pos()
        pos[0] = float(CONVEYOR_SURFACE_CENTER[0]) + CONVEYOR_SURFACE_X_HALF_EXTENT + 0.01
        assert not self.conv.is_on_belt(pos)


# ---------------------------------------------------------------------------
# advance — drift, falloff, held-item exemption
# ---------------------------------------------------------------------------


class TestAdvance:
    def test_drifts_on_belt_item_in_negative_y(self):
        conv = MockConveyor(SPEED)
        obj = LightweightObj("pad_0", position=_on_belt_pos(0.5))
        y_before = obj.get_world_pose()[0][1]
        conv.advance([obj], DT)
        y_after = obj.get_world_pose()[0][1]
        assert y_after == pytest.approx(y_before + SPEED * DT)

    def test_skips_off_belt_item(self):
        conv = MockConveyor(SPEED)
        # Item sitting in the bin (Z way above the belt surface).
        obj = LightweightObj("bin_can", position=np.array([-0.37, 0.6, 0.21]))
        before = obj.get_world_pose()[0].copy()
        conv.advance([obj], DT)
        after = obj.get_world_pose()[0]
        assert np.allclose(before, after)

    def test_drops_z_at_edge_in_one_tick(self):
        conv = MockConveyor(SPEED)
        # Place the item one tick away from the edge.  Next advance() must
        # both push Y past the edge and drop Z below TARGET_MIN_REACHABLE_Z.
        eps = abs(SPEED * DT) * 0.5    # ensure new Y < CONVEYOR_END_Y
        pos = _on_belt_pos(eps)
        obj = LightweightObj("about_to_fall", position=pos)
        conv.advance([obj], DT)
        new_pos = obj.get_world_pose()[0]
        assert new_pos[1] < CONVEYOR_END_Y
        assert new_pos[2] < TARGET_MIN_REACHABLE_Z, (
            f"expected Z teleported below {TARGET_MIN_REACHABLE_Z}, got {new_pos[2]}"
        )

    def test_fallen_item_is_not_advanced_again(self):
        conv = MockConveyor(SPEED)
        eps = abs(SPEED * DT) * 0.5
        obj = LightweightObj("fallen", position=_on_belt_pos(eps))
        conv.advance([obj], DT)            # falls
        pos_after_fall = obj.get_world_pose()[0].copy()
        conv.advance([obj], DT)            # should not move further
        pos_now = obj.get_world_pose()[0]
        assert np.allclose(pos_after_fall, pos_now)

    def test_skips_held_item(self):
        conv = MockConveyor(SPEED)
        held = LightweightObj("held_pad", position=_on_belt_pos(0.5))
        free = LightweightObj("free_pad", position=_on_belt_pos(0.5))
        conv.advance([held, free], DT, skip_names={"held_pad"})
        assert held.get_world_pose()[0][1] == pytest.approx(_on_belt_pos(0.5)[1])
        assert free.get_world_pose()[0][1] == pytest.approx(
            _on_belt_pos(0.5)[1] + SPEED * DT
        )

    def test_rider_drops_with_carrier_at_edge(self):
        """A placed pick (rider) sitting above the belt on top of a pad
        (carrier) must drop in Z when the carrier crosses the belt edge,
        otherwise verification sees the rider hovering above a sunken
        carrier (the bug fixed alongside this test)."""
        conv = MockConveyor(SPEED)
        eps = abs(SPEED * DT) * 0.5    # one tick away from the edge
        carrier_pos = _on_belt_pos(eps)
        # Rider sits ~5 cm above the carrier (e.g. a soup can on a thin pad).
        rider_pos = carrier_pos.copy()
        rider_pos[2] = CONVEYOR_SURFACE_TOP_Z + 0.06
        carrier = LightweightObj("pad", position=carrier_pos)
        rider = LightweightObj("can", position=rider_pos)
        # Mirror the call sequence in the mock loop: picks first (with
        # ride_with), then targets.  Here picks=[rider], targets=[carrier].
        conv.advance([rider], DT, ride_with={"can": carrier})
        conv.advance([carrier], DT)
        rider_after = rider.get_world_pose()[0]
        carrier_after = carrier.get_world_pose()[0]
        assert rider_after[1] < CONVEYOR_END_Y
        assert carrier_after[1] < CONVEYOR_END_Y
        assert rider_after[2] < TARGET_MIN_REACHABLE_Z, (
            f"rider Z should fall with carrier; got {rider_after[2]}"
        )
        # Rider drops by the same delta as carrier so they stay aligned
        # vertically (rider z = carrier z + original stack offset).
        original_offset = rider_pos[2] - carrier_pos[2]
        assert rider_after[2] == pytest.approx(
            carrier_after[2] + original_offset, abs=1e-9
        )

    def test_rider_does_not_drop_before_carrier_crosses_edge(self):
        """Mid-belt: ride_with drifts the rider in Y but does NOT drop Z
        until the rider's new Y crosses the belt edge."""
        conv = MockConveyor(SPEED)
        carrier_pos = _on_belt_pos(0.5)         # well above the edge
        rider_pos = carrier_pos.copy()
        rider_pos[2] = CONVEYOR_SURFACE_TOP_Z + 0.06
        carrier = LightweightObj("pad", position=carrier_pos)
        rider = LightweightObj("can", position=rider_pos)
        conv.advance([rider], DT, ride_with={"can": carrier})
        rider_after = rider.get_world_pose()[0]
        assert rider_after[1] == pytest.approx(rider_pos[1] + SPEED * DT)
        assert rider_after[2] == pytest.approx(rider_pos[2])    # no Z drop


# ---------------------------------------------------------------------------
# End-to-end: spatial-trigger replenishment fires in mock when belt moves
# ---------------------------------------------------------------------------


def _try_import_class(module_name, class_name):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Mock module setup incomplete in full suite: {e}")


class TestSpatialTriggerInMock:
    """End-to-end check that the predicate fires in mock once items drift past
    Y_THRESHOLD, and stays inert when the belt is forced stationary."""

    def test_replenishment_fires_when_belt_moves(self):
        cls = _try_import_class(
            "tasks2.table_task_bottles_to_conveyor_2",
            "TableTaskBottlesToConveyor2",
        )
        from tasks_mock.mock_task_utils import run_mock_task
        # Smaller target_count keeps the run short while still requiring
        # replenishment beyond the initial 6-pad row (initial_count=6).
        try:
            context, _ = run_mock_task(
                cls, seed=1, verbose=False, show_status=False,
                max_ticks=4000, incremental_checks=False, target_count=8,
            )
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Mock module setup incomplete in full suite: {e}")
        strat = context.strategy
        # The strategy was told to expect more targets (suppression lifted).
        # And targets beyond the initial row must have been spawned via
        # spatial-trigger replenishment.
        assert len(strat.target_objs) > 6, (
            f"expected replenishment beyond initial row of 6, "
            f"got {len(strat.target_objs)} targets"
        )

    def test_min_cycle_time_lets_items_drift_off_edge(self):
        """SoupCans2 with a long min_cycle_time should let pads pass the edge.

        Without the cycle gate the BT consumes all spawned pads almost
        instantly; with a wide gate (15 s; > one batch interval of 6 s),
        conveyor drift between cycles carries the lead pad past
        CONVEYOR_END_Y, triggering the instant Z drop.
        """
        cls = _try_import_class(
            "tasks2.table_task_soup_cans_2", "TableTaskSoupCans2",
        )
        from env_config_values import (
            CONVEYOR_END_Y,
            TARGET_MIN_REACHABLE_Z,
        )
        from tasks_mock.mock_task_utils import run_mock_task
        try:
            context, _ = run_mock_task(
                cls, seed=1, verbose=False, show_status=False,
                max_ticks=20000, incremental_checks=False,
                # Gate > batch_interval (6 s) so unpicked pads in earlier
                # bursts get plenty of time to drift past the edge before
                # the BT comes back for them.
                min_cycle_time_s=15.0,
            )
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Mock module setup incomplete in full suite: {e}")

        targets = context.strategy.target_objs
        # Some targets must have crossed the edge AND been Z-dropped.
        fallen = [
            t for t in targets
            if t.get_world_pose()[0][1] < CONVEYOR_END_Y
            and t.get_world_pose()[0][2] < TARGET_MIN_REACHABLE_Z
        ]
        assert fallen, (
            "expected at least one target to drift past CONVEYOR_END_Y and "
            "get the Z drop, but none did — cycle gate may not be slowing "
            "the BT enough"
        )

    def test_replenishment_suppressed_when_belt_stationary(self):
        cls = _try_import_class(
            "tasks2.table_task_bottles_to_conveyor_2",
            "TableTaskBottlesToConveyor2",
        )
        # Force conveyor_speed = None on the resulting spec (the spec is
        # built fresh each time the task is instantiated, so we mutate it
        # via prepare_mock_from_spec).
        from tasks_mock.mock_task_utils import prepare_mock_from_spec
        try:
            task = cls()
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Mock module setup incomplete in full suite: {e}")
        spec = task.get_task_spec()
        spec.conveyor_speed = None
        spec.target_count = 8
        config = prepare_mock_from_spec(spec, task_class_name="TableTaskBottlesToConveyor2")
        # With belt stationary, target spatial scheduler is suppressed →
        # more_targets_expected stays False, only the initial row is present.
        assert config["target_scheduler"] is not None
        assert config["strategy"].more_targets_expected is False
        assert len(config["target_objs"]) == config["target_scheduler"].released_count
