"""Tests for spatial-trigger incremental item generation.

Covers:
- SpatialTriggerRegion: bound semantics (each axis independently None-able).
- SpatialTriggeredItemScheduler:
  - initial batch releases ``initial_count`` items in one call from primary generator
  - tick gating: predicate fires/doesn't fire on synthetic positions
  - invert mode (fire iff no item INSIDE region)
  - replenishment generator is preferred over primary for replenishment items
  - conveyor-zero gating (None or 0.0 suppresses replenishment)
  - count cap (count_range terminates release at MAX)
  - bt_should_start opens after initial batch
  - cooldown via min_spawn_interval
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

from item_generation import (
    ConveyorPositionGenerator,
    FixedValue,
    GridPositionGenerator,
    ItemGenerator,
    SpatialTriggerConfig,
    SpatialTriggeredItemScheduler,
    SpatialTriggerRegion,
)


# -----------------------------------------------------------------------------
# SpatialTriggerRegion
# -----------------------------------------------------------------------------


def test_region_unbounded_contains_everything():
    region = SpatialTriggerRegion()
    assert region.contains(0.0, 0.0)
    assert region.contains(1e9, -1e9)


def test_region_max_y_only():
    region = SpatialTriggerRegion(max_y=1.0)
    assert region.contains(0.0, 0.5)
    assert region.contains(0.0, 1.0)        # boundary inclusive
    assert not region.contains(0.0, 1.0001)
    assert region.contains(1e9, 0.0)        # x unbounded


def test_region_min_y_only():
    region = SpatialTriggerRegion(min_y=-1.0)
    assert region.contains(0.0, 0.0)
    assert region.contains(0.0, -1.0)       # boundary inclusive
    assert not region.contains(0.0, -1.0001)


def test_region_full_rectangle():
    region = SpatialTriggerRegion(min_x=-1.0, max_x=1.0, min_y=-2.0, max_y=2.0)
    assert region.contains(0.0, 0.0)
    assert not region.contains(1.5, 0.0)
    assert not region.contains(0.0, -3.0)
    assert region.contains(1.0, 2.0)        # corner


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _grid_gen(count=6, center=(0.5, 0.0, 0.05)):
    return ItemGenerator(
        position_generator=GridPositionGenerator(
            center=np.array(center),
            rows=count, cols=1,
            spacing_x=0.1, spacing_y=0.1,
            randomize=False,
        ),
        asset_type_strategy=FixedValue("cube"),
    )


def _conveyor_gen(center_y=1.0, capacity=10):
    """Single-point generator with x-jiggle (replenishment-style)."""
    return ItemGenerator(
        position_generator=ConveyorPositionGenerator(
            center_x=0.0,
            center_y=center_y,
            z=0.05,
            spacing=0.0,
            jitter_x=0.01,
            jitter_y=0.0,
        ),
        asset_type_strategy=FixedValue("cube"),
    )


# -----------------------------------------------------------------------------
# SpatialTriggeredItemScheduler — initial batch
# -----------------------------------------------------------------------------


def test_initial_batch_releases_initial_count_items():
    primary = _grid_gen(count=8)
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(max_y=10.0),  # never violated
        initial_count=3,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=8, seed=0)
    initial = sched.get_initial_batch()
    assert len(initial) == 3
    assert sched.released_count == 3
    assert sched.total_count == 8

    # Second call returns nothing — already released.
    assert sched.get_initial_batch() == []


def test_bt_should_start_opens_after_initial_batch():
    primary = _grid_gen(count=4)
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(max_y=10.0),
        initial_count=2,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=4, seed=0)
    assert not sched.bt_should_start(0.0)
    sched.get_initial_batch()
    assert sched.bt_should_start(0.0)


# -----------------------------------------------------------------------------
# SpatialTriggeredItemScheduler — tick gating
# -----------------------------------------------------------------------------


def test_tick_returns_empty_before_initial_batch_released():
    primary = _grid_gen(count=4)
    cfg = SpatialTriggerConfig(region=SpatialTriggerRegion(), initial_count=1)
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=4, seed=0)
    # No initial batch yet.
    assert sched.tick(1.0, current_xy=[], conveyor_speed=-0.01) == []


def test_tick_returns_empty_when_conveyor_stationary():
    primary = _grid_gen(count=4)
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(max_y=1.0),
        initial_count=1,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=4, seed=0)
    sched.get_initial_batch()
    # Predicate fires (no items violate max_y=1.0 at y=0), but speed=0 suppresses.
    assert sched.tick(1.0, current_xy=[(0.0, 0.0)], conveyor_speed=0.0) == []
    assert sched.tick(1.0, current_xy=[(0.0, 0.0)], conveyor_speed=None) == []


def test_tick_fires_when_predicate_satisfied_and_belt_moving():
    primary = _grid_gen(count=4)
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(max_y=1.0),
        initial_count=1,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=4, seed=0)
    sched.get_initial_batch()
    # All items have y < max_y=1.0 → none violate → predicate fires.
    new = sched.tick(1.0, current_xy=[(0.0, 0.0)], conveyor_speed=-0.01)
    assert len(new) == 1
    assert sched.released_count == 2


def test_tick_does_not_fire_when_item_outside_region():
    primary = _grid_gen(count=4)
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(max_y=1.0),
        initial_count=1,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=4, seed=0)
    sched.get_initial_batch()
    # An item at y=1.5 violates max_y=1.0 → predicate does NOT fire.
    assert sched.tick(1.0, current_xy=[(0.0, 1.5)], conveyor_speed=-0.01) == []


def test_invert_mode_fires_when_no_item_inside():
    primary = _grid_gen(count=4)
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0),
        initial_count=1,
        invert=True,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=4, seed=0)
    sched.get_initial_batch()
    # Item OUTSIDE the region (y=5.0) → no item inside → predicate fires.
    new = sched.tick(1.0, current_xy=[(0.0, 5.0)], conveyor_speed=-0.01)
    assert len(new) == 1
    # Item INSIDE the region → predicate does NOT fire.
    assert sched.tick(2.0, current_xy=[(0.0, 0.0)], conveyor_speed=-0.01) == []


# -----------------------------------------------------------------------------
# Replenishment generator
# -----------------------------------------------------------------------------


def test_replenishment_uses_secondary_generator_when_provided():
    primary = _grid_gen(count=2, center=(0.0, 0.0, 0.05))   # row at y=0
    repl = _conveyor_gen(center_y=10.0)                     # spawn at y=10 with jitter
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(max_y=5.0),
        initial_count=2,
        replenishment_generation_strategy=repl,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=4, seed=0)
    initial = sched.get_initial_batch()
    # Initial items come from the primary grid generator (y near 0).
    for item in initial:
        assert abs(item.position[1]) < 1.0
    # Predicate fires (existing items at y near 0 satisfy max_y=5.0).
    new = sched.tick(1.0, current_xy=[(0.0, 0.0), (0.0, 0.05)], conveyor_speed=-0.01)
    assert len(new) == 1
    # Replenishment item comes from the secondary generator (y near 10).
    assert abs(new[0].position[1] - 10.0) < 1.0


def test_replenishment_falls_back_to_primary_when_no_secondary():
    primary = _grid_gen(count=4, center=(0.0, 0.0, 0.05))
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(),  # always-fires
        initial_count=2,
        replenishment_generation_strategy=None,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=4, seed=0)
    sched.get_initial_batch()
    new = sched.tick(1.0, current_xy=[], conveyor_speed=-0.01)
    assert len(new) == 1


# -----------------------------------------------------------------------------
# Cap & cooldown
# -----------------------------------------------------------------------------


def test_count_range_caps_total_release():
    primary = _grid_gen(count=10)
    repl = _conveyor_gen(center_y=10.0)
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(),
        initial_count=2,
        replenishment_generation_strategy=repl,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=5, seed=0)
    assert sched.total_count == 5
    sched.get_initial_batch()
    # 3 replenishments allowed (5 total - 2 initial = 3).
    assert len(sched.tick(1.0, current_xy=[], conveyor_speed=-0.01)) == 1
    assert len(sched.tick(2.0, current_xy=[], conveyor_speed=-0.01)) == 1
    assert len(sched.tick(3.0, current_xy=[], conveyor_speed=-0.01)) == 1
    assert sched.all_released
    # No more — capped.
    assert sched.tick(4.0, current_xy=[], conveyor_speed=-0.01) == []
    assert sched.tick(99.0, current_xy=[], conveyor_speed=-0.01) == []


def test_min_spawn_interval_throttles_replenishment():
    primary = _grid_gen(count=4)
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(),
        initial_count=1,
        min_spawn_interval=0.5,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=4, seed=0)
    sched.get_initial_batch()
    # First replenishment fires.
    assert len(sched.tick(1.0, current_xy=[], conveyor_speed=-0.01)) == 1
    # Cooldown blocks the next call (only 0.1s elapsed).
    assert sched.tick(1.1, current_xy=[], conveyor_speed=-0.01) == []
    # After cooldown, fires again.
    assert len(sched.tick(1.6, current_xy=[], conveyor_speed=-0.01)) == 1


def test_initial_count_clamped_to_total():
    """If count_range < initial_count, scheduler still works (clamps)."""
    primary = _grid_gen(count=10)
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(),
        initial_count=5,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=3, seed=0)
    initial = sched.get_initial_batch()
    assert len(initial) == 3
    assert sched.total_count == 3
    assert sched.all_released


def test_trigger_delay_waits_after_predicate_fires():
    primary = _grid_gen(count=4)
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(),
        initial_count=1,
        trigger_delay=2.0,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=4, seed=0)
    sched.get_initial_batch()
    # Predicate fires at t=1.0 — trigger_delay starts counting.
    assert sched.tick(1.0, current_xy=[], conveyor_speed=-0.01) == []
    # Still within delay window (only 1.0s elapsed).
    assert sched.tick(2.0, current_xy=[], conveyor_speed=-0.01) == []
    # Delay elapsed (2.0s since t=1.0).
    new = sched.tick(3.0, current_xy=[], conveyor_speed=-0.01)
    assert len(new) == 1
    assert sched.released_count == 2


def test_trigger_delay_resets_when_predicate_drops():
    primary = _grid_gen(count=4)
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(min_x=-1.0, max_x=1.0, min_y=-1.0, max_y=1.0),
        initial_count=1,
        invert=True,
        trigger_delay=2.0,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=4, seed=0)
    sched.get_initial_batch()
    # Item outside region at t=1.0 — predicate fires, delay starts.
    assert sched.tick(1.0, current_xy=[(0.0, 5.0)], conveyor_speed=-0.01) == []
    # Item re-enters region at t=2.0 — predicate drops, delay resets.
    assert sched.tick(2.0, current_xy=[(0.0, 0.0)], conveyor_speed=-0.01) == []
    # Item leaves again at t=3.0 — delay restarts from scratch.
    assert sched.tick(3.0, current_xy=[(0.0, 5.0)], conveyor_speed=-0.01) == []
    # Only 1.0s elapsed since restart — still blocked.
    assert sched.tick(4.0, current_xy=[(0.0, 5.0)], conveyor_speed=-0.01) == []
    # 2.0s elapsed since restart at t=3.0 — fires.
    new = sched.tick(5.0, current_xy=[(0.0, 5.0)], conveyor_speed=-0.01)
    assert len(new) == 1


def test_reset_replays_initial_batch():
    primary = _grid_gen(count=4)
    cfg = SpatialTriggerConfig(
        region=SpatialTriggerRegion(),
        initial_count=2,
    )
    sched = SpatialTriggeredItemScheduler(primary, cfg, count_range=4, seed=0)
    sched.get_initial_batch()
    sched.tick(1.0, current_xy=[], conveyor_speed=-0.01)
    sched.reset()
    assert sched.released_count == 0
    initial = sched.get_initial_batch()
    assert len(initial) == 2
