"""Tests for ItemSpawner.

Covers:
- SpawnResult shape and per-tick contents
- more_picks_expected / more_targets_expected flag transitions
- Spatial-trigger vs time-based scheduler dispatch
- bt_should_start gate semantics (OR of configured sides)
- No-op behavior when no scheduler is configured
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
    FixedValue,
    GridPositionGenerator,
    IncrementalGenerationConfig,
    IncrementalItemScheduler,
    ItemGenerator,
    SpatialTriggerConfig,
    SpatialTriggeredItemScheduler,
)
from item_spawner import IsaacPrimFactory, ItemSpawner, SpawnResult
from task_context_base import LightweightObj


def _make_generator(count=4):
    return ItemGenerator(
        position_generator=GridPositionGenerator(
            center=np.array([0.5, 0.0, 0.05]),
            rows=count, cols=1,
            spacing_x=0.1, spacing_y=0.1,
            randomize=False,
        ),
        asset_type_strategy=FixedValue("cube"),
    )


class _RecordingPrimFactory:
    """PrimFactory that materialises ItemSpec lists as LightweightObj prims.

    Records each call so tests can assert which side was invoked and
    with how many items.
    """

    def __init__(self):
        self.pick_calls = []
        self.target_calls = []

    def create_picks(self, items, scene):
        self.pick_calls.append(len(items))
        return [
            LightweightObj(name=item.name or f"pick_{i}", position=item.position)
            for i, item in enumerate(items)
        ]

    def create_targets(self, items, scene):
        self.target_calls.append(len(items))
        return [
            LightweightObj(name=item.name or f"target_{i}", position=item.position)
            for i, item in enumerate(items)
        ]


class TestItemSpawnerBasics:
    def test_no_scheduler_tick_returns_empty_result(self):
        factory = _RecordingPrimFactory()
        spawner = ItemSpawner(prim_factory=factory)
        result = spawner.tick(0.0)
        assert isinstance(result, SpawnResult)
        assert result.new_picks == []
        assert result.new_targets == []
        assert not result.all_picks_released
        assert not result.all_targets_released
        assert spawner.more_picks_expected is False
        assert spawner.more_targets_expected is False

    def test_no_scheduler_bt_should_start_defaults_true(self):
        spawner = ItemSpawner(prim_factory=_RecordingPrimFactory())
        assert spawner.bt_should_start(0.0) is True


class TestItemSpawnerWithIncrementalPicks:
    def _make_spawner(self, total=4):
        gen = _make_generator(count=total)
        cfg = IncrementalGenerationConfig(items_per_batch=2, batch_interval=0.5)
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=total, seed=42)
        # Consume initial batch — task setup phase does this before the
        # spawner is constructed, so model it the same way.
        scheduler.get_initial_batch()
        factory = _RecordingPrimFactory()
        spawner = ItemSpawner(pick_scheduler=scheduler, prim_factory=factory)
        return scheduler, factory, spawner

    def test_first_tick_records_start_time_no_new_items(self):
        scheduler, factory, spawner = self._make_spawner(total=4)
        result = spawner.tick(0.0)
        assert result.new_picks == []
        assert spawner.more_picks_expected is True
        assert factory.pick_calls == []

    def test_tick_after_interval_releases_batch(self):
        scheduler, factory, spawner = self._make_spawner(total=4)
        spawner.tick(0.0)  # set start time
        result = spawner.tick(0.6)  # past the 0.5s interval
        assert len(result.new_picks) == 2
        assert factory.pick_calls == [2]
        assert result.all_picks_released is True
        assert spawner.more_picks_expected is False

    def test_more_picks_expected_flag_clears_on_exhaustion(self):
        scheduler, factory, spawner = self._make_spawner(total=4)
        assert spawner.more_picks_expected is True
        spawner.tick(0.0)
        spawner.tick(0.6)
        assert spawner.more_picks_expected is False
        assert scheduler.all_released is True


class TestItemSpawnerSpatialDispatch:
    def test_spatial_scheduler_receives_current_xy_and_speed(self):
        """Spatial-trigger schedulers must receive ``current_xy`` and
        ``conveyor_speed`` kwargs, derived from the live prim list and
        the spawner's ``conveyor_speed_fn``."""
        gen = _make_generator(count=2)
        cfg = SpatialTriggerConfig(
            region=None,  # never triggers — that's fine, we only check tick args
            initial_count=2,
            items_per_batch=1,
        )
        scheduler = SpatialTriggeredItemScheduler(
            primary_generator=gen,
            config=cfg,
            count_range=4,
            seed=42,
        )
        # Pre-consume initial batch.
        scheduler.get_initial_batch()

        recorded_kwargs = {}

        def fake_tick(simulation_time, **kwargs):
            recorded_kwargs.update(kwargs)
            recorded_kwargs["simulation_time"] = simulation_time
            return []

        scheduler.tick = fake_tick  # type: ignore

        live_picks = [
            LightweightObj(name="p0", position=np.array([0.5, 0.1, 0.05])),
            LightweightObj(name="p1", position=np.array([0.6, 0.2, 0.05])),
        ]
        speed_seen = {"v": 0.015}
        spawner = ItemSpawner(
            pick_scheduler=scheduler,
            prim_factory=_RecordingPrimFactory(),
            conveyor_speed_fn=lambda: speed_seen["v"],
        )
        spawner.tick(1.0, live_picks=live_picks)
        assert recorded_kwargs["simulation_time"] == 1.0
        assert recorded_kwargs["conveyor_speed"] == 0.015
        assert recorded_kwargs["current_xy"] == [(0.5, 0.1), (0.6, 0.2)]

    def test_spatial_scheduler_inert_when_speed_zero(self):
        gen = _make_generator(count=2)
        cfg = SpatialTriggerConfig(region=None, initial_count=1, items_per_batch=1)
        scheduler = SpatialTriggeredItemScheduler(
            primary_generator=gen, config=cfg, count_range=2, seed=42,
        )
        spawner = ItemSpawner(
            pick_scheduler=scheduler,
            prim_factory=_RecordingPrimFactory(),
            conveyor_speed_fn=lambda: 0.0,
        )
        assert spawner.is_spatial(scheduler) is True
        assert spawner.is_spatial_scheduler_inert(scheduler) is True

    def test_time_scheduler_not_treated_as_spatial(self):
        gen = _make_generator(count=2)
        cfg = IncrementalGenerationConfig(items_per_batch=1, batch_interval=0.5)
        scheduler = IncrementalItemScheduler(gen, cfg, count_range=2, seed=42)
        spawner = ItemSpawner(
            pick_scheduler=scheduler, prim_factory=_RecordingPrimFactory(),
        )
        assert spawner.is_spatial(scheduler) is False
        assert spawner.is_spatial_scheduler_inert(scheduler) is False


class TestIsaacPrimFactory:
    """Regression coverage for the self-contained ``IsaacPrimFactory``.

    The factory must own its name counters so naming is independent of
    any external list (notably the strategy's combined scene+virtual
    target list).  These tests stub ``asset_utils`` via ``sys.modules``
    (the real module imports live Isaac Sim APIs that don't exist in
    the mock environment) so the orchestration logic can be exercised
    without standing up an Isaac Sim scene.
    """

    @staticmethod
    def _install_asset_stubs(monkeypatch):
        """Install a stub ``asset_utils`` module in ``sys.modules``.

        ``IsaacPrimFactory.create_picks/create_targets`` import
        ``asset_utils`` lazily on each call, so the in-function import
        picks up the stub registered here.
        """
        import types

        stub = types.ModuleType("asset_utils")

        def fake_add_asset(scene, *, obj_name, position, **kwargs):
            return LightweightObj(name=obj_name, position=position)

        def fake_geom(prim, **kwargs):
            return object()  # opaque sentinel

        stub.add_asset = fake_add_asset
        stub.get_or_compute_prim_geometry = fake_geom
        monkeypatch.setitem(sys.modules, "asset_utils", stub)

    def _make_item(self, asset_type="cube", name=None, color=None):
        from item_generation import ItemSpec
        return ItemSpec(
            asset_type=asset_type,
            position=np.array([0.5, 0.0, 0.05]),
            orientation=None,
            scale=None,
            color=color,
            hidden=False,
            name=name,
        )

    def _make_factory(self, *, pick_name_seq=0, target_name_seq=0):
        prim_geometry = {}
        return IsaacPrimFactory(
            prim_geometry=prim_geometry,
            assets_root_path=None,
            bb_cache_factory=lambda: None,
            pick_name_seq=pick_name_seq,
            target_name_seq=target_name_seq,
        ), prim_geometry

    def test_target_names_are_unique_across_batches(self, monkeypatch):
        """Two successive create_targets calls must NOT collide on names.

        Regression for TableTaskIncrementalTargets crash: prior offset
        calculation reused ``len(configurator._target_objs)`` which
        stayed frozen, producing duplicate ``target_N`` names.
        """
        self._install_asset_stubs(monkeypatch)
        factory, _ = self._make_factory()

        batch1 = factory.create_targets([self._make_item() for _ in range(3)], scene=None)
        batch2 = factory.create_targets([self._make_item() for _ in range(2)], scene=None)

        names = [p.name for p in batch1] + [p.name for p in batch2]
        assert names == ["target_0", "target_1", "target_2", "target_3", "target_4"]
        assert len(set(names)) == len(names)

    def test_pick_names_use_asset_type_prefix_and_advance(self, monkeypatch):
        self._install_asset_stubs(monkeypatch)
        factory, _ = self._make_factory()

        batch1 = factory.create_picks(
            [self._make_item(asset_type="cube") for _ in range(2)], scene=None,
        )
        batch2 = factory.create_picks(
            [self._make_item(asset_type="cube") for _ in range(2)], scene=None,
        )
        names = [p.name for p in batch1] + [p.name for p in batch2]
        assert names == ["cube_0", "cube_1", "cube_2", "cube_3"]

    def test_factory_seeded_with_initial_count(self, monkeypatch):
        """When a factory is seeded with prior counts, subsequent spawns
        continue from the seeded sequence — matches mock parity (where
        we seed from ``len(scene_targets)``)."""
        self._install_asset_stubs(monkeypatch)
        factory, _ = self._make_factory(target_name_seq=5)
        batch = factory.create_targets([self._make_item() for _ in range(2)], scene=None)
        names = [p.name for p in batch]
        assert names == ["target_5", "target_6"]
        assert factory.target_name_seq == 7

    def test_geometry_cache_dict_is_populated(self, monkeypatch):
        self._install_asset_stubs(monkeypatch)
        factory, geom = self._make_factory()
        prims = factory.create_targets([self._make_item() for _ in range(2)], scene=None)
        for prim in prims:
            assert prim.name in geom

    def test_explicit_item_name_bypasses_counter(self, monkeypatch):
        self._install_asset_stubs(monkeypatch)
        factory, _ = self._make_factory()
        prims = factory.create_targets(
            [self._make_item(name="named_target"), self._make_item()],
            scene=None,
        )
        assert [p.name for p in prims] == ["named_target", "target_1"]
        # Counter still advanced for the explicitly-named item (matches
        # prior behavior: counter is positional, not lookup-conditional).
        assert factory.target_name_seq == 2


class TestItemSpawnerBtStartGate:
    def test_or_of_configured_sides(self):
        gen = _make_generator(count=4)
        cfg = IncrementalGenerationConfig(
            items_per_batch=2, batch_interval=0.5, bt_start_threshold=2,
        )
        pick_sched = IncrementalItemScheduler(gen, cfg, count_range=4, seed=42)
        pick_sched.get_initial_batch()  # 2 released → meets threshold

        # Target scheduler that has NOT met its threshold yet (threshold > total).
        gen2 = _make_generator(count=2)
        cfg2 = IncrementalGenerationConfig(
            items_per_batch=1, batch_interval=0.5, bt_start_threshold=10,
        )
        target_sched = IncrementalItemScheduler(gen2, cfg2, count_range=2, seed=42)
        target_sched.get_initial_batch()  # 1 released — below threshold of 10

        spawner = ItemSpawner(
            pick_scheduler=pick_sched, target_scheduler=target_sched,
            prim_factory=_RecordingPrimFactory(),
        )
        # OR is True because picks met their threshold.
        assert spawner.bt_should_start(0.0) is True
