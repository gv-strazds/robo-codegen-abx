"""Tests for incremental target spawning (symmetric to incremental picks).

Verifies:
- TaskSpec.target_incremental_config round-trips through serialization.
- MultiPickStrategy.add_incremental_targets extends the target list and
  recomputes pairings without disturbing completed picks or the active pick.
- CheckTargetAvailable returns RUNNING (rather than latching targets_exhausted)
  while the strategy reports more_targets_expected.
- End-to-end mock execution of TableTaskIncrementalTargets spawns targets
  over time, the BT idles waiting for them, and the task completes.
"""
import numpy as np
import pytest

from tasks_mock.mock_task_utils import setup_mock_modules
setup_mock_modules()

from item_generation import (
    FixedValue,
    GridPositionGenerator,
    IncrementalGenerationConfig,
    ItemGenerator,
)
from multi_pick_strategy import MultiPickStrategy
from task_context_base import LightweightObj, create_lightweight_objs_from_items
from task_spec import TaskSpec


# ---------------------------------------------------------------------------
# TaskSpec serialization
# ---------------------------------------------------------------------------


class TestTaskSpecTargetIncrementalConfig:
    def test_defaults_to_none(self):
        spec = TaskSpec(task_name="t", task_description="d")
        assert spec.target_incremental_config is None

    def test_roundtrip_serialization(self):
        spec = TaskSpec(task_name="t", task_description="d")
        d = spec.to_dict()
        assert "target_incremental_config" in d
        restored = TaskSpec.from_dict(d)
        assert restored.target_incremental_config is None


# ---------------------------------------------------------------------------
# Strategy: add_incremental_targets
# ---------------------------------------------------------------------------


def _make_picks(n):
    return [
        LightweightObj(f"pick_{i}", position=np.array([0.1 * i, 0.0, 0.05]))
        for i in range(n)
    ]


def _make_targets(n, prefix="tgt"):
    return [
        LightweightObj(f"{prefix}_{i}", position=np.array([0.0, 0.1 * i, 0.05]))
        for i in range(n)
    ]


class TestAddIncrementalTargets:
    def test_defaults_more_targets_expected_false(self):
        strat = MultiPickStrategy(pick_objs=_make_picks(2), target_objs=_make_targets(1))
        assert strat.more_targets_expected is False

    def test_extends_target_list_and_pairs_previously_unpaired_pick(self):
        picks = _make_picks(3)
        targets = _make_targets(1)
        strat = MultiPickStrategy(pick_objs=picks, target_objs=targets)
        strat.more_targets_expected = True
        strat.initialize_pairings()

        # With only 1 target and 3 picks, picks 1 and 2 have no target.
        assert strat.get_placing_target_name("pick_0") == "tgt_0"
        assert strat.get_placing_target_name("pick_1") is None

        # Supply 2 more targets; pairings should now cover all 3 picks.
        strat.add_incremental_targets(_make_targets(2, prefix="tgt_new"))
        assert len(strat.target_objs) == 3
        assert strat.get_placing_target_name("pick_1") is not None
        assert strat.get_placing_target_name("pick_2") is not None

    def test_clears_targets_exhausted_latch(self):
        strat = MultiPickStrategy(pick_objs=_make_picks(2), target_objs=_make_targets(1))
        strat.initialize_pairings()
        strat.targets_exhausted = True
        strat.add_incremental_targets(_make_targets(1, prefix="tgt_new"))
        assert strat.targets_exhausted is False

    def test_preserves_completed_picks(self):
        strat = MultiPickStrategy(pick_objs=_make_picks(3), target_objs=_make_targets(1))
        strat.initialize_pairings()
        strat.mark_pick_complete("pick_0")
        strat.add_incremental_targets(_make_targets(2, prefix="tgt_new"))
        # pick_0 stays in completed set
        assert "pick_0" in strat._completed_picks


# ---------------------------------------------------------------------------
# CheckTargetAvailable BT behaviour
# ---------------------------------------------------------------------------


class TestCheckTargetAvailableIdles:
    def _make_context_stub(self, strategy, pick_name):
        class Ctx:
            pass

        ctx = Ctx()
        ctx.strategy = strategy
        ctx.targets_exhausted = False

        def get_current_pick_name():
            return pick_name

        def get_placing_target_name(name):
            return strategy.get_placing_target_name(name)

        ctx.get_current_pick_name = get_current_pick_name
        ctx.get_placing_target_name = get_placing_target_name
        return ctx

    def test_running_while_more_targets_expected(self):
        import py_trees
        from robot_controllers.pt_task_behaviours import CheckTargetAvailable

        # Pick has no target, but more are on the way.
        strat = MultiPickStrategy(pick_objs=_make_picks(2), target_objs=_make_targets(1))
        strat.initialize_pairings()
        strat.more_targets_expected = True

        behaviour = CheckTargetAvailable(name="CheckTargetAvailable")
        behaviour._context = self._make_context_stub(strat, "pick_1")
        status = behaviour.update()
        assert status == py_trees.common.Status.RUNNING
        # Should NOT latch targets_exhausted while we're still waiting.
        assert behaviour._context.targets_exhausted is False

    def test_failure_when_no_more_targets_expected(self):
        import py_trees
        from robot_controllers.pt_task_behaviours import CheckTargetAvailable

        strat = MultiPickStrategy(pick_objs=_make_picks(2), target_objs=_make_targets(1))
        strat.initialize_pairings()
        # Default: more_targets_expected is False.

        behaviour = CheckTargetAvailable(name="CheckTargetAvailable")
        behaviour._context = self._make_context_stub(strat, "pick_1")
        status = behaviour.update()
        assert status == py_trees.common.Status.FAILURE
        assert behaviour._context.targets_exhausted is True


# ---------------------------------------------------------------------------
# End-to-end mock execution of the demo task
# ---------------------------------------------------------------------------


def _try_instantiate_task(module_name, class_name, **kwargs):
    try:
        import importlib
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        return cls(**kwargs)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Mock module setup incomplete in full suite: {e}")


class TestIncrementalTargetsMockExecution:
    def test_demo_task_streams_targets(self):
        task = _try_instantiate_task(
            "tasks.table_task_incremental_targets", "TableTaskIncrementalTargets",
        )
        from tasks_mock.mock_task_utils import prepare_mock_from_spec

        spec = task.get_task_spec()
        config = prepare_mock_from_spec(
            spec, task_class_name="TableTaskIncrementalTargets",
        )

        assert config["target_scheduler"] is not None
        # Only the first batch should exist at setup time.
        assert len(config["target_objs"]) == config["target_scheduler"].released_count
        assert config["target_scheduler"].total_count > config["target_scheduler"].released_count
        assert config["strategy"].more_targets_expected is True

    def test_target_names_match_real_sim_when_virtual_present(self):
        """When virtual targets exist, the mock factory must NOT count
        them toward the scene-target name sequence.  This is the regression
        for the latent mock-only naming offset: prior code seeded the
        factory with ``len(combined_targets)`` (scene + virtual), so the
        first incremental scene target arrived as e.g. ``target_7`` even
        though the real-sim path would produce ``target_5``.  Mock parity
        with real-sim is non-optional — they share the verification path.
        """
        from item_generation import ItemSpec
        from tasks_mock.mock_task_utils import MockPrimFactory

        prim_geometry: dict = {}
        obj_asset_info: dict = {}
        # Pretend: 5 scene targets were spawned plus 2 virtual ones.  The
        # combined list has 7 entries but only scene targets advance the
        # factory's counter.
        factory = MockPrimFactory(
            prim_geometry=prim_geometry,
            obj_asset_info=obj_asset_info,
            pick_name_seq=0,
            target_name_seq=5,
        )

        items = [
            ItemSpec(asset_type="cube", position=np.array([0.5, 0.0, 0.05]))
            for _ in range(2)
        ]
        new_objs = factory.create_targets(items)
        assert [o.name for o in new_objs] == ["target_5", "target_6"]
        assert factory.target_name_seq == 7

    def test_demo_task_completes_end_to_end(self):
        task = _try_instantiate_task(
            "tasks.table_task_incremental_targets", "TableTaskIncrementalTargets",
        )
        try:
            from tasks_mock.mock_task_utils import run_mock_task
            result = run_mock_task(
                type(task), seed=1, verbose=False, show_status=False,
                max_ticks=8000,
                incremental_checks=False,
            )
        except (ImportError, AttributeError) as e:
            # Other tests in the full suite may corrupt mock module state
            # (matches the skip pattern in test_virtual_targets.py).
            pytest.skip(f"Mock module setup incomplete in full suite: {e}")

        context, task_successful = result
        assert task_successful, "mock task did not report success"
        strat = context.strategy
        # The task ends only once at least one of the streaming sides has
        # finished — that's the precondition for "no more pairings possible".
        assert not strat.more_items_expected or not strat.more_targets_expected
        # Pairs are 1:1, so completion is bounded by the smaller spawned side.
        assert len(strat._completed_picks) == min(
            len(strat.pick_objs), len(strat.target_objs)
        )
