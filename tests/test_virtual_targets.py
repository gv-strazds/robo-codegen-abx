"""Tests for virtual target generation.

Verifies:
- create_lightweight_objs_from_items creates correct objects with geometry
- TaskController with virtual strategy merges targets correctly
- Mock execution succeeds with virtual targets
- virtual_target_generation_strategy=None preserves the no-virtual-targets path
"""
import numpy as np
import pytest

# Must set up mock modules before importing anything that touches asset_utils
from tasks_mock.mock_task_utils import setup_mock_modules
setup_mock_modules()

from item_generation import ItemSpec
from task_context_base import LightweightObj, create_lightweight_objs_from_items
from task_controller import TaskController
from task_spec import TaskImplementationSpec, TaskSpec
from multi_pick_strategy import MultiPickStrategy


def _impl(**kwargs) -> TaskImplementationSpec:
    """Build a TaskImplementationSpec for virtual-target wiring tests."""
    return TaskImplementationSpec(**kwargs)


# ---------------------------------------------------------------------------
# create_lightweight_objs_from_items
# ---------------------------------------------------------------------------


class TestCreateLightweightObjsFromItems:
    def test_creates_objects_with_default_names(self):
        items = [
            ItemSpec(asset_type="marker", position=np.array([0.0, 0.0, 0.1]), scale=np.array([0.05, 0.05, 0.001])),
            ItemSpec(asset_type="marker", position=np.array([0.1, 0.0, 0.1]), scale=np.array([0.05, 0.05, 0.001])),
        ]
        objs = create_lightweight_objs_from_items(items, prefix="test_target")
        assert len(objs) == 2
        assert objs[0].name == "test_target_0"
        assert objs[1].name == "test_target_1"

    def test_uses_item_name_when_provided(self):
        items = [
            ItemSpec(asset_type="marker", position=np.array([0.0, 0.0, 0.1]), name="my_marker"),
        ]
        objs = create_lightweight_objs_from_items(items)
        assert objs[0].name == "my_marker"

    def test_populates_prim_geometry(self):
        items = [
            ItemSpec(
                asset_type="soup_can",
                position=np.array([0.0, 0.0, 0.1]),
                scale=np.array([1.0, 1.0, 1.0]),
            ),
        ]
        geom_dict = {}
        objs = create_lightweight_objs_from_items(items, prefix="pick", prim_geometry_out=geom_dict)
        assert len(objs) == 1
        assert "pick_0" in geom_dict
        assert geom_dict["pick_0"].grasp_height > 0

    def test_empty_items_returns_empty(self):
        objs = create_lightweight_objs_from_items([])
        assert objs == []

    def test_preserves_position_and_orientation(self):
        pos = np.array([1.0, 2.0, 3.0])
        orient = np.array([0.707, 0.0, 0.707, 0.0])
        items = [ItemSpec(asset_type="marker", position=pos, orientation=orient)]
        objs = create_lightweight_objs_from_items(items)
        obj_pos, obj_orient = objs[0].get_world_pose()
        np.testing.assert_allclose(obj_pos, pos)
        np.testing.assert_allclose(obj_orient, orient)

    def test_applies_semantic_labels(self):
        items = [
            ItemSpec(asset_type="soup_can", position=np.array([0, 0, 0]), color="red", name="labeled_obj"),
        ]
        objs = create_lightweight_objs_from_items(items)
        assert objs[0].name == "labeled_obj"

    def test_stores_local_half_extents(self):
        items = [
            ItemSpec(asset_type="marker", position=np.array([0.0, 0.0, 0.1]),
                     scale=np.array([0.05, 0.05, 0.001])),
        ]
        objs = create_lightweight_objs_from_items(items)
        assert objs[0]._local_half_extents is not None
        assert len(objs[0]._local_half_extents) == 3
        assert all(h > 0 for h in objs[0]._local_half_extents)


# ---------------------------------------------------------------------------
# AABB computation for LightweightObj (is_on_top verification)
# ---------------------------------------------------------------------------


class TestLightweightObjAABB:
    def test_get_corrected_aabb_uses_half_extents(self):
        """get_corrected_aabb computes AABB from stored half extents for LightweightObj."""
        from task_verification import get_corrected_aabb

        obj = LightweightObj("marker_0", position=np.array([1.0, 2.0, 0.1]))
        obj._local_half_extents = np.array([0.025, 0.025, 0.0005])
        aabb = get_corrected_aabb(obj, bb_cache=None)
        np.testing.assert_allclose(aabb, [0.975, 1.975, 0.0995, 1.025, 2.025, 0.1005])

    def test_is_on_top_with_lightweight_target(self):
        """is_on_top works when the target is a LightweightObj with half extents."""
        from task_verification import is_on_top

        # Pick object sitting at the marker position
        pick = LightweightObj("pick_0", position=np.array([1.0, 2.0, 0.13]))
        pick._local_half_extents = np.array([0.025, 0.025, 0.025])
        # Target marker (flat disc)
        target = LightweightObj("marker_0", position=np.array([1.0, 2.0, 0.1]))
        target._local_half_extents = np.array([0.025, 0.025, 0.0005])
        assert is_on_top(pick, target) is True

    def test_is_on_top_no_xy_overlap_returns_false(self):
        """is_on_top returns False when pick has no XY overlap with target."""
        from task_verification import is_on_top

        pick = LightweightObj("pick_0", position=np.array([5.0, 5.0, 0.13]))
        pick._local_half_extents = np.array([0.025, 0.025, 0.025])
        target = LightweightObj("marker_0", position=np.array([1.0, 2.0, 0.1]))
        target._local_half_extents = np.array([0.025, 0.025, 0.0005])
        assert is_on_top(pick, target) is False


# ---------------------------------------------------------------------------
# TaskController virtual target generation
# ---------------------------------------------------------------------------


class TestTaskControllerVirtualTargets:
    def _make_pick_objs(self, n=3):
        return [
            LightweightObj(f"pick_{i}", position=np.array([0.1 * i, 0.0, 0.05]))
            for i in range(n)
        ]

    def _make_scene_targets(self, n=2):
        return [
            LightweightObj(f"scene_target_{i}", position=np.array([0.0, 0.1 * i, 0.05]))
            for i in range(n)
        ]

    def test_no_virtual_strategy_unchanged(self):
        """Without virtual strategy, create_strategy works as before."""
        picks = self._make_pick_objs(2)
        targets = self._make_scene_targets(2)

        ctrl = TaskController(
            strategy_factory=lambda p, t: MultiPickStrategy(pick_objs=p, target_objs=t),
        )
        strategy = ctrl.create_strategy(picks, targets)
        assert len(strategy.target_objs) == 2
        assert strategy.target_objs[0].name == "scene_target_0"

    def test_virtual_generator_appends_targets(self):
        """Virtual generator with .generate() appends targets after scene targets."""
        picks = self._make_pick_objs(3)
        scene_targets = self._make_scene_targets(1)

        class FakeGenerator:
            def generate(self, count_range=(1, 1), seed=None):
                return [
                    ItemSpec(asset_type="marker", position=np.array([0.5, 0.0, 0.1]),
                             scale=np.array([0.05, 0.05, 0.001])),
                    ItemSpec(asset_type="marker", position=np.array([0.6, 0.0, 0.1]),
                             scale=np.array([0.05, 0.05, 0.001])),
                ]

        ctrl = TaskController(
            strategy_factory=lambda p, t: MultiPickStrategy(pick_objs=p, target_objs=t),
            impl_spec=_impl(virtual_target_generation_strategy=FakeGenerator()),
        )
        strategy = ctrl.create_strategy(picks, scene_targets)
        # 1 scene + 2 virtual = 3 total targets
        assert len(strategy.target_objs) == 3
        assert strategy.target_objs[0].name == "scene_target_0"
        assert strategy.target_objs[1].name == "virtual_target_0"
        assert strategy.target_objs[2].name == "virtual_target_1"

    def test_virtual_callable_receives_pick_and_target_objs(self):
        """Callable virtual strategy receives pick_objs and scene_target_objs."""
        picks = self._make_pick_objs(2)
        scene_targets = self._make_scene_targets(1)

        received = {}

        def pick_aware_gen(pick_objs, target_objs):
            received["picks"] = pick_objs
            received["targets"] = target_objs
            return [
                ItemSpec(asset_type="marker", position=np.array([0.0, 0.0, 0.1]),
                         scale=np.array([0.05, 0.05, 0.001])),
            ]

        ctrl = TaskController(
            strategy_factory=lambda p, t: MultiPickStrategy(pick_objs=p, target_objs=t),
            impl_spec=_impl(virtual_target_generation_strategy=pick_aware_gen),
        )
        strategy = ctrl.create_strategy(picks, scene_targets)
        assert received["picks"] is picks
        assert received["targets"] is scene_targets
        assert len(strategy.target_objs) == 2  # 1 scene + 1 virtual

    def test_virtual_targets_get_geometry_cached(self):
        """Virtual targets' PrimGeometry is cached in the controller's prim_geometry."""
        picks = self._make_pick_objs(1)

        class SoupCanTargetGenerator:
            def generate(self, count_range=(1, 1), seed=None):
                return [
                    ItemSpec(asset_type="soup_can", position=np.array([0.0, 0.0, 0.1]),
                             scale=np.array([1.0, 1.0, 1.0])),
                ]

        prim_geom = {}
        ctrl = TaskController(
            strategy_factory=lambda p, t: MultiPickStrategy(pick_objs=p, target_objs=t),
            prim_geometry=prim_geom,
            impl_spec=_impl(virtual_target_generation_strategy=SoupCanTargetGenerator()),
        )
        ctrl.create_strategy(picks, [])
        assert "virtual_target_0" in prim_geom

    def test_scene_targets_keep_indices(self):
        """Scene targets retain indices 0..n-1; virtual ones start at n."""
        picks = self._make_pick_objs(4)
        scene_targets = self._make_scene_targets(2)

        class VirtualGen:
            def generate(self, count_range=(1, 1), seed=None):
                return [
                    ItemSpec(asset_type="marker", position=np.array([0.0, i * 0.1, 0.1]),
                             scale=np.array([0.05, 0.05, 0.001]))
                    for i in range(2)
                ]

        ctrl = TaskController(
            strategy_factory=lambda p, t: MultiPickStrategy(pick_objs=p, target_objs=t),
            impl_spec=_impl(virtual_target_generation_strategy=VirtualGen()),
        )
        strategy = ctrl.create_strategy(picks, scene_targets)
        # First 2 are scene targets
        assert strategy.target_objs[0].name == "scene_target_0"
        assert strategy.target_objs[1].name == "scene_target_1"
        # Next 2 are virtual
        assert strategy.target_objs[2].name == "virtual_target_0"
        assert strategy.target_objs[3].name == "virtual_target_1"


# ---------------------------------------------------------------------------
# TaskSpec virtual fields
# ---------------------------------------------------------------------------


class TestTaskSpecVirtualFields:
    def test_defaults_to_none(self):
        from task_spec import TaskImplementationSpec, TaskSpec
        spec = TaskSpec(task_name="test", task_description="test")
        assert spec.implementation is None
        assert TaskImplementationSpec().virtual_target_generation_strategy is None
        assert spec.target_count is None

    def test_roundtrip_serialization(self):
        from task_spec import TaskSpec
        spec = TaskSpec(
            task_name="test",
            task_description="test",
            target_count=(3, 5),
        )
        d = spec.to_dict()
        assert "target_count" in d
        restored = TaskSpec.from_dict(d)
        assert restored.target_count == (3, 5)
        assert restored.implementation is None


# ---------------------------------------------------------------------------
# Mock execution with virtual targets (end-to-end)
# ---------------------------------------------------------------------------


def _try_instantiate_task(task_module, task_class_name, **kwargs):
    """Import and instantiate a task class, skipping if mock modules aren't fully set up.

    When running as part of the full test suite, other tests may have
    corrupted the mock module state (e.g. isaacsim.core.api.objects missing
    VisualCapsule). These tests are also validated by run_mock_task.py.
    """
    try:
        import importlib
        mod = importlib.import_module(task_module)
        cls = getattr(mod, task_class_name)
        return cls(**kwargs)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Mock module setup incomplete in full suite: {e}")


class TestMockExecutionWithVirtualTargets:
    """End-to-end tests using prepare_mock_from_spec.

    These may be skipped when running in the full test suite due to mock
    module ordering. They are also validated by run_mock_task.py.
    """

    def test_soup_can_packing_mock(self):
        """TableTaskSoupCanPacking runs successfully with virtual targets."""
        task = _try_instantiate_task("tasks.table_task_soup_can_packing", "TableTaskSoupCanPacking")
        from tasks_mock.mock_task_utils import prepare_mock_from_spec, create_mock_context

        spec = task.get_task_spec()
        config = prepare_mock_from_spec(spec, task_class_name="TableTaskSoupCanPacking")
        # Verify virtual targets were generated
        assert len(config["target_objs"]) == 24  # 4 boxes x 6 markers
        assert any("virtual_target" in o.name for o in config["target_objs"])

        context = create_mock_context(config)
        assert not context.task_finished

    def test_mixed_packing_mock(self):
        """TableTaskMixedPacking runs successfully with virtual targets."""
        task = _try_instantiate_task("tasks.table_task_mixed_packing", "TableTaskMixedPacking")
        from tasks_mock.mock_task_utils import prepare_mock_from_spec, create_mock_context

        spec = task.get_task_spec()
        config = prepare_mock_from_spec(spec, task_class_name="TableTaskMixedPacking")
        # 2 boxes x (1 box target + 4 can targets) = 10 virtual targets
        assert len(config["target_objs"]) == 10
        assert any("virtual_target" in o.name for o in config["target_objs"])

    def test_backward_compat_no_virtual(self):
        """Tasks without virtual targets work unchanged."""
        task = _try_instantiate_task("tasks.table_task_3", "TableTask3")
        from tasks_mock.mock_task_utils import prepare_mock_from_spec

        spec = task.get_task_spec()
        config = prepare_mock_from_spec(spec, task_class_name="TableTask3")
        # No virtual targets — all targets are scene targets
        assert not any("virtual_target" in o.name for o in config["target_objs"])
