"""Tests for TaskSpec dataclass, serialization, and get_task_spec() integration.

Verifies:
- TaskSpec construction and field access
- to_dict() / from_dict() round-trip for data fields
- JSON serialization
- get_task_spec() on representative task classes via mock modules
"""
import json
import os
import sys

import numpy as np
import pytest

# Add extsMock and repo root to path
current_dir = os.path.dirname(__file__)
repo_root = os.path.abspath(os.path.join(current_dir, ".."))
mock_path = os.path.join(repo_root, "extsMock")
sys.path.insert(0, mock_path)
sys.path.insert(0, repo_root)

from task_spec import (
    TaskImplementationSpec,
    TaskSpec,
    _callable_ref,
    _resolve_callable,
    _serialize_value,
    _deserialize_value,
)


# ---------------------------------------------------------------------------
# TaskSpec construction
# ---------------------------------------------------------------------------


class TestTaskSpecConstruction:
    def test_minimal(self):
        spec = TaskSpec(task_name="test", task_description="A test task")
        assert spec.task_name == "test"
        assert spec.task_description == "A test task"
        assert spec.pick_generation_strategy is None
        assert spec.implementation is None
        assert spec.impl.create_strategy is None
        assert spec.containment_check is False
        assert spec.stacking_enabled is False
        assert spec.rationale is None

    def test_with_rationale(self):
        rationale = {
            "create_strategy": "Color matching chosen for cube sorting",
            "pick_generation_strategy": "4x3 grid of cubes in bin",
        }
        spec = TaskSpec(
            task_name="colors",
            task_description="Color cube sort",
            rationale=rationale,
        )
        assert spec.rationale["create_strategy"] == "Color matching chosen for cube sorting"

    def test_with_callable(self):
        def my_factory(picks, targets):
            return None

        spec = TaskSpec(
            task_name="test",
            task_description="test",
            implementation=TaskImplementationSpec(create_strategy=my_factory),
        )
        assert spec.implementation.create_strategy is my_factory

    def test_with_counts(self):
        spec = TaskSpec(
            task_name="test",
            task_description="test",
            pick_count=12,
            target_count=(5, 10),
            seed=42,
        )
        assert spec.pick_count == 12
        assert spec.target_count == (5, 10)
        assert spec.seed == 42


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


class TestSerializationHelpers:
    def test_callable_ref_named_function(self):
        ref = _callable_ref(json.dumps)
        assert ref == "json.dumps"

    def test_callable_ref_lambda(self):
        ref = _callable_ref(lambda x: x)
        assert ref is None

    def test_callable_ref_none(self):
        assert _callable_ref(None) is None

    def test_callable_ref_class(self):
        ref = _callable_ref(TaskSpec)
        assert ref == "task_spec.TaskSpec"

    def test_resolve_callable_known(self):
        fn = _resolve_callable("json.dumps")
        assert fn is json.dumps

    def test_resolve_callable_unknown(self):
        fn = _resolve_callable("nonexistent.module.func")
        assert fn is None

    def test_resolve_callable_none(self):
        assert _resolve_callable("") is None
        assert _resolve_callable(None) is None

    def test_serialize_simple_values(self):
        assert _serialize_value(None) is None
        assert _serialize_value("hello") == "hello"
        assert _serialize_value(42) == 42
        assert _serialize_value(3.14) == 3.14
        assert _serialize_value(True) is True

    def test_serialize_numpy_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = _serialize_value(arr)
        assert result == {"__ndarray__": [1.0, 2.0, 3.0]}

    def test_serialize_tuple(self):
        result = _serialize_value((5, 10))
        assert result == {"__tuple__": [5, 10]}

    def test_serialize_callable(self):
        result = _serialize_value(json.dumps)
        assert result["__callable__"] == "json.dumps"

    def test_serialize_lambda(self):
        result = _serialize_value(lambda x: x)
        assert result["__callable__"] is None
        assert "__repr__" in result

    def test_deserialize_simple_values(self):
        assert _deserialize_value(None) is None
        assert _deserialize_value("hello") == "hello"
        assert _deserialize_value(42) == 42

    def test_deserialize_numpy_array(self):
        result = _deserialize_value({"__ndarray__": [1.0, 2.0, 3.0]})
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_deserialize_tuple(self):
        result = _deserialize_value({"__tuple__": [5, 10]})
        assert result == (5, 10)
        assert isinstance(result, tuple)

    def test_deserialize_callable(self):
        result = _deserialize_value({"__callable__": "json.dumps"})
        assert result is json.dumps

    def test_deserialize_unresolvable_callable(self):
        result = _deserialize_value({"__callable__": None})
        assert result is None

    def test_serialize_numpy_int(self):
        result = _serialize_value(np.int64(42))
        assert result == 42
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# TaskSpec to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestTaskSpecSerialization:
    def test_round_trip_data_fields(self):
        spec = TaskSpec(
            task_name="test_task",
            task_description="A test description",
            pick_count=12,
            target_count=(5, 10),
            seed=42,
            containment_check=True,
            stacking_enabled=True,
            implementation=TaskImplementationSpec(ee_height_for_move=0.35),
            rationale={"create_strategy": "Some reason"},
        )
        d = spec.to_dict()
        restored = TaskSpec.from_dict(d)

        assert restored.task_name == "test_task"
        assert restored.task_description == "A test description"
        assert restored.pick_count == 12
        assert restored.target_count == (5, 10)
        assert restored.seed == 42
        assert restored.containment_check is True
        assert restored.stacking_enabled is True
        assert restored.implementation.ee_height_for_move == 0.35
        assert restored.rationale == {"create_strategy": "Some reason"}

    def test_round_trip_named_callable(self):
        spec = TaskSpec(
            task_name="test",
            task_description="test",
            implementation=TaskImplementationSpec(create_strategy=json.dumps),
        )
        d = spec.to_dict()
        restored = TaskSpec.from_dict(d)
        assert restored.implementation.create_strategy is json.dumps

    def test_round_trip_lambda_loses_callable(self):
        spec = TaskSpec(
            task_name="test",
            task_description="test",
            implementation=TaskImplementationSpec(create_strategy=lambda p, t: None),
        )
        d = spec.to_dict()
        restored = TaskSpec.from_dict(d)
        assert restored.implementation.create_strategy is None

    def test_round_trip_none_fields(self):
        spec = TaskSpec(task_name="test", task_description="test")
        d = spec.to_dict()
        restored = TaskSpec.from_dict(d)
        assert restored.pick_generation_strategy is None
        assert restored.implementation is None
        assert restored.spatial_check_fn is None

    def test_to_json(self):
        spec = TaskSpec(
            task_name="test",
            task_description="test",
            pick_count=6,
            rationale={"strategy": "sequential pairing"},
        )
        j = spec.to_json()
        parsed = json.loads(j)
        assert parsed["task_name"] == "test"
        assert parsed["pick_count"] == 6
        assert parsed["rationale"]["strategy"] == "sequential pairing"

    def test_round_trip_box_verification_info(self):
        box_info = {
            "box_specs": [{"name": "box1", "center_xy": [0.1, 0.2]}],
            "box_inner_size": np.array([0.25, 0.35]),
            "box_height": 0.1,
        }
        spec = TaskSpec(
            task_name="test",
            task_description="test",
            box_verification_info=box_info,
        )
        d = spec.to_dict()
        restored = TaskSpec.from_dict(d)
        assert restored.box_verification_info is not None
        restored_info = restored.box_verification_info
        assert restored_info["box_height"] == 0.1
        assert restored_info["box_specs"][0]["name"] == "box1"
        np.testing.assert_array_almost_equal(
            restored_info["box_inner_size"], [0.25, 0.35]
        )


# ---------------------------------------------------------------------------
# get_task_spec() integration with mock task classes
#
# These tests require the full mock environment because task classes import
# Isaac Sim APIs.  Earlier test files (test_asset_utils, test_prim_geometry)
# may have replaced isaacsim/pxr modules with minimal stubs, so we must
# fully reset and re-import from extsMock before loading task classes.
# ---------------------------------------------------------------------------


def _reset_and_setup_mocks():
    """Fully reset isaacsim/pxr modules and set up clean mocks from extsMock."""
    # 1. Remove all isaacsim and pxr modules so they reimport from extsMock
    for key in list(sys.modules):
        if key == "pxr" or key.startswith("pxr."):
            del sys.modules[key]
        elif key.startswith("isaacsim"):
            del sys.modules[key]
    # Also remove table_setup and task modules that may have cached bad imports
    for key in list(sys.modules):
        if key == "table_setup" or key.startswith("tasks."):
            del sys.modules[key]

    # 2. Re-import from extsMock
    import pxr  # noqa: F401

    # 3. Run setup_mock_modules which handles remaining mocks
    from tasks_mock.mock_task_utils import setup_mock_modules
    setup_mock_modules()

    # 4. Patch any missing attributes on loaded modules (table_setup.py
    #    imports many things at module level that extsMock doesn't fully cover)
    from unittest.mock import MagicMock
    _Dummy = type("_Dummy", (), {"__init__": lambda self, *a, **k: None})
    _patches = {
        "isaacsim.core.api": ["World"],
        "isaacsim.core.api.objects": [
            "VisualCapsule", "VisualSphere", "DynamicCuboid", "DynamicCylinder",
            "FixedCuboid", "FixedCylinder", "VisualCuboid", "DynamicSphere",
            "DynamicCapsule", "DynamicCone",
        ],
        "isaacsim.core.api.objects.ground_plane": ["GroundPlane"],
        "isaacsim.core.api.scenes.scene": ["Scene"],
        "isaacsim.core.prims": ["RigidPrim", "SingleRigidPrim", "SingleXFormPrim"],
        "isaacsim.core.utils.collisions": ["ray_cast"],
        "isaacsim.storage.native": ["get_assets_root_path"],
    }
    for mod_name, attrs in _patches.items():
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        for attr in attrs:
            if not hasattr(mod, attr):
                setattr(mod, attr, _Dummy)


_reset_and_setup_mocks()

# Now import task classes (after clean mocks are in place)
from tasks.table_task_colors_1 import TableTaskColors1
from tasks.table_task_bottles_1 import TableTaskBottles1
from tasks.table_task_3 import TableTask3
from tasks.table_task_layered_cubes import TableTaskLayeredCubes
from tasks.table_task_conveyor_sort import TableTaskConveyorSort
from tasks.table_task_mixed_packing import TableTaskMixedPacking
from tasks.table_task_conveyor_color_stacks import TableTaskConveyorColorStacks
from multi_pickplace_task import UR10MultiPickPlaceTask

# Cache clean modules from extsMock at import time so they can be restored
# before each test method (other test files replace them with minimal stubs).
_clean_isaacsim_modules = {
    key: sys.modules[key] for key in list(sys.modules) if key.startswith("isaacsim")
}
_clean_pxr_submodules = {
    key: sys.modules[key] for key in list(sys.modules) if key.startswith("pxr")
}


@pytest.fixture(autouse=True)
def _restore_mock_modules():
    """Restore extsMock modules that other test files may have replaced.

    test_asset_utils, test_prim_geometry, and test_task_verification create
    minimal pxr and isaacsim stubs that lack attributes needed by task classes.
    """
    # Restore all isaacsim modules
    for key, mod in _clean_isaacsim_modules.items():
        sys.modules[key] = mod
    # Restore pxr modules
    for key, mod in _clean_pxr_submodules.items():
        sys.modules[key] = mod
    yield


class TestGetTaskSpec:
    """Test get_task_spec() on representative task classes."""

    def test_colors1_spec(self):
        task = TableTaskColors1()
        spec = task.get_task_spec()
        assert spec.task_name == "table_task_colors_1"
        assert "colored cubes" in spec.task_description.lower()
        assert spec.pick_generation_strategy is task._configurator.pick_generation_strategy
        assert spec.target_generation_strategy is task._configurator.target_generation_strategy
        assert spec.containment_check is False
        assert spec.stacking_enabled is False
        assert spec.implementation.create_strategy is not None
        assert spec.setup_workspace is not None

    def test_colors1_spatial_check_not_overridden(self):
        task = TableTaskColors1()
        spec = task.get_task_spec()
        assert spec.spatial_check_fn is None

    def test_bottles1_spec(self):
        task = TableTaskBottles1()
        spec = task.get_task_spec()
        assert spec.task_name == "table_task_bottles_1"
        assert spec.implementation.create_strategy is not None
        assert spec.spatial_check_fn is not None

    def test_table_task3_spec(self):
        task = TableTask3()
        spec = task.get_task_spec()
        assert spec.task_name == "table_task_3"
        assert spec.pick_generation_strategy is not None
        assert spec.target_generation_strategy is not None

    def test_layered_cubes_spec(self):
        task = TableTaskLayeredCubes()
        spec = task.get_task_spec()
        assert spec.task_name == "table_task_layered_cubes"
        assert spec.stacking_enabled is True
        assert spec.pick_count == 18
        assert spec.target_count == 18

    def test_conveyor_sort_spec(self):
        task = TableTaskConveyorSort()
        spec = task.get_task_spec()
        assert spec.task_name == "table_task_conveyor_sort"
        assert spec.implementation.create_strategy is not None

    def test_mixed_packing_spec(self):
        task = TableTaskMixedPacking()
        spec = task.get_task_spec()
        assert spec.task_name == "table_task_mixed_packing"
        assert spec.containment_check is True
        assert spec.box_verification_info is not None
        assert spec.placement_constraints_fn is not None

    def test_conveyor_color_stacks_spec(self):
        task = TableTaskConveyorColorStacks()
        spec = task.get_task_spec()
        assert spec.task_name == "table_task_conveyor_color_stacks"
        assert spec.implementation.create_strategy is not None

    def test_base_task_spec(self):
        task = UR10MultiPickPlaceTask(task_name="base_test")
        spec = task.get_task_spec()
        assert spec.task_name == "base_test"
        assert spec.pick_generation_strategy is None
        assert spec.target_generation_strategy is None
        assert spec.spatial_check_fn is None

    def test_metadata_fields_populated(self):
        task = TableTaskColors1()
        spec = task.get_task_spec()
        assert spec.scenario is not None
        assert spec.scenario["source"] == "bin"
        assert spec.implementation.strategy_description is not None
        assert spec.implementation.strategy_description["class"] == "ColorMatchStrategy"
        assert spec.pick_description is not None
        assert spec.pick_description["asset_types"] == ["cube"]
        assert spec.rationale is not None
        assert "create_strategy" in spec.rationale


class TestGetTaskSpecStrategyFactory:
    """Test that the create_strategy callable from get_task_spec works."""

    def test_colors1_strategy_factory(self):
        from task_context_base import LightweightObj
        from asset_utils import _apply_semantic_labels

        task = TableTaskColors1()
        spec = task.get_task_spec()

        picks = []
        for i, color in enumerate(["red", "green", "blue"]):
            obj = LightweightObj(name=f"pick_{i}")
            _apply_semantic_labels(obj, type_label="cube", obj_name=f"pick_{i}", color_labels=[color])
            picks.append(obj)

        targets = []
        for i, color in enumerate(["red", "green", "blue", "yellow"]):
            obj = LightweightObj(name=f"target_{i}")
            _apply_semantic_labels(obj, type_label="cube", obj_name=f"target_{i}", color_labels=[color])
            targets.append(obj)

        strategy = spec.implementation.create_strategy(picks, targets)
        from multi_pick_strategy import ColorMatchStrategy
        assert isinstance(strategy, ColorMatchStrategy)

    def test_bottles1_strategy_factory(self):
        from task_context_base import LightweightObj

        task = TableTaskBottles1()
        spec = task.get_task_spec()

        picks = [LightweightObj(name=f"bottle_{i}") for i in range(3)]
        targets = [LightweightObj(name=f"pad_{i}") for i in range(4)]

        strategy = spec.implementation.create_strategy(picks, targets)
        from multi_pick_strategy import BottlePickStrategy
        assert isinstance(strategy, BottlePickStrategy)

    def test_strategy_factory_does_not_mutate_task(self):
        from task_context_base import LightweightObj

        task = TableTaskBottles1()
        original_picks = list(task._pick_objs)
        original_targets = list(task._target_objs)
        spec = task.get_task_spec()

        picks = [LightweightObj(name=f"test_{i}") for i in range(2)]
        targets = [LightweightObj(name=f"tgt_{i}") for i in range(3)]
        spec.implementation.create_strategy(picks, targets)

        assert task._pick_objs == original_picks
        assert task._target_objs == original_targets


class TestGetTaskSpecToDict:
    """Test to_dict()/from_dict() on specs synthesized from real tasks."""

    def test_colors1_round_trip(self):
        task = TableTaskColors1()
        spec = task.get_task_spec()
        d = spec.to_dict()
        restored = TaskSpec.from_dict(d)

        assert restored.task_name == spec.task_name
        assert restored.task_description == spec.task_description
        assert restored.pick_count == spec.pick_count
        assert restored.target_count == spec.target_count
        assert restored.seed == spec.seed
        assert restored.containment_check == spec.containment_check
        assert restored.stacking_enabled == spec.stacking_enabled

    def test_layered_cubes_round_trip(self):
        task = TableTaskLayeredCubes()
        spec = task.get_task_spec()
        d = spec.to_dict()
        restored = TaskSpec.from_dict(d)

        assert restored.task_name == "table_task_layered_cubes"
        assert restored.stacking_enabled is True
        assert restored.pick_count == 18
        assert restored.target_count == 18

    def test_mixed_packing_round_trip(self):
        task = TableTaskMixedPacking()
        spec = task.get_task_spec()
        d = spec.to_dict()
        restored = TaskSpec.from_dict(d)

        assert restored.task_name == "table_task_mixed_packing"

    def test_to_json_valid(self):
        """Verify to_json produces valid JSON for all representative tasks."""
        tasks = [
            TableTaskColors1(),
            TableTaskBottles1(),
            TableTask3(),
            TableTaskLayeredCubes(),
            TableTaskConveyorSort(),
            TableTaskMixedPacking(),
            TableTaskConveyorColorStacks(),
        ]
        for task in tasks:
            spec = task.get_task_spec()
            j = spec.to_json()
            parsed = json.loads(j)
            assert parsed["task_name"] == spec.task_name, f"Failed for {spec.task_name}"
