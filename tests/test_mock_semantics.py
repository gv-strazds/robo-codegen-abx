"""Tests for the mock semantic label store in extsMock."""
import importlib.util
import os
import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Install minimal stubs for modules that asset_utils imports but are not
# provided by extsMock. Must run before importing asset_utils.
# ---------------------------------------------------------------------------

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_mock_path = os.path.join(_repo_root, "extsMock")

if _mock_path not in sys.path:
    sys.path.insert(0, _mock_path)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


def _install_stubs():
    """Install minimal stubs so asset_utils can be imported."""

    class _Dummy:
        def __init__(self, *a, **kw):
            pass

    missing = [
        "isaacsim.core.api.objects",
        "isaacsim.core.api.scenes",
        "isaacsim.core.api.scenes.scene",
        "isaacsim.core.prims",
        "isaacsim.core.utils.prims",
        "isaacsim.core.utils.viewports",
        "isaacsim.core.utils.extensions",
        "pxr",
    ]

    for mod_name in missing:
        if mod_name not in sys.modules:
            m = MagicMock()
            m.__path__ = []
            sys.modules[mod_name] = m
            if "." in mod_name:
                parent, child = mod_name.rsplit(".", 1)
                if parent in sys.modules:
                    setattr(sys.modules[parent], child, m)

    obj_mod = sys.modules["isaacsim.core.api.objects"]
    for cls_name in ["DynamicCuboid", "FixedCuboid", "VisualCuboid",
                     "DynamicCylinder", "FixedCylinder", "DynamicSphere",
                     "DynamicCapsule", "DynamicCone"]:
        if not hasattr(obj_mod, cls_name) or isinstance(getattr(obj_mod, cls_name), MagicMock):
            setattr(obj_mod, cls_name, _Dummy)

    prims_mod = sys.modules["isaacsim.core.prims"]
    for cls_name in ["SingleRigidPrim", "SingleXFormPrim"]:
        if not hasattr(prims_mod, cls_name) or isinstance(getattr(prims_mod, cls_name), MagicMock):
            setattr(prims_mod, cls_name, _Dummy)

    utils_prims = sys.modules["isaacsim.core.utils.prims"]
    utils_prims.is_prim_path_valid = lambda p: False
    utils_prims.create_prim = lambda *a, **k: None

    scene_mod = sys.modules["isaacsim.core.api.scenes.scene"]
    scene_mod.Scene = _Dummy


def _load_real_mock_semantics():
    """Load the real extsMock semantics module from file and return it."""
    sem_path = os.path.join(_mock_path, "isaacsim", "core", "utils", "semantics.py")
    spec = importlib.util.spec_from_file_location("_real_mock_semantics", sem_path)
    real_sem = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(real_sem)
    return real_sem


_install_stubs()

# Load the real mock semantics module (always from extsMock source file)
_real_sem = _load_real_mock_semantics()

from task_context_mock import MockPickObj


@pytest.fixture(autouse=True)
def _clean_label_store():
    """Clear label store before and after each test."""
    _real_sem.clear_all_labels()
    yield
    _real_sem.clear_all_labels()


# Convenience aliases from the REAL mock module (not from sys.modules which may be stale)
add_labels = _real_sem.add_labels
get_labels = _real_sem.get_labels
set_labels_by_name = _real_sem.set_labels_by_name
clear_all_labels = _real_sem.clear_all_labels
remove_all_semantics = _real_sem.remove_all_semantics
remove_labels = _real_sem.remove_labels
add_update_semantics = _real_sem.add_update_semantics


class TestLabelStore:
    """Basic label store operations."""

    def test_add_and_get_labels(self):
        obj = MockPickObj("cube_0")
        add_labels(obj, labels=["red"], instance_name="color")
        result = get_labels(obj)
        assert result == {"color": ["red"]}

    def test_add_multiple_instance_names(self):
        obj = MockPickObj("cube_1")
        add_labels(obj, labels=["cube"], instance_name="type")
        add_labels(obj, labels=["red"], instance_name="color")
        add_labels(obj, labels=["cube_1"], instance_name="name")
        result = get_labels(obj)
        assert result == {"type": ["cube"], "color": ["red"], "name": ["cube_1"]}

    def test_overwrite_default(self):
        obj = MockPickObj("cube_2")
        add_labels(obj, labels=["red"], instance_name="color")
        add_labels(obj, labels=["blue"], instance_name="color")  # overwrite=True
        result = get_labels(obj)
        assert result == {"color": ["blue"]}

    def test_no_overwrite_appends(self):
        obj = MockPickObj("cube_3")
        add_labels(obj, labels=["red"], instance_name="color", overwrite=True)
        add_labels(obj, labels=["blue"], instance_name="color", overwrite=False)
        result = get_labels(obj)
        assert result["color"] == ["red", "blue"]

    def test_get_labels_unknown_prim(self):
        obj = MockPickObj("nonexistent")
        result = get_labels(obj)
        assert result == {}

    def test_set_labels_by_name(self):
        set_labels_by_name("my_obj", {"type": ["cube"], "color": ["green"]})
        obj = MockPickObj("my_obj")
        result = get_labels(obj)
        assert result == {"type": ["cube"], "color": ["green"]}

    def test_clear_all_labels(self):
        obj = MockPickObj("cube_4")
        add_labels(obj, labels=["red"], instance_name="color")
        clear_all_labels()
        assert get_labels(obj) == {}

    def test_remove_all_semantics(self):
        obj = MockPickObj("cube_5")
        add_labels(obj, labels=["cube"], instance_name="type")
        add_labels(obj, labels=["red"], instance_name="color")
        remove_all_semantics(obj)
        assert get_labels(obj) == {}

    def test_remove_labels_specific_instance(self):
        obj = MockPickObj("cube_6")
        add_labels(obj, labels=["cube"], instance_name="type")
        add_labels(obj, labels=["red"], instance_name="color")
        remove_labels(obj, instance_name="color")
        result = get_labels(obj)
        assert result == {"type": ["cube"]}

    def test_remove_labels_all(self):
        obj = MockPickObj("cube_7")
        add_labels(obj, labels=["cube"], instance_name="type")
        remove_labels(obj, instance_name=None)
        assert get_labels(obj) == {}

    def test_add_update_semantics(self):
        obj = MockPickObj("cube_8")
        add_update_semantics(obj, "red", type_label="color")
        result = get_labels(obj)
        assert result == {"color": ["red"]}


class TestAssetUtilsIntegration:
    """Test that asset_utils label functions work with the mock label store.

    These tests require the real mock semantics module to be active in
    sys.modules so that asset_utils' lazy imports resolve correctly.
    A fixture ensures this before each test.
    """

    @pytest.fixture(autouse=True)
    def _patch_semantics(self):
        """Ensure real mock semantics is active for asset_utils calls."""
        # Save current state
        old_sem = sys.modules.get("isaacsim.core.utils.semantics")
        old_asset = sys.modules.get("asset_utils")

        # Install real mock
        sys.modules["isaacsim.core.utils.semantics"] = _real_sem
        utils_mod = sys.modules.get("isaacsim.core.utils")
        if utils_mod is not None:
            setattr(utils_mod, "semantics", _real_sem)

        # Reimport asset_utils so it binds to real mock
        sys.modules.pop("asset_utils", None)
        import asset_utils as fresh_asset_utils  # noqa: F811
        self._asset_utils = fresh_asset_utils

        yield

        # Restore old state so other tests aren't affected
        if old_sem is not None:
            sys.modules["isaacsim.core.utils.semantics"] = old_sem
        if old_asset is not None:
            sys.modules["asset_utils"] = old_asset

    def test_has_color(self):
        obj = MockPickObj("pick_0")
        add_labels(obj, labels=["red"], instance_name="color")
        assert self._asset_utils.has_color(obj, "red") is True
        assert self._asset_utils.has_color(obj, "blue") is False

    def test_is_of_type(self):
        obj = MockPickObj("pick_1")
        add_labels(obj, labels=["cube"], instance_name="type")
        assert self._asset_utils.is_of_type(obj, "cube") is True
        assert self._asset_utils.is_of_type(obj, "ball") is False

    def test_get_asset_type(self):
        obj = MockPickObj("pick_2")
        add_labels(obj, labels=["cylinder"], instance_name="type")
        assert self._asset_utils.get_asset_type(obj) == "cylinder"

    def test_get_asset_type_default(self):
        obj = MockPickObj("pick_3")
        assert self._asset_utils.get_asset_type(obj, asset_type_default="cube") == "cube"

    def test_is_a_class_label(self):
        obj = MockPickObj("pick_4")
        add_labels(obj, labels=["box"], instance_name="class")
        assert self._asset_utils.is_a(obj, "box") is True
        assert self._asset_utils.is_a(obj, "sphere") is False

    def test_is_a_falls_back_to_type(self):
        obj = MockPickObj("pick_5")
        add_labels(obj, labels=["cube"], instance_name="type")
        assert self._asset_utils.is_a(obj, "cube") is True

    def test_apply_semantic_labels(self):
        obj = MockPickObj("pick_6")
        self._asset_utils._apply_semantic_labels(
            obj,
            type_label="cube",
            obj_name="pick_6",
            color_labels=["green"],
        )
        assert self._asset_utils.has_color(obj, "green") is True
        assert self._asset_utils.is_of_type(obj, "cube") is True
        labels = self._asset_utils.get_labels(obj, "name")
        assert "pick_6" in labels

    def test_has_color_case_insensitive(self):
        obj = MockPickObj("pick_7")
        add_labels(obj, labels=["Red"], instance_name="color")
        assert self._asset_utils.has_color(obj, "red") is True
        assert self._asset_utils.has_color(obj, "Red") is True
