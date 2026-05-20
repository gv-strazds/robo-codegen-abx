"""Shared pytest configuration for all tests.

Ensures extsMock/ is on sys.path before site-packages so the mock
isaacsim package shadows the real one (which requires SimulationApp
initialization before any submodule imports).

Also installs the small set of stubs that extsMock genuinely lacks
(Isaac Sim object classes, prim wrappers, Scene, the `prims`/`viewports`/
`extensions` namespace modules). This is the single source of truth — when a
new asset class is added to ``asset_utils.PRIMS_MAP`` it must also be added to
``_OBJECT_CLASSES`` below. extsMock already provides real implementations of
the utility modules (bounds, stage, semantics, string), so we deliberately do
not stub those.
"""
import os
import sys
from types import ModuleType

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_mock_path = os.path.join(_repo_root, "extsMock")

if _mock_path not in sys.path:
    sys.path.insert(0, _mock_path)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# Classes imported by production code from isaacsim.core.api.objects.
# Mirror of asset_utils.py:33-42 (DynamicCuboid..DynamicCone) plus
# table_setup.py:37-44 (VisualCapsule, VisualSphere). Keep in sync.
_OBJECT_CLASSES = (
    "DynamicCuboid",
    "FixedCuboid",
    "VisualCuboid",
    "DynamicCylinder",
    "FixedCylinder",
    "DynamicSphere",
    "DynamicCapsule",
    "DynamicCone",
    "VisualCapsule",
    "VisualSphere",
)


def _install_isaacsim_stubs():
    # Preload extsMock's real isaacsim subpackages first so subsequent
    # `_get_or_create` calls don't replace them with bare ModuleType stubs
    # (which lack __path__ and break submodule imports).
    try:
        import isaacsim  # noqa: F401
        import isaacsim.core  # noqa: F401
        import isaacsim.core.utils  # noqa: F401
        import isaacsim.core.utils.bounds  # noqa: F401
        import isaacsim.core.utils.rotations  # noqa: F401
        import isaacsim.core.utils.semantics  # noqa: F401
        import isaacsim.core.utils.stage  # noqa: F401
        import isaacsim.core.utils.string  # noqa: F401
    except ImportError:
        pass

    def _get_or_create(name):
        existing = sys.modules.get(name)
        if existing is not None:
            return existing
        mod = ModuleType(name)
        sys.modules[name] = mod
        return mod

    class _Dummy:
        def __init__(self, *args, **kwargs):
            pass

    objects_mod = _get_or_create("isaacsim.core.api.objects")
    for cls_name in _OBJECT_CLASSES:
        if not hasattr(objects_mod, cls_name):
            setattr(objects_mod, cls_name, _Dummy)

    ground_plane_mod = _get_or_create("isaacsim.core.api.objects.ground_plane")
    if not hasattr(ground_plane_mod, "GroundPlane"):
        ground_plane_mod.GroundPlane = _Dummy

    prims_mod = _get_or_create("isaacsim.core.prims")
    for cls_name in ("SingleRigidPrim", "SingleXFormPrim", "RigidPrim"):
        if not hasattr(prims_mod, cls_name):
            setattr(prims_mod, cls_name, _Dummy)

    utils_prims = _get_or_create("isaacsim.core.utils.prims")
    if not hasattr(utils_prims, "is_prim_path_valid"):
        utils_prims.is_prim_path_valid = lambda path: False
    if not hasattr(utils_prims, "create_prim"):
        utils_prims.create_prim = lambda *a, **k: None

    scenes_scene = _get_or_create("isaacsim.core.api.scenes.scene")
    if not hasattr(scenes_scene, "Scene"):
        scenes_scene.Scene = _Dummy

    _get_or_create("isaacsim.core.utils.viewports")
    _get_or_create("isaacsim.core.utils.extensions")


_install_isaacsim_stubs()
