"""Core infrastructure for running real task configurations through the py_trees BT without Isaac Sim.

Provides:
- setup_mock_modules(): Install mock/stub modules so task classes can be imported.
- extract_task_config(): Instantiate a task class and extract objects, geometry, labels, strategy info.
- create_mock_context(): Build a MockTaskContext from extracted config.
- run_mock_task(): End-to-end runner: extract config → build context → tick BT → report results.
"""
import logging
import os
import sys
import time
from types import ModuleType
from typing import Iterable, Optional, Set
from unittest.mock import MagicMock

import numpy as np

# env_config_values is Isaac-Sim-free (per project docs) so it's safe to import
# at module top level even before setup_mock_modules() has installed the
# extsMock shim.
from env_config_values import (
    CONVEYOR_SURFACE_CENTER,
    CONVEYOR_END_Y,
    CONVEYOR_SURFACE_TOP_Z,
    CONVEYOR_SURFACE_X_HALF_EXTENT,
    CONVEYOR_SURFACE_Y_HALF_EXTENT,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock-mode timing
# ---------------------------------------------------------------------------
# Coarse approximation of how many BT ticks correspond to one simulated second
# in mock mode.  Used to convert seconds-based knobs (incremental scheduler
# intervals, conveyor speed, --min-cycle-time) into per-tick deltas.  Mock
# mode does not need fine temporal resolution, so 10 Hz is plenty and keeps
# end-to-end runs cheap.  The same cadence is published to the BT via
# ``context.simulation_time`` so any time-aware behaviour reads one clock.
MOCK_TICK_HZ: int = 10
MOCK_TICK_DT_S: float = 1.0 / MOCK_TICK_HZ

# ---------------------------------------------------------------------------
# Mock conveyor motion
# ---------------------------------------------------------------------------

# Z tolerance for the "on belt" geometric test.  Items rest with their bottom
# on the belt surface so their centre Z is one half-height above
# CONVEYOR_SURFACE_TOP_Z (=0).  5 cm covers everything from 1 cm thin
# rectangles up to ~10 cm tall cans / bottles laid on the belt while still
# excluding picks resting in the bin (BIN_Z_COORD ≈ 0.2 m).
_ON_BELT_Z_TOL = 0.05

# When an item drifts past CONVEYOR_END_Y it is teleported in Z to this value
# (relative to CONVEYOR_SURFACE_TOP_Z) so reachability filters
# (TARGET_MIN_REACHABLE_Z = top - 0.05) treat it as fallen-off and is_on_belt()
# rejects it on subsequent advance() calls.  0.5 m below the belt is well
# beyond any tolerance.
_FALLEN_Z_DROP = 0.5


class MockConveyor:
    """Mock-mode conveyor that drifts on-belt items in -Y each tick.

    Mirrors what PhysX SurfaceVelocityAPI does in real Isaac Sim
    (table_setup.py:181-187), but driven by direct position updates instead
    of physics — same approach as MockTaskContextWithPlaceUpdate uses for
    placement teleporting.

    Items are identified as "on the belt" geometrically: Z near
    ``CONVEYOR_SURFACE_TOP_Z`` and X/Y within the belt's surface bounds.  This
    avoids per-task configuration and works for both pick-on-belt
    (e.g. TableTaskConveyorTypeSort) and target-on-belt
    (e.g. TableTaskBottlesToConveyor2) tasks.  Items not on the belt
    (in the bin, on the cart, lifted by the EE) are skipped naturally.
    """

    def __init__(self, speed: float) -> None:
        self.speed: float = float(speed)  # m/s, signed (negative = -Y)
        self._top_z: float = float(CONVEYOR_SURFACE_TOP_Z)
        self._end_y: float = float(CONVEYOR_END_Y)
        self._start_y: float = self._end_y + 2.0 * float(CONVEYOR_SURFACE_Y_HALF_EXTENT)
        center_x = float(CONVEYOR_SURFACE_CENTER[0])
        self._x_min: float = center_x - float(CONVEYOR_SURFACE_X_HALF_EXTENT)
        self._x_max: float = center_x + float(CONVEYOR_SURFACE_X_HALF_EXTENT)

    def is_on_belt(self, pos) -> bool:
        """True iff the position is close to the belt surface and within X/Y bounds."""
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        if abs(z - self._top_z) > _ON_BELT_Z_TOL:
            return False
        if y < self._end_y or y > self._start_y:
            return False
        return self._x_min <= x <= self._x_max

    def advance(self, items, dt: float, *, skip_names: Iterable[str] = (),
                ride_with: Optional[dict] = None) -> None:
        """Drift each on-belt item by ``speed * dt`` in Y.

        Items whose new Y crosses below ``CONVEYOR_END_Y`` are also instantly
        dropped in Z to a value well below the belt so reachability filters
        (``make_z_reachability_check`` / ``TARGET_MIN_REACHABLE_Z``) treat
        them as fallen-off.  No animated fall — one teleport, then the item
        is skipped on subsequent calls because ``is_on_belt`` returns False.

        Args:
            items: Iterable of objects exposing ``name``, ``get_world_pose()``
                and ``set_position()`` (LightweightObj from task_context_base).
            dt: Mock-time step in seconds.
            skip_names: Names of items that should not drift (typically the
                currently-grasped pick — its mock position is stale until
                ``mark_pick_complete`` snaps it to the target).
            ride_with: Optional ``{rider_name: carrier_obj}`` map.  Each
                rider whose carrier is on the belt is drifted along with
                the carrier — used so a placed pick (resting on top of a
                drifting pad) tracks its pad each tick the same way it
                would in real sim via friction.  Riders are advanced even
                if their own Z would put them above the belt's geometric
                tolerance (e.g. a ~5 cm-tall can sitting on a 1 cm pad).
                When a rider's new Y crosses below the belt edge, the
                rider is also dropped by ``_FALLEN_Z_DROP`` so it stays
                aligned with its now-fallen carrier rather than hovering
                above the conveyor sink.
        """
        delta_y = self.speed * float(dt)
        skip = set(skip_names) if skip_names else set()
        ride_with = ride_with or {}
        for obj in items:
            if obj.name in skip:
                continue
            carrier = ride_with.get(obj.name)
            on_belt_self = self.is_on_belt(obj.get_world_pose()[0])
            on_belt_carrier = (
                carrier is not None
                and self.is_on_belt(carrier.get_world_pose()[0])
            )
            if not (on_belt_self or on_belt_carrier):
                continue
            pos, _ = obj.get_world_pose()
            new_pos = pos.copy()
            new_pos[1] += delta_y
            # Drop in Z when this item's new Y crosses the belt edge.
            # Applies to items resting on the belt directly (on_belt_self
            # — e.g. an unpicked item being carried on the belt, or a
            # target pad) and to riders whose carrier is on the belt and
            # is about to cross with them this tick (e.g. a placed pick
            # sitting on top of a moving target pad).  For belt-resting
            # items, snap Z to ``_top_z - _FALLEN_Z_DROP``.  For elevated
            # riders (z above the belt by the carrier-plus-rider stack
            # height) drop Z by the same amount the carrier will drop,
            # so the rider stays aligned with its now-fallen carrier.
            if new_pos[1] < self._end_y:
                if on_belt_self:
                    new_pos[2] = self._top_z - _FALLEN_Z_DROP
                elif on_belt_carrier:
                    carrier_z = float(carrier.get_world_pose()[0][2])
                    carrier_drop = carrier_z - (self._top_z - _FALLEN_Z_DROP)
                    new_pos[2] = pos[2] - carrier_drop
            obj.set_position(new_pos)


def _currently_held_names(strategy) -> Set[str]:
    """Names of items the strategy has latched as the in-flight pick.

    Used to exempt the carried item from conveyor drift.  Returns an empty
    set if the strategy doesn't expose ``committed_pick_name`` (or it's
    None) — in that case all on-belt items drift, which is the right
    behaviour pre-grasp.
    """
    name = getattr(strategy, "committed_pick_name", None)
    return {name} if name else set()


def _placed_picks_riding_targets(strategy) -> dict:
    """Return ``{placed_pick_name: target_obj}`` for picks resting on a target.

    In real sim a placed item rides along with a moving target via friction;
    in mock we have to teleport it each tick so the verification check sees
    the pick still on its pad.  Pairings come from
    ``strategy.pairings_by_pick_name`` and only completed picks count.
    """
    completed = getattr(strategy, "completed_picks", None) or getattr(
        strategy, "_completed_picks", set()
    )
    pairings = getattr(strategy, "pairings_by_pick_name", None)
    targets_by_name = getattr(strategy, "target_objs_by_name", None)
    if not completed or pairings is None or not targets_by_name:
        return {}
    out = {}
    for name in completed:
        tgt_name = pairings.get(name)
        if tgt_name is None:
            continue
        tgt_obj = targets_by_name.get(tgt_name)
        if tgt_obj is not None:
            out[name] = tgt_obj
    return out


class MockPrimFactory:
    """``PrimFactory`` that materialises items as ``LightweightObj`` mocks.

    Mirrors :class:`IsaacPrimFactory`: owns its own pick/target name
    counters, so naming is independent of any external list contents.
    The counters are seeded from the post-initial-batch scene-object
    counts in :func:`prepare_mock_from_spec` — virtual targets do NOT
    advance the scene-target counter, which keeps mock names aligned
    with real-sim names.
    """

    def __init__(self, *, prim_geometry, obj_asset_info,
                 pick_name_seq: int = 0, target_name_seq: int = 0):
        self._prim_geometry = prim_geometry
        self._obj_asset_info = obj_asset_info
        self._pick_name_seq = pick_name_seq
        self._target_name_seq = target_name_seq

    @property
    def pick_name_seq(self) -> int:
        return self._pick_name_seq

    @property
    def target_name_seq(self) -> int:
        return self._target_name_seq

    def create_picks(self, items, scene=None) -> list:
        objs = _create_objs_from_items(
            items, "pick", self._prim_geometry, self._obj_asset_info,
            start_index=self._pick_name_seq, prefix_from_asset_type=True,
        )
        self._pick_name_seq += len(items)
        return objs

    def create_targets(self, items, scene=None) -> list:
        objs = _create_objs_from_items(
            items, "target", self._prim_geometry, self._obj_asset_info,
            start_index=self._target_name_seq,
        )
        self._target_name_seq += len(items)
        return objs

# ---------------------------------------------------------------------------
# Module mocking (adapted from verify_tasks.py)
# ---------------------------------------------------------------------------


def setup_mock_modules():
    """Install mock/stub sys.modules so task classes can be imported without Isaac Sim.

    Must be called before importing any task module.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exts_mock_path = os.path.join(repo_root, "extsMock")

    if exts_mock_path not in sys.path:
        sys.path.insert(0, exts_mock_path)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # 1. Import real modules from extsMock
    try:
        import isaacsim
        import isaacsim.core
        import isaacsim.core.utils
        import isaacsim.core.utils.stage
        import isaacsim.cortex
        import isaacsim.cortex.framework
    except ImportError:
        pass

    # Patch stage module
    if "isaacsim.core.utils.stage" in sys.modules:
        stage_mod = sys.modules["isaacsim.core.utils.stage"]
        if not hasattr(stage_mod, "get_stage_units"):
            stage_mod.get_stage_units = MagicMock(return_value=1.0)
        if not hasattr(stage_mod, "add_reference_to_stage"):
            stage_mod.add_reference_to_stage = MagicMock()
        if not hasattr(stage_mod, "traverse_stage"):
            stage_mod.traverse_stage = MagicMock()

    try:
        import isaacsim.core.utils.rotations
    except ImportError:
        pass

    # Import real pxr mock (Phase 2.1) — not MagicMock
    try:
        import pxr  # noqa: F401 — from extsMock/pxr/__init__.py
    except ImportError:
        pass

    # 2. Custom mock classes for Isaac Sim API
    class MockBaseTask:
        def __init__(self, name=None, offset=None):
            self._name = name
            self._offset = offset
            self._task_objects = {}
        def set_up_scene(self, scene): pass
        def pre_step(self, time_step_index, simulation_time): pass
        def cleanup(self): pass
        def set_params(self, *args, **kwargs): pass
        def get_params(self): return {}
        def post_reset(self): pass

    class MockBaseController:
        def __init__(self, name=None):
            self._name = name
        def forward(self, *args, **kwargs): return MagicMock()
        def reset(self): pass

    class MockRMPFlowController(MockBaseController):
        def __init__(self, name=None, robot_articulation=None, physics_dt=None):
            super().__init__(name)

    class _Dummy:
        def __init__(self, *a, **kw):
            pass

    def get_custom_mock(name):
        m = MagicMock()
        if name == "isaacsim.core.api.tasks":
            m.BaseTask = MockBaseTask
        elif name == "isaacsim.core.api.controllers.base_controller":
            m.BaseController = MockBaseController
        elif name == "isaacsim.robot.manipulators.examples.universal_robots.controllers.rmpflow_controller":
            m.RMPFlowController = MockRMPFlowController
        elif name == "isaacsim.core.api.objects":
            for cls_name in ["DynamicCuboid", "FixedCuboid", "VisualCuboid",
                             "DynamicCylinder", "DynamicSphere", "DynamicCapsule", "DynamicCone"]:
                setattr(m, cls_name, _Dummy)
        elif name == "isaacsim.core.prims":
            m.SingleRigidPrim = _Dummy
            m.SingleXFormPrim = _Dummy
        elif name == "isaacsim.core.api.scenes.scene":
            m.Scene = _Dummy
        return m

    # 3. Pre-mock modules not in extsMock
    MOCKED_MISSING = [
        "isaacsim.core.api",
        "isaacsim.core.api.objects",
        "isaacsim.core.api.scenes",
        "isaacsim.core.api.scenes.scene",
        "isaacsim.core.api.tasks",
        "isaacsim.core.api.controllers",
        "isaacsim.core.api.controllers.base_controller",
        "isaacsim.robot",
        "isaacsim.robot.manipulators",
        "isaacsim.robot.manipulators.examples",
        "isaacsim.robot.manipulators.examples.universal_robots",
        "isaacsim.robot.manipulators.examples.universal_robots.controllers",
        "isaacsim.robot.manipulators.examples.universal_robots.controllers.rmpflow_controller",
        "isaacsim.robot.manipulators.grippers",
        "isaacsim.robot.manipulators.grippers.gripper",
        "isaacsim.robot.manipulators.grippers.surface_gripper",
        "isaacsim.robot.manipulators.grippers.parallel_gripper",
        "isaacsim.cortex.framework.cortex_utils",
        "isaacsim.core.utils.bounds",
        "isaacsim.core.utils.prims",
        "isaacsim.core.utils.extensions",
        "isaacsim.core.utils.viewports",
        "isaacsim.core.utils.collisions",
        "isaacsim.core.prims",
        "isaacsim.robot.manipulators.examples.universal_robots.controllers.pick_place_controller",
        "isaacsim.robot.manipulators.examples.universal_robots.controllers.stacking_controller",
        "isaacsim.storage",
        "isaacsim.storage.native",
        "isaacsim.core.api.objects.ground_plane",
    ]

    for m in MOCKED_MISSING:
        if m in sys.modules:
            continue
        mock_mod = get_custom_mock(m)
        mock_mod.__path__ = []
        sys.modules[m] = mock_mod

        # Attach to parent
        if "." in m:
            parent_name, child_name = m.rsplit(".", 1)
            if parent_name in sys.modules:
                setattr(sys.modules[parent_name], child_name, mock_mod)

        # Ensure intermediate parents exist
        parts = m.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                sys.modules[parent] = MagicMock()
                sys.modules[parent].__path__ = []
            if i > 1:
                grandparent = ".".join(parts[: i - 1])
                if grandparent in sys.modules:
                    setattr(sys.modules[grandparent], parts[i - 1], sys.modules[parent])

    # 4. Patch return values
    sys.modules["isaacsim.cortex.framework.cortex_utils"].get_assets_root_path_or_die = MagicMock(
        return_value="/tmp/assets"
    )

    # Ensure utils.prims has needed functions
    utils_prims = sys.modules.get("isaacsim.core.utils.prims")
    if utils_prims is not None:
        utils_prims.is_prim_path_valid = lambda p: False
        utils_prims.create_prim = lambda *a, **k: None

    # Ensure string module has find_unique_string_name
    try:
        from isaacsim.core.utils.string import find_unique_string_name  # noqa: F401
    except (ImportError, AttributeError):
        pass
    string_mod = sys.modules.get("isaacsim.core.utils.string")
    if string_mod is not None and not callable(getattr(string_mod, "find_unique_string_name", None)):
        string_mod.find_unique_string_name = lambda initial_name, is_unique_fn: initial_name


# ---------------------------------------------------------------------------
# Mock AABB infrastructure for task verification
# ---------------------------------------------------------------------------

# Module-level AABB registry: {prim_path: np.ndarray[6]}
_mock_aabb_registry = {}


def register_mock_aabb(prim_path, aabb):
    """Register an AABB for a prim path."""
    _mock_aabb_registry[prim_path] = np.array(aabb, dtype=float)


def compute_mock_aabb_from_geometry(position, geom):
    """Synthesize a 6-value AABB from position + PrimGeometry.local_half_extents."""
    hx, hy, hz = geom.local_half_extents
    return np.array([
        position[0] - hx, position[1] - hy, position[2] - hz,
        position[0] + hx, position[1] + hy, position[2] + hz,
    ])


def update_mock_aabb_registry(pick_objs, target_objs, prim_geometry, obj_asset_info=None):
    """(Re)compute AABBs for all objects using current positions + PrimGeometry.

    When obj_asset_info is provided, geometry is recomputed using the object's
    current orientation (from get_world_pose) so that post-placement AABBs
    reflect the placed orientation rather than the spawn orientation.
    """
    from asset_data_utils import lookup_prim_geometry as _lookup_prim_geometry
    _mock_aabb_registry.clear()
    for obj in list(pick_objs) + list(target_objs):
        pos, orient = obj.get_world_pose()
        geom = None
        if obj_asset_info is not None and obj.name in obj_asset_info:
            asset_type, scale = obj_asset_info[obj.name]
            geom = _lookup_prim_geometry(asset_type, obj_scale=scale, orientation=orient)
        if geom is None:
            geom = prim_geometry.get(obj.name)
        if geom is not None:
            aabb = compute_mock_aabb_from_geometry(pos, geom)
            register_mock_aabb(obj.prim_path, aabb)
            # Also update _local_half_extents so that get_corrected_aabb
            # (which short-circuits via this attribute for LightweightObj)
            # reflects the current orientation rather than stale spawn extents.
            if hasattr(obj, '_local_half_extents') and obj._local_half_extents is not None:
                obj._local_half_extents = geom.local_half_extents.copy()


def mock_compute_aabb(bb_cache, prim_path="", include_children=False):
    """Drop-in replacement for isaacsim.core.utils.bounds.compute_aabb."""
    if prim_path in _mock_aabb_registry:
        return _mock_aabb_registry[prim_path].copy()
    return np.zeros(6)


def mock_create_bbox_cache(*args, **kwargs):
    """Drop-in replacement for create_bbox_cache."""
    return None  # PlacementChecker passes this to mock_compute_aabb which ignores it




def _create_mock_verifier(config, context):
    """Create a PlacementChecker configured for mock verification.

    Returns (verifier, monkeypatch_cleanup) where monkeypatch_cleanup is a
    callable that restores the original compute_aabb/create_bbox_cache.
    The caller must call monkeypatch_cleanup() when done with the verifier.
    """
    import task_verification

    pick_objs = config["pick_objs"]
    prim_geometry = config["prim_geometry"]
    strategy = context.strategy
    target_objs = strategy.target_objs

    # Update AABBs from current object poses (post-placement), using current orientation
    obj_asset_info = config.get("obj_asset_info")
    update_mock_aabb_registry(pick_objs, target_objs, prim_geometry, obj_asset_info=obj_asset_info)

    # Check for box containment verification (e.g. TableTaskColorShapes)
    box_info = config.get("box_verification_info")

    # Monkeypatch compute_aabb and create_bbox_cache in task_verification module
    orig_compute = task_verification.compute_aabb
    orig_create = task_verification.create_bbox_cache
    task_verification.compute_aabb = mock_compute_aabb
    task_verification.create_bbox_cache = mock_create_bbox_cache

    def cleanup():
        task_verification.compute_aabb = orig_compute
        task_verification.create_bbox_cache = orig_create

    # Use task-level placement constraints if provided, else strategy default
    from task_verification import make_index_based_strategy_adapters
    valid_targets_fn, default_placement_fn = make_index_based_strategy_adapters(
        strategy, pick_objs, target_objs,
    )
    placement_constraints_fn = config.get("placement_constraints_fn") or default_placement_fn

    if box_info is not None:
        # Box containment verification via shared utility
        from task_verification import build_box_verification_hooks
        box_specs = box_info["box_specs"]
        box_targets, spatial_fn, valid_fn = build_box_verification_hooks(
            box_specs, pick_objs,
            is_pick_expected=strategy.is_pick_expected,
        )
        verifier = task_verification.PlacementChecker(
            pick_objs=pick_objs,
            target_objs=box_targets,
            spatial_check_fn=spatial_fn,
            valid_targets_fn=valid_fn,
            placement_constraints_fn=placement_constraints_fn,
            bb_cache_factory=mock_create_bbox_cache,
            containment_mode=True,
        )
    else:
        # Standard 1:1 marker verification
        # Prefer strategy's custom spatial check when available
        strategy_check_fn = strategy.get_spatial_check_fn()
        if strategy_check_fn is not None:
            spatial_check_fn = strategy_check_fn
        else:
            spatial_check_fn = config.get("spatial_check_fn")
            if spatial_check_fn is None:
                spatial_check_fn = task_verification.is_on_top

        verifier = task_verification.PlacementChecker(
            pick_objs=pick_objs,
            target_objs=target_objs,
            spatial_check_fn=spatial_check_fn,
            valid_targets_fn=valid_targets_fn,
            placement_constraints_fn=placement_constraints_fn,
            bb_cache_factory=mock_create_bbox_cache,
            containment_mode=config.get("containment_check", False),
        )

    return verifier, cleanup


def verify_mock_task(config, context):
    """Run task verification using real PlacementChecker with mock AABBs.

    Returns (success: bool, failures: list[str]) matching
    UR10MultiPickPlaceTask.check_groundtruth_task_success() signature.
    """
    verifier, cleanup = _create_mock_verifier(config, context)
    try:
        result = verifier.verify()
    finally:
        cleanup()

    if not result.success:
        logger.info(result.summary())

    return (result.success, result.failures)


def verify_mock_task_incremental(config, context, pick_names):
    """Run incremental verification for specific pick names.

    Updates AABB registry, creates verifier, and checks only the given picks.
    Logs per-item results: warning for failures, debug for successes.

    Returns (success: bool, checks: list, failures: list).
    """
    verifier, cleanup = _create_mock_verifier(config, context)
    try:
        name_to_idx = {obj.name: i for i, obj in enumerate(config["pick_objs"])}
        pick_indices = [name_to_idx[n] for n in pick_names if n in name_to_idx]
        result = verifier.verify(pick_indices=pick_indices)
    finally:
        cleanup()

    for check in result.checks:
        if check.passed:
            logger.debug(f"Incremental check OK: '{check.pick_name}' -> '{check.target_name}'")
        else:
            logger.warning(f"Incremental check FAIL: '{check.pick_name}': {check.detail}")

    return (result.success, result.checks, result.failures)


# ---------------------------------------------------------------------------
# Config extraction
# ---------------------------------------------------------------------------


def _create_objs_from_items(items, prefix, prim_geometry, obj_asset_info,
                            start_index=0, prefix_from_asset_type=False):
    """Create LightweightObj list from ItemSpec list, populating geometry and asset info dicts.

    Shared helper used by prepare_mock_from_spec() for picks, scene targets,
    and virtual targets.

    Args:
        start_index: Starting index for auto-generated names (avoids
            collisions when adding incremental batches).
        prefix_from_asset_type: When True, unnamed items default to
            ``f"{asset_type}_{i}"``, matching the IsaacSim-side naming in
            ``SimulationConfigurator.add_source_objects()``. Used for picks
            so mock and real runs report the same object names.
    """
    from task_context_base import create_lightweight_objs_from_items

    objs = create_lightweight_objs_from_items(
        items, prefix=prefix, prim_geometry_out=prim_geometry,
        start_index=start_index,
        prefix_from_asset_type=prefix_from_asset_type,
    )
    # Populate obj_asset_info for AABB recomputation during verification
    for obj, item in zip(objs, items):
        obj_asset_info[obj.name] = (item.asset_type, item.scale)
    return objs


def prepare_mock_from_spec(task_spec, task_class_name="Unknown"):
    """Generate mock objects and strategy from a TaskSpec without Isaac Sim.

    Creates LightweightObj instances from the spec's generation strategies,
    applies semantic labels, caches precomputed geometry, then creates and
    initializes the pairing strategy.

    When virtual_target_generation_strategy is set, generates virtual
    targets and appends them to the scene targets before strategy creation.

    Returns a metadata dict compatible with create_mock_context() and
    verify_mock_task().
    """
    from multi_pick_strategy import MultiPickStrategy

    pick_gen = task_spec.pick_generation_strategy
    target_gen = task_spec.target_generation_strategy

    pick_count = task_spec.pick_count
    target_count = task_spec.target_count
    seed = task_spec.seed

    # --- Create lightweight objects, labels, geometry ---
    prim_geometry = {}
    obj_asset_info = {}

    # Incremental pick generation: create scheduler, only use initial batch.
    # Spatial-trigger schedulers fire naturally in mock when conveyor_speed
    # is non-zero (MockConveyor drifts on-belt items each tick); when the
    # belt is stationary, replenishment is still suppressed and the BT
    # completes on the initial batch.
    belt_moving = task_spec.conveyor_speed not in (None, 0.0)
    inc_config = task_spec.pick_incremental_config
    spatial_config = task_spec.pick_spatial_trigger_config
    pick_scheduler = None
    pick_scheduler_is_spatial = False
    if spatial_config is not None and pick_gen is not None:
        from item_generation import SpatialTriggeredItemScheduler
        pick_scheduler = SpatialTriggeredItemScheduler(
            primary_generator=pick_gen,
            config=spatial_config,
            count_range=pick_count,
            seed=seed,
        )
        pick_items = pick_scheduler.get_initial_batch()
        pick_scheduler_is_spatial = True
        logger.info(
            "Spatial-trigger generation (mock): initial batch %d/%d "
            "(replenishment %s)",
            len(pick_items), pick_scheduler.total_count,
            "active — belt moving" if belt_moving
            else "suppressed — belt stationary",
        )
    elif inc_config is not None and pick_gen is not None:
        from item_generation import IncrementalItemScheduler
        pick_scheduler = IncrementalItemScheduler(
            generator=pick_gen,
            config=inc_config,
            count_range=pick_count,
            seed=seed,
        )
        pick_items = pick_scheduler.get_initial_batch()
        logger.info(
            "Incremental generation (mock): initial batch %d/%d",
            len(pick_items), pick_scheduler.total_count,
        )
    else:
        pick_items = pick_gen.generate(count_range=pick_count, seed=seed) if pick_gen else []

    # Incremental target generation: create scheduler, only use initial batch
    tgt_inc_config = task_spec.target_incremental_config
    tgt_spatial_config = task_spec.target_spatial_trigger_config
    target_scheduler = None
    target_scheduler_is_spatial = False
    if tgt_spatial_config is not None and target_gen is not None:
        from item_generation import SpatialTriggeredItemScheduler
        target_scheduler = SpatialTriggeredItemScheduler(
            primary_generator=target_gen,
            config=tgt_spatial_config,
            count_range=target_count,
            seed=seed,
        )
        target_items = target_scheduler.get_initial_batch()
        target_scheduler_is_spatial = True
        logger.info(
            "Spatial-trigger target generation (mock): initial batch %d/%d "
            "(replenishment %s)",
            len(target_items), target_scheduler.total_count,
            "active — belt moving" if belt_moving
            else "suppressed — belt stationary",
        )
    elif tgt_inc_config is not None and target_gen is not None:
        from item_generation import IncrementalItemScheduler
        target_scheduler = IncrementalItemScheduler(
            generator=target_gen,
            config=tgt_inc_config,
            count_range=target_count,
            seed=seed,
        )
        target_items = target_scheduler.get_initial_batch()
        logger.info(
            "Incremental target generation (mock): initial batch %d/%d",
            len(target_items), target_scheduler.total_count,
        )
    else:
        target_items = target_gen.generate(count_range=target_count, seed=seed) if target_gen else []

    pick_objs = _create_objs_from_items(
        pick_items, "pick", prim_geometry, obj_asset_info,
        prefix_from_asset_type=True,
    )
    target_objs = _create_objs_from_items(target_items, "target", prim_geometry, obj_asset_info)

    # --- Virtual target generation ---
    impl_spec = task_spec.implementation
    virtual_gen = impl_spec.virtual_target_generation_strategy if impl_spec is not None else None
    virtual_items = []
    if virtual_gen is not None:
        if hasattr(virtual_gen, 'generate'):
            virtual_items = virtual_gen.generate(
                count_range=task_spec.target_count,
                seed=seed,
            )
        elif callable(virtual_gen):
            virtual_items = virtual_gen(pick_objs, target_objs)
        else:
            logger.warning("virtual_target_generation_strategy is neither a generator nor callable; skipping")

    if virtual_items:
        virtual_objs = _create_objs_from_items(
            virtual_items, "virtual_target", prim_geometry, obj_asset_info,
        )
        combined_targets = target_objs + virtual_objs
        logger.info(f"Generated {len(virtual_objs)} virtual target objects (mock)")
    else:
        combined_targets = target_objs

    # --- Create strategy via TaskSpec factory ---
    create_strategy = impl_spec.create_strategy if impl_spec is not None else None
    if create_strategy is not None:
        strategy = create_strategy(pick_objs, combined_targets)
    else:
        strategy = MultiPickStrategy(pick_objs=pick_objs, target_objs=combined_targets)
    strategy.initialize_pairings()

    # Signal incremental generation to strategy and BT.  Spatial-trigger
    # schedulers need conveyor motion to drive replenishment; if the belt
    # is stationary we still suppress more_*_expected so the BT can
    # complete on the initial batch rather than wait forever.
    suppress_spatial = not belt_moving
    if (pick_scheduler is not None
            and not pick_scheduler.all_released
            and not (pick_scheduler_is_spatial and suppress_spatial)):
        strategy.more_items_expected = True
    if (target_scheduler is not None
            and not target_scheduler.all_released
            and not (target_scheduler_is_spatial and suppress_spatial)):
        strategy.more_targets_expected = True

    spawner = None
    if pick_scheduler is not None or target_scheduler is not None:
        from item_spawner import ItemSpawner
        spawner = ItemSpawner(
            pick_scheduler=pick_scheduler,
            target_scheduler=target_scheduler,
            prim_factory=MockPrimFactory(
                prim_geometry=prim_geometry,
                obj_asset_info=obj_asset_info,
                pick_name_seq=len(pick_objs),
                # Seed from scene targets only — virtual targets share the
                # combined list but must NOT advance the scene-target name
                # sequence (the real-sim factory increments only on scene
                # spawn).  Keeps mock names aligned with the real run.
                target_name_seq=len(target_objs),
            ),
            scene=None,
            conveyor_speed_fn=lambda: task_spec.conveyor_speed,
        )

    return {
        "task_name": task_spec.task_name,
        "task_description": task_spec.task_description or "",
        "task_class_name": task_class_name,
        "seed": seed,
        "pick_items": pick_items,
        "target_items": target_items + virtual_items,
        "pick_objs": pick_objs,
        "target_objs": combined_targets,
        "prim_geometry": prim_geometry,
        "strategy": strategy,
        "obj_asset_info": obj_asset_info,
        "box_verification_info": task_spec.box_verification_info,
        "spatial_check_fn": task_spec.spatial_check_fn,
        "placement_constraints_fn": task_spec.placement_constraints_fn,
        "containment_check": task_spec.containment_check,
        "stacking_enabled": task_spec.stacking_enabled,
        "tree_factory": impl_spec.tree_factory if impl_spec is not None else None,
        "pick_scheduler": pick_scheduler,
        "target_scheduler": target_scheduler,
        "spawner": spawner,
        "conveyor_speed": task_spec.conveyor_speed,
        "task_spec": task_spec,
    }


def extract_task_config(task_class, seed=None, pick_count=None, target_count=None,
                        randomize=None, **kwargs):
    """Instantiate a task class and extract its configuration for mock execution.

    Uses the task's TaskSpec to generate mock objects and strategy without
    requiring Isaac Sim.

    Returns a dict with: task_name, task_description, task_class_name,
    pick_items, target_items, pick_objs, target_objs, prim_geometry,
    strategy, obj_asset_info, box_verification_info, spatial_check_fn,
    stacking_enabled.
    """
    from isaacsim.core.utils.semantics import clear_all_labels
    clear_all_labels()

    # Build kwargs for task __init__
    task_kwargs = {}
    if seed is not None:
        task_kwargs["seed"] = seed
    if pick_count is not None:
        task_kwargs["pick_count"] = pick_count
    if target_count is not None:
        task_kwargs["target_count"] = target_count
    if randomize is not None:
        task_kwargs["randomize"] = randomize
    task_kwargs.update(kwargs)

    task = task_class(**task_kwargs)
    spec = task.get_task_spec()

    # CLI counts/seed always override the task's defaults
    if pick_count is not None:
        spec.pick_count = pick_count
    if target_count is not None:
        spec.target_count = target_count
    if seed is not None:
        spec.seed = seed

    metadata = prepare_mock_from_spec(spec, task_class_name=type(task).__name__)

    # Populate the configurator so the task's ``_pick_objs`` / ``_target_objs``
    # forwarders (which read from the configurator during set_up_scene)
    # return the mock lists.  Closures referencing ``self._pick_objs``
    # see them through the same forwarders.
    task._configurator._pick_objs = metadata["pick_objs"]
    task._configurator._target_objs = metadata["target_objs"]
    task._configurator._prim_geometry = metadata["prim_geometry"]

    return metadata


# ---------------------------------------------------------------------------
# Mock context with place-update behaviour
# ---------------------------------------------------------------------------


class MockTaskContextWithPlaceUpdate:
    """Extends MockTaskContext to update object positions after each pick is placed."""

    def __new__(cls, *args, **kwargs):
        # Deferred import to avoid circular imports at module level
        from task_context_mock import MockTaskContext
        # Dynamically create a subclass of MockTaskContext
        if not hasattr(cls, "_real_class"):
            cls._real_class = type(
                "MockTaskContextWithPlaceUpdate",
                (MockTaskContext,),
                {
                    "mark_pick_complete": cls._mark_pick_complete_impl,
                },
            )
        return cls._real_class(*args, **kwargs)

    @staticmethod
    def _mark_pick_complete_impl(self, pick_name):
        """Override mark_pick_complete to update object pose after placing."""
        # Get placing info before marking complete
        target_name = self.get_placing_target_name(pick_name)
        drop_orient = self.get_end_effector_orientation_for_drop(pick_name, target_name)
        target_name, place_pos, place_orient = self.get_placing_info(pick_name, drop_orient)

        # Call base implementation
        from task_context_mock import MockTaskContext
        MockTaskContext.mark_pick_complete(self, pick_name)

        # Update mock object position to reflect placement
        if place_pos is not None:
            pick_obj = self._strategy.pick_objs_by_name.get(pick_name)
            if pick_obj is not None:
                pick_obj.set_position(place_pos)
                if place_orient is not None:
                    pick_obj.set_orientation(place_orient)
                logger.debug(f"  Placed {pick_name} at {place_pos}")


def create_mock_context(config, strategy=None):
    """Build a MockTaskContext from extracted config.

    Uses MockTaskContextWithPlaceUpdate so object positions are updated after placement.
    """
    from robot_controllers.mock_robot import MockGripper
    from task_context_mock import MockTaskContext

    if strategy is None:
        strategy = config.get("strategy")

    pick_objs = config["pick_objs"]
    target_objs = config["target_objs"]

    mock_gripper = MockGripper()

    # Build names and positions dicts for MockTaskContext
    pick_names = [o.name for o in pick_objs]
    target_names = [o.name for o in target_objs]
    pick_positions = {o.name: o._position for o in pick_objs}
    target_positions = {o.name: o._position for o in target_objs}

    # We pass the pre-built strategy so MockTaskContext doesn't create its own default.
    # ``impl_spec`` lets MockTaskContext detect ``use_curobo=True`` and wire a
    # NullArmMotionDriver so cuRobo BT setup() finds an arm_motion_driver.
    task_spec_obj = config.get("task_spec")
    impl_spec = task_spec_obj.implementation if task_spec_obj is not None else None
    context = MockTaskContextWithPlaceUpdate(
        pick_names=pick_names,
        target_names=target_names,
        pick_positions=pick_positions,
        target_positions=target_positions,
        prim_geometry=config["prim_geometry"],
        gripper=mock_gripper,
        strategy=strategy,
        impl_spec=impl_spec,
    )
    return context


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


def run_mock_task(task_class, seed=None, render=False, max_ticks=20000,
                  pick_count=None, target_count=None, randomize=None,
                  verbose=True, show_status=False, tick_interval:float=-1.0,
                  incremental_checks=True, min_cycle_time_s: float = 0.0,
                  **kwargs):
    """Run a task through the mock py_trees BT end-to-end.

    Args:
        min_cycle_time_s: Minimum simulated seconds between pick-place
            cycles (sets ``context.min_cycle_time_s``).  0 disables the
            gate.  See ``WaitForCycleTime`` in pt_task_behaviours.

    Returns the context for inspection.
    """
    import py_trees
    from robot_controllers.pt_task_tree import make_task_controller_tree as default_tree_factory
    from isaacsim.core.utils.semantics import clear_all_labels

    # 0. Enable DEBUG logging when showing status (matches run_mock_pickplace_task.py)
    if show_status:
        py_trees.logging.level = py_trees.logging.Level.DEBUG
        if tick_interval < 0:
            tick_interval = 0.05  # default to 20 ticks/sec

    # 1. Clean state
    clear_all_labels()

    # 2. Extract config
    config = extract_task_config(
        task_class, seed=seed, pick_count=pick_count,
        target_count=target_count, randomize=randomize, **kwargs,
    )

    pick_scheduler = config.get("pick_scheduler")
    target_scheduler = config.get("target_scheduler")

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Task: {config['task_class_name']}")
        if config["task_description"]:
            print(f"  {config['task_description'][:100]}")
        print(f"  Strategy: {config['strategy'].__class__.__name__}")
        print(f"  Pick objects: {len(config['pick_objs'])}")
        if pick_scheduler is not None:
            print(f"    (incremental: {pick_scheduler.released_count}/{pick_scheduler.total_count} initial)")
        print(f"  Target objects: {len(config['target_objs'])}")
        if target_scheduler is not None:
            print(f"    (incremental: {target_scheduler.released_count}/{target_scheduler.total_count} initial)")
        print(f"  Geometry entries: {len(config['prim_geometry'])}")
        print(f"{'=' * 60}")

    # 3. Create context (strategy already created by prepare_mock_from_spec)
    strategy = config["strategy"]
    context = create_mock_context(config, strategy=strategy)
    context.spawner = config.get("spawner")
    # Cycle-time gate (consumed by WaitForCycleTime).  Setting on the
    # context rather than the BT keeps the tree factories untouched and
    # lets the gate flip on/off purely from CLI input.
    context.min_cycle_time_s = float(min_cycle_time_s or 0.0)

    # 4. Build BT — use task's tree_factory if available
    tree_factory = config.get("tree_factory") or default_tree_factory
    root = tree_factory(fake_fast=True)
    tree = py_trees.trees.BehaviourTree(root=root)

    if render:
        py_trees.display.render_dot_tree(root)
        print("Rendered dot tree to file.")
        return context, None

    # 5. Setup
    tree.setup(
        timeout=15,
        context=context,
        arm_commander=context.arm_commander,
        gripper_commander=context.gripper_commander,
    )

    # 6. Tick loop
    if verbose:
        print(f"\nRunning BT (max {max_ticks} ticks)...")

    start_time = time.time()
    final_tick = 0
    prev_completed = set()
    incremental_failures = []
    mock_time = 0.0
    mock_dt = MOCK_TICK_DT_S  # see MOCK_TICK_HZ at module top
    # Conveyor motion: construct once if the task has a non-zero belt speed.
    # MockConveyor.advance() runs after each tree.tick() so on-belt items
    # drift in -Y at the same cadence used for scheduler ticks.
    conveyor_speed = config.get("conveyor_speed")
    conveyor = (
        MockConveyor(conveyor_speed)
        if conveyor_speed not in (None, 0.0) else None
    )
    # True when no scheduler is configured (BT can start immediately).
    spawner = context.spawner
    bt_started = spawner is None or not (
        spawner.has_pick_scheduler or spawner.has_target_scheduler
    )
    for i in range(1, max_ticks + 1):
        mock_time += mock_dt
        # Publish current simulated time to the context so WaitForCycleTime
        # (and any other time-aware behaviour) reads the same clock the
        # schedulers use.  Cadence = MOCK_TICK_HZ (see module constants).
        context.simulation_time = mock_time

        if spawner is not None:
            result = spawner.tick(
                mock_time,
                live_picks=config["pick_objs"],
                live_targets=config["target_objs"],
            )
            if result.new_picks:
                strategy.add_incremental_picks(result.new_picks)
            if result.new_targets:
                strategy.add_incremental_targets(result.new_targets)
            if (result.all_picks_released and spawner.pick_scheduler is not None
                    and strategy.more_items_expected):
                strategy.more_items_expected = False
                if verbose:
                    print(f"  [tick {i}] Incremental generation complete: "
                          f"all {spawner.pick_scheduler.total_count} picks spawned")
            if (result.all_targets_released and spawner.target_scheduler is not None
                    and strategy.more_targets_expected):
                strategy.more_targets_expected = False
                if verbose:
                    print(f"  [tick {i}] Incremental target generation complete: "
                          f"all {spawner.target_scheduler.total_count} targets spawned")

        # BT-start gate: see UR10MultiPickPlaceTask.task_step for the
        # invariant about not defaulting missing schedulers to "ready".
        if not bt_started:
            if spawner.bt_should_start(mock_time):
                bt_started = True
                if verbose:
                    print(f"  [tick {i}] BT start gate opened "
                          f"(picks={len(config['pick_objs'])}, targets={len(config['target_objs'])})")
            else:
                continue  # skip BT tick

        tree.tick()
        # Advance mock arm commander so cortex-style behaviors can detect arrival
        if hasattr(context.arm_commander, 'tick'):
            context.arm_commander.tick()
        final_tick = i

        # Incremental verification after each newly completed pick.  Runs
        # BEFORE conveyor.advance so a just-placed item is still aligned
        # with its (currently-drifting) target pad — otherwise the next
        # tick's drift would offset the pad by ``speed * dt`` and trip
        # is_within on tight targets.
        if incremental_checks:
            current_completed = set(strategy.completed_picks)
            new_picks = current_completed - prev_completed
            if new_picks:
                ok, checks, fails = verify_mock_task_incremental(
                    config, context, list(new_picks),
                )
                if not ok:
                    incremental_failures.extend(fails)
                prev_completed = current_completed

        # Drift on-belt items in -Y so spatial-trigger predicates fire and
        # tasks like TableTaskSoupCans2 see items reach the fall-off edge.
        # The currently-grasped pick (if any) is exempted so its stale
        # mock position does not phantom-drift while it's "being carried".
        # Placed picks "ride" their target pad so the per-tick spatial
        # check still sees them aligned.
        if conveyor is not None:
            held = _currently_held_names(strategy)
            ride_with = _placed_picks_riding_targets(strategy)
            conveyor.advance(
                config["pick_objs"], mock_dt,
                skip_names=held, ride_with=ride_with,
            )
            conveyor.advance(config["target_objs"], mock_dt, skip_names=held)

        if show_status:
            print(f"\n--------- Tick {i} ---------\n")
            print(py_trees.display.unicode_tree(root, show_status=True))
        if tick_interval > 0:
            time.sleep(tick_interval)

        if tree.root.status != py_trees.common.Status.RUNNING:
            break

    elapsed = time.time() - start_time

    # 7. Report results
    if verbose:
        print(f"\n{'─' * 60}")
        print(f"Completed in {final_tick} ticks ({elapsed:.2f}s)")
        print(f"Final tree status: {tree.root.status}")
        print(f"task_finished: {context.task_finished}")
        print(f"completed_picks: {context._completed_picks}")
        print(f"all_picks_done: {context.all_picks_done}")
        print(f"targets_exhausted: {context.targets_exhausted}")

        # Show object positions
        print(f"\nPick objects ({len(config['pick_objs'])}):")
        for obj in config["pick_objs"]:
            pos, _ = obj.get_world_pose()
            is_placed = obj.name in context._completed_picks
            status = "PLACED" if is_placed else "not placed"
            print(f"  {obj.name}: pos={pos} [{status}]")

        # Show pairings
        print(f"\nPairings:")
        for pick_name, tgt_name in strategy.pairings_by_pick_name.items():
            if tgt_name is not None:
                print(f"  {pick_name} -> {tgt_name}")
            else:
                print(f"  {pick_name} -> (no target)")

        if incremental_checks and incremental_failures:
            print(f"\nIncremental check failures ({len(incremental_failures)}):")
            for msg in incremental_failures:
                print(f"  {msg}")

        print(f"{'─' * 60}\n")

    # 8. Verify task success (matching run_task.py logging)
    task_successful = None
    if context.task_finished:
        task_successful, failures = verify_mock_task(config, context)
        task_name = config["task_name"]
        seed = config.get("seed")
        if task_successful:
            logger.warning(f"Task {task_name} Completed successfully (seed: {seed}).")
        else:
            for msg in failures:
                logger.warning(msg)
            logger.warning(f"Task {task_name} Verification checks reported UNSUCCESSFUL completion (seed: {seed}).")

    return context, task_successful
