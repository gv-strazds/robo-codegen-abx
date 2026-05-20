"""TaskSpec + TaskImplementationSpec: split task description from execution policy.

``TaskSpec`` captures WHAT the task is: scene/item generation, workspace
setup, conveyor configuration, verification semantics, and scene-side
metadata.  ``TaskImplementationSpec`` (nested under ``TaskSpec.implementation``)
captures HOW the task is executed: pairing strategy factory, BT tree
factory, RMPFlow/cortex tuning, watchdog timeouts, postures, reachability
gates, and cuRobo configuration.

The split lets the ``SimulationConfigurator`` be built from ``TaskSpec``
alone, with ``TaskController`` / ``TaskContext`` constructed later from
the combination of ``TaskSpec`` + ``TaskImplementationSpec`` + scene state.

JSON serialization: ``to_dict()`` / ``from_dict()`` round-trip for data
fields.  Callable fields are serialized as qualified name references
(e.g., ``"multi_pick_strategy.ColorMatchStrategy"``) and resolved via
``importlib`` on deserialization.  Lambdas and closures serialize as
metadata only.
"""
import dataclasses
import importlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from env_config_values import DEFAULT_CONVEYOR_SPEED  # noqa: F401 — re-export for callers

logger = logging.getLogger(__name__)


# Shared sentinel for "task did not override posture config" on
# ``TaskImplementationSpec`` fields — distinct from ``None`` which
# explicitly disables the null-space bias.  Imported from
# ``task_context_base`` so the same object identity is used both here and
# when comparing against the ``TaskContextBase.__init__`` default.
from task_context_base import POSTURE_UNSET  # noqa: E402 — keep near uses


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _callable_ref(fn) -> Optional[str]:
    """Return a qualified name reference for a callable, or None if not resolvable.

    Returns a resolvable reference only for top-level functions and classes
    (not lambdas, closures, or locally-defined classes).
    """
    if fn is None:
        return None
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname:
        return None
    # Reject lambdas, local functions, local classes
    if "<lambda>" in qualname or "<locals>" in qualname:
        return None
    return f"{module}.{qualname}"


def _resolve_callable(ref: str) -> Optional[Callable]:
    """Resolve a qualified name reference (e.g. 'module.Class') to a callable."""
    if not ref:
        return None
    parts = ref.rsplit(".", 1)
    if len(parts) != 2:
        return None
    module_path, attr_name = parts
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, attr_name, None)
    except (ImportError, AttributeError):
        return None


def _serialize_value(val) -> Any:
    """Serialize a value for JSON compatibility."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return float(val)
    if isinstance(val, np.ndarray):
        return {"__ndarray__": val.tolist()}
    if isinstance(val, tuple):
        return {"__tuple__": [_serialize_value(v) for v in val]}
    if isinstance(val, list):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _serialize_value(v) for k, v in val.items()}
    if callable(val):
        ref = _callable_ref(val)
        result = {"__callable__": ref}
        if ref is None:
            result["__repr__"] = repr(val)
        return result
    # Complex objects: store type info and repr
    return {"__type__": f"{type(val).__module__}.{type(val).__qualname__}", "__repr__": repr(val)}


def _deserialize_value(val) -> Any:
    """Deserialize a JSON-compatible value back to Python objects."""
    if val is None or isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, dict):
        if "__ndarray__" in val:
            return np.array(val["__ndarray__"])
        if "__tuple__" in val:
            return tuple(_deserialize_value(v) for v in val["__tuple__"])
        if "__callable__" in val:
            ref = val["__callable__"]
            return _resolve_callable(ref) if ref else None
        if "__type__" in val:
            return None  # Can't reconstruct complex objects from repr
        if "__repr__" in val:
            return None
        return {k: _deserialize_value(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_deserialize_value(v) for v in val]
    return val


# ---------------------------------------------------------------------------
# TaskImplementationSpec
# ---------------------------------------------------------------------------


@dataclass
class TaskImplementationSpec:
    """Execution-policy configuration for a pick-and-place task.

    Holds the knobs the controller layer reads — pairing strategy factory,
    BT tree factory, motion tuning, postures, reachability gates, cuRobo
    flags.  Created independently of the scene side so that
    ``SimulationConfigurator`` can be built from ``TaskSpec`` alone.
    """

    # Pairing strategy factory: (pick_objs, target_objs) -> MultiPickStrategy.
    # The task constructs a fresh strategy from the spawned object lists.
    create_strategy: Optional[Callable] = None

    # Virtual target generation: hidden/utility markers generated at pairing
    # time by the TaskController, not spawned as USD prims.  Uses
    # ``TaskSpec.target_count`` for the generation count.  These are policy
    # helpers (the pairing strategy's auxiliary targets), not scene objects.
    virtual_target_generation_strategy: Any = None

    # Tree factory: (fake_fast: bool) -> py_trees root node
    # If None, the default make_task_controller_tree is used.
    tree_factory: Optional[Callable] = None

    # Target reachability predicate: Callable(target_obj) -> bool.
    # When set, targets that fail are excluded from BT target selection.
    # Use env_config_values.make_z_reachability_check() for conveyor tasks.
    target_reachable_fn: Optional[Callable] = None

    # Pick reachability gate (consumed by CheckPickReachable in the cortex BT).
    # Both are optional; when None the gate uses library defaults
    # (UR10_WORKING_RADIUS for the radius, no Z-floor check).
    # pick_min_reachable_z: items below this Z are considered permanently
    # unreachable — the gate calls strategy.defer_pick() and returns FAILURE so
    # the BT skips them.  Set this for conveyor tasks where items can slide off
    # (e.g. DROPZONE_Z - 0.10).
    # pick_max_reachable_radius_xy: items beyond this radial XY distance from
    # the robot base are treated as not-yet-reachable; the gate returns
    # RUNNING so the BT idles at altitude until the item drifts into reach.
    pick_min_reachable_z: Optional[float] = None
    pick_max_reachable_radius_xy: Optional[float] = None

    # Per-task overrides for the cortex tree's CortexExecuteApproach behaviour.
    # pick_approach_p_thresh: position-tolerance for declaring the final grasp
    # approach SUCCESS (metres).  Default 0.005 m (set in
    # robot_controllers/pt_cortex_behaviours.py for moving-conveyor robustness).
    # Tighten to ~0.002 m for tasks where the gripper must close flush against
    # the item to avoid a dangling air gap (e.g. socket-insertion placements).
    pick_approach_p_thresh: Optional[float] = None

    # Per-task override for the lateral std-dev of the pick-side approach
    # funnel (metres).  Default perception_utils.DEFAULT_PICK_APPROACH_STD_DEV
    # (0.015 m, widened so RMPFlow descends cleanly against moving targets).
    pick_approach_std_dev: Optional[float] = None

    # Per-task overrides for the cortex BT's sim-time watchdog timeouts.
    # ``None`` keeps the module defaults (DEFAULT_*_TIMEOUT_S).
    move_timeout_s: Optional[float] = None
    approach_timeout_s: Optional[float] = None
    insert_timeout_s: Optional[float] = None

    # Execution tuning
    ee_height_for_move: Optional[float] = None

    # Per-task overrides for the cortex-tree place-side hover/approach geometry.
    # ``None`` (default) keeps the module defaults (DEFAULT_PLACE_HOVER_ABOVE_Z
    # and DEFAULT_PLACE_APPROACH_DISTANCE in perception_utils).  TaskContextBase
    # clamps ``place_hover_above_z >= place_approach_distance``.
    place_hover_above_z: Optional[float] = None
    place_approach_distance: Optional[float] = None

    # Per-task null-space posture overrides threaded into TaskContextBase.
    # ``POSTURE_UNSET`` (default) leaves the module-level
    # ``PICK_POSTURE_CONFIG`` / ``PLACE_POSTURE_CONFIG`` in effect.
    # ``None`` explicitly disables the posture bias for that phase.
    pick_posture_config: Any = POSTURE_UNSET
    place_posture_config: Any = POSTURE_UNSET

    # cuRobo motion-planner integration. ``use_curobo=True`` swaps the
    # arm-command path from Cortex MotionCommander to a CuroboMotionGenDriver.
    # The selected ``tree_factory`` should be
    # ``make_curobo_task_controller_tree`` — the cortex tree's ApproachParams
    # have no cuRobo equivalent.
    use_curobo: bool = False
    # Filename of the cuRobo robot config YAML (relative to cuRobo's
    # configs/robot path). ``ur10e.yml`` ships with cuRobo.
    curobo_robot_yaml: str = "ur10e.yml"
    # Optional ``(impl_spec, configurator) -> list[curobo.geom.types.Cuboid]``
    # hook to add task-specific obstacles to the cuRobo collision world.
    curobo_obstacles_fn: Optional[Callable] = None

    # Sim-time seconds to wait after the BT-start gate opens before the
    # behaviour tree begins ticking.  Lets gravity-spawned items settle so
    # teleport-mode placements don't carry residual momentum.  ``None``
    # means no delay.
    startup_delay_seconds: Optional[float] = None

    # Per-task override of ``AssetMetaData.grasp_offset_local`` for one
    # or more asset types in this task.  Keys are asset_type strings
    # (matching ``AssetMetaData.asset_type``); values are 3-vectors in
    # the asset's native (pre-spawn-rotation, unscaled) local frame.
    # Applied by ``SimulationConfigurator`` when populating the geometry
    # cache for each pick: when a pick's asset_type is in this dict the
    # override replaces the asset-default offset (scale is applied
    # afterwards, matching the asset-default path).  ``None`` or a
    # missing key falls through to the asset default.
    grasp_offset_local_overrides: Optional[Dict[str, np.ndarray]] = None

    # Implementation-side metadata (documentation, not executable config)
    strategy_description: Optional[dict] = None
    # e.g. {"class": "ColorMatchStrategy", "pairing": "color_match",
    #        "details": "color_palette=['red','green','blue']"}
    rationale: Optional[Dict[str, str]] = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "create_strategy": _serialize_value(self.create_strategy),
            "virtual_target_generation_strategy": _serialize_value(self.virtual_target_generation_strategy),
            "tree_factory": _serialize_value(self.tree_factory),
            "target_reachable_fn": _serialize_value(self.target_reachable_fn),
            "pick_min_reachable_z": self.pick_min_reachable_z,
            "pick_max_reachable_radius_xy": self.pick_max_reachable_radius_xy,
            "pick_approach_p_thresh": self.pick_approach_p_thresh,
            "pick_approach_std_dev": self.pick_approach_std_dev,
            "move_timeout_s": self.move_timeout_s,
            "approach_timeout_s": self.approach_timeout_s,
            "insert_timeout_s": self.insert_timeout_s,
            "ee_height_for_move": self.ee_height_for_move,
            "place_hover_above_z": self.place_hover_above_z,
            "place_approach_distance": self.place_approach_distance,
            "pick_posture_config": _serialize_value(self.pick_posture_config),
            "place_posture_config": _serialize_value(self.place_posture_config),
            "use_curobo": self.use_curobo,
            "curobo_robot_yaml": self.curobo_robot_yaml,
            "curobo_obstacles_fn": _serialize_value(self.curobo_obstacles_fn),
            "startup_delay_seconds": self.startup_delay_seconds,
            "grasp_offset_local_overrides": (
                {k: _serialize_value(v) for k, v in self.grasp_offset_local_overrides.items()}
                if self.grasp_offset_local_overrides is not None else None
            ),
            "strategy_description": self.strategy_description,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskImplementationSpec":
        """Reconstruct from a serialized dict."""
        return cls(
            create_strategy=_deserialize_value(d.get("create_strategy")),
            virtual_target_generation_strategy=_deserialize_value(
                d.get("virtual_target_generation_strategy")
            ),
            tree_factory=_deserialize_value(d.get("tree_factory")),
            target_reachable_fn=_deserialize_value(d.get("target_reachable_fn")),
            pick_min_reachable_z=d.get("pick_min_reachable_z"),
            pick_max_reachable_radius_xy=d.get("pick_max_reachable_radius_xy"),
            pick_approach_p_thresh=d.get("pick_approach_p_thresh"),
            pick_approach_std_dev=d.get("pick_approach_std_dev"),
            move_timeout_s=d.get("move_timeout_s"),
            approach_timeout_s=d.get("approach_timeout_s"),
            insert_timeout_s=d.get("insert_timeout_s"),
            ee_height_for_move=d.get("ee_height_for_move"),
            place_hover_above_z=d.get("place_hover_above_z"),
            place_approach_distance=d.get("place_approach_distance"),
            pick_posture_config=_deserialize_value(d.get("pick_posture_config", POSTURE_UNSET))
                if "pick_posture_config" in d else POSTURE_UNSET,
            place_posture_config=_deserialize_value(d.get("place_posture_config", POSTURE_UNSET))
                if "place_posture_config" in d else POSTURE_UNSET,
            use_curobo=d.get("use_curobo", False),
            curobo_robot_yaml=d.get("curobo_robot_yaml", "ur10e.yml"),
            curobo_obstacles_fn=_deserialize_value(d.get("curobo_obstacles_fn")),
            startup_delay_seconds=d.get("startup_delay_seconds"),
            grasp_offset_local_overrides=(
                {k: _deserialize_value(v)
                 for k, v in d["grasp_offset_local_overrides"].items()}
                if d.get("grasp_offset_local_overrides") is not None else None
            ),
            strategy_description=d.get("strategy_description"),
            rationale=d.get("rationale"),
        )


# ---------------------------------------------------------------------------
# TaskSpec
# ---------------------------------------------------------------------------


@dataclass
class TaskSpec:
    """Abstract task description for a pick-and-place task.

    Captures the scene-side of the task: item generation, workspace setup,
    conveyor configuration, verification semantics, and human-readable
    metadata.  Execution policy lives on the nested ``implementation``
    field (a :class:`TaskImplementationSpec`).

    A ``SimulationConfigurator`` can be built from this object alone — it
    never touches the ``implementation`` field.  ``TaskController`` /
    ``TaskContext`` consume both halves after the configurator has run.
    """

    task_name: str
    task_description: str

    # Item generation
    pick_generation_strategy: Any = None
    target_generation_strategy: Any = None
    pick_count: Optional[Union[int, tuple]] = None
    target_count: Optional[Union[int, tuple]] = None
    seed: Optional[int] = None

    # Workspace setup: (scene, assets_root_path) -> None
    setup_workspace: Optional[Callable] = None

    # Whether pick objects are pre-stacked at source — scene fact that the
    # strategy reads when pairing.
    stacking_enabled: bool = False

    # Conveyor surface velocity (m/s). None → stationary (0.0). Assign
    # DEFAULT_CONVEYOR_SPEED (imported from env_config_values or re-exported
    # from task_spec) to request the library-default moving-conveyor speed.
    conveyor_speed: Optional[float] = None

    # Conveyor fall-off verification snapshot. When enabled, target objects
    # that approach the -Y edge of the conveyor are checked for placement
    # correctness before physics can disrupt the spatial relationship.
    # conveyor_falloff_enabled: None = auto-on whenever conveyor_speed is
    # truthy; True/False explicitly overrides.
    conveyor_falloff_enabled: Optional[bool] = None
    conveyor_falloff_snapshot_margin: float = 0.0
    conveyor_falloff_hide_after: bool = True  # set_visibility(False) past edge
    conveyor_end_y: Optional[float] = None  # override; else env_config_values.CONVEYOR_END_Y

    # Verification semantics — what counts as a successful placement
    spatial_check_fn: Optional[Callable] = None
    placement_constraints_fn: Optional[Callable] = None
    containment_check: bool = False
    box_verification_info: Optional[dict] = None

    # Incremental generation (optional)
    pick_incremental_config: Optional[Any] = None  # IncrementalGenerationConfig
    target_incremental_config: Optional[Any] = None  # IncrementalGenerationConfig

    # Spatial-trigger generation (optional, mutually exclusive per side with
    # the time-based ``*_incremental_config``).
    pick_spatial_trigger_config: Optional[Any] = None    # SpatialTriggerConfig
    target_spatial_trigger_config: Optional[Any] = None  # SpatialTriggerConfig

    # Human-readable metadata (documentation, not executable config)
    scenario: Optional[Dict[str, str]] = None
    # e.g. {"source": "bin", "destination": "dropzone_grid", "workspace": "two_tables"}
    pick_description: Optional[dict] = None
    target_description: Optional[dict] = None
    verification_description: Optional[dict] = None
    # e.g. {"spatial_check": "is_within + is_vertical",
    #        "placement_constraints": "is_vertical (for USD asset types)"}

    # Rationale / justification strings (scene-side field name -> reason)
    rationale: Optional[Dict[str, str]] = None

    # Execution policy (HOW). ``None`` means "use library defaults".
    implementation: Optional[TaskImplementationSpec] = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def impl(self) -> TaskImplementationSpec:
        """Return ``self.implementation`` or a fresh default.

        Caller code can read impl fields without null-guarding the
        ``implementation`` attribute.  Never caches the default onto
        ``self`` — a description-only spec stays description-only.
        """
        return self.implementation if self.implementation is not None else TaskImplementationSpec()

    def with_impl(self, **kw) -> "TaskSpec":
        """Return a copy of this spec with implementation fields overridden.

        Builds a fresh ``TaskImplementationSpec`` when no implementation is
        currently set, or copies the existing one with the given overrides.
        """
        impl = self.implementation if self.implementation is not None else TaskImplementationSpec()
        return dataclasses.replace(self, implementation=dataclasses.replace(impl, **kw))

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialize to a JSON-compatible dict.

        Data fields round-trip perfectly.  Callable fields are serialized as
        qualified name references when possible (top-level functions/classes),
        or as metadata (repr) for lambdas/closures.  Complex objects like
        ItemGenerator are stored as type + repr.  The nested
        ``implementation`` is serialized under the ``"implementation"`` key.
        """
        return {
            "task_name": self.task_name,
            "task_description": self.task_description,
            "pick_count": _serialize_value(self.pick_count),
            "target_count": _serialize_value(self.target_count),
            "seed": self.seed,
            "pick_generation_strategy": _serialize_value(self.pick_generation_strategy),
            "target_generation_strategy": _serialize_value(self.target_generation_strategy),
            "setup_workspace": _serialize_value(self.setup_workspace),
            "stacking_enabled": self.stacking_enabled,
            "conveyor_speed": self.conveyor_speed,
            "conveyor_falloff_enabled": self.conveyor_falloff_enabled,
            "conveyor_falloff_snapshot_margin": self.conveyor_falloff_snapshot_margin,
            "conveyor_falloff_hide_after": self.conveyor_falloff_hide_after,
            "conveyor_end_y": self.conveyor_end_y,
            "spatial_check_fn": _serialize_value(self.spatial_check_fn),
            "placement_constraints_fn": _serialize_value(self.placement_constraints_fn),
            "containment_check": self.containment_check,
            "box_verification_info": _serialize_value(self.box_verification_info),
            "pick_incremental_config": _serialize_value(self.pick_incremental_config),
            "target_incremental_config": _serialize_value(self.target_incremental_config),
            "pick_spatial_trigger_config": _serialize_value(self.pick_spatial_trigger_config),
            "target_spatial_trigger_config": _serialize_value(self.target_spatial_trigger_config),
            "scenario": self.scenario,
            "pick_description": self.pick_description,
            "target_description": _serialize_value(self.target_description),
            "verification_description": self.verification_description,
            "rationale": self.rationale,
            "implementation": (
                self.implementation.to_dict() if self.implementation is not None else None
            ),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskSpec":
        """Reconstruct a TaskSpec from a serialized dict."""
        impl_dict = d.get("implementation")
        return cls(
            task_name=d.get("task_name", ""),
            task_description=d.get("task_description", ""),
            pick_count=_deserialize_value(d.get("pick_count")),
            target_count=_deserialize_value(d.get("target_count")),
            seed=d.get("seed"),
            pick_generation_strategy=_deserialize_value(d.get("pick_generation_strategy")),
            target_generation_strategy=_deserialize_value(d.get("target_generation_strategy")),
            setup_workspace=_deserialize_value(d.get("setup_workspace")),
            stacking_enabled=d.get("stacking_enabled", False),
            conveyor_speed=d.get("conveyor_speed"),
            conveyor_falloff_enabled=d.get("conveyor_falloff_enabled"),
            conveyor_falloff_snapshot_margin=d.get("conveyor_falloff_snapshot_margin", 0.0),
            conveyor_falloff_hide_after=d.get("conveyor_falloff_hide_after", True),
            conveyor_end_y=d.get("conveyor_end_y"),
            spatial_check_fn=_deserialize_value(d.get("spatial_check_fn")),
            placement_constraints_fn=_deserialize_value(d.get("placement_constraints_fn")),
            containment_check=d.get("containment_check", False),
            box_verification_info=_deserialize_value(d.get("box_verification_info")),
            pick_incremental_config=_deserialize_value(d.get("pick_incremental_config")),
            target_incremental_config=_deserialize_value(d.get("target_incremental_config")),
            pick_spatial_trigger_config=_deserialize_value(d.get("pick_spatial_trigger_config")),
            target_spatial_trigger_config=_deserialize_value(d.get("target_spatial_trigger_config")),
            scenario=d.get("scenario"),
            pick_description=d.get("pick_description"),
            target_description=_deserialize_value(d.get("target_description")),
            verification_description=d.get("verification_description"),
            rationale=d.get("rationale"),
            implementation=(
                TaskImplementationSpec.from_dict(impl_dict) if impl_dict is not None else None
            ),
        )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    # ------------------------------------------------------------------
    # Derived accessors
    # ------------------------------------------------------------------

    def falloff_is_enabled(self) -> bool:
        """Whether conveyor fall-off snapshot verification should run for this task.

        Auto-on whenever a non-zero conveyor_speed is configured; explicit
        conveyor_falloff_enabled True/False overrides the auto-decision.
        """
        if self.conveyor_falloff_enabled is not None:
            return bool(self.conveyor_falloff_enabled)
        return bool(self.conveyor_speed)
