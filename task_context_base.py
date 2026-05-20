"""TaskContextBase: shared base class for TaskContext and MockTaskContext.

Contains hardware methods (robot, gripper, arm/gripper commanders)
and thin delegation to MultiPickStrategy for pairing, pick iteration,
placing info, and completion tracking.

Subclasses are responsible for constructing the objects (robot, gripper, etc.)
and passing them along with a strategy to this base __init__.
"""
import json
import logging
import os
import time
from typing import List, Optional, Tuple

import numpy as np

from isaacsim.cortex.framework.motion_commander import MotionCommand, PosePq, ApproachParams
import isaacsim.cortex.framework.math_util as math_util

from multi_pick_strategy import MultiPickStrategy
import perception_utils

logger = logging.getLogger(__name__)


# Sentinel for "caller did not supply a per-instance posture override"
# (distinct from "caller supplied None", which explicitly disables the
# null-space posture bias).  Used by TaskContextBase.__init__ kwargs.
POSTURE_UNSET = object()


# Default approach-funnel parameters moved to ``perception_utils`` as
# ``DEFAULT_PICK_APPROACH_*`` / ``DEFAULT_PLACE_APPROACH_*`` module
# constants.  The cortex-style command builders read per-call values
# off the cached ``GraspPose`` / ``PlacePose`` rather than using a
# module constant here; overriding per-task flows through
# ``perception_utils.compute_grasp_pose`` kwargs or a future
# per-strategy hook.


# -----------------------------------------------------------------------------
# Null-space posture config defaults
# -----------------------------------------------------------------------------
# Single home for pick and place posture configs consumed by the cortex-style
# command builders via ``get_posture_config(phase)``.  Both admit a JSON-encoded
# env-var override so ``utility_scripts/optimize_posture.py`` can inject
# candidates without editing source.  Per-instance overrides (supplied through
# ``TaskContextBase.__init__``) take precedence over these module defaults.

# Pick posture — matches the upstream Cortex bin_stacking reference
# (exts/.../bin_stacking_behavior.py), previously duplicated as
# ``CortexMoveToPick.POSTURE_CONFIG``.
PICK_POSTURE_CONFIG = np.array(
    [-1.2654234, -2.9708025, -2.219733, 0.6445836, 1.5186214, 0.30098662]
)
# Place posture — hand-tuned for TableTaskBottlesToConveyor2; kept here so
# per-task overrides can live on ``TaskSpec`` / ``TaskContextBase`` rather than
# the tree factory.
PLACE_POSTURE_CONFIG = np.array(
    [-1.503686, -2.273453, -1.591299, -1.651287, -2.205517, 3.042854]
)


def _load_posture_override(env_var: str) -> Optional[np.ndarray]:
    raw = os.environ.get(env_var)
    if not raw:
        return None
    try:
        parsed = np.array(json.loads(raw))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Ignoring invalid %s: %s", env_var, exc)
        return None
    logger.info("%s overridden from env: %s", env_var, parsed)
    return parsed


_pick_env_override = _load_posture_override("PICK_POSTURE_CONFIG_OVERRIDE")
if _pick_env_override is not None:
    PICK_POSTURE_CONFIG = _pick_env_override

_place_env_override = _load_posture_override("PLACE_POSTURE_CONFIG_OVERRIDE")
if _place_env_override is not None:
    PLACE_POSTURE_CONFIG = _place_env_override


# Default sim-time watchdog timeouts (seconds) for the cortex BT's
# CortexMove* behaviours, applied as SimTimeout decorators in
# pt_cortex_tree.make_cortex_task_controller_tree.  Per-task overrides flow
# through TaskSpec.{move,approach,insert}_timeout_s and are returned by the
# corresponding TaskContextBase.get_*_timeout_s() methods below.
#
# Values mirror the previous behaviour-internal constants in
# pt_cortex_behaviours.CortexMove (15 s) and CortexExecuteApproach (8 s) so
# the refactor preserves the prior semantics by default.  Tighten via
# TaskSpec for tasks where a faster bail-out helps recovery (e.g. dense
# multi-pick scenarios); loosen for tasks with intentionally slow paths.
DEFAULT_MOVE_TIMEOUT_S = 15.0
DEFAULT_APPROACH_TIMEOUT_S = 8.0
DEFAULT_INSERT_TIMEOUT_S = 15.0


# How far ahead (seconds) to aim along the target's measured XY drift when
# composing the place command.  At the default conveyor speed (0.015 m/s)
# this yields a 2.25 mm lead; at 2× belt speed, 4.5 mm.  Stationary targets
# produce zero drift and therefore no lead.  The lead is recomputed every
# BT tick and added to the live target XY, so commanded XY tracks the
# target throughout the descent — the residual on landing is approximately
# (LEAD - EE_controller_lag).  Lowered from the historical 0.5 s after
# _estimate_target_drift_xy was switched to sim-time finite-differences
# (previously the wall-clock estimator under-reported drift in GUI mode
# where sim < real-time, masking the over-aggressive constant).
# Interim value pending direct measurement of EE-vs-commanded XY lag.
PLACE_LEAD_HORIZON_S = 0.15

# Same idea on the pick side, applied in ``refresh_grasp_pose_position`` so
# the cached grasp pose's ``ee_position`` aims slightly ahead of the live
# item position when the item is drifting (e.g. riding a conveyor).  Without
# this, ``CortexExecuteApproach`` chases a target that keeps moving in Y, the
# tight -Z approach funnel keeps re-centering, and the steady-state Y/Z error
# is large enough to time out the 2 mm position threshold even when the EE
# is visually on top of the item.  Stationary picks produce zero drift and
# therefore no lead.  See ``PLACE_LEAD_HORIZON_S`` for why this dropped
# from 0.5 s to 0.15 s.
PICK_LEAD_HORIZON_S = 0.15

# Hard cap on the XY lead magnitude (metres) applied to the pick target.
# Realistic conveyor speeds (~15 mm/s default, up to ~30 mm/s) yield
# leads of 7-15 mm, comfortably below this cap.  The cap exists as a
# defence-in-depth: if drift estimation ever produces an implausibly
# large velocity (e.g. a held-and-then-launched item, or a finite-
# difference glitch), the lead is clipped rather than slewing the EE
# target metres away from the actual pick.
PICK_LEAD_MAX_MAGNITUDE_M = 0.05


# Quaternion helpers live in ``perception_utils``.  Keep thin aliases
# here so the private names used in-file still resolve.
_quaternions_equivalent = perception_utils._quaternions_equivalent
_rotate_pick_offset_to_drop = perception_utils._rotate_offset_by_rel_quat


class LightweightObj:
    """Lightweight object with name and pose, usable without Isaac Sim.

    Provides the same pose interface as Isaac Sim prim objects so it can be
    used as a drop-in substitute in mock/test workflows.
    """

    def __init__(self, name, position=None, orientation=None):
        self.name = name
        self.prim_path = f"/World/{name}"
        self._position = position if position is not None else np.array([0.5, 0.0, 0.05])
        self._orientation = orientation if orientation is not None else np.array([1.0, 0.0, 0.0, 0.0])
        # In-memory semantic labels — used instead of USD API since
        # LightweightObj has no real USD prim.  Keyed by instance_name
        # (e.g. "type", "color", "name") → list of label strings.
        self._semantic_labels = {}
        # World-aligned half extents for AABB computation without USD queries.
        # Set by create_lightweight_objs_from_items() from PrimGeometry.
        self._local_half_extents = None

    def get_local_pose(self):
        return self._position.copy(), self._orientation.copy()

    def get_world_pose(self):
        return self._position.copy(), self._orientation.copy()

    def set_position(self, position):
        self._position = position.copy()

    def set_orientation(self, orientation):
        self._orientation = orientation.copy()

    def set_world_pose(self, position=None, orientation=None):
        if position is not None:
            self._position = position.copy()
        if orientation is not None:
            self._orientation = orientation.copy()

    def get_local_scale(self):
        return np.array([1.0, 1.0, 1.0])


def _safe_local_scale(obj) -> Optional[np.ndarray]:
    """Return the object's local scale as a 3-vector, or None on failure.

    Used by ``_grasp_offset_world`` to scale a per-task grasp-offset
    override to the spawned instance.  Defensive: minimal test mocks
    may not implement ``get_local_scale``; in that case we fall through
    to "no scaling" rather than crash the pose lookup.
    """
    getter = getattr(obj, "get_local_scale", None)
    if getter is None:
        return None
    try:
        scale = getter()
    except Exception:
        return None
    if scale is None:
        return None
    arr = np.asarray(scale, dtype=float).reshape(-1)
    if arr.shape[0] != 3:
        return None
    return arr


def create_lightweight_objs_from_items(items, prefix="virtual_target", prim_geometry_out=None,
                                      start_index=0, prefix_from_asset_type=False):
    """Create LightweightObj instances from ItemSpec list with semantic labels and geometry.

    For each ItemSpec, creates a LightweightObj with the item's pose, stores
    semantic labels directly on the object (avoiding USD API calls that would
    fail on non-prim objects), and looks up precomputed geometry.

    Args:
        items: List of ItemSpec instances.
        prefix: Name prefix for items without an explicit name.
        prim_geometry_out: Optional dict to populate with {name: PrimGeometry}.
        start_index: Starting index for auto-generated names (default 0).
            Use a non-zero value when adding incremental batches to avoid
            name collisions with previously created objects.
        prefix_from_asset_type: If True, items without an explicit name use
            ``f"{item.asset_type}_{i}"`` (mirroring the IsaacSim path in
            ``SimulationConfigurator.add_source_objects()``) instead of the
            fixed ``prefix``. Falls back to ``prefix`` when the item has no
            ``asset_type``. Explicit ``item.name`` always wins.

    Returns:
        List of LightweightObj instances.
    """
    from asset_data_utils import lookup_prim_geometry

    objs = []
    for i, item in enumerate(items):
        if item.name:
            name = item.name
        elif prefix_from_asset_type and item.asset_type:
            name = f"{item.asset_type}_{start_index + i}"
        else:
            name = f"{prefix}_{start_index + i}"
        obj = LightweightObj(
            name=name,
            position=item.position,
            orientation=item.orientation if item.orientation is not None else np.array([1.0, 0.0, 0.0, 0.0]),
        )
        # Store semantic labels directly on the LightweightObj (no USD API).
        # _get_semantic_labels() in asset_utils checks _semantic_labels first.
        if item.asset_type:
            obj._semantic_labels["type"] = [item.asset_type]
        if name:
            obj._semantic_labels["name"] = [name]
        if isinstance(item.color, str):
            obj._semantic_labels["color"] = [item.color]
        geom = lookup_prim_geometry(item.asset_type, obj_scale=item.scale, orientation=item.orientation)
        if geom is not None:
            obj._local_half_extents = geom.local_half_extents.copy()
            if prim_geometry_out is not None:
                prim_geometry_out[name] = geom
        objs.append(obj)
    return objs


class TaskContextBase:
    """Base class holding all shared state and behaviour for TaskContext variants.

    Args:
        robot: Robot articulation object (real or mock).
        strategy: MultiPickStrategy instance owning pairing and pick iteration.
        gripper: Gripper object; if None, falls back to robot.gripper.
        ee_height_for_move: Height (in scene units) used for horizontal moves.
        arm_commander: IArmCommander for end-effector motion (Cortex-aligned).
        gripper_commander: IGripperCommander for gripper control (Cortex-aligned).
    """

    # Fallback EE offset (2cm Z-lift) when no PrimGeometry is cached for the
    # pick.  Should almost never be hit in normal operation — get_end_effector_offset
    # logs a warning when it is.  Subclasses may override if they need a different
    # fallback, but prefer ensuring prim_geometry is populated instead.
    _EE_OFFSET_FALLBACK = np.array([0.0, 0.0, 0.02])

    def __init__(
        self,
        robot,
        strategy: MultiPickStrategy,
        gripper=None,
        ee_height_for_move: float = 0.3,
        teleport_mode: bool = False,
        prim_geometry: Optional[dict] = None,
        arm_commander=None,
        gripper_commander=None,
        pick_posture_config=POSTURE_UNSET,
        place_posture_config=POSTURE_UNSET,
        place_hover_above_z: Optional[float] = None,
        place_approach_distance: Optional[float] = None,
        pick_min_reachable_z: Optional[float] = None,
        pick_max_reachable_radius_xy: Optional[float] = None,
        pick_approach_p_thresh: Optional[float] = None,
        pick_approach_std_dev: Optional[float] = None,
        move_timeout_s: Optional[float] = None,
        approach_timeout_s: Optional[float] = None,
        insert_timeout_s: Optional[float] = None,
        grasp_offset_local_overrides: Optional[dict] = None,
    ) -> None:
        self._robot = robot
        self._strategy = strategy
        self._gripper = gripper if gripper is not None else getattr(self._robot, 'gripper', None)
        self._ee_height_for_move_min = ee_height_for_move
        self._teleport_mode = teleport_mode
        self._prim_geometry = prim_geometry or {}
        self._arm_commander = arm_commander
        self._gripper_commander = gripper_commander
        # Pick reachability gate knobs (consumed by CheckPickReachable).
        # Both ``None`` keeps library defaults (no Z-floor check;
        # env_config_values.UR10_WORKING_RADIUS for the radius).
        self._pick_min_reachable_z = pick_min_reachable_z
        self._pick_max_reachable_radius_xy = pick_max_reachable_radius_xy
        # Per-task pick-approach knobs (consumed by CortexExecuteApproach
        # and PrepareGrasp).  ``None`` keeps the library defaults — 5 mm
        # SUCCESS threshold and DEFAULT_PICK_APPROACH_STD_DEV — which suit
        # moving-conveyor picks; tighten per-task for stationary picks
        # where flush gripper close matters.
        self._pick_approach_p_thresh = pick_approach_p_thresh
        self._pick_approach_std_dev = pick_approach_std_dev
        # Sim-time watchdog timeout overrides for the cortex BT.  ``None``
        # keeps the module-level DEFAULT_*_TIMEOUT_S values; per-task
        # overrides flow through TaskSpec.{move,approach,insert}_timeout_s.
        self._move_timeout_s = move_timeout_s
        self._approach_timeout_s = approach_timeout_s
        self._insert_timeout_s = insert_timeout_s
        # Per-task asset_type -> 3-vector grasp-offset override (in the
        # object's native local frame, unscaled).  Consumed by
        # ``_grasp_offset_world``: when a pick's asset_type appears in
        # this dict the override (scaled by the pick's local scale) wins
        # over the asset-level default on ``PrimGeometry.default_grasp_offset``.
        # ``None`` => no overrides for this task.
        self._grasp_offset_local_overrides = grasp_offset_local_overrides or {}
        # Per-pick cache of world-frame grasp offsets (lazy memoization).
        # The world-frame value depends only on the (override-or-default,
        # reference_orientation, pick scale) triple, all of which are
        # static once a pick is spawned — so the cache is correct for the
        # lifetime of this context.
        self._grasp_offset_world_cache: dict = {}
        # Cached on first :meth:`get_robot_base_xy` call — the UR10 base
        # prim does not move during the task, so a single world-pose query
        # at gate-evaluation time is sufficient.
        self._robot_base_xy: Optional[np.ndarray] = None
        # Cached on first :meth:`get_robot_base_z` call — used by
        # ``CheckGraspPoseReachable`` to anchor the 3D reachability sphere
        # at the actual mount Z (so a robot mounted on a tall pedestal
        # gets a correctly-shifted envelope).
        self._robot_base_z: Optional[float] = None
        # Per-instance posture-config overrides.  Sentinel ``POSTURE_UNSET`` means
        # "use the module default"; passing ``None`` explicitly disables the
        # null-space posture bias for that phase.
        self._pick_posture_override = pick_posture_config
        self._place_posture_override = place_posture_config
        # Per-instance overrides for the cortex-tree place-side hover/approach
        # geometry.  ``None`` keeps the module defaults.  Read by the cortex
        # behaviours via :meth:`get_place_hover_above_z` and
        # :meth:`get_place_approach_distance`.
        self._place_hover_above_z = place_hover_above_z
        self._place_approach_distance = place_approach_distance
        # Per-cycle perception cache.  Populated as a side effect by the
        # command builders today; once the perception behaviours (steps 7+)
        # are in place they become the primary writers and the builders
        # shift to pure read-the-cache consumers.  Cleared by
        # ``reset_cycle_cache`` between attempts.
        self._current_grasp_pose: Optional[perception_utils.GraspPose] = None
        self._current_item_in_ee: Optional[perception_utils.ItemInEEPose] = None
        self._current_place_pose: Optional[perception_utils.PlacePose] = None
        # "Grasp succeeded" signal, decoupled from ``_current_item_in_ee``.
        # Teleport mode sets this True without populating
        # ``_current_item_in_ee`` so ``compute_place_pose`` falls back to
        # its nominal branch.  Non-teleport VerifyGrasp sets both.
        self._is_holding_item: bool = False
        # (target_obj id) → (xy[2], sim_time) from the previous tick.
        # Used by ``_estimate_target_drift_xy`` to finite-difference the
        # target's world-frame XY velocity so
        # ``compute_dynamic_place_command_for_active_item`` can lead a
        # moving target without overshooting a stationary one.
        # The "now" timestamp is provided by ``get_current_sim_time()`` —
        # subclasses with a live World override it to read sim time, so
        # finite-differences are dt-invariant (i.e. the lead is the same
        # whether sim runs at 0.4× or 1.4× wall-clock real-time).  The
        # base class falls back to ``time.time()`` for the mock path.
        self._target_drift_cache: dict = {}
        # Cycle-time gate (consumed by WaitForCycleTime in pt_task_behaviours).
        # ``simulation_time`` is updated each step by the host loop (mock
        # tick loop or real-sim pre_step). ``min_cycle_time_s`` is set
        # from the CLI ``--min-cycle-time`` arg (0 = no gating).
        # ``last_cycle_start_time`` is the simulation_time at which the
        # previous pick-place cycle started — set by WaitForCycleTime
        # itself when it releases.  The next cycle waits until
        # simulation_time >= last_cycle_start_time + min_cycle_time_s,
        # so a cycle that already takes ≥ min_cycle_time_s sees no
        # extra delay.
        self.simulation_time: float = 0.0
        self.min_cycle_time_s: float = 0.0
        self.last_cycle_start_time: Optional[float] = None
        # Runtime services attached once the task has built strategy +
        # scene state.  Pick / target lists are accessed through the
        # ``pick_objs`` / ``target_objs`` properties below, which
        # delegate to the strategy — the strategy is the sole owner
        # (and mutation site) of those lists.
        self.verifier = None
        self.spawner = None

    @property
    def teleport_mode(self) -> bool:
        """True when the task is running in teleport mode.

        In teleport mode the arm/gripper commanders are no-ops and
        ``mark_pick_complete`` snaps the held item onto its target.
        Perception behaviours that rely on live FK (e.g. VerifyGrasp's
        pose-deviation check) should short-circuit when this is set.
        """
        return self._teleport_mode

    @property
    def mock_mode(self) -> bool:
        """True when the task is running under the pure-Python mock harness.

        Set by ``MockTaskContext``.  Real-sim ``TaskContext`` leaves it
        False (the default).  The mock arm simulates motion via a
        tick-based countdown that does not faithfully represent
        contact physics — in particular, the held item never tracks
        the EE during a lift.  Perception behaviours that compare EE
        pose vs item world pose (e.g. ``VerifyGrasp``'s pose-deviation
        check) must short-circuit in this mode to avoid spurious
        FAILUREs that the retry decorator then compounds into infinite
        deferrals.
        """
        return False

    # -------------------------------------------------------------------
    # Per-cycle perception cache
    # -------------------------------------------------------------------

    def get_current_grasp_pose(self) -> Optional["perception_utils.GraspPose"]:
        return self._current_grasp_pose

    def set_current_grasp_pose(
        self, pose: Optional["perception_utils.GraspPose"],
    ) -> None:
        self._current_grasp_pose = pose

    def get_current_item_in_ee(self) -> Optional["perception_utils.ItemInEEPose"]:
        return self._current_item_in_ee

    def set_current_item_in_ee(
        self, pose: Optional["perception_utils.ItemInEEPose"],
    ) -> None:
        self._current_item_in_ee = pose

    def get_current_place_pose(self) -> Optional["perception_utils.PlacePose"]:
        return self._current_place_pose

    def set_current_place_pose(
        self, pose: Optional["perception_utils.PlacePose"],
    ) -> None:
        self._current_place_pose = pose

    def reset_cycle_cache(self) -> None:
        """Clear cached grasp/item-in-ee/place poses for the next attempt."""
        self._current_grasp_pose = None
        self._current_item_in_ee = None
        self._current_place_pose = None
        self._is_holding_item = False
        self._target_drift_cache.clear()

    def get_current_sim_time(self) -> float:
        """Return current simulation time in seconds.

        Default implementation returns wall-clock time as a fallback for the
        mock path (no live World).  ``TaskContext`` overrides this to read
        ``World.instance().current_time`` so drift estimates and any other
        rate-sensitive computation track sim time, not wall-clock time.
        Without the override, finite-differences on a moving conveyor target
        would scale with the sim/wall-clock ratio (which itself depends on
        ``physics_dt``) and yield dt-dependent lead magnitudes.
        """
        return time.time()

    def _estimate_target_drift_xy(self, target_obj, current_xy) -> np.ndarray:
        """World-frame XY velocity of *target_obj*, finite-differenced across ticks.

        Returns a zero vector on the first observation, on sub-ms dt, when
        the gap between observations is too large to be a reliable
        finite-difference estimate (|dt| > 0.5 s — typical between BT
        attempts wrapped in Retry, where a stale cache entry from many
        seconds ago would otherwise yield a wildly inflated velocity), or
        when |v| < 2 mm/s (below that it reads as encoder jitter on an
        effectively stationary target — keeps stationary pads from
        producing a ghost lead).
        """
        now = self.get_current_sim_time()
        key = id(target_obj)
        curr = np.asarray(current_xy[:2], dtype=float)
        prev = self._target_drift_cache.get(key)
        self._target_drift_cache[key] = (curr, now)
        if prev is None:
            return np.zeros(2, dtype=float)
        prev_xy, prev_t = prev
        dt = now - prev_t
        if dt <= 1e-3 or dt > 0.5:
            return np.zeros(2, dtype=float)
        vel = (curr - prev_xy) / dt
        if np.linalg.norm(vel) < 0.002:
            return np.zeros(2, dtype=float)
        return vel

    def is_holding_item(self) -> bool:
        """Return True when a grasp has been verified for the current cycle.

        Set by ``VerifyGrasp`` on SUCCESS (both real-sim and teleport
        paths).  Cleared by ``reset_cycle_cache`` between attempts.
        Decoupled from ``get_current_item_in_ee()`` presence because
        teleport mode has no meaningful measurement to cache but still
        needs to gate the place subtree open.
        """
        return self._is_holding_item

    def set_holding_item(self, held: bool) -> None:
        """Set the grasp-succeeded flag (see ``is_holding_item``)."""
        self._is_holding_item = bool(held)

    def refresh_grasp_pose_position(self) -> Optional["perception_utils.GraspPose"]:
        """Refresh only the position of the cached ``GraspPose`` from the live pick pose.

        Simulates a perception module that re-measures the item's world
        position while the robot is already moving toward it — useful when
        the item has not yet settled at ``PrepareGrasp`` time (e.g. items
        spawned above the bin that are still falling under gravity).

        Keeps orientation, approach direction / distance / std-dev, and
        the grasp-time EE offset frozen at the values captured by the
        initial ``PrepareGrasp`` — only ``ee_position`` tracks the live
        pick position (= ``live_pick_pos + ee_offset_world_at_grasp``).
        This matches the "semi-realistic" model where only fast position
        re-estimation is cheap; re-deriving the approach plan is not.

        No-op when no cache exists (``PrepareGrasp`` has not yet run),
        or when the current pick / live pick position cannot be resolved.
        """
        cached = self._current_grasp_pose
        if cached is None:
            return None
        pick_name = self.get_current_pick_name()
        if pick_name is None:
            return cached
        pick_pos = self.get_picking_position(pick_name)
        if pick_pos is None:
            return cached
        new_ee_position = (
            np.asarray(pick_pos, dtype=float) + cached.ee_offset_world_at_grasp
        )
        # Drift lead for moving picks (e.g. items riding a conveyor).  Mirrors
        # the place-side lead in compute_dynamic_place_command_for_active_item:
        # tracking the live item alone leaves a steady-state lag that
        # CortexExecuteApproach cannot close within p_thresh, and the constantly
        # re-centering -Z approach funnel stalls the descent.  Aiming the EE
        # ee_position ahead by ``PICK_LEAD_HORIZON_S * drift_xy`` lets RMPFlow
        # match the item's velocity rather than chase it.  Stationary picks
        # produce zero drift and therefore no lead.
        pick_obj = self._strategy.pick_objs_by_name.get(pick_name)
        if pick_obj is not None:
            drift_xy = self._estimate_target_drift_xy(pick_obj, pick_pos)
            lead_xy = PICK_LEAD_HORIZON_S * drift_xy
            lead_norm = float(np.linalg.norm(lead_xy))
            if lead_norm > PICK_LEAD_MAX_MAGNITUDE_M:
                lead_xy *= PICK_LEAD_MAX_MAGNITUDE_M / lead_norm
            new_ee_position[:2] += lead_xy
        refreshed = perception_utils.GraspPose(
            ee_position=new_ee_position,
            ee_orientation=cached.ee_orientation,
            approach_direction=cached.approach_direction,
            approach_distance=cached.approach_distance,
            approach_std_dev=cached.approach_std_dev,
            ee_offset_world_at_grasp=cached.ee_offset_world_at_grasp,
            item_position_at_grasp=cached.item_position_at_grasp,
            item_orientation_at_grasp=cached.item_orientation_at_grasp,
        )
        self._current_grasp_pose = refreshed
        return refreshed

    # -------------------------------------------------------------------
    # Strategy access
    # -------------------------------------------------------------------

    @property
    def strategy(self) -> MultiPickStrategy:
        return self._strategy

    # -------------------------------------------------------------------
    # Scene state — canonical accessors.  Storage lives on the strategy
    # (its ``add_incremental_*`` / ``_extend_*`` helpers are the sole
    # mutation sites); the context exposes them so callers don't reach
    # into the strategy directly.
    # -------------------------------------------------------------------

    @property
    def pick_objs(self) -> list:
        return self._strategy.pick_objs

    @property
    def target_objs(self) -> list:
        return self._strategy.target_objs

    @property
    def pick_objs_by_name(self) -> dict:
        return self._strategy.pick_objs_by_name

    @property
    def target_objs_by_name(self) -> dict:
        return self._strategy.target_objs_by_name

    # -------------------------------------------------------------------
    # State query methods (called by ContextMonitorBehaviour)
    # -------------------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        joints_state = self._robot.get_joints_state()
        return joints_state.positions

    def get_end_effector_position(self) -> np.ndarray:
        pos, _ = self._robot.end_effector.get_local_pose()
        return pos

    def get_current_pick_name(self) -> Optional[str]:
        return self._strategy.get_current_pick_name()

    def get_picking_position(self, pick_name: str) -> Optional[np.ndarray]:
        return self._strategy.get_picking_position(pick_name)

    def get_placing_target_name(self, pick_name: str) -> Optional[str]:
        """Return the target name for the given pick. Delegates to strategy."""
        return self._strategy.get_placing_target_name(pick_name)

    def get_placing_info(
        self, pick_name: str,
        end_effector_orientation_for_drop: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute target info (name, position, orientation) for the given pick.

        Uses cached PrimGeometry for the drop Z calculation:
            drop_z = target_world_z + target.top_surface_height + pick.rest_height
        """
        target_name = self._strategy.get_placing_target_name(pick_name)
        if target_name is None:
            return None, None, None

        target_obj = self._strategy.target_objs_by_name.get(target_name)
        if target_obj is None:
            return None, None, None

        target_pos, target_orient = target_obj.get_world_pose()

        # Determine placement orientation.
        if end_effector_orientation_for_drop is not None:
            place_orient = target_orient
        else:
            pick_obj = self._strategy.pick_objs_by_name.get(pick_name)
            if pick_obj is not None:
                _, place_orient = pick_obj.get_world_pose()
            else:
                place_orient = target_orient

        # Compute drop Z from geometry
        pick_geom = self._prim_geometry.get(pick_name)
        target_geom = self._prim_geometry.get(target_name)
        if pick_geom is not None and target_geom is not None:
            drop_z = target_pos[2] + target_geom.top_surface_height + pick_geom.rest_height
            return target_name, np.array([target_pos[0], target_pos[1], drop_z]), place_orient

        return target_name, target_pos, place_orient

    def get_end_effector_offset(self, pick_name: str) -> np.ndarray:
        """Return an object-aware end-effector offset for the given pick.

        Composes the geometry-derived ``[0, 0, grasp_height]`` (along
        world +Z at pick orientation) with the per-pick grasp offset
        rotated into world frame (``_grasp_offset_world``).  Falls back
        to ``_EE_OFFSET_FALLBACK`` when no geometry is available (logs a
        warning, since this usually indicates a missing precomputed
        entry or an item spawned outside the configurator path).
        """
        if pick_name and pick_name in self._prim_geometry:
            geom = self._prim_geometry[pick_name]
            base = np.array([0.0, 0.0, geom.grasp_height])
            return base + self._grasp_offset_world(pick_name)
        if not hasattr(self, "_ee_offset_fallback_warned"):
            self._ee_offset_fallback_warned = set()
        key = pick_name or "<empty>"
        if key not in self._ee_offset_fallback_warned:
            self._ee_offset_fallback_warned.add(key)
            logger.warning(
                "get_end_effector_offset: no PrimGeometry for pick_name=%r; "
                "using fallback offset %s",
                pick_name, self._EE_OFFSET_FALLBACK.tolist(),
            )
        return self._EE_OFFSET_FALLBACK.copy()

    def _grasp_offset_world(self, pick_name: Optional[str]) -> np.ndarray:
        """Per-pick grasp offset rotated into the world frame.

        Resolution order:
          1. Per-task override in ``_grasp_offset_local_overrides``
             (keyed by the pick's asset_type, scaled by the pick's
             local scale).
          2. Asset-level default on ``PrimGeometry.default_grasp_offset``
             (already scaled at lookup time).

        Whichever applies is rotated by the geometry's recorded
        ``reference_orientation``.  Returns a zero 3-vector when no
        offset applies — preserves "grasp at center" for legacy assets.

        Results are memoized per pick_name; the world-frame value is
        static for the lifetime of the context (override, default,
        scale, and reference_orientation are all fixed once spawned).
        """
        if not pick_name:
            return np.zeros(3)
        cached = self._grasp_offset_world_cache.get(pick_name)
        if cached is not None:
            return cached
        geom = self._prim_geometry.get(pick_name)
        if geom is None:
            return np.zeros(3)

        offset_local = None
        # Per-task override wins over the asset-level default.
        if self._grasp_offset_local_overrides:
            pick_obj = self._strategy.pick_objs_by_name.get(pick_name)
            if pick_obj is not None:
                from asset_utils import get_asset_type
                asset_type = get_asset_type(pick_obj, asset_type_default=None)
                override = (
                    self._grasp_offset_local_overrides.get(asset_type)
                    if asset_type else None
                )
                if override is not None:
                    arr = np.asarray(override, dtype=float).reshape(-1)
                    if arr.shape[0] == 3:
                        scale = _safe_local_scale(pick_obj)
                        if scale is not None:
                            arr = arr * scale
                        offset_local = arr

        if offset_local is None:
            offset_local = np.asarray(geom.default_grasp_offset, dtype=float)

        if not np.any(offset_local):
            result = np.zeros(3)
        else:
            ref = geom.reference_orientation
            if ref is None:
                result = offset_local.copy()
            else:
                from asset_data_utils import _quat_to_rotation_matrix
                R = _quat_to_rotation_matrix(np.asarray(ref, dtype=float))
                result = R @ offset_local

        self._grasp_offset_world_cache[pick_name] = result
        return result

    def get_end_effector_orientation(self, pick_name: Optional[str] = None) -> np.ndarray:
        """End-effector orientation for pick phases. Delegates to strategy."""
        return self._strategy.get_end_effector_orientation(pick_name or "")

    def get_end_effector_orientation_for_drop(
        self, pick_name: Optional[str] = None, target_name: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """End-effector orientation for drop phase. Delegates to strategy."""
        return self._strategy.get_end_effector_orientation_for_drop(
            pick_name or "", target_name
        )

    def get_end_effector_offset_for_drop(
        self, pick_name: Optional[str] = None,
        end_effector_orientation_for_drop: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """World-frame EE-to-item-center offset for the drop phase.

        The held item's center sits at a fixed location in the EE's local
        frame (set at grasp time as ``[0, 0, grasp_height]``, i.e.
        ``grasp_height`` along the EE's "up" axis).  When the EE rotates
        from its pick orientation to a different drop orientation, that
        local-frame vector rotates with the gripper, so the *world-frame*
        offset is the pick offset rotated by ``R_drop · R_pick⁻¹``.

        Concrete bottle case to anchor the geometry: the bottle is picked
        from a side-lying pose with the EE descending vertically; the
        offset from item center to flange is ``grasp_height`` (= bottle
        radius) along world +Z.  At drop the EE rotates 90° so the bottle
        is held from the side and placed upright; the same vector now
        points horizontally — flange one ``grasp_height`` to the side of
        the upright bottle's center, not above it.

        Returns:
            ``None`` if no drop orientation is supplied (caller should
            fall back to the pick-side offset).  Otherwise the rotated
            world-frame offset.  When ``drop_orient ≡ pick_orient`` the
            relative rotation is identity and the result is identical to
            the pick offset, so callers do not need a special-case check.
        """
        if end_effector_orientation_for_drop is None:
            return None
        geom = self._prim_geometry.get(pick_name) if pick_name else None
        if geom is None:
            return None
        pick_offset_world = (
            np.array([0.0, 0.0, geom.grasp_height])
            + self._grasp_offset_world(pick_name)
        )
        pick_orient = self.get_end_effector_orientation(pick_name)
        if _quaternions_equivalent(end_effector_orientation_for_drop, pick_orient):
            # Identity rotation — short-circuit the matrix math (and
            # avoid floating-point dust in the unchanged-orientation
            # case, which is by far the most common).
            return pick_offset_world
        return _rotate_pick_offset_to_drop(
            pick_offset_world, pick_orient, end_effector_orientation_for_drop,
        )

    def get_posture_config(self, phase: str) -> Optional[np.ndarray]:
        """Null-space posture config (joint-angle bias) for the given phase.

        Args:
            phase: ``"pick"`` or ``"place"``.

        Resolution order:
          1. Per-instance override passed into ``__init__`` (``None`` means
             "explicitly disable posture bias" and wins over the defaults).
          2. Module-level ``PICK_POSTURE_CONFIG`` / ``PLACE_POSTURE_CONFIG``
             (env-var overridable via ``PICK_POSTURE_CONFIG_OVERRIDE`` and
             ``PLACE_POSTURE_CONFIG_OVERRIDE``).
          3. ``None`` for any unrecognised phase.

        Called directly by the command builders; there is no longer a
        ``posture_config`` kwarg on those methods.  To override per task,
        pass ``pick_posture_config=`` / ``place_posture_config=`` to the
        context constructor (usually via ``TaskSpec``).
        """
        if phase == "pick":
            if self._pick_posture_override is not POSTURE_UNSET:
                return self._pick_posture_override
            return PICK_POSTURE_CONFIG
        if phase == "place":
            if self._place_posture_override is not POSTURE_UNSET:
                return self._place_posture_override
            return PLACE_POSTURE_CONFIG
        return None

    def get_ee_height_for_move(self) -> float:
        recommended = self._strategy.get_recommended_ee_height(prim_geometry=self._prim_geometry)
        if recommended is not None:
            return max(self._ee_height_for_move_min, recommended)
        return self._ee_height_for_move_min

    def get_place_approach_distance(self) -> float:
        """Return the RMPFlow approach-funnel length used by the place-side phases.

        ``CortexDownToInsert`` consumes this as its descent-funnel length;
        ``CortexMoveToPlace`` uses it (via :meth:`get_place_hover_above_z`)
        to enforce the "hover at least the funnel length above the target"
        invariant so the descent funnel has room to engage from above.
        """
        if self._place_approach_distance is not None:
            return float(self._place_approach_distance)
        return float(perception_utils.DEFAULT_PLACE_APPROACH_DISTANCE)

    def get_place_hover_above_z(self) -> float:
        """Return the relative Z hover above the place target during transport.

        Used by ``CortexMoveToPlace``.  Always clamped to be at least
        :meth:`get_place_approach_distance` so that the EE enters the
        descent funnel from above without overshoot — the
        ``CortexMoveToPlace -> CortexDownToInsert`` handoff stays clean
        regardless of per-task overrides.
        """
        if self._place_hover_above_z is not None:
            base = float(self._place_hover_above_z)
        else:
            base = float(perception_utils.DEFAULT_PLACE_HOVER_ABOVE_Z)
        return max(base, self.get_place_approach_distance())

    def get_pick_min_reachable_z(self) -> Optional[float]:
        """Return the world-Z floor below which a pick is considered dropped.

        ``CheckPickReachable`` calls ``strategy.defer_pick`` and returns
        FAILURE for picks whose world Z falls below this floor.  ``None``
        (default) disables the floor check entirely.  Per-task override
        flows through ``TaskSpec.pick_min_reachable_z``.
        """
        return (
            float(self._pick_min_reachable_z)
            if self._pick_min_reachable_z is not None else None
        )

    def get_pick_max_reachable_radius_xy(self) -> float:
        """Return the reach-radius value used by the pick reachability gates.

        Two gates consume this number, with different geometric
        interpretations:

        - ``CheckPickReachable`` (in pick_attempt) treats it as a 2D XY
          cylinder radius around the robot base.  Items with XY distance
          beyond it cause the gate to return RUNNING (BT idles, waiting
          for a moving conveyor to drift the item into reach).
        - ``CheckGraspPoseReachable`` (also in pick_attempt, after
          ``PrepareGrasp``) treats it as a 3D sphere radius around the
          mount point ``(base_xy, base_z)``.  This catches the high-Z
          case where a pose at the cylinder edge climbs out the top of
          the kinematic shell when the EE has to lift to the pre-grasp
          altitude.

        Default is :data:`env_config_values.UR10_WORKING_RADIUS`; per-task
        override flows through ``TaskSpec.pick_max_reachable_radius_xy``.
        """
        if self._pick_max_reachable_radius_xy is not None:
            return float(self._pick_max_reachable_radius_xy)
        import env_config_values
        return float(env_config_values.UR10_WORKING_RADIUS)

    def get_robot_base_xy(self) -> np.ndarray:
        """Return the world-frame XY position of the robot base (cached).

        The UR10 base prim does not move during the task, so the first
        query is cached and reused for the rest of the run.  Returns
        ``[0, 0]`` if the world-pose query fails (degraded fallback —
        better to keep the gate working than to crash).
        """
        if self._robot_base_xy is None:
            try:
                pos, _ = self._robot.get_world_pose()
                self._robot_base_xy = np.asarray(pos[:2], dtype=float)
            except Exception:
                self._robot_base_xy = np.zeros(2, dtype=float)
        return self._robot_base_xy

    def get_robot_base_z(self) -> float:
        """Return the world-frame Z position of the robot mount (cached).

        Used by ``CheckGraspPoseReachable`` to anchor the 3D reachability
        sphere.  The UR10 mount does not move during the task, so the first
        query is cached for the rest of the run.  Returns ``0.0`` if the
        world-pose query fails (degraded fallback — keeps the gate working
        rather than crashing).
        """
        if self._robot_base_z is None:
            try:
                pos, _ = self._robot.get_world_pose()
                self._robot_base_z = float(pos[2])
            except Exception:
                self._robot_base_z = 0.0
        return self._robot_base_z

    def get_pick_approach_p_thresh(self) -> float:
        """Return the position-tolerance for ``CortexExecuteApproach`` SUCCESS.

        Per-task override flows through ``TaskSpec.pick_approach_p_thresh``.
        Default 0.005 m matches the looser value chosen for moving-conveyor
        picks where RMPFlow has a small steady-state tracking error; tighten
        per-task (e.g. 0.002 m) when the gripper must close flush against
        the item — for example on socket-insertion placements where any
        suction-tip air gap propagates into a misaligned drop.
        """
        if self._pick_approach_p_thresh is not None:
            return float(self._pick_approach_p_thresh)
        return 0.005

    def get_pick_approach_std_dev(self) -> float:
        """Return the lateral std-dev for the pick-side RMPFlow approach funnel.

        Per-task override flows through ``TaskSpec.pick_approach_std_dev``.
        Default delegates to ``perception_utils.DEFAULT_PICK_APPROACH_STD_DEV``
        (0.015 m, widened so RMPFlow descends cleanly against moving
        targets).  Narrow this for stationary picks needing tighter lateral
        alignment during the descent.
        """
        if self._pick_approach_std_dev is not None:
            return float(self._pick_approach_std_dev)
        return float(perception_utils.DEFAULT_PICK_APPROACH_STD_DEV)

    def get_move_timeout_s(self) -> float:
        """Sim-time watchdog (seconds) for freespace and transport moves.

        Wraps ``CortexMoveToPreGrasp``, ``CortexMoveToPlace``, and
        ``CortexMoveRelative`` in :func:`pt_sim_time_decorators.sim_timeout_to_success`.
        Per-task override flows through ``TaskSpec.move_timeout_s``;
        default :data:`DEFAULT_MOVE_TIMEOUT_S` (15 s) preserves the
        prior behaviour-internal force-success deadline.
        """
        if self._move_timeout_s is not None:
            return float(self._move_timeout_s)
        return DEFAULT_MOVE_TIMEOUT_S

    def get_approach_timeout_s(self) -> float:
        """Sim-time watchdog (seconds) for the final grasp approach.

        Wraps ``CortexExecuteApproach``.  Tighter than the general
        move timeout because the approach segment is short and the
        ``p_thresh`` is tight (a 15 s wait usually means the EE is
        oscillating in the funnel).  Per-task override flows through
        ``TaskSpec.approach_timeout_s``; default
        :data:`DEFAULT_APPROACH_TIMEOUT_S` (8 s).
        """
        if self._approach_timeout_s is not None:
            return float(self._approach_timeout_s)
        return DEFAULT_APPROACH_TIMEOUT_S

    def get_insert_timeout_s(self) -> float:
        """Sim-time watchdog (seconds) for the place-side descent.

        Wraps ``CortexDownToInsert``.  Wrap is non-optional in the
        factory because this behaviour now demotes SUCCESS→RUNNING on
        any Z gap; without the watchdog it could RUNNING indefinitely
        on a target the EE genuinely cannot reach.  Per-task override
        flows through ``TaskSpec.insert_timeout_s``; default
        :data:`DEFAULT_INSERT_TIMEOUT_S` (15 s).
        """
        if self._insert_timeout_s is not None:
            return float(self._insert_timeout_s)
        return DEFAULT_INSERT_TIMEOUT_S

    # -------------------------------------------------------------------
    # Mutation methods (called by task behaviours) — delegated to strategy
    # -------------------------------------------------------------------

    def advance_pick_index(self) -> Optional[str]:
        """Increment the current pick index. Return new pick name or None."""
        return self._strategy.advance_pick_index()

    def mark_pick_complete(self, pick_name: str) -> None:
        """Mark the given pick as completed.

        In teleport mode, also moves the pick object to its target position.
        """
        if self._teleport_mode:
            target_name = self.get_placing_target_name(pick_name)
            drop_orient = self.get_end_effector_orientation_for_drop(pick_name, target_name)
            target_name, place_pos, place_orient = self.get_placing_info(pick_name, drop_orient)
        self._strategy.mark_pick_complete(pick_name)
        if self._teleport_mode and place_pos is not None:
            pick_obj = self._strategy.pick_objs_by_name.get(pick_name)
            if pick_obj is not None:
                pick_obj.set_world_pose(position=place_pos, orientation=place_orient)
                logger.info(f"Teleported '{pick_name}' to target '{target_name}'")

    # -------------------------------------------------------------------
    # Properties — delegated to strategy
    # -------------------------------------------------------------------

    @property
    def task_finished(self) -> bool:
        return self._strategy.task_finished

    @task_finished.setter
    def task_finished(self, value: bool) -> None:
        self._strategy.task_finished = value

    @property
    def all_picks_done(self) -> bool:
        return self._strategy.all_picks_done

    @property
    def targets_exhausted(self) -> bool:
        return self._strategy.targets_exhausted

    @targets_exhausted.setter
    def targets_exhausted(self, value: bool) -> None:
        self._strategy.targets_exhausted = value

    @property
    def picking_order_item_names(self) -> List[str]:
        return self._strategy.picking_order_item_names

    @property
    def _completed_picks(self) -> set:
        """Backward-compatible access to completed picks (used by tests)."""
        return self._strategy.completed_picks

    # -------------------------------------------------------------------
    # Hardware properties (remain on TaskContextBase)
    # -------------------------------------------------------------------

    @property
    def robot(self):
        """Read-only access to the robot articulation."""
        return self._robot

    @property
    def gripper(self):
        """Read-only access to the gripper."""
        return self._gripper

    @property
    def arm_commander(self):
        """IArmCommander for end-effector motion (Cortex-aligned)."""
        return self._arm_commander

    @property
    def gripper_commander(self):
        """IGripperCommander for gripper control (Cortex-aligned)."""
        return self._gripper_commander

    # -------------------------------------------------------------------
    # Robot hardware convenience methods
    # -------------------------------------------------------------------

    def reset_gripper(self) -> None:
        """Reset the gripper to its open position."""
        if hasattr(self._gripper, 'set_joint_positions') and hasattr(self._gripper, 'joint_opened_positions'):
            self._gripper.set_joint_positions(self._gripper.joint_opened_positions)

    # -------------------------------------------------------------------
    # Motion command methods
    # -------------------------------------------------------------------

    def robot_at_target(self, command: MotionCommand, p_thresh: float, R_thresh: float) -> bool:
        """Check if the robot end-effector is close to the target specified in a MotionCommand.

        Args:
            command: MotionCommand whose target_pose defines the desired pose.
            p_thresh: Position threshold (metres) for "close enough".
            R_thresh: Rotation threshold for "close enough" (average axis error).

        Returns:
            True if the current FK pose is within thresholds of the command target.
        """
        fk_pq = self._arm_commander.get_fk_pq()
        fk_T = fk_pq.to_T()

        if not hasattr(command, "target_pose") or command.target_pose is None:
            return False
        target_T = command.target_pose.to_T()
        return math_util.transforms_are_close(fk_T, target_T, p_thresh=p_thresh, R_thresh=R_thresh)

    def compute_motion_command_to_target(self, target_pq: PosePq) -> MotionCommand:
        """Create a simple MotionCommand to reach the given pose.

        Args:
            target_pq: Target pose as PosePq.

        Returns:
            A MotionCommand with the given target pose.
        """
        try:
            return MotionCommand(target_pose=target_pq)
        except Exception as e:
            logger.error(f"compute_motion_command_to_target: p={target_pq.p} q={target_pq.q}")
            raise

    def get_relative_pq(self, offset: np.ndarray, use_world_frame: bool = True) -> PosePq:
        """Compute a PosePq offset from the current end-effector pose.

        Args:
            offset: 3-D offset vector.
            use_world_frame: If True, offset is added in world frame.
                If False, offset is rotated by the current EE orientation first.

        Returns:
            A new PosePq representing the offset target.
        """
        fk_pq = self._arm_commander.get_fk_pq()

        if use_world_frame:
            return PosePq(fk_pq.p + offset, fk_pq.q)

        # Offset is in the end-effector frame — rotate by current EE orientation.
        T = fk_pq.to_T()
        R, p = math_util.unpack_T(T)
        target_p = p + R.dot(offset)
        target_q = math_util.matrix_to_quat(R)
        return PosePq(target_p, target_q)

    def _build_motion_command(
        self,
        target_pose: PosePq,
        approach_params: ApproachParams,
        posture_config: Optional[np.ndarray],
    ) -> MotionCommand:
        """Single construction point for MotionCommand objects owned by the context."""
        return MotionCommand(
            target_pose=target_pose,
            approach_params=approach_params,
            posture_config=posture_config,
        )

    def _resolve_grasp_pose(self) -> Optional["perception_utils.GraspPose"]:
        """Return the cached ``GraspPose`` or compute + cache one.

        Shared by the pre-grasp and approach command builders so they
        always agree on the same pose.  Preserves the warn-once log for
        missing ``PrimGeometry``.
        """
        if self._current_grasp_pose is not None:
            return self._current_grasp_pose
        pick_name = self.get_current_pick_name()
        if pick_name is None:
            return None
        pick_pos = self.get_picking_position(pick_name)
        if pick_pos is None:
            return None
        pick_geom = self._prim_geometry.get(pick_name)
        if pick_geom is None:
            self.get_end_effector_offset(pick_name)  # fires the warn-once log
        grasp_pose = perception_utils.compute_grasp_pose(
            pick_name,
            pick_position=pick_pos,
            pick_geometry=pick_geom,
            pick_orientation_preference=self.get_end_effector_orientation(pick_name),
            grasp_offset_world=self._grasp_offset_world(pick_name),
        )
        self._current_grasp_pose = grasp_pose
        return grasp_pose

    def compute_pick_command_for_active_item(self) -> Optional[MotionCommand]:
        """Compute a MotionCommand to execute the final grasp approach.

        Uses the cached ``GraspPose`` when ``PrepareGrasp`` has populated
        one this cycle; otherwise computes + caches on demand.  The
        resulting command targets the grasp pose with a tight approach
        funnel and the ``"pick"`` posture.

        Returns:
            A MotionCommand with approach-from-above parameters, or None
            when there is no current pick.
        """
        grasp_pose = self._resolve_grasp_pose()
        if grasp_pose is None:
            return None
        target_pose = PosePq(grasp_pose.ee_position, grasp_pose.ee_orientation)
        approach = ApproachParams(
            direction=grasp_pose.approach_direction * grasp_pose.approach_distance,
            std_dev=grasp_pose.approach_std_dev,
        )
        return self._build_motion_command(
            target_pose, approach, self.get_posture_config("pick"),
        )

    def compute_pregrasp_command_for_active_item(
        self,
        use_approach_funnel: bool = False,
        approach_direction: Optional[np.ndarray] = None,
        approach_distance: Optional[float] = None,
        approach_std_dev: Optional[float] = None,
        min_z: Optional[float] = None,
    ) -> Optional[MotionCommand]:
        """Compute a MotionCommand to the pre-grasp staging pose.

        The pre-grasp pose sits ``approach_distance`` upstream of the
        grasp pose along ``approach_direction`` — i.e. where the
        ``CortexMoveToPreGrasp`` behaviour parks the flange in free
        space before ``CortexExecuteApproach`` descends into the grasp.
        Uses the ``"pick"`` posture since the arm is settling into its
        grasp configuration.

        Args:
            use_approach_funnel: When False (default), a freespace move
                with no ``ApproachParams`` (legacy behaviour).  When True,
                an ``ApproachParams`` is attached so RMPFlow biases the
                final segment to approach the pre-grasp pose along the
                supplied direction (e.g. a horizontal direction from the
                live EE to the pre-grasp staging position, mirroring
                ``CortexMoveToPlace``'s transport-hover funnel).
            approach_direction: Unit vector for the funnel direction.
                When ``None`` falls back to the cached
                ``GraspPose.approach_direction`` (typically -Z).
            approach_distance: Funnel length (metres).  When ``None``
                falls back to the cached ``GraspPose.approach_distance``.
            approach_std_dev: Funnel std-dev (metres).  When ``None``
                falls back to the cached ``GraspPose.approach_std_dev``.
            min_z: Optional Z-floor for the target position.  When set
                and the natural pre-grasp Z is below ``min_z``, the
                target is raised to ``(pre_grasp_xy, min_z)`` so the
                pre-grasp transit stays at altitude.  ``CortexMoveToPreGrasp``
                passes ``ee_height_for_move`` so freespace travel from
                the post-place lift to the next pick stays at transport
                altitude (avoids sloping descents that knock over
                destination stacks).  ``CortexExecuteApproach`` is
                unaffected — its target is the grasp pose, with a vertical
                funnel that descends from whatever Z the EE entered at.

        Returns:
            A MotionCommand at the pre-grasp pose, or ``None`` when
            there is no current pick.
        """
        grasp_pose = self._resolve_grasp_pose()
        if grasp_pose is None:
            return None
        target_position = grasp_pose.pre_grasp_position
        if min_z is not None and float(target_position[2]) < float(min_z):
            target_position = np.array(target_position, dtype=float).copy()
            target_position[2] = float(min_z)
        target_pose = PosePq(target_position, grasp_pose.ee_orientation)
        posture_config = self.get_posture_config("pick")
        if use_approach_funnel:
            direction = (approach_direction if approach_direction is not None
                         else grasp_pose.approach_direction)
            distance = (approach_distance if approach_distance is not None
                        else grasp_pose.approach_distance)
            std_dev = (approach_std_dev if approach_std_dev is not None
                       else grasp_pose.approach_std_dev)
            approach = ApproachParams(
                direction=direction * distance,
                std_dev=std_dev,
            )
            return self._build_motion_command(
                target_pose, approach, posture_config,
            )
        # No ApproachParams — this is a freespace move; the tight funnel
        # applies only to the final approach segment.
        return MotionCommand(
            target_pose=target_pose,
            posture_config=posture_config,
        )

    def get_pre_grasp_position(self) -> Optional[np.ndarray]:
        """Return the world-frame pre-grasp staging position for the active pick.

        Wraps the cached ``GraspPose`` (populated by ``PrepareGrasp`` and
        refreshed by ``refresh_grasp_pose_position``).  Returns ``None``
        when no grasp pose is available (e.g. no current pick).
        """
        grasp_pose = self._resolve_grasp_pose()
        if grasp_pose is None:
            return None
        return grasp_pose.pre_grasp_position

    def compute_dynamic_place_command_for_active_item(
        self,
        target_p: Optional[np.ndarray] = None,
        target_obj=None,
        above: float = 0.0,
        use_approach_funnel: bool = True,
        approach_distance: Optional[float] = None,
        approach_std_dev: Optional[float] = None,
        approach_direction: Optional[np.ndarray] = None,
    ) -> Optional[MotionCommand]:
        """Compute a MotionCommand to place the current active item.

        If *target_p* is not supplied it is derived from
        :meth:`get_placing_info`.  When *target_obj* is given, its live world
        pose is used for XY fine-alignment with the held object.  The
        null-space posture config comes from :meth:`get_posture_config`
        ``("place")`` (module default or per-instance override); there is no
        kwarg to pass one in.

        Args:
            target_p: Explicit placement position (world frame).  Overrides the
                value from ``get_placing_info`` when provided.
            target_obj: Optional scene object for dynamic XY alignment.
            above: Extra height (metres) added to the placement Z.
            use_approach_funnel: When True (default, used by
                ``CortexDownToInsert``), the MotionCommand carries
                ``ApproachParams`` so RMPFlow biases the final segment
                to approach the target along the approach direction.
                When False, the command is a freespace move with no
                funnel.  Note: when ``approach_direction`` is horizontal
                the funnel does not stack vertical height on top of
                ``above`` (peak EE Z stays at ``insert_z + above``), so
                ``CortexMoveToPlace`` can enable the funnel to damp
                transport overshoot without paying extra hover height.
            approach_direction: Optional unit vector for the approach
                segment.  When ``None`` uses the default (-Z) from
                ``perception_utils.DEFAULT_PLACE_APPROACH_DIRECTION``.
                Set to a horizontal vector in ``CortexMoveToPlace`` to
                damp overshoot along the dominant travel axis.

        Returns:
            A MotionCommand with approach-from-above parameters, or None if there
            is no current pick or no valid target.
        """
        pick_name = self.get_current_pick_name()
        if pick_name is None:
            return None

        # Determine target name and orientation from strategy.  Keep the
        # raw (unresolved) value from the strategy separate from the
        # resolved drop orientation used to build the pose — the former
        # drives the "did the orientation actually change?" decision for
        # the drop-side EE offset below.
        target_name = self.get_placing_target_name(pick_name)
        drop_orient_raw = self.get_end_effector_orientation_for_drop(pick_name, target_name)
        drop_orient = (
            drop_orient_raw if drop_orient_raw is not None
            else self.get_end_effector_orientation(pick_name)
        )

        # Determine placement position.
        if target_p is None:
            _, place_pos, _ = self.get_placing_info(pick_name, drop_orient_raw)
            if place_pos is None:
                return None
            target_p = place_pos

        # Dynamic XY lead when a live target object is provided.  The
        # commanded target is nudged along the target's *own* measured
        # XY drift (not the item-to-target gap, which produces a large
        # ghost lead while the held item is still being transported).
        # A stationary pad has zero drift → zero nudge → no overshoot /
        # pull-back seam between CortexMoveToPlace and CortexDownToInsert.
        # A conveyor pad produces a Y-only drift and gets a Y-only lead
        # automatically, without hardcoding an axis.
        if target_obj is not None:
            target_obj_p, _ = target_obj.get_world_pose()
            drift_xy = self._estimate_target_drift_xy(target_obj, target_obj_p)
            target_p = target_p.copy()
            target_p[:2] += PLACE_LEAD_HORIZON_S * drift_xy

        # Assemble the drop-side EE offset + above-hover + drop orientation
        # via perception_utils.compute_place_pose.  ``target_p`` is the
        # drop-Z-adjusted target position (insert_z base); perception adds
        # the drop-rotated flange offset on top of it.  Missing geometry
        # still routes through get_end_effector_offset for the warn-once
        # log on the legacy fallback path.
        pick_geom = self._prim_geometry.get(pick_name)
        if pick_geom is None:
            self.get_end_effector_offset(pick_name)  # fires the warn-once log
        target_name_for_log = target_name if target_name is not None else ""
        compute_kwargs = dict(
            pick_geometry=pick_geom,
            pick_orientation=self.get_end_effector_orientation(pick_name),
            drop_orientation=drop_orient_raw,
            item_in_ee=self._current_item_in_ee,
            above=above,
            grasp_offset_world=self._grasp_offset_world(pick_name),
        )
        if approach_distance is not None:
            compute_kwargs["approach_distance"] = float(approach_distance)
        if approach_std_dev is not None:
            compute_kwargs["approach_std_dev"] = float(approach_std_dev)
        if approach_direction is not None:
            compute_kwargs["approach_direction"] = np.asarray(approach_direction, dtype=float)
        place_pose = perception_utils.compute_place_pose(
            pick_name, target_name_for_log,
            target_position=target_p,
            **compute_kwargs,
        )
        # Side-effect cache — matches the grasp-side behaviour so
        # PreparePlacement/CortexMoveToPlace/CortexDownToInsert can all
        # share the same resolved pose once step 9 wires them through.
        self._current_place_pose = place_pose

        target_pose = PosePq(place_pose.ee_position, place_pose.ee_orientation)
        posture_config = self.get_posture_config("place")
        if use_approach_funnel:
            approach = ApproachParams(
                direction=place_pose.approach_direction * place_pose.approach_distance,
                std_dev=place_pose.approach_std_dev,
            )
            return self._build_motion_command(target_pose, approach, posture_config)
        # Freespace move (no funnel) — mirrors compute_pregrasp_command_for_active_item.
        return MotionCommand(
            target_pose=target_pose,
            posture_config=posture_config,
        )

    def send_motion_command(self, cmd: MotionCommand) -> None:
        """Send a MotionCommand via the arm commander.

        Used by cortex-style behaviors that compute full MotionCommand objects
        (with approach_params, posture_config) rather than bare position/orientation.
        """
        self._arm_commander.send_motion_command(cmd)

    def get_placement_target(self):
        """Return (target_p, target_obj) for the current pick's placement target.

        Convenience wrapper around :meth:`get_placing_info` for cortex-style
        behaviors that need the target position and object separately (for
        dynamic XY alignment in MoveToPlace/DownToInsert).

        Returns:
            Tuple of (target_position, target_object), or (None, None) if
            no current pick or no valid target.
        """
        pick_name = self.get_current_pick_name()
        if pick_name is None:
            return None, None
        target_name = self.get_placing_target_name(pick_name)
        if target_name is None:
            return None, None
        drop_orient = self.get_end_effector_orientation_for_drop(pick_name, target_name)
        _, target_p, _ = self.get_placing_info(pick_name, drop_orient)
        if target_p is None:
            return None, None
        # Also return the target object for dynamic XY alignment
        target_obj = self._strategy.target_objs_by_name.get(target_name)
        return target_p, target_obj

    # -------------------------------------------------------------------
    # Reset / reorder — delegated to strategy
    # -------------------------------------------------------------------

    def reset(self, picking_order_item_names: Optional[List[str]] = None) -> None:
        self._strategy.reset(picking_order_item_names)

    def reorder_picks(self, new_order_names: List[str], current_pick_name: Optional[str] = None) -> None:
        self._strategy.reorder_picks(new_order_names, current_pick_name)

    def update_pairings(self, pairings_by_pick_name: dict) -> None:
        self._strategy.update_pairings(pairings_by_pick_name)
