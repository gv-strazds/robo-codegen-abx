"""Cortex-style py_trees behaviours for pick-and-place.

Adapted from pt_experiments/pt_pickplace.py to use the existing TaskContextBase
API (arm_commander, gripper_commander, compute_* methods) rather than accessing
context.robot.arm / context.robot.suction_gripper directly.

Key differences from the time-interpolated behaviours in pt_pick_place_behaviours.py:
- Movement behaviours send MotionCommands and check robot_at_target() for completion
  (threshold-based) rather than stepping through time interpolation (t += dt).
- No blackboard intermediary — behaviours query TaskContextBase directly.
- Simpler structure: 4 movement phases vs 9 time-interpolated phases.
"""
import logging
import time
from typing import Optional

import numpy as np
import py_trees

from isaacsim.cortex.framework.motion_commander import PosePq

import perception_utils

logger = logging.getLogger(__name__)


class CortexActionBase(py_trees.behaviour.Behaviour):
    """Base class for cortex-style behaviours that use TaskContextBase.

    Receives ``context`` (TaskContextBase) via setup(), and exposes
    ``arm_commander`` and ``gripper_commander`` for convenience.
    """

    def __init__(self, name: str):
        super().__init__(name)
        self.context = None
        self.arm_commander = None
        self.gripper_commander = None

    def setup(self, **kwargs) -> None:
        self.logger.debug(f"  {self.name} [{type(self).__name__}::setup()]")
        if 'context' in kwargs:
            self.context = kwargs['context']
        else:
            raise RuntimeError(f"{type(self).__name__}.setup() missing REQUIRED kwarg: context")
        if 'arm_commander' in kwargs and kwargs['arm_commander'] is not None:
            self.arm_commander = kwargs['arm_commander']
        elif self.context is not None:
            self.arm_commander = self.context.arm_commander
        if 'gripper_commander' in kwargs and kwargs['gripper_commander'] is not None:
            self.gripper_commander = kwargs['gripper_commander']
        elif self.context is not None:
            self.gripper_commander = self.context.gripper_commander

    def initialise(self) -> None:
        self.logger.debug(f"  {self.name} [{type(self).__name__}::initialise()]")


class CortexMove(CortexActionBase):
    """Base movement behaviour: sends a MotionCommand and checks robot_at_target().

    Subclasses must set ``self.command`` (a MotionCommand) before calling
    ``super().update()``.  Returns RUNNING until the arm is within
    (p_thresh, R_thresh) of the target, then SUCCESS.

    Watchdog timeouts (formerly: 10 s warning + 15 s force-success
    baked into ``update``) live outside this class now — wrap with
    :func:`pt_sim_time_decorators.sim_timeout_to_success` at the
    tree-construction layer (see ``pt_cortex_tree.py``).  The timeout
    decorator surfaces a rich diagnostic on expiry via the
    :meth:`_timeout_diagnostic` callback and lets the parent Sequence
    advance via ``FailureIsSuccess``-on-FAILURE.
    """

    def __init__(self, name: str, p_thresh: float, R_thresh: float, fake_fast: bool = False):
        super().__init__(name)
        self.p_thresh = p_thresh
        self.R_thresh = R_thresh
        self.command = None
        self._fake_fast = fake_fast

    def _timeout_diagnostic(self) -> str:
        """Build a diagnostic string for timeout warnings.

        Used by the SimTimeout decorator wrapping this behaviour
        (passed as the ``on_timeout`` callback in the tree factory).
        Reports target pose, current FK pose, and distance — the
        information that lets a maintainer distinguish "RMPFlow is
        oscillating in the funnel" from "command target is wrong".
        """
        parts = []
        if self.command is not None and hasattr(self.command, 'target_pose') and self.command.target_pose is not None:
            target_p = self.command.target_pose.p
            parts.append(f"target_p={target_p}")
            try:
                fk_p = self.arm_commander.get_fk_p()
                dist = np.linalg.norm(fk_p - target_p)
                parts.append(f"fk_p={fk_p}, dist={dist:.4f}")
            except Exception:
                pass
        parts.append(f"p_thresh={self.p_thresh}, R_thresh={self.R_thresh}")
        return "; ".join(parts)

    def update(self) -> py_trees.common.Status:
        if self.command is None:
            self.logger.warning(f"  {self.name}: no command to send")
            return py_trees.common.Status.FAILURE

        if not hasattr(self.command, 'target_pose') or self.command.target_pose is None:
            logger.warning(f"{self.name}: command has no target_pose, motion will be a no-op")
            return py_trees.common.Status.FAILURE

        self.context.send_motion_command(self.command)

        if self._fake_fast:
            return py_trees.common.Status.SUCCESS

        if self.context.robot_at_target(self.command, p_thresh=self.p_thresh, R_thresh=self.R_thresh):
            return py_trees.common.Status.SUCCESS

        return py_trees.common.Status.RUNNING


class CortexMoveToPreGrasp(CortexMove):
    """Free-space move to the pre-grasp staging pose.

    Reads the cached ``GraspPose`` (populated by ``PrepareGrasp``) via
    ``context.compute_pregrasp_command_for_active_item`` and sends a
    MotionCommand with a *horizontal* approach funnel — direction is
    the unit XY vector from the live EE to the pre-grasp position.
    The funnel damps motion along the dominant transit axis without
    stacking vertical height (mirrors the pattern used by
    ``CortexMoveToPlace`` for transport-hover; see :pyattr:`update`).
    ``CortexExecuteApproach`` then handles the tight final descent.

    The looser ``p_thresh`` (1 cm) reflects the freespace nature: we
    do not need sub-millimetre precision here; the approach funnel in
    the following behaviour will tighten XY during descent.
    """

    # Loose lateral std-dev for the pre-grasp transport funnel.  Wider
    # than ``CortexMoveToPlace.MOVE_TO_PLACE_APPROACH_STD_DEV`` (0.04)
    # because pre-grasp staging has even less precision pressure than
    # transport hover — the tight final alignment belongs to
    # ``CortexExecuteApproach``.
    PRE_GRASP_APPROACH_STD_DEV = 0.05

    def __init__(self, name: str = "CortexMoveToPreGrasp", fake_fast: bool = False):
        super().__init__(name, p_thresh=0.01, R_thresh=2.0, fake_fast=fake_fast)

    def update(self) -> py_trees.common.Status:
        # Retarget each tick so the pre-grasp waypoint tracks an item that
        # may still be settling (e.g. a soup can that hasn't yet finished
        # falling into the bin at PrepareGrasp time).  Position-only refresh
        # — orientation and approach plan stay frozen at initial grasp.
        self.context.refresh_grasp_pose_position()

        # Horizontal approach direction from live EE XY to the pre-grasp
        # XY.  Falls back to a freespace move (no funnel) when the EE is
        # essentially already at the pre-grasp XY — at that point there
        # is no transport velocity to damp.  Mirrors the live-direction
        # block in ``CortexMoveToPlace.update``.
        approach_dir = None
        use_funnel = False
        try:
            pre_grasp_p = self.context.get_pre_grasp_position()
            if pre_grasp_p is not None:
                fk_p = self.arm_commander.get_fk_p()
                dir_xy = pre_grasp_p[:2] - fk_p[:2]
                n = float(np.linalg.norm(dir_xy))
                if n > 1e-3:
                    approach_dir = np.array([dir_xy[0] / n, dir_xy[1] / n, 0.0])
                    use_funnel = True
        except Exception:
            pass

        # Floor the pre-grasp Z to the live transport altitude so the
        # freespace transit from the previous lift_after_place to the
        # next pick area stays at altitude (no sloping descent that
        # would clip a tall destination stack).  CortexExecuteApproach
        # then descends from this altitude into the grasp via the
        # vertical funnel.
        self.command = self.context.compute_pregrasp_command_for_active_item(
            use_approach_funnel=use_funnel,
            approach_direction=approach_dir,
            approach_std_dev=self.PRE_GRASP_APPROACH_STD_DEV,
            min_z=self.context.get_ee_height_for_move(),
        )
        if self.command is None:
            pick_name = self.context.get_current_pick_name()
            if self.context.strategy.more_items_expected:
                logger.debug(
                    f"{self.name}: no current pick (pick='{pick_name}') — waiting"
                )
                return py_trees.common.Status.RUNNING
            logger.warning(
                f"{self.name}: compute_pregrasp_command returned None (pick='{pick_name}')"
            )
            return py_trees.common.Status.FAILURE
        # Stash the committed pick name (matches CortexMoveToPick behaviour)
        # so LatchCurrentPick latches the right item once the gripper closes.
        self.context.strategy.committed_pick_name = self.context.get_current_pick_name()
        return super().update()


class CortexExecuteApproach(CortexMove):
    """Final grasp approach: move from pre-grasp to the grasp pose.

    Reads the same cached ``GraspPose`` as ``CortexMoveToPreGrasp`` via
    ``context.compute_pick_command_for_active_item``.  Tight ``p_thresh``
    (2 mm) and the grasp-time approach funnel.  Intended to run immediately
    after ``CortexMoveToPreGrasp`` so RMPFlow can exploit the straight-line
    final segment.
    """

    def __init__(self, name: str = "CortexExecuteApproach", fake_fast: bool = False):
        # 5 mm default p_thresh suits moving-conveyor picks where RMPFlow
        # has a small steady-state tracking error.  Tasks that need a flush
        # gripper close (e.g. socket-insertion placements that propagate
        # any suction-tip air gap into a misaligned drop) override via
        # TaskSpec.pick_approach_p_thresh — see TaskContextBase.get_pick_approach_p_thresh.
        # The override is applied each time the behaviour is initialised
        # (initialise() below), not at construction time, because the
        # context isn't bound until setup().
        super().__init__(name, p_thresh=0.005, R_thresh=2.0, fake_fast=fake_fast)

    def initialise(self) -> None:
        super().initialise()
        if self.context is not None:
            # Refresh per-attempt so a TaskSpec hot-swap or context
            # replacement is reflected immediately.  CortexMove.update
            # reads self.p_thresh each tick.
            self.p_thresh = self.context.get_pick_approach_p_thresh()

    def update(self) -> py_trees.common.Status:
        # Retarget each tick (position only — the approach direction /
        # distance / std-dev stay frozen at the initial PrepareGrasp
        # values).  First tick satisfies "refresh before starting the
        # approach"; subsequent ticks let the funnel track a still-
        # settling item as RMPFlow descends.
        self.context.refresh_grasp_pose_position()
        self.command = self.context.compute_pick_command_for_active_item()
        if self.command is None:
            pick_name = self.context.get_current_pick_name()
            if self.context.strategy.more_items_expected:
                logger.debug(
                    f"{self.name}: no current pick (pick='{pick_name}') — waiting"
                )
                return py_trees.common.Status.RUNNING
            logger.warning(
                f"{self.name}: compute_pick_command returned None (pick='{pick_name}')"
            )
            return py_trees.common.Status.FAILURE
        # Refresh committed pick stash each tick (the pre-grasp behaviour
        # already set it, but re-stashing is safe and keeps the pattern
        # consistent with the legacy CortexMoveToPick).
        self.context.strategy.committed_pick_name = self.context.get_current_pick_name()
        return super().update()


class CortexMoveToPlace(CortexMove):
    """Move above the placement target for the current active item.

    Refreshes (target_p, target_obj) from the context on every tick so the
    commanded pose tracks a target that may be moving (e.g. an object
    riding a conveyor).  The MotionCommand is recomputed each tick with
    the live target position.

    Horizontal approach funnel: the approach direction is the horizontal
    unit vector from the live EE XY to the target XY, so RMPFlow biases
    and damps motion along the dominant travel axis rather than along
    -Z.  Because the direction has no vertical component, the funnel
    does not stack ``approach_distance`` on top of ``above`` — peak EE
    Z stays at ``insert_z + above`` — so we keep the "no extra hover
    height" property of the original freespace design while gaining
    along-direction deceleration that prevents overshoot/pull-back at
    the seam into ``CortexDownToInsert``.

    A wider ``approach_std_dev`` (0.04 m vs the 0.02 m default) is used
    here because this is transport hover, not a precision insertion —
    the subsequent ``CortexDownToInsert`` owns the tight final XY
    alignment and vertical descent.

    Hover altitude is governed by two context-driven knobs (no
    behaviour-level overrides):

    * ``context.get_place_hover_above_z()`` — the relative Z above the
      live target (already clamped to be ``>=`` the place-side
      approach-funnel length so the descent funnel always engages from
      above).  Per-task override flows through
      ``TaskSpec.place_hover_above_z``.

    * ``context.get_ee_height_for_move()`` — an absolute floor on the
      commanded EE Z.  When the natural ``target_z + hover_above`` is
      below this floor, the commanded above-target offset is *raised*
      so the EE transports at the configured altitude (parallel to the
      ``CortexMoveRelative`` lift cap).  Per-task override flows
      through ``TaskSpec.ee_height_for_move``.

    The null-space posture config is supplied by the context (via
    ``TaskContextBase.get_posture_config("place")``) and no longer
    threaded through behaviour instance state.
    """

    # Wider lateral std-dev for the transport-hover funnel.  Tight
    # alignment belongs to CortexDownToInsert; a narrow funnel here
    # would slow transport by rejecting legitimate mid-flight poses.
    MOVE_TO_PLACE_APPROACH_STD_DEV = 0.04

    def __init__(self, name: str = "CortexMoveToPlace", fake_fast: bool = False):
        super().__init__(name, p_thresh=0.02, R_thresh=2.0, fake_fast=fake_fast)
        self.target_p = None
        self.target_obj = None

    def initialise(self) -> None:
        super().initialise()
        self.target_p, self.target_obj = self.context.get_placement_target()

    def update(self) -> py_trees.common.Status:
        self.target_p, self.target_obj = self.context.get_placement_target()
        if self.target_p is None:
            pick_name = self.context.get_current_pick_name()
            # Hold the carried item rather than dropping it while more
            # targets are still scheduled to arrive (e.g. conveyor batches).
            # The last commanded MotionCommand remains active so RMPFlow
            # keeps the arm in place until a target becomes available.
            if self.context.strategy.more_targets_expected:
                logger.debug(
                    f"{self.name}: no placement target (pick='{pick_name}') "
                    f"— waiting for more targets"
                )
                return py_trees.common.Status.RUNNING
            logger.warning(f"{self.name}: no placement target position (pick='{pick_name}')")
            return py_trees.common.Status.FAILURE

        # Horizontal approach direction from live EE XY to target XY.
        # Damps motion along the dominant travel axis without stacking
        # vertical height.  Falls back to the default (-Z) direction
        # when EE is essentially already at the target XY — at that
        # point there is no transport velocity to damp.
        approach_dir = None
        use_funnel = False
        try:
            fk_p = self.arm_commander.get_fk_p()
            dir_xy = self.target_p[:2] - fk_p[:2]
            n = float(np.linalg.norm(dir_xy))
            if n > 1e-3:
                approach_dir = np.array([dir_xy[0] / n, dir_xy[1] / n, 0.0])
                use_funnel = True
        except Exception:
            pass

        # Effective above-target offset: at least the configured relative
        # hover, and at least enough to lift the absolute commanded Z to
        # ee_height_for_move.  Both knobs are per-task tunable via
        # TaskSpec; the floor logic here parallels the lift-cap behaviour
        # in CortexMoveRelative(cap_to_ee_height_for_move=True).
        hover_above = self.context.get_place_hover_above_z()
        ee_height_floor = self.context.get_ee_height_for_move()
        target_z = float(self.target_p[2])
        effective_above = max(hover_above, ee_height_floor - target_z)

        self.command = self.context.compute_dynamic_place_command_for_active_item(
            self.target_p, self.target_obj, above=effective_above,
            use_approach_funnel=use_funnel,
            approach_direction=approach_dir,
            approach_distance=self.context.get_place_approach_distance(),
            approach_std_dev=self.MOVE_TO_PLACE_APPROACH_STD_DEV,
        )
        if self.command is None:
            pick_name = self.context.get_current_pick_name()
            logger.warning(f"{self.name}: compute_dynamic_place_command returned None (pick='{pick_name}')")
            return py_trees.common.Status.FAILURE
        return super().update()


class CortexDownToInsert(CortexMove):
    """Lower into the placement target (insertion move).

    Like CortexMoveToPlace but lowering to the exact placement Z
    (``above=0.0``) with a tighter position threshold.  The placement Z
    from ``get_placing_info`` already accounts for target surface height
    and object rest height, so no extra offset is needed.

    Also like CortexMoveToPlace, refreshes (target_p, target_obj) on
    every tick so the descent tracks a moving target (e.g. a target
    riding a conveyor).

    Z-axis tolerance vs XY tolerance:
        ``p_thresh`` is a 3D-distance check, which conflates XY tracking
        lag with Z descent error.  For descent the *Z component* is what
        matters — releasing the held item is meaningful only when the
        item is genuinely close to the target's top surface — so this
        behaviour adds a stricter ``z_thresh`` requirement on top of the
        loose 3D check.  Setting ``z_thresh`` to a small value (e.g.
        0.005 m) prevents SUCCESS from firing while the EE is still
        hovering well above the commanded descent Z.

    Args:
        z_thresh: Maximum allowed |fk_z - cmd_z| to accept SUCCESS.  When
            ``None`` (default) it tracks ``p_thresh``, preserving legacy
            behaviour.  Set to a tighter value (e.g. 0.005 m) for tasks
            where the held item must be released very close to the target
            top surface to avoid bounce or rolling.

    The null-space posture config is supplied by the context (via
    ``TaskContextBase.get_posture_config("place")``) and no longer
    threaded through behaviour instance state.
    """

    def __init__(self, name: str = "CortexDownToInsert", above: float = 0.0,
                 loose_fit: bool = False, fake_fast: bool = False,
                 z_thresh: float = None,
                 approach_std_dev: Optional[float] = None,
                 approach_distance: Optional[float] = None,
                 use_approach_funnel: bool = True,
                 gap_log_interval_s: float = 1.0):
        p_thresh = 0.02 if loose_fit else 0.01
        super().__init__(name, p_thresh=p_thresh, R_thresh=2.0, fake_fast=fake_fast)
        self.above = above
        self.z_thresh = z_thresh if z_thresh is not None else p_thresh
        self.target_p = None
        self.target_obj = None
        # Place-approach tuning knobs.  None → use perception_utils defaults
        # (DEFAULT_PLACE_APPROACH_DISTANCE=0.20, DEFAULT_PLACE_APPROACH_STD_DEV=0.02).
        # Widen std_dev when the cone appears to be rejecting the descent on
        # a moving target (bottle/can keeps bouncing out of the funnel).
        self._approach_std_dev = approach_std_dev
        self._approach_distance = approach_distance
        self._use_approach_funnel = use_approach_funnel
        # Per-tick (fk_z, cmd_z) diagnostic throttled to this wall-clock
        # interval.  Set to 0 to log every tick, or a large value to silence.
        self._gap_log_interval_s = float(gap_log_interval_s)
        self._last_gap_log_time = 0.0

    def initialise(self) -> None:
        super().initialise()
        self.target_p, self.target_obj = self.context.get_placement_target()
        # Wall-clock start time used only by the gap-log diagnostic (the
        # control-flow watchdog lives in a SimTimeout decorator wrapping
        # this behaviour at the tree-construction layer; see
        # pt_cortex_tree.make_cortex_task_controller_tree).
        now = time.time()
        self._last_gap_log_time = now
        self._descent_start_wall_t = now

    def update(self) -> py_trees.common.Status:
        self.target_p, self.target_obj = self.context.get_placement_target()
        if self.target_p is None:
            pick_name = self.context.get_current_pick_name()
            if self.context.strategy.more_targets_expected:
                logger.debug(
                    f"{self.name}: no placement target (pick='{pick_name}') "
                    f"— waiting for more targets"
                )
                return py_trees.common.Status.RUNNING
            logger.warning(f"{self.name}: no placement target position (pick='{pick_name}')")
            return py_trees.common.Status.FAILURE
        # Approach-distance precedence:
        #   1. Explicit constructor kwarg (factory ``down_to_insert_approach_distance``)
        #   2. Per-task override on the context (TaskSpec.place_approach_distance)
        #   3. perception_utils.DEFAULT_PLACE_APPROACH_DISTANCE (the context's own fallback)
        # The context's get_place_approach_distance() handles (2) and (3); we only
        # short-circuit when (1) was supplied.
        approach_distance = (
            self._approach_distance if self._approach_distance is not None
            else self.context.get_place_approach_distance()
        )
        self.command = self.context.compute_dynamic_place_command_for_active_item(
            self.target_p, self.target_obj, above=self.above,
            use_approach_funnel=self._use_approach_funnel,
            approach_distance=approach_distance,
            approach_std_dev=self._approach_std_dev,
        )
        if self.command is None:
            pick_name = self.context.get_current_pick_name()
            logger.warning(f"{self.name}: compute_dynamic_place_command returned None (pick='{pick_name}')")
            return py_trees.common.Status.FAILURE

        # Throttled per-tick gap diagnostic.  Reveals whether the descent is
        # slowly closing (posture-bias drag → fk_z trending down) or
        # oscillating near a fixed Z (cone-rejection retry → fk_z bouncing
        # back up every few samples).
        #
        # Also logs the held item's pose vs the live target socket pose so
        # we can distinguish "EE can't reach target" (control failure) from
        # "bottle is physically resting on the socket rim" (collision
        # blocking the descent).  In the collision case the bottle is
        # approximately stationary above the socket regardless of what the
        # EE / cmd are doing, and item_z - target_z is the rim height.
        if not self._fake_fast and self._gap_log_interval_s >= 0:
            now = time.time()
            if now - self._last_gap_log_time >= self._gap_log_interval_s:
                self._last_gap_log_time = now
                try:
                    fk_p = self.arm_commander.get_fk_p()
                    cmd_p = self.command.target_pose.p
                    z_gap = float(fk_p[2] - cmd_p[2])
                    xy_gap = float(np.linalg.norm(fk_p[:2] - cmd_p[:2]))
                    elapsed = now - self._descent_start_wall_t
                    item_diag = ""
                    try:
                        pick_name = self.context.get_current_pick_name()
                        pick_obj = (
                            self.context.strategy.pick_objs_by_name.get(pick_name)
                            if pick_name is not None else None
                        )
                        if pick_obj is not None and self.target_obj is not None:
                            item_p, _ = pick_obj.get_world_pose()
                            tgt_p, _ = self.target_obj.get_world_pose()
                            item_p = np.asarray(item_p, dtype=float)
                            tgt_p = np.asarray(tgt_p, dtype=float)
                            d_xy = float(np.linalg.norm(item_p[:2] - tgt_p[:2]))
                            d_z = float(item_p[2] - tgt_p[2])
                            item_diag = (
                                f" item_p=[{item_p[0]:.4f}, {item_p[1]:.4f}, {item_p[2]:.4f}]"
                                f" tgt_p=[{tgt_p[0]:.4f}, {tgt_p[1]:.4f}, {tgt_p[2]:.4f}]"
                                f" item_vs_tgt: d_xy={d_xy:.4f} d_z={d_z:+.4f}"
                            )
                    except Exception:
                        pass
                    logger.info(
                        f"{self.name}: descent t={elapsed:.1f}s fk_z={fk_p[2]:.4f} "
                        f"cmd_z={cmd_p[2]:.4f} z_gap={z_gap:+.4f} xy_gap={xy_gap:.4f}"
                        f"{item_diag}"
                    )
                except Exception:
                    pass

        status = super().update()
        # Apply the stricter Z-axis check on top of the 3D distance check.
        # If the EE is still well above the commanded descent Z, demote
        # the SUCCESS back to RUNNING so the cortex motion commander
        # keeps pressing toward the target.  Always demote — the
        # SimTimeout decorator wrapping this behaviour at the tree
        # layer is the watchdog that breaks an unrecoverable descent.
        if status == py_trees.common.Status.SUCCESS and not self._fake_fast:
            try:
                fk_z = float(self.arm_commander.get_fk_p()[2])
                cmd_z = float(self.command.target_pose.p[2])
                if abs(fk_z - cmd_z) > self.z_thresh:
                    status = py_trees.common.Status.RUNNING
            except Exception:
                pass
        if status == py_trees.common.Status.SUCCESS and not self._fake_fast:
            try:
                joints = self.context.get_joint_positions()
                pick_name = self.context.get_current_pick_name()
                # XY-diag: log world coords of held item, live target, EE, and commanded EE.
                def _fmt(p):
                    return "[" + ", ".join(f"{v:.4f}" for v in p) + "]"
                pick_p_s = "n/a"
                try:
                    pick_obj = self.context._strategy.pick_objs_by_name.get(pick_name)
                    if pick_obj is not None:
                        pick_p, _ = pick_obj.get_world_pose()
                        pick_p_s = _fmt(pick_p)
                except Exception:
                    pass
                tgt_p_s = "n/a"
                try:
                    if self.target_obj is not None:
                        tgt_p, _ = self.target_obj.get_world_pose()
                        tgt_p_s = _fmt(tgt_p)
                    elif self.target_p is not None:
                        tgt_p_s = _fmt(self.target_p)
                except Exception:
                    pass
                fk_p_s = "n/a"
                try:
                    fk_p_s = _fmt(self.arm_commander.get_fk_p())
                except Exception:
                    pass
                cmd_p_s = "n/a"
                try:
                    if self.command is not None and self.command.target_pose is not None:
                        cmd_p_s = _fmt(self.command.target_pose.p)
                except Exception:
                    pass
                logger.info(
                    f"{self.name}: SUCCESS joint_angles=[{', '.join(f'{j:.6f}' for j in joints[:6])}] "
                    f"pick={pick_name} item_p={pick_p_s} target_p={tgt_p_s} "
                    f"fk_p={fk_p_s} cmd_p={cmd_p_s}"
                )
            except Exception:
                pass
        return status


class CortexMoveRelative(CortexMove):
    """Move the end-effector by a relative offset from current position.

    On initialise(), computes the target PosePq from context.get_relative_pq().
    Each tick sends the resulting MotionCommand.

    Two optional clamps re-shape the target Z relative to
    ``context.get_ee_height_for_move()`` (the configured transport altitude):

    * ``cap_to_ee_height_for_move=True`` — *floor* clamp.  The lift target
      Z is raised to at least ``ee_height_for_move`` when the natural
      offset would land below it.  Use this on the post-pick lift so a
      low grasp still rises to transport altitude before the next move.

    * ``cap_max_to_ee_height_for_move=True`` — *ceiling* clamp.  The lift
      target Z is clipped to no more than ``ee_height_for_move`` when the
      natural offset would land above it.  Use this on the post-place
      lift so placing on a tall stack does not push the wrist beyond
      reach (``natural = place_z + grasp_height + relative_offset`` can
      exceed the UR10's reach for tall stacks; clamping to the
      already-cleared transport altitude is sufficient).

    With both flags True at the same site, the lift target collapses to
    *exactly* ``ee_height_for_move`` regardless of the natural offset —
    the same semantic the default tree uses for every horizontal move.
    With both False (the default), the behaviour is the legacy "lift by
    the relative offset, no clamping" pattern.
    """

    def __init__(self, name: str, offset: np.ndarray, use_world_frame: bool = True,
                 fake_fast: bool = False, cap_to_ee_height_for_move: bool = False,
                 cap_max_to_ee_height_for_move: bool = False):
        super().__init__(name, p_thresh=0.02, R_thresh=2.0, fake_fast=fake_fast)
        self.offset = offset
        self.use_world_frame = use_world_frame
        self.cap_to_ee_height_for_move = cap_to_ee_height_for_move
        self.cap_max_to_ee_height_for_move = cap_max_to_ee_height_for_move
        self.target_pq = None

    def initialise(self) -> None:
        super().initialise()
        self.target_pq = self.context.get_relative_pq(
            offset=self.offset, use_world_frame=self.use_world_frame,
        )
        if self.target_pq is None:
            return
        # Read the transport altitude once — both clamps consult the same value.
        ee_height = (
            self.context.get_ee_height_for_move()
            if (self.cap_to_ee_height_for_move
                or self.cap_max_to_ee_height_for_move)
            else None
        )
        if ee_height is None:
            return

        natural_z = float(self.target_pq.p[2])
        capped_z = natural_z
        if self.cap_to_ee_height_for_move and capped_z < ee_height:
            capped_z = float(ee_height)
        if self.cap_max_to_ee_height_for_move and capped_z > ee_height:
            capped_z = float(ee_height)

        if capped_z != natural_z:
            new_p = np.array(self.target_pq.p, dtype=float).copy()
            new_p[2] = capped_z
            self.target_pq = PosePq(new_p, self.target_pq.q)
            direction = "raised" if capped_z > natural_z else "clipped"
            logger.debug(
                f"{self.name}: lift target z {direction} to ee_height_for_move "
                f"({ee_height:.3f}) from natural {natural_z:.3f}"
            )

    def update(self) -> py_trees.common.Status:
        if self.target_pq is None:
            logger.warning(f"{self.name}: get_relative_pq returned None (offset={self.offset})")
            return py_trees.common.Status.FAILURE
        self.command = self.context.compute_motion_command_to_target(self.target_pq)
        if self.command is None:
            logger.warning(f"{self.name}: compute_motion_command_to_target returned None")
            return py_trees.common.Status.FAILURE
        return super().update()


class CortexCloseGripper(CortexActionBase):
    """Close the gripper.  Returns SUCCESS immediately."""

    def update(self) -> py_trees.common.Status:
        self.gripper_commander.close()
        return py_trees.common.Status.SUCCESS


class CortexOpenGripper(CortexActionBase):
    """Open the gripper.  Returns SUCCESS immediately."""

    def update(self) -> py_trees.common.Status:
        self.gripper_commander.open()
        return py_trees.common.Status.SUCCESS
