"""py_trees Behaviour classes for the 9 phases of a pick-place cycle.

Each behaviour corresponds to one phase of the pick-place sequence.
They share data via the py_trees blackboard (namespace /pickplace/) and
send commands to the robot via an IArmCommander / IGripperCommander
received during ``setup()``.
"""
import logging
from typing import Optional

import numpy as np
import py_trees

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sim-time-based phase progression
# ---------------------------------------------------------------------------

# Per-tick advance for ``self.t`` is ``sim_dt * BT_TICK_REFERENCE_HZ / num_steps``,
# so each phase completes in ``num_steps / BT_TICK_REFERENCE_HZ`` sim seconds
# regardless of physics_dt and psteps_per_render.  The reference rate is set
# to 60 Hz to match the BT tick rate of the original (pre-PHYSICS_DT /
# RENDERING_DT separation) loop, which ticked the BT once per physics step
# at the IsaacSim default physics_dt = 1/60.  Existing per-phase
# ``num_steps`` tunings therefore keep their original sim-time durations.
BT_TICK_REFERENCE_HZ = 60.0


def _try_get_sim_time() -> Optional[float]:
    """Return World.instance().current_time, or None if no live World.

    The mock test path has no live World — returning None makes
    ``PickPlaceBehaviour.update`` fall back to the fixed per-tick ``self.dt``
    so existing tests keep their tick-counted semantics.
    """
    try:
        from isaacsim.core.api.world import World
        world = World.instance()
        if world is not None:
            return float(world.current_time)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Interpolation helper class
# ---------------------------------------------------------------------------

class SinusoidalInterpolator:
    """Smooth sinusoidal interpolation from a fixed start to a dynamic end.

    Uses cosine easing: 0 at t=0, 1 at t=1, with smooth acceleration
    and deceleration.  Works with scalars and numpy arrays.

    Construct once (typically in ``initialise()``) with the start value,
    then call ``evaluate(end, t)`` each tick with the current target.
    """

    def __init__(self, start):
        self.start = start

    def evaluate(self, end, t):
        """Return interpolated value at time *t* toward *end* (t clamped to [0, 1])."""
        alpha = 0.5 * (1 - np.cos(max(0.0, t) * np.pi))
        return (1 - alpha) * self.start + alpha * end


# ---------------------------------------------------------------------------
# Keys written/read on the /pickplace/ blackboard namespace
# ---------------------------------------------------------------------------

# Written by ContextMonitorBehaviour each tick
INPUT_KEYS = [
    "picking_position",
    "placing_position",
    "current_joint_positions",
    "end_effector_offset",
    "end_effector_orientation",
    "end_effector_offset_for_drop",
    "end_effector_orientation_for_drop",
    "ee_height_for_move",
]

# Written by CloseGripper (latched at grasp time), read by LiftPicked / MoveToPlaceXY
PICK_POSITION_KEY = "pick_position"

# Phase 0: MoveToPickXY
# Phase 1: LowerToPick
# Phase 2: WaitSettling
# Phase 3: CloseGripper
# Phase 4: LiftPicked
# Phase 5: MoveToPlaceXY
# Phase 6: LowerToPlace
# Phase 7: OpenGripper
# Phase 8: LiftAfterPlace
PICKPLACE_PHASE_DTs = {
    'MoveToPickXY': 0.01,       # num_steps=100
    'LowerToPick':  0.005,      # num_steps=200
    'WaitSettling': 0.1,        # num_steps=10
    'CloseGripper': 1.0,        # num_steps=1
    'LiftPicked':   0.008,      # num_steps=125
    'MoveToPlaceXY': 0.005,     # num_steps=200
    'LowerToPlace':  0.005,     # num_steps=200
    'OpenGripper':   1.0,       # num_steps=1
    'LiftAfterPlace': 0.08,     # num_steps=13
}

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class PickPlaceBehaviour(py_trees.behaviour.Behaviour):
    """Base class for pick-place phase behaviours.

    Handles the progression pattern shared by all phases: each tick
    advances an internal phase-progress ``t`` toward 1.0.  While
    ``t < 1`` the behaviour returns RUNNING; once ``t >= 1`` the next
    ``update`` returns SUCCESS without sending a new command.

    Per-tick advance defaults to ``sim_dt * BT_TICK_REFERENCE_HZ / num_steps``
    when a live IsaacSim ``World`` is present (sim-time-based — phase
    duration is constant in sim seconds regardless of ``physics_dt`` or
    ``psteps_per_render``).  Falls back to the legacy fixed
    ``1.0 / num_steps`` per tick when no World is live (mock tests) or
    when ``fake_fast`` is set.

    Commands are sent directly to an IArmCommander / IGripperCommander
    received via ``setup(arm_commander=..., gripper_commander=...)``.
    """

    def __init__(self, name: str, num_steps: int, fake_fast=False):
        super().__init__(name=name)
        self._num_steps = num_steps
        self._fake_fast = fake_fast
        if fake_fast:
            self.dt = 0.5 if num_steps > 1 else 1.0
        else:
            self.dt = 1.0 / num_steps
        self.t = 0.0
        self._last_tick_sim_time: Optional[float] = None

        # Blackboard client (namespace="/pickplace")
        self.bb = self.attach_blackboard_client(
            name=f"{self.name}", namespace="/pickplace"
        )
        for key in INPUT_KEYS:
            self.bb.register_key(key=key, access=py_trees.common.Access.READ)

        # Commander interfaces, set via setup()
        self._arm_commander = None
        self._gripper_commander = None

    def setup(self, **kwargs) -> None:
        """Receive arm_commander and gripper_commander via kwargs."""
        if "arm_commander" in kwargs:
            self._arm_commander = kwargs["arm_commander"]
        if "gripper_commander" in kwargs:
            self._gripper_commander = kwargs["gripper_commander"]

    def initialise(self) -> None:
        """Reset phase progress at the start of each activation."""
        self.t = 0.0
        self._last_tick_sim_time = None

    def _compute_tick_increment(self) -> float:
        """Per-tick advance for ``self.t``.

        Sim-time-aware so phase duration in sim seconds is invariant
        across ``physics_dt`` and ``psteps_per_render``.  Falls back to
        the fixed ``self.dt`` (legacy tick-counted) when ``fake_fast`` is
        set or no live World is available (mock tests).
        """
        if self._fake_fast:
            return self.dt
        now = _try_get_sim_time()
        if now is None:
            return self.dt
        if self._last_tick_sim_time is None:
            # First tick after initialise — record the sim time so we can
            # finite-difference on subsequent ticks, but don't advance
            # progress (we'd have to assume a tick rate, and getting that
            # wrong over-credits or under-credits at non-default rates).
            self._last_tick_sim_time = now
            return 0.0
        sim_dt = now - self._last_tick_sim_time
        self._last_tick_sim_time = now
        if sim_dt <= 0.0:
            # Sim time hasn't advanced (e.g. paused, or two BT ticks
            # within one physics step) — don't advance phase progress.
            return 0.0
        return sim_dt * BT_TICK_REFERENCE_HZ / self._num_steps

    def update(self) -> py_trees.common.Status:
        """Send command, advance phase progress.

        Returns RUNNING while the phase is active.  Returns SUCCESS once
        the phase has completed *and* its final command has been sent on
        a previous tick.
        """
        if self.t >= 1.0:
            return py_trees.common.Status.SUCCESS

        self.send_command()
        self.t += self._compute_tick_increment()
        return py_trees.common.Status.RUNNING

    def send_command(self) -> None:
        """Override in subclasses to send the appropriate command."""
        raise NotImplementedError

    def _send_ee_target(self, xy, height, offset, orientation) -> None:
        """Build EE position target from components and send to arm commander."""
        from isaacsim.cortex.framework.motion_commander import MotionCommand, PosePq

        position_target = np.array([
            xy[0] + offset[0],
            xy[1] + offset[1],
            height + offset[2],
        ])
        cmd = MotionCommand(target_pose=PosePq(position_target, orientation))
        self._arm_commander.send_motion_command(cmd)


# ---------------------------------------------------------------------------
# Phase 0: MoveToPickXY
# ---------------------------------------------------------------------------

class MoveToPickXYBehaviour(PickPlaceBehaviour):
    """Move end-effector above the pick item's center at move height."""

    def send_command(self) -> None:
        self._send_ee_target(
            xy=self.bb.picking_position[:2],
            height=self.bb.ee_height_for_move,
            offset=self.bb.end_effector_offset,
            orientation=self.bb.end_effector_orientation,
        )


# ---------------------------------------------------------------------------
# Phase 1: LowerToPick
# ---------------------------------------------------------------------------

class LowerToPickBehaviour(PickPlaceBehaviour):
    """Lower end-effector to encircle the pick item."""

    def initialise(self) -> None:
        super().initialise()
        self._h_interp = SinusoidalInterpolator(self.bb.ee_height_for_move)

    def send_command(self) -> None:
        h = self._h_interp.evaluate(self.bb.picking_position[2], self.t)
        self._send_ee_target(
            xy=self.bb.picking_position[:2],
            height=h,
            offset=self.bb.end_effector_offset,
            orientation=self.bb.end_effector_orientation,
        )


# ---------------------------------------------------------------------------
# Phase 2: WaitSettling
# ---------------------------------------------------------------------------

class WaitSettlingBehaviour(PickPlaceBehaviour):
    """Wait for robot's inertia to settle before grasping.

    Sends no command — the arm holds its current position.
    """

    def send_command(self) -> None:
        pass  # No command needed; arm holds position


# ---------------------------------------------------------------------------
# Phase 3: CloseGripper
# ---------------------------------------------------------------------------

class CloseGripperBehaviour(PickPlaceBehaviour):
    """Close gripper to grasp the item.

    Latches pick_position from picking_position at grasp time, recording
    the object's location for use by later interpolation phases.
    """

    def __init__(self, name: str, num_steps: int, fake_fast: bool = False):
        super().__init__(name=name, num_steps=num_steps, fake_fast=fake_fast)
        self.bb.register_key(key=PICK_POSITION_KEY, access=py_trees.common.Access.WRITE)

    def initialise(self) -> None:
        super().initialise()
        self.bb.pick_position = self.bb.picking_position.copy()

    def send_command(self) -> None:
        self._gripper_commander.close()


# ---------------------------------------------------------------------------
# Phase 4: LiftPicked
# ---------------------------------------------------------------------------

class LiftPickedBehaviour(PickPlaceBehaviour):
    """Lift end-effector upward while maintaining grip.

    By default, holds XY at the latched grasp position
    (``bb.pick_position``) so the EE rises straight up regardless of
    whether the original item location is moving (e.g. a conveyor surface
    drags the held item along).  Set
    ``track_picked_item_during_lift=True`` to follow the *live* item XY
    during the lift — useful only when the gripped item is still being
    carried by a moving surface and the EE needs to track that motion
    while lifting (no current task requires this).
    """

    def __init__(self, name: str, num_steps: int, fake_fast: bool = False,
                 track_picked_item_during_lift: bool = False):
        super().__init__(name=name, num_steps=num_steps, fake_fast=fake_fast)
        self.bb.register_key(key=PICK_POSITION_KEY, access=py_trees.common.Access.READ)
        self._track_live_xy = bool(track_picked_item_during_lift)

    def initialise(self) -> None:
        super().initialise()
        self._h_interp = SinusoidalInterpolator(self.bb.pick_position[2])

    def send_command(self) -> None:
        h = self._h_interp.evaluate(self.bb.ee_height_for_move, self.t)
        xy = (self.bb.picking_position[:2] if self._track_live_xy
              else self.bb.pick_position[:2])
        self._send_ee_target(
            xy=xy,
            height=h,
            offset=self.bb.end_effector_offset,
            orientation=self.bb.end_effector_orientation,
        )


# ---------------------------------------------------------------------------
# Phase 5: MoveToPlaceXY
# ---------------------------------------------------------------------------

class MoveToPlaceXYBehaviour(PickPlaceBehaviour):
    """Smoothly move end-effector from pick to place XY at move height."""

    def __init__(self, name: str, num_steps: int, fake_fast: bool = False):
        super().__init__(name=name, num_steps=num_steps, fake_fast=fake_fast)
        self.bb.register_key(key=PICK_POSITION_KEY, access=py_trees.common.Access.READ)

    def initialise(self) -> None:
        super().initialise()
        self._xy_interp = SinusoidalInterpolator(
            self.bb.pick_position[:2]
        )

    def send_command(self) -> None:
        xy_target = self._xy_interp.evaluate(
            np.array([self.bb.placing_position[0], self.bb.placing_position[1]]),
            self.t,
        )
        self._send_ee_target(
            xy=xy_target,
            height=self.bb.ee_height_for_move,
            offset=self.bb.end_effector_offset_for_drop,
            orientation=self.bb.end_effector_orientation_for_drop,
        )


# ---------------------------------------------------------------------------
# Phase 6: LowerToPlace
# ---------------------------------------------------------------------------

class LowerToPlaceBehaviour(PickPlaceBehaviour):
    """Lower end-effector to placement height."""

    def initialise(self) -> None:
        super().initialise()
        self._h_interp = SinusoidalInterpolator(self.bb.ee_height_for_move)

    def send_command(self) -> None:
        h = self._h_interp.evaluate(self.bb.placing_position[2], self.t)
        self._send_ee_target(
            xy=self.bb.placing_position[:2],
            height=h,
            offset=self.bb.end_effector_offset_for_drop,
            orientation=self.bb.end_effector_orientation_for_drop,
        )


# ---------------------------------------------------------------------------
# Phase 7: OpenGripper
# ---------------------------------------------------------------------------

class OpenGripperBehaviour(PickPlaceBehaviour):
    """Open gripper to release the held item."""

    def send_command(self) -> None:
        self._gripper_commander.open()


# ---------------------------------------------------------------------------
# Phase 8: LiftAfterPlace
# ---------------------------------------------------------------------------

class LiftAfterPlaceBehaviour(PickPlaceBehaviour):
    """Lift end-effector after releasing the item."""

    def send_command(self) -> None:
        self._send_ee_target(
            xy=self.bb.placing_position[:2],
            height=self.bb.ee_height_for_move,
            offset=self.bb.end_effector_offset_for_drop,
            orientation=self.bb.end_effector_orientation_for_drop,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_pick_place_sequence(
    fake_fast: bool = False,
    track_picked_item_during_lift: bool = False,
) -> py_trees.composites.Sequence:
    """Build the pick-place behavior sequence.

    Args:
        fake_fast: Use unrealistically quick behavior completion times (for testing).
        track_picked_item_during_lift: When True, ``LiftPickedBehaviour``
            follows the live world XY of the picked item during the lift.
            Default False — the lift holds the latched grasp XY so the EE
            rises straight up regardless of whether the original item
            location is moving (e.g. a conveyor surface drags the held
            item along).  See :class:`LiftPickedBehaviour`.

    Returns:
        A py_trees Sequence(memory=True) implementing pick then place.
    """
    pick_item = py_trees.composites.Sequence(
        name="pick_item", memory=True, children=[
            MoveToPickXYBehaviour(name="MoveToPickXY", num_steps=100, fake_fast=fake_fast),
            LowerToPickBehaviour(name="LowerToPick", num_steps=200, fake_fast=fake_fast),
            WaitSettlingBehaviour(name="WaitSettling", num_steps=10, fake_fast=fake_fast),
            CloseGripperBehaviour(name="CloseGripper", num_steps=1, fake_fast=fake_fast),
            LiftPickedBehaviour(
                name="LiftPicked", num_steps=125, fake_fast=fake_fast,
                track_picked_item_during_lift=track_picked_item_during_lift,
            ),
        ])
    place_item = py_trees.composites.Sequence(
        name="place_item", memory=True, children=[
            MoveToPlaceXYBehaviour(name="MoveToPlaceXY", num_steps=200, fake_fast=fake_fast),
            LowerToPlaceBehaviour(name="LowerToPlace", num_steps=200, fake_fast=fake_fast),
            OpenGripperBehaviour(name="OpenGripper", num_steps=1, fake_fast=fake_fast),
            LiftAfterPlaceBehaviour(name="LiftAfterPlace", num_steps=13, fake_fast=fake_fast),
        ])

    return py_trees.composites.Sequence(
        name="pick_then_place", memory=True, children=[pick_item, place_item])
