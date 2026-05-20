"""Tests for individual pick-place py_trees behaviours.

Verifies that each movement behaviour uses the correct EE orientation
(standard for pick phases, drop for place phases) by setting up blackboard
inputs and ticking the behaviour once.

Behaviours send commands to an IArmCommander / IGripperCommander;
tests use MockArmCommander / MockGripperCommander to inspect sent targets.
"""
import sys
import os
import numpy as np
import pytest

# Add extsMock and repo root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'extsMock'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import py_trees
from isaacsim.core.utils.rotations import euler_angles_to_quat
from robot_controllers.mock_robot import MockArmCommander, MockGripperCommander
from robot_controllers.pt_pick_place_behaviours import (
    MoveToPickXYBehaviour,
    MoveToPlaceXYBehaviour,
    LowerToPlaceBehaviour,
    LiftAfterPlaceBehaviour,
    LiftPickedBehaviour,
    CloseGripperBehaviour,
    OpenGripperBehaviour,
)


@pytest.fixture(autouse=True)
def clear_blackboard():
    """Clear py_trees blackboard before and after each test."""
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

STD_OFFSET = np.array([0.0, 0.0, 0.0])
DROP_OFFSET = np.array([-0.03, 0.0, 0.0])
PICKING_POS = np.array([0.5, 0.0, 0.05])
PLACING_POS = np.array([-0.5, 0.0, 0.05])
CURRENT_JOINTS = np.zeros(6)


@pytest.fixture
def orientations():
    """Compute standard and drop orientations."""
    std = euler_angles_to_quat(np.array([0, np.pi / 2, 0]))
    drop = euler_angles_to_quat(np.array([np.pi / 2, 0, 0]))
    return std, drop


def _setup_bb_inputs(picking_pos, placing_pos, current_joints,
                     std_orient, std_offset, drop_orient=None, drop_offset=None):
    """Write pickplace inputs to the py_trees blackboard for testing."""
    bb = py_trees.blackboard.Client(name="test_writer", namespace="/pickplace")
    for key in [
        "picking_position", "placing_position", "current_joint_positions",
        "end_effector_offset", "end_effector_orientation",
        "end_effector_offset_for_drop", "end_effector_orientation_for_drop",
        "ee_height_for_move",
        "pick_position",
    ]:
        bb.register_key(key=key, access=py_trees.common.Access.WRITE)

    bb.picking_position = picking_pos
    bb.placing_position = placing_pos
    bb.current_joint_positions = current_joints
    bb.end_effector_offset = std_offset
    bb.end_effector_orientation = std_orient
    bb.end_effector_offset_for_drop = drop_offset if drop_offset is not None else std_offset
    bb.end_effector_orientation_for_drop = drop_orient if drop_orient is not None else std_orient
    bb.ee_height_for_move = 0.3
    bb.pick_position = picking_pos.copy()
    return bb


# ---------------------------------------------------------------------------
# Tests: pick phases should use standard orientation
# ---------------------------------------------------------------------------

def test_move_to_pick_uses_standard_orientation(orientations):
    """Phase 0 (MoveToPickXY) should use standard orientation."""
    std_orient, drop_orient = orientations
    arm = MockArmCommander()
    _setup_bb_inputs(PICKING_POS, PLACING_POS, CURRENT_JOINTS,
                     std_orient, STD_OFFSET, drop_orient, DROP_OFFSET)

    behaviour = MoveToPickXYBehaviour(name="test_move_to_pick", num_steps=2)
    behaviour.setup(arm_commander=arm, gripper_commander=None)
    behaviour.tick_once()

    assert np.allclose(arm.last_target_orientation, std_orient), \
        "MoveToPickXY should use standard orientation"


# ---------------------------------------------------------------------------
# Tests: place phases should use drop orientation
# ---------------------------------------------------------------------------

def test_move_to_place_uses_drop_orientation(orientations):
    """Phase 5 (MoveToPlaceXY) should use drop orientation."""
    std_orient, drop_orient = orientations
    arm = MockArmCommander()
    _setup_bb_inputs(PICKING_POS, PLACING_POS, CURRENT_JOINTS,
                     std_orient, STD_OFFSET, drop_orient, DROP_OFFSET)

    behaviour = MoveToPlaceXYBehaviour(name="test_move_to_place", num_steps=2)
    behaviour.setup(arm_commander=arm, gripper_commander=None)
    behaviour.tick_once()

    assert np.allclose(arm.last_target_orientation, drop_orient), \
        "MoveToPlaceXY should use drop orientation"


def test_lower_to_place_uses_drop_orientation(orientations):
    """Phase 6 (LowerToPlace) should use drop orientation."""
    std_orient, drop_orient = orientations
    arm = MockArmCommander()
    _setup_bb_inputs(PICKING_POS, PLACING_POS, CURRENT_JOINTS,
                     std_orient, STD_OFFSET, drop_orient, DROP_OFFSET)

    behaviour = LowerToPlaceBehaviour(name="test_lower_to_place", num_steps=2)
    behaviour.setup(arm_commander=arm, gripper_commander=None)
    behaviour.tick_once()

    assert np.allclose(arm.last_target_orientation, drop_orient), \
        "LowerToPlace should use drop orientation"


def test_lift_after_place_uses_drop_orientation(orientations):
    """Phase 8 (LiftAfterPlace) should use drop orientation."""
    std_orient, drop_orient = orientations
    arm = MockArmCommander()
    _setup_bb_inputs(PICKING_POS, PLACING_POS, CURRENT_JOINTS,
                     std_orient, STD_OFFSET, drop_orient, DROP_OFFSET)

    behaviour = LiftAfterPlaceBehaviour(name="test_lift_after_place", num_steps=2)
    behaviour.setup(arm_commander=arm, gripper_commander=None)
    behaviour.tick_once()

    assert np.allclose(arm.last_target_orientation, drop_orient), \
        "LiftAfterPlace should use drop orientation"


# ---------------------------------------------------------------------------
# Test: without drop args, all phases use standard orientation
# ---------------------------------------------------------------------------

def test_no_drop_args_uses_standard_orientation(orientations):
    """Without drop orientation args, place phases fall back to standard."""
    std_orient, _drop_orient = orientations
    arm = MockArmCommander()
    # No drop_orient / drop_offset — should default to std
    _setup_bb_inputs(PICKING_POS, PLACING_POS, CURRENT_JOINTS,
                     std_orient, STD_OFFSET)

    behaviour = MoveToPlaceXYBehaviour(name="test_no_drop", num_steps=2)
    behaviour.setup(arm_commander=arm, gripper_commander=None)
    behaviour.tick_once()

    assert np.allclose(arm.last_target_orientation, std_orient), \
        "Without drop args, place phases should use standard orientation"


# ---------------------------------------------------------------------------
# Tests: gripper behaviours call commander directly
# ---------------------------------------------------------------------------

def test_close_gripper_calls_commander(orientations):
    """CloseGripperBehaviour should call gripper_commander.close()."""
    std_orient, _ = orientations
    gripper = MockGripperCommander()
    _setup_bb_inputs(PICKING_POS, PLACING_POS, CURRENT_JOINTS, std_orient, STD_OFFSET)

    behaviour = CloseGripperBehaviour(name="test_close", num_steps=1)
    behaviour.setup(arm_commander=None, gripper_commander=gripper)
    behaviour.tick_once()

    assert gripper.is_closed
    assert gripper.close_count == 1


def test_open_gripper_calls_commander(orientations):
    """OpenGripperBehaviour should call gripper_commander.open()."""
    std_orient, _ = orientations
    gripper = MockGripperCommander()
    gripper.close()  # start closed
    _setup_bb_inputs(PICKING_POS, PLACING_POS, CURRENT_JOINTS, std_orient, STD_OFFSET)

    behaviour = OpenGripperBehaviour(name="test_open", num_steps=1)
    behaviour.setup(arm_commander=None, gripper_commander=gripper)
    behaviour.tick_once()

    assert not gripper.is_closed
    assert gripper.open_count == 1


def test_close_gripper_fake_fast_calls_commander(orientations):
    """Even with fake_fast=True, should call gripper_commander.close()."""
    std_orient, _ = orientations
    gripper = MockGripperCommander()
    _setup_bb_inputs(PICKING_POS, PLACING_POS, CURRENT_JOINTS, std_orient, STD_OFFSET)

    behaviour = CloseGripperBehaviour(name="test_close_ff", num_steps=1, fake_fast=True)
    behaviour.setup(arm_commander=None, gripper_commander=gripper)
    behaviour.tick_once()

    assert gripper.is_closed
    assert gripper.close_count == 1


# ---------------------------------------------------------------------------
# Test: CloseGripper latches pick position from picking_position
# ---------------------------------------------------------------------------

def test_close_gripper_latches_pick_position(orientations):
    """CloseGripperBehaviour should latch pick_position from picking_position."""
    std_orient, _ = orientations
    gripper = MockGripperCommander()
    bb = _setup_bb_inputs(PICKING_POS, PLACING_POS, CURRENT_JOINTS, std_orient, STD_OFFSET)

    # Overwrite picking_position to differ from the pre-written pick_position
    new_pick_pos = np.array([0.7, 0.1, 0.08])
    bb.picking_position = new_pick_pos

    behaviour = CloseGripperBehaviour(name="test_latch", num_steps=1)
    behaviour.setup(arm_commander=None, gripper_commander=gripper)
    behaviour.tick_once()

    reader = py_trees.blackboard.Client(name="latch_reader", namespace="/pickplace")
    reader.register_key(key="pick_position", access=py_trees.common.Access.READ)

    assert np.allclose(reader.pick_position, new_pick_pos)


# ---------------------------------------------------------------------------
# Test: phase progression is sim-time-based (FRAME_RATE-invariant)
# ---------------------------------------------------------------------------

def test_phase_progression_is_sim_time_based(monkeypatch, orientations):
    """``PickPlaceBehaviour.update`` should advance ``self.t`` at a rate
    proportional to ``sim_dt × BT_TICK_REFERENCE_HZ / num_steps`` so that
    each phase completes in ``num_steps / BT_TICK_REFERENCE_HZ`` sim
    seconds regardless of the BT tick rate (FRAME_RATE) it runs at.
    """
    from robot_controllers import pt_pick_place_behaviours as ppb
    std_orient, _ = orientations
    NUM_STEPS = 30
    BT_REF = ppb.BT_TICK_REFERENCE_HZ  # = 30.0
    # Phase duration in sim seconds (independent of tick rate):
    phase_duration_s = NUM_STEPS / BT_REF  # = 1.0 s

    for bt_tick_hz in (15.0, 30.0, 60.0, 150.0):
        # Drive _try_get_sim_time with a controlled sim clock.
        sim_t = [0.0]
        monkeypatch.setattr(ppb, "_try_get_sim_time", lambda: sim_t[0])

        arm = MockArmCommander()
        _setup_bb_inputs(PICKING_POS, PLACING_POS, CURRENT_JOINTS,
                         std_orient, STD_OFFSET)
        behaviour = ppb.MoveToPickXYBehaviour(
            name=f"test_progression_{bt_tick_hz}", num_steps=NUM_STEPS,
        )
        behaviour.setup(arm_commander=arm, gripper_commander=None)

        # Tick until SUCCESS, advancing the sim clock by 1/bt_tick_hz per tick.
        bt_tick_dt = 1.0 / bt_tick_hz
        n_ticks = 0
        py_trees.blackboard.Blackboard.clear  # noqa — keep bb between iters
        while True:
            n_ticks += 1
            behaviour.tick_once()
            if behaviour.status == py_trees.common.Status.SUCCESS:
                break
            sim_t[0] += bt_tick_dt
            assert n_ticks < 10_000, "phase progression looped indefinitely"

        # Total sim time elapsed = (n_ticks - 1) * bt_tick_dt + first-tick
        # contribution.  Phase finishes when t >= 1.0; for the sim-time path
        # the first tick advances by 1/num_steps, subsequent ticks by
        # bt_tick_dt * BT_REF / num_steps.  So total sim time at completion
        # is approximately phase_duration_s, +/- one tick.
        sim_time_to_complete = sim_t[0]
        assert abs(sim_time_to_complete - phase_duration_s) <= 2 * bt_tick_dt, (
            f"BT@{bt_tick_hz}Hz: phase took {sim_time_to_complete:.4f}s sim, "
            f"expected ~{phase_duration_s:.4f}s (±2 ticks of {bt_tick_dt:.4f}s)"
        )


# ---------------------------------------------------------------------------
# Test: LiftPicked XY source — latched (default) vs. live (opt-in)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "track_live, expected_second_xy",
    [
        (False, np.array([0.5, 0.5])),  # default: hold latched pick XY
        (True, np.array([0.5, 0.3])),   # opt-in: follow live picking XY
    ],
)
def test_lift_picked_xy_source(orientations, track_live, expected_second_xy):
    """Phase 4 (LiftPicked) default holds the latched ``pick_position`` XY;
    with ``track_picked_item_during_lift=True`` it follows the live
    ``picking_position`` XY (e.g. for a held item still being dragged
    by a moving conveyor surface)."""
    std_orient, _ = orientations
    arm = MockArmCommander()
    pick_xy = np.array([0.5, 0.5, 0.1])
    _setup_bb_inputs(picking_pos=pick_xy.copy(), placing_pos=PLACING_POS,
                     current_joints=CURRENT_JOINTS,
                     std_orient=std_orient, std_offset=STD_OFFSET)

    behaviour = LiftPickedBehaviour(
        name=f"test_lift_picked_{track_live}", num_steps=2,
        track_picked_item_during_lift=track_live,
    )
    behaviour.setup(arm_commander=arm, gripper_commander=None)
    behaviour.tick_once()
    assert np.allclose(arm.last_target_position[:2], pick_xy[:2]), (
        f"first tick XY should match latched pick XY; got "
        f"{arm.last_target_position[:2]}"
    )

    # Simulate live picking_position drifting (e.g. conveyor drag)
    bb = py_trees.blackboard.Client(name="lift_xy_writer", namespace="/pickplace")
    bb.register_key(key="picking_position", access=py_trees.common.Access.WRITE)
    bb.picking_position = np.array([0.5, 0.3, 0.1])
    behaviour.tick_once()
    assert np.allclose(arm.last_target_position[:2], expected_second_xy), (
        f"second tick XY (track_live={track_live}) should be {expected_second_xy}; "
        f"got {arm.last_target_position[:2]}"
    )


# ---------------------------------------------------------------------------
# Tests: IGripperCommander.grasp_state()
# ---------------------------------------------------------------------------


class TestGraspState:
    """Step 5: best-effort grasp-state reporting on IGripperCommander."""

    def test_mock_defaults_to_unknown(self):
        gripper = MockGripperCommander()
        assert gripper.grasp_state() == "unknown"
        gripper.close()
        assert gripper.grasp_state() == "unknown"

    def test_mock_override_to_holding(self):
        gripper = MockGripperCommander()
        gripper.close()
        gripper.grasp_state_override = "holding"
        assert gripper.grasp_state() == "holding"

    def test_mock_override_to_empty(self):
        gripper = MockGripperCommander()
        gripper.close()
        gripper.grasp_state_override = "empty"
        assert gripper.grasp_state() == "empty"

    def test_mock_reset_clears_override(self):
        gripper = MockGripperCommander()
        gripper.grasp_state_override = "holding"
        gripper.reset()
        assert gripper.grasp_state() == "unknown"

    def test_null_gripper_reports_unknown(self):
        from robot_controllers.cortex_adapters import NullGripperCommander
        gripper = NullGripperCommander()
        assert gripper.grasp_state() == "unknown"
        gripper.close()
        assert gripper.grasp_state() == "unknown"

    def test_cortex_adapter_reports_unknown(self):
        # We don't need a real SurfaceGripper — CortexGripperAdapter never
        # consults it for grasp_state() today (follow-up could).
        from robot_controllers.cortex_adapters import CortexGripperAdapter

        class _DummySurfaceGripper:
            def open(self): pass
            def close(self): pass

        adapter = CortexGripperAdapter(_DummySurfaceGripper())
        assert adapter.grasp_state() == "unknown"
        adapter.close()
        assert adapter.grasp_state() == "unknown"

    def test_legacy_adapter_reports_unknown(self):
        from robot_controllers.cortex_adapters import LegacyGripperAdapter

        class _DummyGripper:
            def forward(self, action): return None

        class _DummyArtic:
            def apply_action(self, action): pass

        adapter = LegacyGripperAdapter(_DummyGripper(), _DummyArtic())
        assert adapter.grasp_state() == "unknown"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
