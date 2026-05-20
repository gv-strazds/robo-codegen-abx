"""Tests for motion command methods on TaskContextBase."""

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

from isaacsim.cortex.framework.motion_commander import MotionCommand, PosePq, ApproachParams
from task_context_mock import MockTaskContext


def _make_ctx(**kwargs):
    """Create a MockTaskContext with sensible defaults for motion tests."""
    return MockTaskContext(**kwargs)


class TestRobotAtTarget:
    """Tests for TaskContextBase.robot_at_target()."""

    def test_returns_true_when_at_target(self):
        ctx = _make_ctx()
        # Set the mock arm to a known position/orientation
        arm = ctx.arm_commander
        arm._position = np.array([0.5, 0.1, 0.3])
        arm._orientation = np.array([1.0, 0.0, 0.0, 0.0])

        target = PosePq(np.array([0.5, 0.1, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
        cmd = MotionCommand(target_pose=target)
        assert ctx.robot_at_target(cmd, p_thresh=0.01, R_thresh=0.01)

    def test_returns_false_when_far(self):
        ctx = _make_ctx()
        arm = ctx.arm_commander
        arm._position = np.array([0.0, 0.0, 0.0])
        arm._orientation = np.array([1.0, 0.0, 0.0, 0.0])

        target = PosePq(np.array([1.0, 1.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        cmd = MotionCommand(target_pose=target)
        assert not ctx.robot_at_target(cmd, p_thresh=0.01, R_thresh=0.01)

    def test_returns_true_within_threshold(self):
        ctx = _make_ctx()
        arm = ctx.arm_commander
        arm._position = np.array([0.5, 0.1, 0.3])
        arm._orientation = np.array([1.0, 0.0, 0.0, 0.0])

        # Target slightly offset — within threshold
        target = PosePq(np.array([0.505, 0.1, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
        cmd = MotionCommand(target_pose=target)
        assert ctx.robot_at_target(cmd, p_thresh=0.01, R_thresh=0.01)

    def test_returns_false_with_rotation_mismatch(self):
        ctx = _make_ctx()
        arm = ctx.arm_commander
        arm._position = np.array([0.5, 0.1, 0.3])
        arm._orientation = np.array([1.0, 0.0, 0.0, 0.0])

        # Same position, very different orientation
        target = PosePq(
            np.array([0.5, 0.1, 0.3]),
            np.array([0.0, 1.0, 0.0, 0.0]),  # 180 deg rotation around X
        )
        cmd = MotionCommand(target_pose=target)
        assert not ctx.robot_at_target(cmd, p_thresh=0.1, R_thresh=0.01)

    def test_handles_default_orientation(self):
        """MockArmCommander with no orientation set still works (defaults to identity)."""
        ctx = _make_ctx()
        # arm._orientation is None by default; get_fk_pq() should return identity quat
        target = PosePq(np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
        cmd = MotionCommand(target_pose=target)
        assert ctx.robot_at_target(cmd, p_thresh=0.01, R_thresh=0.01)


class TestComputeMotionCommandToTarget:
    """Tests for TaskContextBase.compute_motion_command_to_target()."""

    def test_returns_motion_command_with_correct_pose(self):
        ctx = _make_ctx()
        target = PosePq(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        cmd = ctx.compute_motion_command_to_target(target)
        assert isinstance(cmd, MotionCommand)
        np.testing.assert_array_equal(cmd.target_pose.p, target.p)
        np.testing.assert_array_equal(cmd.target_pose.q, target.q)

    def test_no_approach_or_posture(self):
        ctx = _make_ctx()
        target = PosePq(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))
        cmd = ctx.compute_motion_command_to_target(target)
        assert not cmd.has_approach_params
        assert not cmd.has_posture_config


class TestGetRelativePq:
    """Tests for TaskContextBase.get_relative_pq()."""

    def test_world_frame_offset(self):
        ctx = _make_ctx()
        arm = ctx.arm_commander
        arm._position = np.array([0.5, 0.0, 0.3])
        arm._orientation = np.array([1.0, 0.0, 0.0, 0.0])

        result = ctx.get_relative_pq(np.array([0.1, 0.0, 0.0]))
        np.testing.assert_array_almost_equal(result.p, [0.6, 0.0, 0.3])

    def test_world_frame_preserves_orientation(self):
        ctx = _make_ctx()
        arm = ctx.arm_commander
        arm._position = np.array([0.5, 0.0, 0.3])
        arm._orientation = np.array([0.707, 0.707, 0.0, 0.0])

        result = ctx.get_relative_pq(np.array([0.0, 0.0, 0.1]))
        np.testing.assert_array_almost_equal(result.q, arm._orientation, decimal=3)

    def test_ee_frame_offset_identity_rotation(self):
        """With identity rotation, EE-frame offset equals world-frame offset."""
        ctx = _make_ctx()
        arm = ctx.arm_commander
        arm._position = np.array([0.5, 0.0, 0.3])
        arm._orientation = np.array([1.0, 0.0, 0.0, 0.0])

        world_result = ctx.get_relative_pq(np.array([0.1, 0.0, 0.0]), use_world_frame=True)
        ee_result = ctx.get_relative_pq(np.array([0.1, 0.0, 0.0]), use_world_frame=False)
        np.testing.assert_array_almost_equal(world_result.p, ee_result.p, decimal=5)

    def test_ee_frame_offset_rotated(self):
        """With 90-deg rotation around Z, X-offset in EE frame becomes Y-offset in world."""
        from isaacsim.core.utils.rotations import euler_angles_to_quat
        ctx = _make_ctx()
        arm = ctx.arm_commander
        arm._position = np.array([1.0, 0.0, 0.0])
        # 90 deg around Z axis
        arm._orientation = euler_angles_to_quat(np.array([0.0, 0.0, np.pi / 2]))

        result = ctx.get_relative_pq(np.array([0.1, 0.0, 0.0]), use_world_frame=False)
        # X in EE frame → Y in world frame (with 90 deg Z rotation)
        np.testing.assert_array_almost_equal(result.p, [1.0, 0.1, 0.0], decimal=4)


class TestComputePickCommand:
    """Tests for TaskContextBase.compute_pick_command_for_active_item()."""

    def test_basic_pick_command(self):
        ctx = _make_ctx()
        cmd = ctx.compute_pick_command_for_active_item()
        assert isinstance(cmd, MotionCommand)
        assert cmd.target_pose is not None

        # Target position should be pick_pos + ee_offset
        pick_pos = ctx.get_picking_position("pick_0")
        ee_offset = ctx.get_end_effector_offset("pick_0")
        expected_p = pick_pos + ee_offset
        np.testing.assert_array_almost_equal(cmd.target_pose.p, expected_p)

    def test_orientation_from_strategy(self):
        ctx = _make_ctx()
        cmd = ctx.compute_pick_command_for_active_item()
        expected_orient = ctx.get_end_effector_orientation("pick_0")
        np.testing.assert_array_almost_equal(cmd.target_pose.q, expected_orient)

    def test_approach_params_downward(self):
        ctx = _make_ctx()
        cmd = ctx.compute_pick_command_for_active_item()
        assert cmd.has_approach_params
        # Direction should be pointing downward (-Z)
        direction = cmd.approach_params.direction
        assert direction[2] < 0  # negative Z
        np.testing.assert_array_almost_equal(
            direction / np.linalg.norm(direction), [0, 0, -1]
        )

    def test_posture_config_from_context(self):
        # Posture config now flows via get_posture_config(); override via
        # ctx.get_posture_config monkeypatch (or TaskContextBase.__init__
        # kwargs) rather than a method parameter.
        ctx = _make_ctx()
        posture = np.array([0.0, -1.0, 0.0, -2.0, 0.0, 1.0])
        ctx.get_posture_config = (
            lambda phase: posture if phase == "pick" else None
        )
        cmd = ctx.compute_pick_command_for_active_item()
        assert cmd.has_posture_config
        np.testing.assert_array_equal(cmd.posture_config, posture)

    def test_returns_none_when_no_current_pick(self):
        ctx = _make_ctx(pick_names=["p0"], target_names=["t0"])
        # Advance past the only pick
        ctx.mark_pick_complete("p0")
        ctx.advance_pick_index()
        cmd = ctx.compute_pick_command_for_active_item()
        assert cmd is None


class TestComputeDynamicPlaceCommand:
    """Tests for TaskContextBase.compute_dynamic_place_command_for_active_item()."""

    def test_basic_place_command(self):
        ctx = _make_ctx()
        cmd = ctx.compute_dynamic_place_command_for_active_item()
        assert isinstance(cmd, MotionCommand)
        assert cmd.target_pose is not None

    def test_target_from_placing_info(self):
        ctx = _make_ctx()
        cmd = ctx.compute_dynamic_place_command_for_active_item()
        # The place position should come from get_placing_info, shifted by
        # the drop-side EE offset (falls back to _EE_OFFSET_FALLBACK when
        # no PrimGeometry is cached).
        pick_name = ctx.get_current_pick_name()
        drop_orient = ctx.get_end_effector_orientation_for_drop(pick_name, "target_0")
        if drop_orient is None:
            drop_orient = ctx.get_end_effector_orientation(pick_name)
        _, expected_pos, _ = ctx.get_placing_info(pick_name, drop_orient)
        if expected_pos is not None:
            expected_pos = expected_pos + ctx.get_end_effector_offset(pick_name)
            np.testing.assert_array_almost_equal(cmd.target_pose.p, expected_pos)

    def test_above_offset(self):
        ctx = _make_ctx()
        cmd_low = ctx.compute_dynamic_place_command_for_active_item(above=0.0)
        cmd_high = ctx.compute_dynamic_place_command_for_active_item(above=0.1)
        assert cmd_high.target_pose.p[2] == pytest.approx(cmd_low.target_pose.p[2] + 0.1)

    def test_explicit_target_p(self):
        ctx = _make_ctx()
        explicit_p = np.array([0.0, 0.0, 0.5])
        cmd = ctx.compute_dynamic_place_command_for_active_item(target_p=explicit_p)
        # Explicit target_p is still shifted by the drop-side EE offset.
        pick_name = ctx.get_current_pick_name()
        expected_p = explicit_p + ctx.get_end_effector_offset(pick_name)
        np.testing.assert_array_almost_equal(cmd.target_pose.p, expected_p)

    def test_approach_params_downward(self):
        ctx = _make_ctx()
        cmd = ctx.compute_dynamic_place_command_for_active_item()
        assert cmd.has_approach_params
        direction = cmd.approach_params.direction
        assert direction[2] < 0  # negative Z

    def test_posture_config_from_context(self):
        ctx = _make_ctx()
        posture = np.array([0.0, -1.0, 0.0, -2.0, 0.0, 1.0])
        ctx.get_posture_config = (
            lambda phase: posture if phase == "place" else None
        )
        cmd = ctx.compute_dynamic_place_command_for_active_item()
        assert cmd.has_posture_config
        np.testing.assert_array_equal(cmd.posture_config, posture)

    def test_returns_none_when_no_current_pick(self):
        ctx = _make_ctx(pick_names=["p0"], target_names=["t0"])
        ctx.mark_pick_complete("p0")
        ctx.advance_pick_index()
        cmd = ctx.compute_dynamic_place_command_for_active_item()
        assert cmd is None

    def test_stationary_target_obj_applies_no_lead(self):
        """A stationary target_obj produces zero drift → no XY nudge."""
        from task_context_base import LightweightObj

        ctx = _make_ctx()
        pick_obj = ctx.strategy.pick_objs_by_name["pick_0"]
        pick_obj.set_world_pose(position=np.array([0.1, 0.2, 0.3]))

        target_obj = LightweightObj("fine_target", position=np.array([0.11, 0.21, 0.3]))
        # Two calls with no target movement: first seeds the drift cache,
        # second measures zero velocity → zero lead.
        ctx.compute_dynamic_place_command_for_active_item(
            target_p=np.array([0.1, 0.2, 0.3]),
            target_obj=target_obj,
        )
        cmd = ctx.compute_dynamic_place_command_for_active_item(
            target_p=np.array([0.1, 0.2, 0.3]),
            target_obj=target_obj,
        )
        np.testing.assert_array_almost_equal(
            cmd.target_pose.p[:2], [0.1, 0.2], decimal=4
        )

    def test_moving_target_obj_applies_drift_lead(self):
        """A target_obj whose own XY drifts between ticks gets a lead along that drift."""
        import time as _time
        from task_context_base import LightweightObj, PLACE_LEAD_HORIZON_S

        ctx = _make_ctx()
        pick_obj = ctx.strategy.pick_objs_by_name["pick_0"]
        pick_obj.set_world_pose(position=np.array([0.1, 0.2, 0.3]))

        target_obj = LightweightObj("moving_target", position=np.array([0.10, 0.20, 0.3]))
        # Seed the drift cache with the initial position.
        ctx.compute_dynamic_place_command_for_active_item(
            target_p=np.array([0.1, 0.2, 0.3]),
            target_obj=target_obj,
        )
        # Simulate a tick's worth of belt motion in -Y.
        _time.sleep(0.05)
        target_obj.set_world_pose(position=np.array([0.10, 0.19, 0.3]))
        cmd = ctx.compute_dynamic_place_command_for_active_item(
            target_p=np.array([0.1, 0.2, 0.3]),
            target_obj=target_obj,
        )
        # Lead is PLACE_LEAD_HORIZON_S * measured_velocity; velocity is
        # dominated by the -Y step, so x stays put and y shifts negative.
        assert cmd.target_pose.p[0] == pytest.approx(0.1, abs=1e-4)
        assert cmd.target_pose.p[1] < 0.2
        # Sanity bound: lead magnitude ≤ horizon × observed y-step / dt,
        # and dt ≥ 0.05 s, so |lead_y| ≤ PLACE_LEAD_HORIZON_S * 0.01 / 0.05 = 0.1.
        assert cmd.target_pose.p[1] > 0.2 - PLACE_LEAD_HORIZON_S * 0.01 / 0.05 - 1e-6


class TestPostureConfigSeam:
    """Tests for get_posture_config() + per-instance override mechanism."""

    def test_default_pick_posture_is_module_constant(self):
        from task_context_base import PICK_POSTURE_CONFIG

        ctx = _make_ctx()
        cmd = ctx.compute_pick_command_for_active_item()
        assert cmd.has_posture_config
        np.testing.assert_array_equal(cmd.posture_config, PICK_POSTURE_CONFIG)

    def test_default_place_posture_is_module_constant(self):
        from task_context_base import PLACE_POSTURE_CONFIG

        ctx = _make_ctx()
        cmd = ctx.compute_dynamic_place_command_for_active_item()
        assert cmd.has_posture_config
        np.testing.assert_array_equal(cmd.posture_config, PLACE_POSTURE_CONFIG)

    def test_pick_posture_override_via_init_kwarg(self):
        from task_context_mock import MockTaskContext

        pick_posture = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        ctx = MockTaskContext(pick_posture_config=pick_posture)
        cmd = ctx.compute_pick_command_for_active_item()
        assert cmd.has_posture_config
        np.testing.assert_array_equal(cmd.posture_config, pick_posture)

    def test_place_posture_override_via_init_kwarg(self):
        from task_context_mock import MockTaskContext

        place_posture = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        ctx = MockTaskContext(place_posture_config=place_posture)
        cmd = ctx.compute_dynamic_place_command_for_active_item()
        assert cmd.has_posture_config
        np.testing.assert_array_equal(cmd.posture_config, place_posture)

    def test_pick_explicit_none_disables_posture(self):
        from task_context_mock import MockTaskContext

        ctx = MockTaskContext(pick_posture_config=None)
        cmd = ctx.compute_pick_command_for_active_item()
        assert not cmd.has_posture_config

    def test_place_explicit_none_disables_posture(self):
        from task_context_mock import MockTaskContext

        ctx = MockTaskContext(place_posture_config=None)
        cmd = ctx.compute_dynamic_place_command_for_active_item()
        assert not cmd.has_posture_config


class TestPerceptionCache:
    """Step 4: cache slots for current grasp/item-in-ee/place poses."""

    def test_cache_starts_empty(self):
        ctx = _make_ctx()
        assert ctx.get_current_grasp_pose() is None
        assert ctx.get_current_item_in_ee() is None
        assert ctx.get_current_place_pose() is None

    def test_compute_pick_populates_grasp_pose_cache(self):
        from perception_utils import GraspPose

        ctx = _make_ctx()
        cmd = ctx.compute_pick_command_for_active_item()
        assert cmd is not None
        grasp_pose = ctx.get_current_grasp_pose()
        assert isinstance(grasp_pose, GraspPose)
        # The command's target pose should match the cached pose.
        np.testing.assert_array_equal(cmd.target_pose.p, grasp_pose.ee_position)
        np.testing.assert_array_equal(cmd.target_pose.q, grasp_pose.ee_orientation)

    def test_compute_place_populates_place_pose_cache(self):
        from perception_utils import PlacePose

        ctx = _make_ctx()
        cmd = ctx.compute_dynamic_place_command_for_active_item(above=0.1)
        assert cmd is not None
        place_pose = ctx.get_current_place_pose()
        assert isinstance(place_pose, PlacePose)
        # The command's target pose matches the cached hover position (incl. above).
        np.testing.assert_array_equal(cmd.target_pose.p, place_pose.ee_position)

    def test_reset_cycle_cache_clears_all_slots(self):
        ctx = _make_ctx()
        ctx.compute_pick_command_for_active_item()
        ctx.compute_dynamic_place_command_for_active_item()
        assert ctx.get_current_grasp_pose() is not None
        assert ctx.get_current_place_pose() is not None
        ctx.reset_cycle_cache()
        assert ctx.get_current_grasp_pose() is None
        assert ctx.get_current_item_in_ee() is None
        assert ctx.get_current_place_pose() is None

    def test_set_current_item_in_ee_roundtrip(self):
        from perception_utils import ItemInEEPose

        ctx = _make_ctx()
        pose = ItemInEEPose(
            position_in_ee=np.array([0.0, 0.0, -0.05]),
            orientation_in_ee=np.array([1.0, 0.0, 0.0, 0.0]),
            position_error=0.001,
            orientation_error_rad=0.01,
        )
        ctx.set_current_item_in_ee(pose)
        assert ctx.get_current_item_in_ee() is pose
