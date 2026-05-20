"""Unit tests for PrepareGrasp / VerifyGrasp / DeferPickAndRelease / HaveItemInGripper.

Step 7 of the grasp-affordance refactor.  These behaviours are not
wired into any tree yet — they are tested in isolation with a mock
TaskContext.
"""
import os
import sys

import numpy as np
import py_trees
import pytest

# Ensure extsMock shadows real isaacsim and repo root is importable.
_current_dir = os.path.dirname(__file__)
_repo_root = os.path.abspath(os.path.join(_current_dir, ".."))
_mock_path = os.path.join(_repo_root, "extsMock")
for _p in (_mock_path, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from perception_utils import GraspPose, ItemInEEPose
from robot_controllers.pt_cortex_perception_behaviours import (
    CheckGraspPoseReachable,
    CheckPickReachable,
    DeferPickAndRelease,
    HaveItemInGripper,
    PrepareGrasp,
    PreparePlacement,
    VerifyGrasp,
)
from task_context_mock import MockTaskContext


def _make_ctx(**kwargs):
    return MockTaskContext(**kwargs)


class TestPrepareGrasp:
    def test_caches_grasp_pose_on_context(self):
        ctx = _make_ctx()
        b = PrepareGrasp()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        pose = ctx.get_current_grasp_pose()
        assert isinstance(pose, GraspPose)

    def test_failure_when_no_current_pick(self):
        ctx = _make_ctx(pick_names=["p0"])
        ctx.mark_pick_complete("p0")
        ctx.advance_pick_index()
        b = PrepareGrasp()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE

    def test_failure_when_no_context(self):
        b = PrepareGrasp()
        # No setup() → context is None.
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE

    def test_records_item_world_pose_at_grasp(self):
        ctx = _make_ctx()
        pick_name = ctx.get_current_pick_name()
        # Plant a non-trivial item world pose so we can tell the capture ran.
        pick_obj = ctx.strategy.pick_objs_by_name[pick_name]
        pick_obj.set_world_pose(
            position=np.array([0.42, -0.11, 0.07]),
            orientation=np.array([np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0]),
        )
        b = PrepareGrasp()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        pose = ctx.get_current_grasp_pose()
        assert pose.item_position_at_grasp is not None
        assert pose.item_orientation_at_grasp is not None
        np.testing.assert_allclose(pose.item_position_at_grasp, [0.42, -0.11, 0.07])
        np.testing.assert_allclose(
            pose.item_orientation_at_grasp,
            [np.cos(np.pi / 8), np.sin(np.pi / 8), 0.0, 0.0],
            atol=1e-12,
        )

    def test_cached_grasp_pose_includes_per_task_grasp_offset(self):
        """Regression: PrepareGrasp must pass ``grasp_offset_world`` so the
        cached ``GraspPose.ee_offset_world_at_grasp`` reflects the per-task
        override.  Without this, the cortex tree picks bottles at centre
        even when the task declares a grasp_offset_local_overrides entry
        (TableTaskBottles2 used to silently lose the offset)."""
        from asset_data_utils import PrimGeometry
        # PrimGeometry with grasp_height=0.04 and identity reference orientation
        # so the offset stays untransformed in world frame.
        geom = PrimGeometry(
            grasp_height=0.04,
            rest_height=0.025,
            top_surface_height=0.025,
            local_half_extents=np.array([0.02, 0.02, 0.07]),
            needs_aabb_scale_correction=False,
            reference_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        ctx = _make_ctx(
            pick_names=["bottle_0"],
            target_names=["t0"],
            prim_geometry={"bottle_0": geom},
            grasp_offset_local_overrides={"madara_bottle": np.array([0.0, 0.0, 0.015])},
        )
        # Wire the asset_type semantic label on the MockPickObj that
        # MockTaskContext built internally.
        ctx.strategy.pick_objs_by_name["bottle_0"]._semantic_labels["type"] = ["madara_bottle"]
        b = PrepareGrasp()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        pose = ctx.get_current_grasp_pose()
        # Canonical [0, 0, grasp_height] + override [0, 0, 0.015]
        np.testing.assert_allclose(
            pose.ee_offset_world_at_grasp, [0.0, 0.0, 0.055], atol=1e-9,
        )


class TestVerifyGrasp:
    def _prepare(self, ctx):
        PrepareGrasp().tick_once_helper = None  # avoid linter complaints
        b = PrepareGrasp()
        b.setup(context=ctx)
        b.tick_once()
        return b

    def test_successful_verification_when_item_tracks_ee(self):
        # Use mock_mode=False to exercise the real pose-check logic
        # instead of the mock-mode short-circuit.
        ctx = _make_ctx(mock_mode=False)
        self._prepare(ctx)
        # Simulate the item having moved with the EE after the lift:
        # position pick_obj at the current expected location.
        pick_name = ctx.get_current_pick_name()
        grasp_pose = ctx.get_current_grasp_pose()
        pick_obj = ctx.strategy.pick_objs_by_name[pick_name]
        # "Move" the EE upward by 0.2 m (simulate lift).
        ctx.arm_commander._position = grasp_pose.ee_position + np.array([0, 0, 0.2])
        ctx.arm_commander._orientation = grasp_pose.ee_orientation.copy()
        # Item tracks the lift: new item position = original pick_pos + lift.
        pick_obj.set_world_pose(
            position=grasp_pose.ee_position + np.array([0, 0, 0.2])
            - grasp_pose.ee_offset_world_at_grasp
        )
        b = VerifyGrasp(slip_threshold=0.01)
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        assert isinstance(ctx.get_current_item_in_ee(), ItemInEEPose)

    def test_failure_when_item_left_behind(self):
        ctx = _make_ctx(mock_mode=False)
        self._prepare(ctx)
        pick_name = ctx.get_current_pick_name()
        grasp_pose = ctx.get_current_grasp_pose()
        # Lift the EE — item did NOT come (still at original position).
        ctx.arm_commander._position = grasp_pose.ee_position + np.array([0, 0, 0.2])
        ctx.arm_commander._orientation = grasp_pose.ee_orientation.copy()
        # pick_obj stays at its original world pose (from MockTaskContext
        # default) → measured offset ≈ [0, 0, 0.2 + grasp_height] ≠ expected.
        b = VerifyGrasp(slip_threshold=0.01)
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE

    def test_failure_when_gripper_reports_empty(self):
        ctx = _make_ctx(mock_mode=False)
        self._prepare(ctx)
        # Force the mock gripper to report "empty" — should veto SUCCESS
        # even if the pose happens to match.
        ctx.gripper_commander.grasp_state_override = "empty"
        b = VerifyGrasp(slip_threshold=10.0)  # pose threshold irrelevant here
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE

    def test_unknown_gripper_state_does_not_veto(self):
        ctx = _make_ctx(mock_mode=False)
        self._prepare(ctx)
        pick_name = ctx.get_current_pick_name()
        grasp_pose = ctx.get_current_grasp_pose()
        pick_obj = ctx.strategy.pick_objs_by_name[pick_name]
        ctx.arm_commander._position = grasp_pose.ee_position + np.array([0, 0, 0.2])
        ctx.arm_commander._orientation = grasp_pose.ee_orientation.copy()
        pick_obj.set_world_pose(
            position=grasp_pose.ee_position + np.array([0, 0, 0.2])
            - grasp_pose.ee_offset_world_at_grasp
        )
        # Default: grasp_state() returns "unknown" — pose check should pass.
        assert ctx.gripper_commander.grasp_state() == "unknown"
        b = VerifyGrasp(slip_threshold=0.01)
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS

    def test_failure_without_cached_grasp_pose(self):
        ctx = _make_ctx()
        # Skip PrepareGrasp — no grasp_pose cached.
        b = VerifyGrasp()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE

    def test_teleport_mode_bypasses_pose_check(self):
        """In teleport mode the arm is a no-op; VerifyGrasp should SUCCESS regardless."""
        ctx = _make_ctx(teleport_mode=True, mock_mode=False)
        # Populate grasp_pose via PrepareGrasp.  No need to move the
        # mock arm or pick object — teleport mode should short-circuit.
        pg = PrepareGrasp()
        pg.setup(context=ctx)
        pg.tick_once()
        assert ctx.teleport_mode is True

        b = VerifyGrasp(slip_threshold=0.01)
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        # Flag is set so HaveItemInGripper gates pass; item_in_ee is
        # deliberately None so compute_place_pose uses its nominal branch.
        assert ctx.is_holding_item() is True
        assert ctx.get_current_item_in_ee() is None

    def test_mock_mode_bypasses_pose_check(self):
        """In mock mode the arm tick-countdown does not align with
        wall-clock Timer durations.  VerifyGrasp must short-circuit the
        same way as teleport mode to avoid retry-deferral cascades."""
        ctx = _make_ctx()  # default mock_mode=True
        assert ctx.mock_mode is True
        assert ctx.teleport_mode is False
        pg = PrepareGrasp()
        pg.setup(context=ctx)
        pg.tick_once()
        b = VerifyGrasp(slip_threshold=0.01)
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        assert ctx.is_holding_item() is True
        assert ctx.get_current_item_in_ee() is None


class TestVerifyGraspPersistentFailure:
    """Per-pick consecutive-failure counter promotes to permanent."""

    def _setup_failing_grasp(self, ctx):
        """Configure ctx so VerifyGrasp fails the pose check."""
        pg = PrepareGrasp()
        pg.setup(context=ctx)
        pg.tick_once()
        pick_name = ctx.get_current_pick_name()
        grasp_pose = ctx.get_current_grasp_pose()
        # Lift the EE — pick stays put → position_error large.
        ctx.arm_commander._position = grasp_pose.ee_position + np.array([0, 0, 0.2])
        ctx.arm_commander._orientation = grasp_pose.ee_orientation.copy()
        return pick_name

    def test_first_failure_does_not_promote(self):
        ctx = _make_ctx(mock_mode=False)
        pick_name = self._setup_failing_grasp(ctx)
        b = VerifyGrasp(slip_threshold=0.01)
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE
        assert not ctx.strategy.is_pick_permanently_unreachable(pick_name)
        assert b._fail_counts.get(pick_name) == 1

    def test_threshold_consecutive_failures_promote_to_permanent(self):
        ctx = _make_ctx(mock_mode=False)
        pick_name = self._setup_failing_grasp(ctx)
        b = VerifyGrasp(slip_threshold=0.01)
        b.setup(context=ctx)
        # Run failures up to threshold.
        for _ in range(b.MAX_CONSECUTIVE_FAILURES - 1):
            b.tick_once()
            assert not ctx.strategy.is_pick_permanently_unreachable(pick_name)
        # Final failure promotes.
        b.tick_once()
        assert ctx.strategy.is_pick_permanently_unreachable(pick_name)
        # Counter cleared on promotion.
        assert pick_name not in b._fail_counts

    def test_success_resets_failure_counter(self):
        ctx = _make_ctx(mock_mode=False)
        pick_name = self._setup_failing_grasp(ctx)
        b = VerifyGrasp(slip_threshold=0.01)
        b.setup(context=ctx)
        # One failure.
        b.tick_once()
        assert b._fail_counts.get(pick_name) == 1
        # Now arrange a SUCCESS: position the pick at the expected post-lift offset.
        grasp_pose = ctx.get_current_grasp_pose()
        pick_obj = ctx.strategy.pick_objs_by_name[pick_name]
        pick_obj.set_world_pose(
            position=ctx.arm_commander._position - grasp_pose.ee_offset_world_at_grasp,
        )
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        # Counter cleared on SUCCESS so a later failure starts from 0.
        assert pick_name not in b._fail_counts

    def test_count_survives_initialise_for_retry_restarts(self):
        """The wrapping Retry restarts pick_attempt on FAILURE; that
        invokes initialise() on every Behaviour in the sequence.  The
        consecutive-failure counter must NOT reset on initialise(),
        otherwise the Retry budget would multiply the threshold."""
        ctx = _make_ctx(mock_mode=False)
        pick_name = self._setup_failing_grasp(ctx)
        b = VerifyGrasp(slip_threshold=0.01)
        b.setup(context=ctx)
        b.tick_once()
        assert b._fail_counts.get(pick_name) == 1
        # Simulate a Retry restart.
        b.initialise()
        assert b._fail_counts.get(pick_name) == 1


class TestDeferPickAndRelease:
    def test_opens_gripper_and_defers_pick(self):
        ctx = _make_ctx()
        pick_name = ctx.get_current_pick_name()
        ctx.gripper_commander.close()
        open_count_before = ctx.gripper_commander.open_count

        b = DeferPickAndRelease()
        b.setup(context=ctx)
        b.tick_once()

        assert b.status == py_trees.common.Status.SUCCESS
        # Gripper was opened (release any partial grip).
        assert ctx.gripper_commander.open_count == open_count_before + 1
        # Pick was deferred.
        assert ctx.strategy.is_pick_deferred(pick_name)

    def test_resets_cycle_cache(self):
        ctx = _make_ctx()
        PrepareGrasp().setup(context=ctx)
        PrepareGrasp().tick_once()
        # Manually set caches to simulate a partial cycle.
        from perception_utils import GraspPose, ItemInEEPose
        ctx.set_current_grasp_pose(GraspPose(
            ee_position=np.array([0.0, 0.0, 0.0]),
            ee_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            approach_direction=np.array([0.0, 0.0, -1.0]),
            approach_distance=0.1,
            approach_std_dev=0.005,
            ee_offset_world_at_grasp=np.array([0.0, 0.0, 0.05]),
        ))
        ctx.set_current_item_in_ee(ItemInEEPose(
            position_in_ee=np.array([0.0, 0.0, -0.05]),
            orientation_in_ee=np.array([1.0, 0.0, 0.0, 0.0]),
            position_error=0.5,
            orientation_error_rad=0.0,
        ))

        b = DeferPickAndRelease()
        b.setup(context=ctx)
        b.tick_once()

        assert ctx.get_current_grasp_pose() is None
        assert ctx.get_current_item_in_ee() is None


class TestHaveItemInGripper:
    def test_success_when_flag_set_with_item_pose(self):
        # Normal grasp: flag True AND item_in_ee populated.
        ctx = _make_ctx()
        ctx.set_holding_item(True)
        ctx.set_current_item_in_ee(ItemInEEPose(
            position_in_ee=np.array([0.0, 0.0, -0.05]),
            orientation_in_ee=np.array([1.0, 0.0, 0.0, 0.0]),
            position_error=0.001,
            orientation_error_rad=0.0,
        ))
        b = HaveItemInGripper()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS

    def test_success_when_flag_set_without_item_pose(self):
        # Teleport grasp: flag True, item_in_ee is None.  Gate still opens.
        ctx = _make_ctx()
        ctx.set_holding_item(True)
        assert ctx.get_current_item_in_ee() is None
        b = HaveItemInGripper()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS

    def test_failure_when_flag_not_set(self):
        ctx = _make_ctx()
        ctx.reset_cycle_cache()
        assert ctx.is_holding_item() is False
        b = HaveItemInGripper()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE

    def test_failure_when_item_cached_but_flag_not_set(self):
        # Defensive: only the flag gates the tree. Stale item_in_ee
        # without the flag must not open the gate.
        ctx = _make_ctx()
        ctx.reset_cycle_cache()
        ctx.set_current_item_in_ee(ItemInEEPose(
            position_in_ee=np.array([0.0, 0.0, -0.05]),
            orientation_in_ee=np.array([1.0, 0.0, 0.0, 0.0]),
            position_error=0.001,
            orientation_error_rad=0.0,
        ))
        assert ctx.is_holding_item() is False
        b = HaveItemInGripper()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE


class TestPreparePlacement:
    def test_caches_place_pose_on_context(self):
        from perception_utils import PlacePose

        ctx = _make_ctx()
        b = PreparePlacement()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        assert isinstance(ctx.get_current_place_pose(), PlacePose)

    def test_failure_when_no_current_pick(self):
        ctx = _make_ctx(pick_names=["p0"])
        ctx.mark_pick_complete("p0")
        ctx.advance_pick_index()
        b = PreparePlacement()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE


class TestCortexMoveToPreGraspAndApproach:
    """Step 8 split: pre-grasp + approach share the cached GraspPose."""

    def test_pre_grasp_targets_upstream_of_grasp(self):
        from robot_controllers.pt_cortex_behaviours import CortexMoveToPreGrasp

        ctx = _make_ctx()
        PrepareGrasp().setup(context=ctx)
        PrepareGrasp().tick_once()
        # Manually run PrepareGrasp via a fresh instance so the cache is set.
        pg = PrepareGrasp()
        pg.setup(context=ctx)
        pg.tick_once()
        grasp_pose = ctx.get_current_grasp_pose()
        assert grasp_pose is not None

        pre = CortexMoveToPreGrasp(fake_fast=True)
        pre.setup(context=ctx, arm_commander=ctx.arm_commander)
        pre.tick_once()

        # The sent command should target the pre-grasp XY, with Z
        # floored to the live transport altitude (so the freespace
        # transit from the previous lift_after_place to the next pick
        # area stays at altitude — see the ``min_z`` kwarg passed to
        # ``compute_pregrasp_command_for_active_item``).
        assert pre.command is not None
        np.testing.assert_allclose(
            pre.command.target_pose.p[:2], grasp_pose.pre_grasp_position[:2],
        )
        expected_z = max(
            float(grasp_pose.pre_grasp_position[2]),
            float(ctx.get_ee_height_for_move()),
        )
        assert pre.command.target_pose.p[2] == pytest.approx(expected_z)
        np.testing.assert_allclose(
            pre.command.target_pose.q, grasp_pose.ee_orientation,
        )
        # The pre-grasp move now carries a horizontal RMPFlow approach
        # funnel (mirrors CortexMoveToPlace's transport-hover pattern):
        # direction = unit XY vector from live EE to pre-grasp XY, with
        # z=0 so vertical height is not stacked on top of the funnel
        # length.  The mock arm starts at the origin and the pre-grasp
        # pose sits over the bin, so XY distance is well above the 1e-3
        # threshold and the funnel fires.
        assert pre.command.has_approach_params
        approach_vec = pre.command.approach_params.direction
        assert abs(float(approach_vec[2])) < 1e-9, (
            f"funnel direction must be horizontal, got z={approach_vec[2]}"
        )
        # std_dev matches the class-level loose default.
        assert pre.command.approach_params.std_dev == \
            pytest.approx(CortexMoveToPreGrasp.PRE_GRASP_APPROACH_STD_DEV)

    def test_execute_approach_targets_grasp_pose(self):
        from robot_controllers.pt_cortex_behaviours import CortexExecuteApproach

        ctx = _make_ctx()
        pg = PrepareGrasp()
        pg.setup(context=ctx)
        pg.tick_once()
        grasp_pose = ctx.get_current_grasp_pose()

        approach = CortexExecuteApproach(fake_fast=True)
        approach.setup(context=ctx, arm_commander=ctx.arm_commander)
        approach.tick_once()

        assert approach.command is not None
        np.testing.assert_allclose(
            approach.command.target_pose.p, grasp_pose.ee_position,
        )
        # Approach funnel present.
        assert approach.command.has_approach_params

    def test_pre_grasp_and_approach_share_cached_pose(self):
        """Both motion behaviours keep the non-position portion of the grasp pose stable.

        The pre-grasp / approach behaviours refresh ``ee_position`` from
        the live pick position each tick (to track items that may still
        be settling), but ``ee_orientation``, approach plan, and the
        grasp-time EE offset must remain identical to the values
        captured by the initial ``PrepareGrasp``.
        """
        from robot_controllers.pt_cortex_behaviours import (
            CortexExecuteApproach,
            CortexMoveToPreGrasp,
        )

        ctx = _make_ctx()
        pg = PrepareGrasp()
        pg.setup(context=ctx)
        pg.tick_once()
        cached = ctx.get_current_grasp_pose()

        pre = CortexMoveToPreGrasp(fake_fast=True)
        pre.setup(context=ctx, arm_commander=ctx.arm_commander)
        pre.tick_once()

        approach = CortexExecuteApproach(fake_fast=True)
        approach.setup(context=ctx, arm_commander=ctx.arm_commander)
        approach.tick_once()

        # The non-position portion of the grasp pose is frozen across ticks.
        final = ctx.get_current_grasp_pose()
        assert final is not None
        np.testing.assert_allclose(final.ee_orientation, cached.ee_orientation)
        np.testing.assert_allclose(final.approach_direction, cached.approach_direction)
        assert final.approach_distance == cached.approach_distance
        assert final.approach_std_dev == cached.approach_std_dev
        np.testing.assert_allclose(
            final.ee_offset_world_at_grasp, cached.ee_offset_world_at_grasp,
        )


class TestCheckPickReachable:
    """Z-floor branch must promote the failure to permanently unreachable."""

    def _make_ctx_with_floor(self, pick_z, **kwargs):
        # mock_mode=False so the z-floor check runs (mock-mode branch
        # short-circuits to SUCCESS).  pick_min_reachable_z gives us a
        # known threshold so we can place pick_0 below it.
        positions = {"pick_0": np.array([0.5, 0.0, pick_z])}
        return _make_ctx(
            mock_mode=False,
            pick_min_reachable_z=-0.10,
            pick_max_reachable_radius_xy=2.0,
            pick_positions=positions,
            **kwargs,
        )

    def test_below_floor_returns_failure(self):
        ctx = self._make_ctx_with_floor(pick_z=-0.4)
        b = CheckPickReachable()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE

    def test_below_floor_marks_permanent(self):
        ctx = self._make_ctx_with_floor(pick_z=-0.4)
        pick_name = ctx.get_current_pick_name()
        assert not ctx.strategy.is_pick_permanently_unreachable(pick_name)
        b = CheckPickReachable()
        b.setup(context=ctx)
        b.tick_once()
        assert ctx.strategy.is_pick_permanently_unreachable(pick_name)

    def test_above_floor_returns_success(self):
        ctx = self._make_ctx_with_floor(pick_z=0.05)
        b = CheckPickReachable()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS

    def test_short_circuit_when_already_permanent(self):
        """A pick already flagged permanent should fail-fast without re-marking."""
        ctx = self._make_ctx_with_floor(pick_z=0.05)  # above floor, but pre-flagged
        pick_name = ctx.get_current_pick_name()
        ctx.strategy.mark_pick_permanently_unreachable(pick_name)
        b = CheckPickReachable()
        b.setup(context=ctx)
        b.tick_once()
        # Even though z is above the floor, the short-circuit fires.
        assert b.status == py_trees.common.Status.FAILURE

    # --- Radial grace-window branch -------------------------------------------------

    def _make_ctx_out_of_radius(self, **kwargs):
        # Item at radial 1.5 with max_radius=1.0 → radial branch fires.
        positions = {"pick_0": np.array([1.5, 0.0, 0.05])}
        return _make_ctx(
            mock_mode=False,
            pick_min_reachable_z=-0.10,
            pick_max_reachable_radius_xy=1.0,
            pick_positions=positions,
            **kwargs,
        )

    def test_out_of_radius_first_tick_returns_running(self):
        """First out-of-radius tick yields RUNNING (start of grace window)."""
        ctx = self._make_ctx_out_of_radius()
        b = CheckPickReachable()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.RUNNING
        pick_name = ctx.get_current_pick_name()
        assert pick_name in b._unreachable_since

    def test_out_of_radius_after_grace_marks_permanent(self):
        """Grace expiry promotes the pick to permanently-unreachable.

        Tick once to enter the grace window (RUNNING), then back-date
        the seed timestamp past the window and re-tick.
        """
        ctx = self._make_ctx_out_of_radius()
        b = CheckPickReachable()
        b.setup(context=ctx)
        b.tick_once()
        assert b.status == py_trees.common.Status.RUNNING
        pick_name = ctx.get_current_pick_name()
        assert not ctx.strategy.is_pick_permanently_unreachable(pick_name)
        b._unreachable_since[pick_name] = (
            ctx.get_current_sim_time() - (b.UNREACHABLE_GRACE_S + 1.0)
        )
        b.tick_once()
        assert b.status == py_trees.common.Status.FAILURE
        assert ctx.strategy.is_pick_permanently_unreachable(pick_name)
        # Per-pick state cleared on terminal promotion.
        assert pick_name not in b._unreachable_since

    def test_recovery_clears_grace_state(self):
        """If the item drifts back into reach, per-pick grace state is cleared."""
        ctx = self._make_ctx_out_of_radius()
        b = CheckPickReachable()
        b.setup(context=ctx)
        b.tick_once()
        pick_name = ctx.get_current_pick_name()
        assert pick_name in b._unreachable_since
        # Drift the item deep into the workspace.
        ctx.strategy.pick_objs_by_name[pick_name].set_world_pose(
            position=np.array([0.3, 0.0, 0.05]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        b.tick_once()
        assert b.status == py_trees.common.Status.SUCCESS
        assert pick_name not in b._unreachable_since

    def test_initialise_does_not_clear_grace_state(self):
        """initialise() must NOT clear _unreachable_since.

        The wrapping Retry decorator restarts pick_attempt on every
        FAILURE, which re-initialises every Behaviour in the sequence.
        Clearing the grace map there would multiply the wait by
        PICK_RETRY_BUDGET; the grace window must survive Retry restarts
        so a single window bounds the cost.
        """
        ctx = self._make_ctx_out_of_radius()
        b = CheckPickReachable()
        b.setup(context=ctx)
        b.tick_once()
        pick_name = ctx.get_current_pick_name()
        assert pick_name in b._unreachable_since
        b.initialise()
        assert pick_name in b._unreachable_since


class TestCheckGraspPoseReachable:
    """3D pre-grasp reachability gate.  Tests run with mock_mode=False so
    the gate is not short-circuited; ``PrepareGrasp`` is run first to
    cache a ``GraspPose`` on the context."""

    def _ctx_with_pick(self, pick_pos, r_max):
        positions = {"pick_0": np.array(pick_pos, dtype=float)}
        return _make_ctx(
            mock_mode=False,
            pick_max_reachable_radius_xy=r_max,
            pick_positions=positions,
        )

    def _prepare_and_gate(self, ctx):
        PrepareGrasp().tick_once  # noqa — keeps the linter quiet
        prep = PrepareGrasp()
        prep.setup(context=ctx)
        prep.tick_once()
        assert prep.status == py_trees.common.Status.SUCCESS, "PrepareGrasp must seed GraspPose"
        gate = CheckGraspPoseReachable()
        gate.setup(context=ctx)
        return gate

    def test_in_sphere_returns_success(self):
        # pick at radial 0.5, low z → pre-grasp dist3d well under r_max=1.0.
        ctx = self._ctx_with_pick(pick_pos=[0.5, 0.0, 0.05], r_max=1.0)
        gate = self._prepare_and_gate(ctx)
        gate.tick_once()
        assert gate.status == py_trees.common.Status.SUCCESS

    def test_out_of_sphere_first_tick_returns_running(self):
        """First out-of-sphere tick yields RUNNING (start of grace window)."""
        # pick at radial 1.5 → pre-grasp dist3d > r_max=1.0.
        ctx = self._ctx_with_pick(pick_pos=[1.5, 0.0, 0.05], r_max=1.0)
        gate = self._prepare_and_gate(ctx)
        gate.tick_once()
        assert gate.status == py_trees.common.Status.RUNNING

    def test_out_of_sphere_after_grace_returns_failure(self):
        """Once the grace window expires the gate yields FAILURE.

        Tick once to enter the grace window (RUNNING), then back-date
        the seed timestamp past the window and re-tick.
        """
        ctx = self._ctx_with_pick(pick_pos=[1.5, 0.0, 0.05], r_max=1.0)
        gate = self._prepare_and_gate(ctx)
        gate.tick_once()
        assert gate.status == py_trees.common.Status.RUNNING
        pick_name = ctx.get_current_pick_name()
        # Back-date so the next tick observes elapsed > grace.
        gate._unreachable_since[pick_name] = ctx.get_current_sim_time() - (gate.UNREACHABLE_GRACE_S + 1.0)
        gate.tick_once()
        assert gate.status == py_trees.common.Status.FAILURE

    def test_recovery_clears_grace_state(self):
        """If the pose becomes reachable again the per-pick grace state is cleared.

        Mirrors a moving conveyor: the item drifts deeper into the
        workspace between ticks.  We simulate that by editing the cached
        grasp pose.
        """
        ctx = self._ctx_with_pick(pick_pos=[1.5, 0.0, 0.05], r_max=1.0)
        gate = self._prepare_and_gate(ctx)
        gate.tick_once()
        assert gate.status == py_trees.common.Status.RUNNING
        pick_name = ctx.get_current_pick_name()
        assert pick_name in gate._unreachable_since
        # Move the pick deep into the workspace, refresh the grasp pose.
        ctx.strategy.pick_objs_by_name[pick_name].set_world_pose(
            position=np.array([0.3, 0.0, 0.05]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        prep = PrepareGrasp()
        prep.setup(context=ctx)
        prep.tick_once()
        gate.tick_once()
        assert gate.status == py_trees.common.Status.SUCCESS
        assert pick_name not in gate._unreachable_since

    def test_grace_expiry_marks_permanent(self):
        """When the grace window expires, the gate marks the pick
        permanently unreachable so the wrapping Retry / defer cycle
        cannot keep re-charging fresh grace windows."""
        ctx = self._ctx_with_pick(pick_pos=[1.5, 0.0, 0.05], r_max=1.0)
        gate = self._prepare_and_gate(ctx)
        gate.tick_once()  # populates _unreachable_since (RUNNING)
        pick_name = ctx.get_current_pick_name()
        assert not ctx.strategy.is_pick_permanently_unreachable(pick_name)
        # Force grace expiry on the next tick.
        gate._unreachable_since[pick_name] = ctx.get_current_sim_time() - (gate.UNREACHABLE_GRACE_S + 1.0)
        gate.tick_once()
        assert gate.status == py_trees.common.Status.FAILURE
        assert ctx.strategy.is_pick_permanently_unreachable(pick_name)
        # State for that pick is cleared (the entry would be irrelevant
        # since the pick is now permanent).
        assert pick_name not in gate._unreachable_since

    def test_mock_mode_short_circuits_to_success(self):
        """In mock_mode (default) the gate must pass-through unconditionally."""
        ctx = _make_ctx()  # mock_mode=True default
        prep = PrepareGrasp()
        prep.setup(context=ctx)
        prep.tick_once()
        gate = CheckGraspPoseReachable()
        gate.setup(context=ctx)
        gate.tick_once()
        assert gate.status == py_trees.common.Status.SUCCESS

    def test_failure_when_no_grasp_pose(self):
        """If PrepareGrasp didn't run (no cached pose) the gate fails."""
        ctx = self._ctx_with_pick(pick_pos=[0.5, 0.0, 0.05], r_max=1.0)
        # Do NOT run PrepareGrasp.
        gate = CheckGraspPoseReachable()
        gate.setup(context=ctx)
        gate.tick_once()
        assert gate.status == py_trees.common.Status.FAILURE

    def test_initialise_does_not_clear_grace_state(self):
        """initialise() must NOT clear _unreachable_since.

        The wrapping Retry decorator restarts pick_attempt on every
        FAILURE, which re-initialises every Behaviour in the sequence.
        If we cleared _unreachable_since on initialise(), each retry
        would start a fresh 10 s grace window, multiplying the wait by
        PICK_RETRY_BUDGET (currently 5) for free.  The grace window must
        survive Retry restarts so a single 10 s window bounds the cost.
        """
        ctx = self._ctx_with_pick(pick_pos=[1.5, 0.0, 0.05], r_max=1.0)
        gate = self._prepare_and_gate(ctx)
        gate.tick_once()
        pick_name = ctx.get_current_pick_name()
        assert pick_name in gate._unreachable_since
        # initialise() must not wipe the grace timestamp.
        gate.initialise()
        assert pick_name in gate._unreachable_since

    def test_grace_survives_retry_restart(self):
        """Simulate a Retry restart by calling initialise() between ticks.
        The grace window must persist so 5×Retry doesn't multiply the wait."""
        ctx = self._ctx_with_pick(pick_pos=[1.5, 0.0, 0.05], r_max=1.0)
        gate = self._prepare_and_gate(ctx)
        gate.tick_once()
        pick_name = ctx.get_current_pick_name()
        first_t = gate._unreachable_since[pick_name]
        # Simulate a Retry restart.
        gate.initialise()
        gate.tick_once()
        # Same timestamp survives — no fresh grace window.
        assert gate._unreachable_since[pick_name] == first_t

    def test_live_pose_refresh_recovers_when_item_drifts_in(self):
        """A moving-conveyor item that enters the cylinder marginally
        out-of-sphere may drift fully into the sphere during the grace
        window.  The gate must read the LIVE item position each tick
        (not the snapshot cached by PrepareGrasp) so it sees the drift
        and SUCCESSes mid-window instead of incorrectly promoting."""
        # Pick starts at radial 1.5 → out of sphere (dist3d ≈ 1.5+ at low z).
        ctx = self._ctx_with_pick(pick_pos=[1.5, 0.0, 0.05], r_max=1.0)
        gate = self._prepare_and_gate(ctx)
        gate.tick_once()
        assert gate.status == py_trees.common.Status.RUNNING
        pick_name = ctx.get_current_pick_name()
        assert pick_name in gate._unreachable_since
        # Conveyor "drift": move item deep into the workspace.  The
        # cached GraspPose is still at the original (out-of-sphere)
        # location; the gate must consult the live picking_position.
        ctx.strategy.pick_objs_by_name[pick_name].set_world_pose(
            position=np.array([0.3, 0.0, 0.05]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        gate.tick_once()
        assert gate.status == py_trees.common.Status.SUCCESS
        # No permanent flag set, grace state cleared.
        assert not ctx.strategy.is_pick_permanently_unreachable(pick_name)
        assert pick_name not in gate._unreachable_since

    def test_live_pose_unaffected_when_item_stays_static(self):
        """When the item doesn't move the live-pose path must still see
        the same out-of-sphere distance and progress through the grace
        window as before — i.e. the live-pose refactor doesn't accidentally
        forgive static unreachable items."""
        ctx = self._ctx_with_pick(pick_pos=[1.5, 0.0, 0.05], r_max=1.0)
        gate = self._prepare_and_gate(ctx)
        gate.tick_once()
        assert gate.status == py_trees.common.Status.RUNNING
        pick_name = ctx.get_current_pick_name()
        # Don't move the item.  Force grace expiry.
        gate._unreachable_since[pick_name] = ctx.get_current_sim_time() - (gate.UNREACHABLE_GRACE_S + 1.0)
        gate.tick_once()
        assert gate.status == py_trees.common.Status.FAILURE
        assert ctx.strategy.is_pick_permanently_unreachable(pick_name)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
