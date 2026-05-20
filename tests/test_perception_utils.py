"""Unit tests for perception_utils.

Step 1 of the grasp-affordance refactor: dataclasses + low-level helpers
only.  The three ``compute_*`` entry points land in later steps; those
will get their own, larger test modules.
"""
import numpy as np
import pytest

from asset_data_utils import AssetSymmetry, PrimGeometry
from perception_utils import (
    DEFAULT_GRASP_HEIGHT_FALLBACK,
    DEFAULT_PICK_APPROACH_DIRECTION,
    DEFAULT_PICK_APPROACH_DISTANCE,
    DEFAULT_PICK_APPROACH_STD_DEV,
    DEFAULT_PLACE_APPROACH_DIRECTION,
    DEFAULT_PLACE_APPROACH_DISTANCE,
    DEFAULT_PLACE_APPROACH_STD_DEV,
    GraspPose,
    ItemInEEPose,
    PlacePose,
    _quat_angle,
    _quat_conjugate,
    _quat_multiply,
    _quaternions_equivalent,
    _rotate_offset_by_rel_quat,
    _swing_twist_decomp,
    compute_grasp_pose,
    compute_item_in_ee_pose,
    compute_place_pose,
)


IDENTITY_Q = np.array([1.0, 0.0, 0.0, 0.0])
HALF_TURN_X_Q = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # 90° around X
HALF_TURN_Y_Q = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # 90° around Y
GRIPPER_DOWN_Q = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # same as Y-90°
FLIP_Q = np.array([0.0, 0.0, 0.0, 1.0])  # 180° around Z


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


class TestQuaternionsEquivalent:
    def test_both_none(self):
        assert _quaternions_equivalent(None, None) is True

    def test_one_none(self):
        assert _quaternions_equivalent(None, IDENTITY_Q) is False
        assert _quaternions_equivalent(IDENTITY_Q, None) is False

    def test_identical(self):
        assert _quaternions_equivalent(IDENTITY_Q, IDENTITY_Q) is True

    def test_double_cover(self):
        # q and -q represent the same rotation.
        assert _quaternions_equivalent(HALF_TURN_X_Q, -HALF_TURN_X_Q) is True

    def test_different(self):
        assert _quaternions_equivalent(HALF_TURN_X_Q, HALF_TURN_Y_Q) is False

    def test_shape_mismatch(self):
        assert _quaternions_equivalent(IDENTITY_Q, np.array([1.0, 0.0, 0.0])) is False


class TestRotateOffsetByRelQuat:
    def test_identity_rotation_is_noop(self):
        offset = np.array([0.0, 0.0, 0.05])
        rotated = _rotate_offset_by_rel_quat(offset, IDENTITY_Q, IDENTITY_Q)
        np.testing.assert_allclose(rotated, offset, atol=1e-9)

    def test_bottle_drop_case(self):
        # Pick: gripper-down (Y-90°), offset = [0, 0, grasp_height] in world.
        # Drop: bottle-on-side (X-90° composed with gripper-down).  The
        # world-frame vertical offset becomes horizontal after the relative
        # rotation.  We only check magnitude preservation + that the Z
        # component collapses to zero (pure horizontal at the bottle drop).
        offset = np.array([0.0, 0.0, 0.05])
        pick_orient = GRIPPER_DOWN_Q
        # Drop = pick composed with 90° around X, expressed as a single quat.
        drop_orient = _quat_multiply(HALF_TURN_X_Q, GRIPPER_DOWN_Q)
        rotated = _rotate_offset_by_rel_quat(offset, pick_orient, drop_orient)
        # Magnitude must be preserved — it is a rigid rotation.
        np.testing.assert_allclose(
            np.linalg.norm(rotated), np.linalg.norm(offset), atol=1e-9,
        )
        # The 90° relative rotation about X sends +Z to -Y (or +Y depending
        # on handedness); check |Z| component becomes near-zero.
        assert abs(rotated[2]) < 1e-9
        # And |Y| picks up the full magnitude.
        assert abs(abs(rotated[1]) - np.linalg.norm(offset)) < 1e-9


class TestQuatUtilities:
    def test_conjugate(self):
        q = np.array([0.1, 0.2, 0.3, 0.4])
        np.testing.assert_allclose(
            _quat_conjugate(q), [0.1, -0.2, -0.3, -0.4], atol=1e-12,
        )

    def test_multiply_identity(self):
        q = HALF_TURN_X_Q
        np.testing.assert_allclose(_quat_multiply(IDENTITY_Q, q), q, atol=1e-12)
        np.testing.assert_allclose(_quat_multiply(q, IDENTITY_Q), q, atol=1e-12)

    def test_angle_identity(self):
        assert _quat_angle(IDENTITY_Q) == pytest.approx(0.0, abs=1e-12)

    def test_angle_half_turn(self):
        # 90° rotation → π/2 rad.
        assert _quat_angle(HALF_TURN_X_Q) == pytest.approx(np.pi / 2, abs=1e-9)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


def _make_grasp_pose(**overrides):
    defaults = dict(
        ee_position=np.array([0.5, 0.0, 0.15]),
        ee_orientation=GRIPPER_DOWN_Q,
        approach_direction=np.array([0.0, 0.0, -1.0]),
        approach_distance=0.1,
        approach_std_dev=0.005,
        ee_offset_world_at_grasp=np.array([0.0, 0.0, 0.05]),
    )
    defaults.update(overrides)
    return GraspPose(**defaults)


class TestGraspPose:
    def test_construct_with_defaults(self):
        pose = _make_grasp_pose()
        np.testing.assert_allclose(pose.ee_position, [0.5, 0.0, 0.15])
        np.testing.assert_allclose(pose.ee_orientation, GRIPPER_DOWN_Q)
        np.testing.assert_allclose(pose.approach_direction, [0.0, 0.0, -1.0])
        assert pose.approach_distance == 0.1
        assert pose.approach_std_dev == 0.005
        np.testing.assert_allclose(pose.ee_offset_world_at_grasp, [0.0, 0.0, 0.05])

    def test_pre_grasp_position(self):
        pose = _make_grasp_pose(
            ee_position=np.array([0.5, 0.0, 0.15]),
            approach_direction=np.array([0.0, 0.0, -1.0]),
            approach_distance=0.1,
        )
        # pre_grasp = ee_position - direction * distance = 0.15 - (-1.0 * 0.1) = 0.25
        np.testing.assert_allclose(pose.pre_grasp_position, [0.5, 0.0, 0.25])

    def test_normalises_approach_direction(self):
        # A non-unit vector must be normalised in place.
        pose = _make_grasp_pose(approach_direction=np.array([0.0, 0.0, -2.0]))
        np.testing.assert_allclose(pose.approach_direction, [0.0, 0.0, -1.0])

    def test_zero_approach_direction_raises(self):
        with pytest.raises(ValueError, match="approach_direction"):
            _make_grasp_pose(approach_direction=np.array([0.0, 0.0, 0.0]))

    def test_defensive_copy_of_inputs(self):
        ee_pos = np.array([0.5, 0.0, 0.15])
        offset = np.array([0.0, 0.0, 0.05])
        pose = _make_grasp_pose(ee_position=ee_pos, ee_offset_world_at_grasp=offset)
        ee_pos[0] = 99.0
        offset[2] = 99.0
        # Mutating the caller's arrays must not affect the cached pose.
        assert pose.ee_position[0] == 0.5
        assert pose.ee_offset_world_at_grasp[2] == 0.05

    def test_item_at_grasp_defaults_to_none(self):
        pose = _make_grasp_pose()
        assert pose.item_position_at_grasp is None
        assert pose.item_orientation_at_grasp is None

    def test_item_at_grasp_round_trip(self):
        item_pos = np.array([0.5, 0.1, 0.1])
        item_q = HALF_TURN_X_Q
        pose = _make_grasp_pose(
            item_position_at_grasp=item_pos,
            item_orientation_at_grasp=item_q,
        )
        np.testing.assert_allclose(pose.item_position_at_grasp, [0.5, 0.1, 0.1])
        np.testing.assert_allclose(pose.item_orientation_at_grasp, HALF_TURN_X_Q)

    def test_item_at_grasp_defensive_copy(self):
        item_pos = np.array([0.5, 0.1, 0.1])
        item_q = np.array(HALF_TURN_X_Q, copy=True)
        pose = _make_grasp_pose(
            item_position_at_grasp=item_pos,
            item_orientation_at_grasp=item_q,
        )
        item_pos[0] = 99.0
        item_q[0] = 99.0
        assert pose.item_position_at_grasp[0] == 0.5
        assert pose.item_orientation_at_grasp[0] == pytest.approx(HALF_TURN_X_Q[0])


def _make_item_in_ee(**overrides):
    defaults = dict(
        position_in_ee=np.array([0.0, 0.0, -0.05]),
        orientation_in_ee=IDENTITY_Q,
        position_error=0.001,
        orientation_error_rad=0.01,
    )
    defaults.update(overrides)
    return ItemInEEPose(**defaults)


class TestItemInEEPose:
    def test_construct(self):
        pose = _make_item_in_ee()
        np.testing.assert_allclose(pose.position_in_ee, [0.0, 0.0, -0.05])
        np.testing.assert_allclose(pose.orientation_in_ee, IDENTITY_Q)
        assert pose.position_error == 0.001
        assert pose.orientation_error_rad == 0.01

    def test_defensive_copy(self):
        p = np.array([0.0, 0.0, -0.05])
        pose = _make_item_in_ee(position_in_ee=p)
        p[2] = 99.0
        assert pose.position_in_ee[2] == -0.05


def _make_place_pose(**overrides):
    defaults = dict(
        ee_position=np.array([0.7, 0.2, 0.3]),
        ee_orientation=GRIPPER_DOWN_Q,
        approach_direction=np.array([0.0, 0.0, -1.0]),
        approach_distance=0.2,
        approach_std_dev=0.02,
        insert_z=0.1,
    )
    defaults.update(overrides)
    return PlacePose(**defaults)


class TestPlacePose:
    def test_construct(self):
        pose = _make_place_pose()
        np.testing.assert_allclose(pose.ee_position, [0.7, 0.2, 0.3])
        assert pose.approach_distance == 0.2
        assert pose.insert_z == 0.1

    def test_pre_place_position(self):
        pose = _make_place_pose(
            ee_position=np.array([0.7, 0.2, 0.3]),
            approach_direction=np.array([0.0, 0.0, -1.0]),
            approach_distance=0.2,
        )
        np.testing.assert_allclose(pose.pre_place_position, [0.7, 0.2, 0.5])

    def test_normalises_approach_direction(self):
        pose = _make_place_pose(approach_direction=np.array([0.0, 0.0, -3.0]))
        np.testing.assert_allclose(pose.approach_direction, [0.0, 0.0, -1.0])

    def test_zero_approach_direction_raises(self):
        with pytest.raises(ValueError, match="approach_direction"):
            _make_place_pose(approach_direction=np.array([0.0, 0.0, 0.0]))


# ---------------------------------------------------------------------------
# compute_grasp_pose
# ---------------------------------------------------------------------------


def _make_geom(grasp_height=0.05):
    return PrimGeometry(
        grasp_height=grasp_height,
        rest_height=grasp_height,
        top_surface_height=grasp_height,
        local_half_extents=np.array([0.05, 0.05, grasp_height]),
        needs_aabb_scale_correction=False,
    )


class TestComputeGraspPose:
    def test_basic_vertical_grasp(self):
        pick_pos = np.array([0.5, 0.1, 0.1])
        geom = _make_geom(grasp_height=0.05)
        pose = compute_grasp_pose(
            "pick_0",
            pick_position=pick_pos,
            pick_geometry=geom,
            pick_orientation_preference=GRIPPER_DOWN_Q,
        )
        # EE sits at pick_pos + [0, 0, grasp_height]
        np.testing.assert_allclose(pose.ee_position, [0.5, 0.1, 0.15])
        # Orientation passed through unchanged.
        np.testing.assert_allclose(pose.ee_orientation, GRIPPER_DOWN_Q)
        # Default approach params.
        np.testing.assert_allclose(pose.approach_direction, DEFAULT_PICK_APPROACH_DIRECTION)
        assert pose.approach_distance == DEFAULT_PICK_APPROACH_DISTANCE
        assert pose.approach_std_dev == DEFAULT_PICK_APPROACH_STD_DEV
        # Offset is the grasp-height vertical vector.
        np.testing.assert_allclose(pose.ee_offset_world_at_grasp, [0.0, 0.0, 0.05])

    def test_fallback_when_geometry_none(self):
        pose = compute_grasp_pose(
            "pick_0",
            pick_position=np.array([0.0, 0.0, 0.0]),
            pick_geometry=None,
            pick_orientation_preference=GRIPPER_DOWN_Q,
        )
        np.testing.assert_allclose(
            pose.ee_offset_world_at_grasp, [0.0, 0.0, DEFAULT_GRASP_HEIGHT_FALLBACK],
        )
        np.testing.assert_allclose(
            pose.ee_position, [0.0, 0.0, DEFAULT_GRASP_HEIGHT_FALLBACK],
        )

    def test_custom_approach_overrides(self):
        geom = _make_geom()
        pose = compute_grasp_pose(
            "pick_0",
            pick_position=np.array([0.5, 0.0, 0.1]),
            pick_geometry=geom,
            pick_orientation_preference=GRIPPER_DOWN_Q,
            approach_direction=np.array([0.0, -1.0, 0.0]),  # horizontal approach
            approach_distance=0.15,
            approach_std_dev=0.01,
        )
        np.testing.assert_allclose(pose.approach_direction, [0.0, -1.0, 0.0])
        assert pose.approach_distance == 0.15
        assert pose.approach_std_dev == 0.01

    def test_pre_grasp_position_is_upstream_of_approach(self):
        geom = _make_geom(grasp_height=0.05)
        pose = compute_grasp_pose(
            "pick_0",
            pick_position=np.array([0.5, 0.0, 0.1]),
            pick_geometry=geom,
            pick_orientation_preference=GRIPPER_DOWN_Q,
        )
        # Default approach is [0,0,-1] * 0.2; pre-grasp sits 0.2 above ee_position.
        np.testing.assert_allclose(
            pose.pre_grasp_position,
            pose.ee_position - DEFAULT_PICK_APPROACH_DIRECTION * DEFAULT_PICK_APPROACH_DISTANCE,
        )

    def test_passes_through_item_at_grasp(self):
        geom = _make_geom(grasp_height=0.05)
        pose = compute_grasp_pose(
            "pick_0",
            pick_position=np.array([0.5, 0.1, 0.1]),
            pick_geometry=geom,
            pick_orientation_preference=GRIPPER_DOWN_Q,
            item_position_at_grasp=np.array([0.5, 0.1, 0.1]),
            item_orientation_at_grasp=HALF_TURN_X_Q,
        )
        np.testing.assert_allclose(pose.item_position_at_grasp, [0.5, 0.1, 0.1])
        np.testing.assert_allclose(pose.item_orientation_at_grasp, HALF_TURN_X_Q)

    def test_item_at_grasp_none_by_default(self):
        geom = _make_geom(grasp_height=0.05)
        pose = compute_grasp_pose(
            "pick_0",
            pick_position=np.array([0.5, 0.1, 0.1]),
            pick_geometry=geom,
            pick_orientation_preference=GRIPPER_DOWN_Q,
        )
        assert pose.item_position_at_grasp is None
        assert pose.item_orientation_at_grasp is None


class TestComputeGraspPoseWithOffset:
    """``grasp_offset_world`` shifts the EE flange by an additional world-frame
    vector relative to the item centre.  The shifted offset is also latched
    onto the resulting ``GraspPose.ee_offset_world_at_grasp`` so downstream
    place math sees the same expected offset.
    """

    def test_offset_shifts_ee_position(self):
        pick_pos = np.array([0.5, 0.1, 0.1])
        geom = _make_geom(grasp_height=0.05)
        # Bottle "toward cap" offset, rotated into world frame:
        # for a bottle laid horizontally along world Y, +Z-local maps to
        # +Y-world.  Here we simulate that with a pure world-Y offset.
        offset_world = np.array([0.0, 0.015, 0.0])
        pose = compute_grasp_pose(
            "pick_0",
            pick_position=pick_pos,
            pick_geometry=geom,
            pick_orientation_preference=IDENTITY_Q,
            grasp_offset_world=offset_world,
        )
        np.testing.assert_allclose(
            pose.ee_position, [0.5, 0.115, 0.15], atol=1e-12,
        )
        np.testing.assert_allclose(
            pose.ee_offset_world_at_grasp, [0.0, 0.015, 0.05], atol=1e-12,
        )

    def test_zero_offset_is_noop(self):
        pick_pos = np.array([0.5, 0.0, 0.1])
        geom = _make_geom(grasp_height=0.05)
        baseline = compute_grasp_pose(
            "pick_0",
            pick_position=pick_pos,
            pick_geometry=geom,
            pick_orientation_preference=IDENTITY_Q,
        )
        with_zero = compute_grasp_pose(
            "pick_0",
            pick_position=pick_pos,
            pick_geometry=geom,
            pick_orientation_preference=IDENTITY_Q,
            grasp_offset_world=np.zeros(3),
        )
        np.testing.assert_allclose(with_zero.ee_position, baseline.ee_position)
        np.testing.assert_allclose(
            with_zero.ee_offset_world_at_grasp, baseline.ee_offset_world_at_grasp,
        )


# ---------------------------------------------------------------------------
# compute_place_pose
# ---------------------------------------------------------------------------


class TestComputePlacePose:
    """``compute_place_pose`` takes a pre-adjusted target position (``[2]`` is
    the drop-Z base after target.top_surface + pick.rest) and handles the
    drop-side EE offset + above hover offset.
    """

    def test_same_orientation_simple_insert(self):
        # Caller has pre-adjusted target_position[2] to drop_z = 0.07.
        # pick_geom grasp_height = 0.04 → drop_offset = [0, 0, 0.04].
        # insert_z = 0.07 + 0.04 = 0.11; ee_position[2] = 0.11 + 0.2 above = 0.31.
        pick_geom = _make_geom(grasp_height=0.04)
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=np.array([0.7, 0.2, 0.07]),
            pick_geometry=pick_geom,
            pick_orientation=GRIPPER_DOWN_Q,
            drop_orientation=None,       # same-as-pick
            above=0.2,
        )
        np.testing.assert_allclose(pose.ee_position[:2], [0.7, 0.2])
        assert pose.insert_z == pytest.approx(0.11, abs=1e-9)
        assert pose.ee_position[2] == pytest.approx(0.11 + 0.2, abs=1e-9)
        np.testing.assert_allclose(pose.ee_orientation, GRIPPER_DOWN_Q)

    def test_different_drop_orientation_rotates_offset(self):
        # Bottle case: pick with EE down, drop with 90° X-flip.  The
        # vertical grasp offset rotates into a horizontal offset.
        pick_geom = _make_geom(grasp_height=0.04)
        pick_orient = GRIPPER_DOWN_Q
        drop_orient = _quat_multiply(HALF_TURN_X_Q, GRIPPER_DOWN_Q)
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=np.array([0.7, 0.2, 0.05]),
            pick_geometry=pick_geom,
            pick_orientation=pick_orient,
            drop_orientation=drop_orient,
            above=0.0,
        )
        np.testing.assert_allclose(pose.ee_orientation, drop_orient, atol=1e-12)
        # Rotated offset: vertical [0,0,0.04] becomes horizontal; Z component
        # ~0, Y picks up the magnitude.
        diff = pose.ee_position - np.array([0.7, 0.2, 0.05])
        assert abs(diff[2]) < 1e-9
        assert abs(abs(diff[1]) - 0.04) < 1e-9

    def test_fallback_when_pick_geometry_missing(self):
        # Without pick_geometry the drop offset uses the fallback
        # [0, 0, DEFAULT_GRASP_HEIGHT_FALLBACK] (matches legacy
        # _EE_OFFSET_FALLBACK behaviour).
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=np.array([0.7, 0.2, 0.05]),
            pick_geometry=None,
            pick_orientation=GRIPPER_DOWN_Q,
            drop_orientation=None,
            above=0.0,
        )
        np.testing.assert_allclose(
            pose.ee_position, [0.7, 0.2, 0.05 + DEFAULT_GRASP_HEIGHT_FALLBACK],
        )
        assert pose.insert_z == pytest.approx(
            0.05 + DEFAULT_GRASP_HEIGHT_FALLBACK, abs=1e-12,
        )

    def test_default_approach_params(self):
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=np.array([0.7, 0.2, 0.05]),
            pick_geometry=None,
            pick_orientation=GRIPPER_DOWN_Q,
            drop_orientation=None,
        )
        np.testing.assert_allclose(pose.approach_direction, DEFAULT_PLACE_APPROACH_DIRECTION)
        assert pose.approach_distance == DEFAULT_PLACE_APPROACH_DISTANCE
        assert pose.approach_std_dev == DEFAULT_PLACE_APPROACH_STD_DEV

    def test_equivalent_drop_short_circuits_identity(self):
        # When drop_orientation == pick_orientation (double-cover-safe),
        # no rotation is applied — even though drop_orientation is not None.
        pick_geom = _make_geom(grasp_height=0.04)
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=np.array([0.0, 0.0, 0.1]),
            pick_geometry=pick_geom,
            pick_orientation=GRIPPER_DOWN_Q,
            drop_orientation=-GRIPPER_DOWN_Q,  # same rotation via double cover
            above=0.0,
        )
        np.testing.assert_allclose(pose.ee_position, [0.0, 0.0, 0.1 + 0.04])

    def test_offset_shifts_nominal_drop_same_orientation(self):
        # When drop orientation == pick, the offset adds directly.
        pick_geom = _make_geom(grasp_height=0.04)
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=np.array([0.7, 0.2, 0.05]),
            pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q,
            drop_orientation=None,
            above=0.0,
            grasp_offset_world=np.array([0.0, 0.015, 0.0]),
        )
        # ee_position = target + [0,0,grasp_height] + offset_world
        np.testing.assert_allclose(
            pose.ee_position, [0.7, 0.215, 0.09], atol=1e-12,
        )

    def test_offset_rotates_with_drop_orientation(self):
        # Bottle case: pick with identity, drop with 90° X-flip.  The
        # combined offset ([0,0,grasp_height] + grasp_offset_world) rotates
        # under R_drop · R_pickᵀ.  For X-90° rotation, +Y stays as +Y? No:
        # quaternion HALF_TURN_X_Q rotates +Z → +Y? Check: with q=cos45 +
        # sin45 X, R rotates +Y → +Z and +Z → -Y.  So a [0, 0.015, 0.04]
        # composite world-frame offset at pick orientation becomes
        # [0, ?, ?] post-rotation — Y component flips into Z, Z into -Y.
        pick_geom = _make_geom(grasp_height=0.04)
        target = np.array([0.7, 0.2, 0.05])
        offset_world = np.array([0.0, 0.015, 0.0])
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=target,
            pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q,
            drop_orientation=HALF_TURN_X_Q,
            above=0.0,
            grasp_offset_world=offset_world,
        )
        # Composite pre-rotation offset is [0, 0.015, 0.04].
        # R(HALF_TURN_X_Q) maps (x, y, z) -> (x, -z, y).
        expected_offset = np.array([0.0, -0.04, 0.015])
        np.testing.assert_allclose(
            pose.ee_position, target + expected_offset, atol=1e-12,
        )

    def test_offset_zero_matches_no_offset(self):
        # Passing grasp_offset_world=zeros must match the no-offset call.
        pick_geom = _make_geom(grasp_height=0.04)
        target = np.array([0.7, 0.2, 0.05])
        no_offset = compute_place_pose(
            "pick_0", "target_0",
            target_position=target,
            pick_geometry=pick_geom,
            pick_orientation=GRIPPER_DOWN_Q,
            drop_orientation=None,
        )
        zero_offset = compute_place_pose(
            "pick_0", "target_0",
            target_position=target,
            pick_geometry=pick_geom,
            pick_orientation=GRIPPER_DOWN_Q,
            drop_orientation=None,
            grasp_offset_world=np.zeros(3),
        )
        np.testing.assert_allclose(zero_offset.ee_position, no_offset.ee_position)


def _make_rest_geom(grasp_height=0.04, rest_height=None):
    """PrimGeometry with independent grasp_height/rest_height for the
    measured-branch tests (nominal fixtures use grasp==rest)."""
    if rest_height is None:
        rest_height = grasp_height
    return PrimGeometry(
        grasp_height=grasp_height,
        rest_height=rest_height,
        top_surface_height=grasp_height,
        local_half_extents=np.array([0.05, 0.05, rest_height]),
        needs_aabb_scale_correction=False,
    )


class TestComputePlacePoseMeasured:
    """Step 5: ``compute_place_pose`` with a measured ``item_in_ee``.

    Together these exercise (A) orientation-aware insert Z and
    (B) item-centre-over-target position correction.
    """

    def test_nominal_measurement_matches_nominal_branch(self):
        # When the measurement matches the expected grasp geometry
        # (no slip, no tilt), the measured branch produces the same
        # ee_position as the nominal branch — regression guard.
        pick_geom = _make_rest_geom(grasp_height=0.04, rest_height=0.04)
        target = np.array([0.7, 0.2, 0.09])  # caller baked in rest_height

        nominal = compute_place_pose(
            "pick_0", "target_0",
            target_position=target, pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q, drop_orientation=None,
            item_in_ee=None, above=0.0,
        )
        # Expected item-in-ee for identity grasp-orientation: item sits
        # at [0, 0, -grasp_height] relative to the flange (along -Z of
        # the EE local frame = world -Z for identity EE orientation).
        measured_item = _make_item_in_ee(
            position_in_ee=np.array([0.0, 0.0, -0.04]),
            orientation_in_ee=IDENTITY_Q,
        )
        measured = compute_place_pose(
            "pick_0", "target_0",
            target_position=target, pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q, drop_orientation=None,
            item_in_ee=measured_item, above=0.0,
        )
        np.testing.assert_allclose(
            measured.ee_position, nominal.ee_position, atol=1e-9,
        )
        assert measured.insert_z == pytest.approx(nominal.insert_z, abs=1e-9)

    def test_lateral_slip_shifts_ee_xy_toward_slipped_item(self):
        # Item sits 2 cm off the flange along EE local +X (identity
        # drop orientation → world +X).  To put the item centre on the
        # target, the EE must shift by -2 cm in world X.
        pick_geom = _make_rest_geom(grasp_height=0.04, rest_height=0.04)
        target = np.array([0.5, 0.0, 0.09])
        measured_item = _make_item_in_ee(
            position_in_ee=np.array([0.02, 0.0, -0.04]),
            orientation_in_ee=IDENTITY_Q,
        )
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=target, pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q, drop_orientation=None,
            item_in_ee=measured_item, above=0.0,
        )
        # ee_x = target_x - R_drop @ position_in_ee[0] = 0.5 - 0.02 = 0.48
        assert pose.ee_position[0] == pytest.approx(0.48, abs=1e-9)
        assert pose.ee_position[1] == pytest.approx(0.0, abs=1e-9)

    def test_tilted_grasp_does_not_change_insert_z(self):
        # Orientation-aware insert-Z correction is a follow-up.  For
        # now, ``orientation_in_ee`` on the measured ItemInEEPose does
        # not influence the place Z — only ``position_in_ee`` does.
        # A 10° tilt (with nominal position_in_ee) therefore produces
        # the same ee_position as an untilted grasp.
        rest_height = 0.1
        pick_geom = _make_rest_geom(grasp_height=0.04, rest_height=rest_height)
        target = np.array([0.5, 0.0, 0.12])
        tilt_angle = np.deg2rad(10.0)
        tilt_q = np.array(
            [np.cos(tilt_angle / 2), np.sin(tilt_angle / 2), 0.0, 0.0]
        )
        measured_tilted = _make_item_in_ee(
            position_in_ee=np.array([0.0, 0.0, -0.04]),
            orientation_in_ee=tilt_q,
        )
        measured_untilted = _make_item_in_ee(
            position_in_ee=np.array([0.0, 0.0, -0.04]),
            orientation_in_ee=IDENTITY_Q,
        )
        pose_tilted = compute_place_pose(
            "pick_0", "target_0",
            target_position=target, pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q, drop_orientation=None,
            item_in_ee=measured_tilted, above=0.0,
        )
        pose_untilted = compute_place_pose(
            "pick_0", "target_0",
            target_position=target, pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q, drop_orientation=None,
            item_in_ee=measured_untilted, above=0.0,
        )
        np.testing.assert_allclose(
            pose_tilted.ee_position, pose_untilted.ee_position, atol=1e-9,
        )
        # Sanity: ee_z matches the nominal target_z + grasp_height form
        # (target_z = 0.12, R_drop @ position_in_ee = [0, 0, -0.04]
        # → ee_z = 0.12 - (-0.04) = 0.16).
        assert pose_tilted.ee_position[2] == pytest.approx(0.16, abs=1e-9)

    def test_slip_only_shifts_ee_for_tilted_item(self):
        # Orientation is tilted, but only XY slip influences ee_position.
        # (Equivalent test to the non-tilted slip case — regression guard
        # that tilt doesn't couple into the XY correction either.)
        rest_height = 0.1
        pick_geom = _make_rest_geom(grasp_height=0.04, rest_height=rest_height)
        target = np.array([0.5, 0.3, 0.12])
        tilt_angle = np.deg2rad(10.0)
        tilt_q = np.array(
            [np.cos(tilt_angle / 2), np.sin(tilt_angle / 2), 0.0, 0.0]
        )
        measured_item = _make_item_in_ee(
            position_in_ee=np.array([0.02, -0.01, -0.04]),
            orientation_in_ee=tilt_q,
        )
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=target, pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q, drop_orientation=None,
            item_in_ee=measured_item, above=0.0,
        )
        # R_drop = I (pick_orient = IDENTITY), so
        # R_drop @ [0.02, -0.01, -0.04] = [0.02, -0.01, -0.04].
        # ee = target - that = [0.48, 0.31, 0.16].
        np.testing.assert_allclose(
            pose.ee_position, [0.48, 0.31, 0.16], atol=1e-9,
        )

    def test_no_pick_geometry_skips_oriented_z_correction(self):
        # With pick_geometry None, rest_height is unknown — orientation-aware
        # Z correction is skipped.  Position correction still applies.
        target = np.array([0.5, 0.0, 0.1])
        measured_item = _make_item_in_ee(
            position_in_ee=np.array([0.02, 0.0, -0.04]),
            orientation_in_ee=IDENTITY_Q,
        )
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=target, pick_geometry=None,
            pick_orientation=IDENTITY_Q, drop_orientation=None,
            item_in_ee=measured_item, above=0.0,
        )
        # ee = target - R_drop @ pos_in_ee = [0.5, 0.0, 0.1] - [0.02, 0.0, -0.04]
        #    = [0.48, 0.0, 0.14]
        np.testing.assert_allclose(pose.ee_position, [0.48, 0.0, 0.14], atol=1e-9)

    def test_zero_magnitude_position_falls_back_to_nominal(self):
        # Defensive: a placeholder ItemInEEPose with exactly zero
        # position_in_ee falls back to the nominal branch (so a stray
        # teleport placeholder can't collapse the EE onto the target).
        pick_geom = _make_rest_geom(grasp_height=0.04, rest_height=0.04)
        target = np.array([0.5, 0.0, 0.09])
        placeholder = _make_item_in_ee(
            position_in_ee=np.zeros(3),
            orientation_in_ee=IDENTITY_Q,
        )
        measured = compute_place_pose(
            "pick_0", "target_0",
            target_position=target, pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q, drop_orientation=None,
            item_in_ee=placeholder, above=0.0,
        )
        nominal = compute_place_pose(
            "pick_0", "target_0",
            target_position=target, pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q, drop_orientation=None,
            item_in_ee=None, above=0.0,
        )
        np.testing.assert_allclose(
            measured.ee_position, nominal.ee_position, atol=1e-9,
        )

    def test_insert_z_uses_pre_above_value(self):
        # ``insert_z`` is the EE flange Z at drop (pre-``above``); the
        # returned ee_position[2] adds ``above`` on top.  For a perfect
        # identity-orientation grasp with position_in_ee = [0, 0, -gh],
        # insert_z == target[2] + grasp_height == nominal branch.
        pick_geom = _make_rest_geom(grasp_height=0.04, rest_height=0.04)
        target = np.array([0.5, 0.0, 0.09])
        measured_item = _make_item_in_ee(
            position_in_ee=np.array([0.0, 0.0, -0.04]),
            orientation_in_ee=IDENTITY_Q,
        )
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=target, pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q, drop_orientation=None,
            item_in_ee=measured_item, above=0.2,
        )
        assert pose.insert_z == pytest.approx(0.13, abs=1e-9)
        assert pose.ee_position[2] == pytest.approx(0.33, abs=1e-9)


def _axis_angle_quat(axis, angle_rad):
    """Build a [w, x, y, z] unit quaternion from an axis / angle pair."""
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    half = float(angle_rad) / 2.0
    s = np.sin(half)
    return np.array([np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s])


class TestSwingTwistDecomp:
    """Unit tests for the swing-twist helper used by the symmetry-aware correction."""

    def test_pure_twist_gives_identity_swing(self):
        # Rotation 30° about +Z decomposed w.r.t. Z-axis → swing == identity.
        q = _axis_angle_quat([0, 0, 1], np.deg2rad(30.0))
        q_swing, q_twist = _swing_twist_decomp(q, np.array([0.0, 0.0, 1.0]))
        assert _quaternions_equivalent(q_swing, IDENTITY_Q)
        assert _quaternions_equivalent(q_twist, q)

    def test_pure_swing_gives_identity_twist(self):
        # Rotation 30° about +X decomposed w.r.t. Z-axis → twist == identity.
        q = _axis_angle_quat([1, 0, 0], np.deg2rad(30.0))
        q_swing, q_twist = _swing_twist_decomp(q, np.array([0.0, 0.0, 1.0]))
        assert _quaternions_equivalent(q_twist, IDENTITY_Q)
        assert _quaternions_equivalent(q_swing, q)

    def test_mixed_rotation_round_trip(self):
        # Arbitrary rotation decomposed → q_swing * q_twist reconstructs q.
        q = _axis_angle_quat([0.3, 0.4, 0.5], np.deg2rad(40.0))
        axis = np.array([0.0, 0.0, 1.0])
        q_swing, q_twist = _swing_twist_decomp(q, axis)
        reconstructed = _quat_multiply(q_swing, q_twist)
        assert _quaternions_equivalent(reconstructed, q)
        # q_twist's axis component must be aligned with the decomposition axis.
        v = q_twist[1:4]
        if np.linalg.norm(v) > 1e-9:
            cos_sim = abs(float(np.dot(v / np.linalg.norm(v), axis)))
            assert cos_sim > 1 - 1e-6

    def test_non_canonical_axis(self):
        # Sanity: swing-twist works with a non-canonical (tilted) axis.
        axis = np.array([0.0, 1.0, 1.0])
        axis = axis / np.linalg.norm(axis)
        q = _axis_angle_quat(axis, np.deg2rad(25.0))
        q_swing, q_twist = _swing_twist_decomp(q, axis)
        # Pure rotation about axis → swing == identity.
        assert _quaternions_equivalent(q_swing, IDENTITY_Q)


class TestComputePlacePoseOrientationRestore:
    """Tests for the symmetry-aware orientation-restoring EE correction.

    Behavior:
      - Asset with no symmetry tag → correction disabled, EE orient
        unchanged (current behaviour preserved).
      - Asset tagged 'full' (sphere) → correction disabled, every
        rotation is a symmetry.
      - Asset tagged 'continuous_axis' → the observable tilt (swing)
        of the symmetry axis is corrected by pre-rotating the drop EE
        by the inverse swing; the twist about the symmetry axis is
        absorbed and leaves EE orientation unchanged.
    """

    def _make_geom(self, ref_q, symmetry=None):
        return PrimGeometry(
            grasp_height=0.04, rest_height=0.04, top_surface_height=0.04,
            local_half_extents=np.array([0.05, 0.05, 0.04]),
            needs_aabb_scale_correction=False,
            reference_orientation=ref_q,
            symmetry=symmetry,
        )

    def test_untagged_asset_no_correction(self):
        # symmetry=None (default) → current-behaviour preservation:
        # ee_orientation unchanged even when the item is jostled.
        pick_geom = self._make_geom(IDENTITY_Q, symmetry=None)
        q_jostled = _axis_angle_quat([0, 0, 1], np.deg2rad(15.0))
        measured = _make_item_in_ee(
            position_in_ee=np.array([0.0, 0.0, -0.04]),
            orientation_in_ee=q_jostled,
        )
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=np.array([0.5, 0.0, 0.09]),
            pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q,
            drop_orientation=None,
            item_in_ee=measured, above=0.0,
        )
        assert _quaternions_equivalent(pose.ee_orientation, IDENTITY_Q)

    def test_full_symmetry_no_correction(self):
        # Every rotation is a symmetry → ee_orientation unchanged for ANY jostle.
        pick_geom = self._make_geom(
            IDENTITY_Q, symmetry=AssetSymmetry(kind="full"),
        )
        # Pick an arbitrary off-axis jostle.
        q_jostled = _axis_angle_quat([0.3, 0.4, 0.5], np.deg2rad(22.0))
        measured = _make_item_in_ee(
            position_in_ee=np.array([0.0, 0.0, -0.04]),
            orientation_in_ee=q_jostled,
        )
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=np.array([0.5, 0.0, 0.09]),
            pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q,
            drop_orientation=None,
            item_in_ee=measured, above=0.0,
        )
        assert _quaternions_equivalent(pose.ee_orientation, IDENTITY_Q)

    def test_continuous_axis_twist_absorbed(self):
        # Item with axis symmetry about local Z, jostled purely ABOUT that
        # axis (15° roll) → ee_orientation unchanged (twist absorbed).
        # This is the bottle-rolling-in-bin case that previously drove the
        # wrist toward unreachable poses under the naive correction.
        sym = AssetSymmetry(
            kind="continuous_axis", axis_local=np.array([0.0, 0.0, 1.0]),
        )
        pick_geom = self._make_geom(IDENTITY_Q, symmetry=sym)
        q_jostled = _axis_angle_quat([0, 0, 1], np.deg2rad(15.0))
        measured = _make_item_in_ee(
            position_in_ee=np.array([0.0, 0.0, -0.04]),
            orientation_in_ee=q_jostled,
        )
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=np.array([0.5, 0.0, 0.09]),
            pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q,
            drop_orientation=None,
            item_in_ee=measured, above=0.0,
        )
        assert _quaternions_equivalent(pose.ee_orientation, IDENTITY_Q)

    def test_continuous_axis_tilt_corrected(self):
        # Item with axis symmetry about local Z, jostled purely
        # ORTHOGONAL to that axis (10° tilt about +X) → ee_orientation
        # gets pre-rotated by the inverse tilt so the symmetry axis
        # lands vertical.
        sym = AssetSymmetry(
            kind="continuous_axis", axis_local=np.array([0.0, 0.0, 1.0]),
        )
        pick_geom = self._make_geom(IDENTITY_Q, symmetry=sym)
        tilt = np.deg2rad(10.0)
        q_jostled = _axis_angle_quat([1, 0, 0], tilt)
        measured = _make_item_in_ee(
            position_in_ee=np.array([0.0, 0.0, -0.04]),
            orientation_in_ee=q_jostled,
        )
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=np.array([0.5, 0.0, 0.09]),
            pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q,
            drop_orientation=None,
            item_in_ee=measured, above=0.0,
        )
        # Expected correction: final_orient = conj(q_jostled) since the jostle
        # is pure swing (orthogonal to axis) with IDENTITY reference_in_ee.
        expected = _quat_conjugate(q_jostled)
        assert _quaternions_equivalent(pose.ee_orientation, expected)

    def test_continuous_axis_mixed_jostle(self):
        # Mixed jostle (tilt + twist): only the tilt (swing) component is
        # corrected; the twist about the symmetry axis is absorbed.
        sym = AssetSymmetry(
            kind="continuous_axis", axis_local=np.array([0.0, 0.0, 1.0]),
        )
        pick_geom = self._make_geom(IDENTITY_Q, symmetry=sym)
        q_tilt = _axis_angle_quat([1, 0, 0], np.deg2rad(8.0))
        q_twist = _axis_angle_quat([0, 0, 1], np.deg2rad(20.0))
        # Jostle = swing * twist about EE-frame axis (which equals local axis
        # here because reference_orientation is identity).
        q_jostled = _quat_multiply(q_tilt, q_twist)
        measured = _make_item_in_ee(
            position_in_ee=np.array([0.0, 0.0, -0.04]),
            orientation_in_ee=q_jostled,
        )
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=np.array([0.5, 0.0, 0.09]),
            pick_geometry=pick_geom,
            pick_orientation=IDENTITY_Q,
            drop_orientation=None,
            item_in_ee=measured, above=0.0,
        )
        # Verify the commanded EE orientation is reachable AND that
        # applying it to the jostled item orients the symmetry axis
        # vertically (modulo twist, which is irrelevant).
        R_ee = np.array([
            [1 - 2 * (pose.ee_orientation[2] ** 2 + pose.ee_orientation[3] ** 2),
             2 * (pose.ee_orientation[1] * pose.ee_orientation[2] - pose.ee_orientation[3] * pose.ee_orientation[0]),
             2 * (pose.ee_orientation[1] * pose.ee_orientation[3] + pose.ee_orientation[2] * pose.ee_orientation[0])],
            [2 * (pose.ee_orientation[1] * pose.ee_orientation[2] + pose.ee_orientation[3] * pose.ee_orientation[0]),
             1 - 2 * (pose.ee_orientation[1] ** 2 + pose.ee_orientation[3] ** 2),
             2 * (pose.ee_orientation[2] * pose.ee_orientation[3] - pose.ee_orientation[1] * pose.ee_orientation[0])],
            [2 * (pose.ee_orientation[1] * pose.ee_orientation[3] - pose.ee_orientation[2] * pose.ee_orientation[0]),
             2 * (pose.ee_orientation[2] * pose.ee_orientation[3] + pose.ee_orientation[1] * pose.ee_orientation[0]),
             1 - 2 * (pose.ee_orientation[1] ** 2 + pose.ee_orientation[2] ** 2)],
        ])
        # Item axis in EE frame (post-jostle): rotate local +Z by measured
        # in-EE rotation = q_jostled since reference_in_ee == identity.
        R_jostle = np.array([
            [1 - 2 * (q_jostled[2] ** 2 + q_jostled[3] ** 2),
             2 * (q_jostled[1] * q_jostled[2] - q_jostled[3] * q_jostled[0]),
             2 * (q_jostled[1] * q_jostled[3] + q_jostled[2] * q_jostled[0])],
            [2 * (q_jostled[1] * q_jostled[2] + q_jostled[3] * q_jostled[0]),
             1 - 2 * (q_jostled[1] ** 2 + q_jostled[3] ** 2),
             2 * (q_jostled[2] * q_jostled[3] - q_jostled[1] * q_jostled[0])],
            [2 * (q_jostled[1] * q_jostled[3] - q_jostled[2] * q_jostled[0]),
             2 * (q_jostled[2] * q_jostled[3] + q_jostled[1] * q_jostled[0]),
             1 - 2 * (q_jostled[1] ** 2 + q_jostled[2] ** 2)],
        ])
        axis_in_ee_after_jostle = R_jostle @ np.array([0.0, 0.0, 1.0])
        # World-frame axis after applying corrected EE orient and rigid grasp:
        axis_world = R_ee @ axis_in_ee_after_jostle
        # Axis should point straight up (item lands with axis vertical).
        np.testing.assert_allclose(axis_world, [0.0, 0.0, 1.0], atol=1e-6)

    def test_soup_can_axis_local_y(self):
        # Sanity: axis_local is interpreted in the item's native frame.
        # Soup can has native Y-long axis (YCB), spawn-rotated -90° about X
        # to stand upright.  We simulate that by setting reference_orientation
        # to the -90°X quaternion.  A pure twist about world +Z (which after
        # the -90°X mapping corresponds to rotation about the local Y axis)
        # should be absorbed without correction.
        sym = AssetSymmetry(
            kind="continuous_axis", axis_local=np.array([0.0, 1.0, 0.0]),
        )
        # -90° about X quaternion:
        ref_q = _axis_angle_quat([1, 0, 0], -np.pi / 2)
        pick_geom = self._make_geom(ref_q, symmetry=sym)
        # Grasp the can with EE matching the can's upright frame (reference).
        pick_orient = ref_q.copy()
        # A pure twist about the local Y axis — in EE frame (which equals
        # reference), rotate about local Y:
        q_roll_local_y = _axis_angle_quat([0, 1, 0], np.deg2rad(30.0))
        measured = _make_item_in_ee(
            position_in_ee=np.array([0.0, 0.0, -0.04]),
            orientation_in_ee=q_roll_local_y,
        )
        pose = compute_place_pose(
            "pick_0", "target_0",
            target_position=np.array([0.5, 0.0, 0.09]),
            pick_geometry=pick_geom,
            pick_orientation=pick_orient,
            drop_orientation=None,
            item_in_ee=measured, above=0.0,
        )
        # Roll about the local Y axis (symmetry) must be absorbed —
        # ee_orientation stays at pick_orient.
        assert _quaternions_equivalent(pose.ee_orientation, pick_orient)


# ---------------------------------------------------------------------------
# compute_item_in_ee_pose
# ---------------------------------------------------------------------------


class _PoseStub:
    """Minimal PosePq-compatible stand-in for tests."""
    def __init__(self, p, q):
        self.p = np.asarray(p, dtype=float)
        self.q = np.asarray(q, dtype=float)


class _PickObjStub:
    """Minimal scene-object stand-in with get_world_pose."""
    def __init__(self, p, q):
        self._p = np.asarray(p, dtype=float)
        self._q = np.asarray(q, dtype=float)

    def get_world_pose(self):
        return self._p.copy(), self._q.copy()


class TestComputeItemInEEPose:
    def test_successful_grasp_has_zero_position_error(self):
        # Grasp at pick_pos = [0.5, 0.1, 0.1], grasp_height = 0.05.
        # Expected offset = [0, 0, 0.05].  Lift moves EE to [0.5, 0.1, 0.35].
        # The item came along for the ride: item at [0.5, 0.1, 0.30].
        # Measured offset = [0, 0, 0.05] = expected → position_error = 0.
        grasp = compute_grasp_pose(
            "pick_0",
            pick_position=np.array([0.5, 0.1, 0.1]),
            pick_geometry=_make_geom(grasp_height=0.05),
            pick_orientation_preference=GRIPPER_DOWN_Q,
        )
        pick_obj = _PickObjStub([0.5, 0.1, 0.30], IDENTITY_Q)
        ee_pose = _PoseStub([0.5, 0.1, 0.35], GRIPPER_DOWN_Q)
        measured = compute_item_in_ee_pose(
            pick_obj=pick_obj, ee_pose=ee_pose, expected_grasp_pose=grasp,
        )
        assert measured.position_error == pytest.approx(0.0, abs=1e-9)

    def test_failed_grasp_leaves_item_behind(self):
        # Item did NOT come with the gripper — stayed at original position.
        # EE lifted to [0.5, 0.1, 0.35], item still at [0.5, 0.1, 0.10].
        # Measured offset = [0, 0, 0.25]; expected = [0, 0, 0.05]
        # → position_error = 0.20.
        grasp = compute_grasp_pose(
            "pick_0",
            pick_position=np.array([0.5, 0.1, 0.1]),
            pick_geometry=_make_geom(grasp_height=0.05),
            pick_orientation_preference=GRIPPER_DOWN_Q,
        )
        pick_obj = _PickObjStub([0.5, 0.1, 0.10], IDENTITY_Q)
        ee_pose = _PoseStub([0.5, 0.1, 0.35], GRIPPER_DOWN_Q)
        measured = compute_item_in_ee_pose(
            pick_obj=pick_obj, ee_pose=ee_pose, expected_grasp_pose=grasp,
        )
        assert measured.position_error == pytest.approx(0.20, abs=1e-9)

    def test_lateral_slip_reports_xy_error(self):
        # Item slipped laterally by 3 cm along +X (would be detected by
        # VerifyGrasp's position-error threshold).
        grasp = compute_grasp_pose(
            "pick_0",
            pick_position=np.array([0.5, 0.1, 0.1]),
            pick_geometry=_make_geom(grasp_height=0.05),
            pick_orientation_preference=GRIPPER_DOWN_Q,
        )
        pick_obj = _PickObjStub([0.53, 0.1, 0.30], IDENTITY_Q)
        ee_pose = _PoseStub([0.5, 0.1, 0.35], GRIPPER_DOWN_Q)
        measured = compute_item_in_ee_pose(
            pick_obj=pick_obj, ee_pose=ee_pose, expected_grasp_pose=grasp,
        )
        assert measured.position_error == pytest.approx(0.03, abs=1e-9)

    def test_orientation_error_zero_when_grasp_time_orientation_missing(self):
        # Back-compat: when item_orientation_at_grasp is None, the
        # orientation_error_rad stays at 0.0 even if the measured
        # orientation differs from nominal.
        grasp = compute_grasp_pose(
            "pick_0",
            pick_position=np.array([0.5, 0.1, 0.1]),
            pick_geometry=_make_geom(grasp_height=0.05),
            pick_orientation_preference=GRIPPER_DOWN_Q,
        )
        assert grasp.item_orientation_at_grasp is None
        # Item has rotated wildly post-lift — still reports 0.0.
        pick_obj = _PickObjStub([0.5, 0.1, 0.30], HALF_TURN_X_Q)
        ee_pose = _PoseStub([0.5, 0.1, 0.35], GRIPPER_DOWN_Q)
        measured = compute_item_in_ee_pose(
            pick_obj=pick_obj, ee_pose=ee_pose, expected_grasp_pose=grasp,
        )
        assert measured.orientation_error_rad == 0.0

    def test_orientation_error_zero_for_rigid_grasp(self):
        # Grasp-time item orientation recorded; lift leaves the item-in-EE
        # relation unchanged (pure translation).  orientation_error_rad = 0.
        item_q = HALF_TURN_X_Q
        grasp = compute_grasp_pose(
            "pick_0",
            pick_position=np.array([0.5, 0.1, 0.1]),
            pick_geometry=_make_geom(grasp_height=0.05),
            pick_orientation_preference=GRIPPER_DOWN_Q,
            item_orientation_at_grasp=item_q,
        )
        # Lift: EE translates by +0.2 Z, item tracks with the same world quat.
        pick_obj = _PickObjStub([0.5, 0.1, 0.30], item_q)
        ee_pose = _PoseStub([0.5, 0.1, 0.35], GRIPPER_DOWN_Q)
        measured = compute_item_in_ee_pose(
            pick_obj=pick_obj, ee_pose=ee_pose, expected_grasp_pose=grasp,
        )
        assert measured.orientation_error_rad == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.parametrize("angle_rad", [0.1, 0.5, np.pi / 4, np.pi / 2])
    def test_orientation_error_matches_injected_rotation(self, angle_rad):
        # Record an initial item orientation; rotate the world item by
        # `angle_rad` about X between grasp and lift (simulating a slip).
        # With EE unchanged, orientation_error_rad should equal angle_rad.
        item_q_grasp = IDENTITY_Q
        slip_q = np.array([np.cos(angle_rad / 2), np.sin(angle_rad / 2), 0.0, 0.0])
        # Post-slip world item orientation = slip_q · item_q_grasp
        item_q_post = _quat_multiply(slip_q, item_q_grasp)
        grasp = compute_grasp_pose(
            "pick_0",
            pick_position=np.array([0.5, 0.1, 0.1]),
            pick_geometry=_make_geom(grasp_height=0.05),
            pick_orientation_preference=GRIPPER_DOWN_Q,
            item_orientation_at_grasp=item_q_grasp,
        )
        pick_obj = _PickObjStub([0.5, 0.1, 0.30], item_q_post)
        ee_pose = _PoseStub([0.5, 0.1, 0.35], GRIPPER_DOWN_Q)
        measured = compute_item_in_ee_pose(
            pick_obj=pick_obj, ee_pose=ee_pose, expected_grasp_pose=grasp,
        )
        assert measured.orientation_error_rad == pytest.approx(angle_rad, abs=1e-9)
