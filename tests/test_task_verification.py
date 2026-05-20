"""Tests for task_verification module (PlacementChecker, spatial functions, dataclasses)."""

import numpy as np
import pytest

from task_verification import (
    PlacementChecker,
    VerificationResult,
    PlacementCheck,
    ranges_overlap,
    get_corrected_aabb,
    is_on_top,
    is_within,
    is_within_box_geometry,
    is_vertical,
    is_horizontal,
)
from asset_data_utils import scale_aabb


# ---------------------------------------------------------------------------
# Mock prim helpers
# ---------------------------------------------------------------------------

class MockPrim:
    """Minimal prim mock for PlacementChecker tests."""

    def __init__(self, name, prim_path=None, labels=None):
        self.name = name
        self.prim_path = prim_path or f"/World/{name}"
        self._labels = labels or {}

    @property
    def prim(self):
        return self


class MockLabeledPrim(MockPrim):
    """Mock prim with semantic labels for testing get_corrected_aabb."""

    def __init__(self, name, type_label=None, **kwargs):
        labels = {}
        if type_label:
            labels["type"] = [type_label] if isinstance(type_label, str) else type_label
        super().__init__(name, labels=labels, **kwargs)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRangesOverlap:
    def test_overlapping(self):
        assert ranges_overlap(0, 2, 1, 3) is True

    def test_non_overlapping(self):
        assert ranges_overlap(0, 1, 2, 3) is False

    def test_touching(self):
        assert ranges_overlap(0, 1, 1, 2) is True


class TestScaleAABB:
    def test_identity_scale(self):
        aabb = np.array([0, 0, 0, 2, 2, 2], dtype=float)
        result = scale_aabb(aabb, np.array([1, 1, 1]))
        np.testing.assert_array_almost_equal(result, aabb)

    def test_double_scale(self):
        aabb = np.array([0, 0, 0, 2, 2, 2], dtype=float)
        result = scale_aabb(aabb, np.array([2, 2, 2]))
        np.testing.assert_array_almost_equal(result, np.array([-1, -1, -1, 3, 3, 3]))


class TestPlacementChecker:
    """Tests for PlacementChecker using a custom spatial_check_fn to control placement."""

    @staticmethod
    def _make_spatial_fn(occupancy_map):
        """Create a spatial_check_fn from a dict {(pick_idx, target_idx): True/False}.

        The function receives prim objects; we map back to indices via name convention.
        """
        def check(pick, target, bb_cache=None, obj_scale=None):
            pi = int(pick.name.split("_")[1])
            ti = int(target.name.split("_")[1])
            return occupancy_map.get((pi, ti), False)
        return check

    @staticmethod
    def _make_picks(n):
        return [MockPrim(f"pick_{i}") for i in range(n)]

    @staticmethod
    def _make_targets(n):
        return [MockPrim(f"target_{i}") for i in range(n)]

    def test_all_picks_placed(self):
        picks = self._make_picks(3)
        targets = self._make_targets(3)
        spatial = self._make_spatial_fn({(0, 0): True, (1, 1): True, (2, 2): True})

        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial, bb_cache_factory=lambda: None,
        )
        result = verifier.verify()

        assert result.success is True
        assert len(result.failures) == 0
        assert len(result.checks) == 3
        assert all(c.passed for c in result.checks)

    def test_pick_not_placed_targets_available(self):
        picks = self._make_picks(2)
        targets = self._make_targets(2)
        # pick_0 on target_0, pick_1 NOT on anything
        spatial = self._make_spatial_fn({(0, 0): True})

        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial, bb_cache_factory=lambda: None,
        )
        result = verifier.verify()

        assert result.success is False
        assert len(result.failures) == 1
        assert "pick_1" in result.failures[0]

    def test_pick_not_placed_no_targets_available(self):
        """If all valid targets are occupied, unplaced pick is acceptable."""
        picks = self._make_picks(3)
        targets = self._make_targets(2)
        # pick_0 on target_0, pick_1 on target_1, pick_2 on nothing
        spatial = self._make_spatial_fn({(0, 0): True, (1, 1): True})

        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial, bb_cache_factory=lambda: None,
        )
        result = verifier.verify()

        assert result.success is True
        assert len(result.failures) == 0

    def test_custom_spatial_check(self):
        """Pass is_within-like function as spatial_check_fn."""
        picks = self._make_picks(1)
        targets = self._make_targets(1)
        # Custom check that always returns True
        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=lambda p, t, bb_cache=None, obj_scale=None: True,
            bb_cache_factory=lambda: None,
        )
        result = verifier.verify()
        assert result.success is True

    def test_valid_targets_filtering(self):
        """Only certain targets are valid for certain picks."""
        picks = self._make_picks(2)
        targets = self._make_targets(3)
        # pick_0 on target_2, pick_1 on target_0
        spatial = self._make_spatial_fn({(0, 2): True, (1, 0): True})

        # pick_0 can only go to targets [0, 1] — so target_2 doesn't count
        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial,
            valid_targets_fn=lambda pi: [0, 1] if pi == 0 else [0, 1, 2],
            bb_cache_factory=lambda: None,
        )
        result = verifier.verify()

        # pick_0 is on target_2 but that's not in its valid set → failure
        assert result.success is False
        assert len(result.failures) == 1
        assert "pick_0" in result.failures[0]

    def test_placement_constraints(self):
        """Reject specific pick-target pairs via placement_constraints_fn."""
        picks = self._make_picks(1)
        targets = self._make_targets(2)
        # pick_0 is on target_0 spatially
        spatial = self._make_spatial_fn({(0, 0): True})

        # But constraints reject the (0, 0) pair with a reason
        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial,
            placement_constraints_fn=lambda pi, ti: (ti != 0, "" if ti != 0 else "target 0 rejected"),
            bb_cache_factory=lambda: None,
        )
        result = verifier.verify()

        assert result.success is False
        assert len(result.failures) == 1

    def test_placement_constraints_reason_in_failure(self):
        """Constraint failure reason appears in detail and failure message."""
        picks = self._make_picks(1)
        targets = self._make_targets(2)
        spatial = self._make_spatial_fn({(0, 0): True})

        reason = "item is not vertical (upright orientation required)"
        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial,
            placement_constraints_fn=lambda pi, ti: (False, reason),
            bb_cache_factory=lambda: None,
        )
        result = verifier.verify()

        assert result.success is False
        assert len(result.failures) == 1
        assert reason in result.failures[0]
        failed_check = [c for c in result.checks if not c.passed][0]
        assert failed_check.detail == reason

    def test_placement_constraints_tuple_success(self):
        """Constraint returning (True, '') passes verification."""
        picks = self._make_picks(1)
        targets = self._make_targets(1)
        spatial = self._make_spatial_fn({(0, 0): True})

        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial,
            placement_constraints_fn=lambda pi, ti: (True, ""),
            bb_cache_factory=lambda: None,
        )
        result = verifier.verify()

        assert result.success is True
        assert len(result.failures) == 0

    def test_summary_output(self):
        picks = self._make_picks(2)
        targets = self._make_targets(2)
        spatial = self._make_spatial_fn({(0, 0): True})

        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial, bb_cache_factory=lambda: None,
        )
        result = verifier.verify()

        summary = result.summary()
        assert "FAILED" in summary
        assert "pick_0" in summary
        assert "pick_1" in summary
        # Summary tags each check by provenance: [LIVE] for real-time checks,
        # [SNAPSHOT@...] for frozen conveyor-falloff snapshots.
        assert "[LIVE]" in summary
        assert "FAIL" in summary

    def test_summary_all_passed(self):
        picks = self._make_picks(2)
        targets = self._make_targets(2)
        spatial = self._make_spatial_fn({(0, 0): True, (1, 1): True})

        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial, bb_cache_factory=lambda: None,
        )
        result = verifier.verify()

        summary = result.summary()
        assert "PASSED" in summary
        assert "[FAIL]" not in summary

    def test_empty_picks(self):
        result = PlacementChecker(
            pick_objs=[], target_objs=[MockPrim("t_0")],
            bb_cache_factory=lambda: None,
        ).verify()
        assert result.success is True
        assert len(result.checks) == 0

    def test_empty_targets(self):
        result = PlacementChecker(
            pick_objs=[MockPrim("p_0")], target_objs=[],
            bb_cache_factory=lambda: None,
        ).verify()
        assert result.success is True
        assert len(result.checks) == 0

    def test_check_occupancy(self):
        picks = self._make_picks(2)
        targets = self._make_targets(2)
        spatial = self._make_spatial_fn({(0, 1): True, (1, 0): True})

        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial, bb_cache_factory=lambda: None,
        )
        occ = verifier.check_occupancy()

        assert occ == {1: 0, 0: 1}


class TestGetCorrectedAABB:
    """Tests for the get_corrected_aabb utility."""

    def test_no_scale_no_correction(self):
        """Without obj_scale, AABB should be returned as-is from compute_aabb."""
        prim = MockPrim("test_prim")
        result = get_corrected_aabb(prim, bb_cache=None, obj_scale=None)
        # Default stub returns [0, 0, 0, 1, 1, 1]
        np.testing.assert_array_almost_equal(result, np.array([0, 0, 0, 1, 1, 1]))

    def test_non_cuboid_no_correction(self):
        """Non-cuboid prims should not get scale correction even with obj_scale."""
        prim = MockPrim("cylinder_prim")
        scale = np.array([2.0, 2.0, 2.0])
        result = get_corrected_aabb(prim, bb_cache=None, obj_scale=scale)
        # No correction applied - is_of_type returns False for non-labeled prims
        np.testing.assert_array_almost_equal(result, np.array([0, 0, 0, 1, 1, 1]))

    def test_cuboid_with_scale_correction(self, monkeypatch):
        """Cuboid prims with obj_scale should get inverse scale correction."""
        import task_verification
        prim = MockPrim("cube_prim")
        scale = np.array([2.0, 2.0, 2.0])

        # Monkey-patch needs_aabb_scale_correction to return True for this prim
        monkeypatch.setattr(task_verification, "needs_aabb_scale_correction", lambda p: True)

        result = get_corrected_aabb(prim, bb_cache=None, obj_scale=scale)
        # scale_aabb with inv_scale [0.5, 0.5, 0.5] on [0,0,0,1,1,1]:
        # center = [0.5, 0.5, 0.5], half_extents = [0.5, 0.5, 0.5]
        # scaled_half_extents = [0.25, 0.25, 0.25]
        # new_min = [0.25, 0.25, 0.25], new_max = [0.75, 0.75, 0.75]
        np.testing.assert_array_almost_equal(result, np.array([0.25, 0.25, 0.25, 0.75, 0.75, 0.75]))


class MockPoseObj:
    """Minimal mock with get_world_pose() + semantic labels for pose-based checks.

    Mirrors ``LightweightObj``'s interface for the slice used by is_vertical /
    is_horizontal: a quaternion in ``[w, x, y, z]`` order and a ``"type"``
    semantic label that ``get_asset_type`` can resolve.
    """

    def __init__(self, name, asset_type=None, position=None, orientation=None):
        self.name = name
        self.prim_path = f"/World/{name}"
        self._position = np.asarray(
            position if position is not None else [0.0, 0.0, 0.0],
            dtype=float,
        )
        self._orientation = np.asarray(
            orientation if orientation is not None else [1.0, 0.0, 0.0, 0.0],
            dtype=float,
        )
        self._semantic_labels = {}
        if asset_type:
            self._semantic_labels["type"] = [asset_type]

    def get_world_pose(self):
        return self._position.copy(), self._orientation.copy()


def _quat_rot_x(deg):
    """Quaternion [w, x, y, z] for rotation about world X by ``deg`` degrees."""
    import math
    theta = math.radians(deg)
    return np.array([math.cos(theta / 2), math.sin(theta / 2), 0.0, 0.0])


def _quat_rot_y(deg):
    """Quaternion [w, x, y, z] for rotation about world Y by ``deg`` degrees."""
    import math
    theta = math.radians(deg)
    return np.array([math.cos(theta / 2), 0.0, math.sin(theta / 2), 0.0])


class TestIsVertical:
    """Pose-based is_vertical tests using mock objects with semantic labels."""

    def test_upright_cylinder_is_vertical(self):
        """Cylinder (up_axis=+Z) with identity quat passes."""
        obj = MockPoseObj("c", asset_type="cylinder")
        assert is_vertical(obj, max_tilt_deg=15) is True

    def test_tilted_10deg_passes(self):
        """10° tilt about X passes max_tilt_deg=15."""
        obj = MockPoseObj("c", asset_type="cylinder", orientation=_quat_rot_x(10))
        assert is_vertical(obj, max_tilt_deg=15) is True

    def test_tilted_20deg_fails(self):
        """20° tilt about X fails max_tilt_deg=15."""
        obj = MockPoseObj("c", asset_type="cylinder", orientation=_quat_rot_x(20))
        assert is_vertical(obj, max_tilt_deg=15) is False

    def test_default_threshold_15(self):
        """Default max_tilt_deg=15: 14° passes, 16° fails."""
        obj_pass = MockPoseObj("c1", asset_type="cylinder", orientation=_quat_rot_x(14))
        obj_fail = MockPoseObj("c2", asset_type="cylinder", orientation=_quat_rot_x(16))
        assert is_vertical(obj_pass) is True
        assert is_vertical(obj_fail) is False

    def test_on_side_fails(self):
        """90° rot about X puts up-axis horizontal → fails verticality."""
        obj = MockPoseObj("c", asset_type="cylinder", orientation=_quat_rot_x(90))
        assert is_vertical(obj, max_tilt_deg=15) is False

    def test_upside_down_fails_strict_sign(self):
        """180° flip points up-axis to -Z → strict-sign rejects (tilt=180°)."""
        obj = MockPoseObj("c", asset_type="cylinder", orientation=_quat_rot_x(180))
        assert is_vertical(obj, max_tilt_deg=15) is False
        # Even a generous threshold (90°) still rejects an inverted object.
        assert is_vertical(obj, max_tilt_deg=89) is False

    def test_ycb_spawn_rotation_makes_can_upright(self):
        """soup_can has up_axis_local=[0,-1,0]; the standard -90° X spawn maps it to +Z."""
        spawn_q = _quat_rot_x(-90)
        obj = MockPoseObj("c", asset_type="soup_can", orientation=spawn_q)
        assert is_vertical(obj, max_tilt_deg=15) is True

    def test_ycb_identity_orientation_fails(self):
        """soup_can at identity orientation (native -Y is up, so -Y points world -Y, not +Z) fails."""
        obj = MockPoseObj("c", asset_type="soup_can")
        assert is_vertical(obj, max_tilt_deg=15) is False

    def test_missing_asset_type_returns_false(self):
        """Object without semantic 'type' label returns False (no metadata)."""
        obj = MockPoseObj("unknown")
        assert is_vertical(obj, max_tilt_deg=15) is False

    def test_unknown_asset_type_returns_false(self):
        """Object with a 'type' label not in ITEMS_MAP returns False."""
        obj = MockPoseObj("x", asset_type="not_a_real_asset_type")
        assert is_vertical(obj, max_tilt_deg=15) is False

    def test_tilt_about_y_is_symmetric(self):
        """Rotation axis (X vs Y) does not affect the verticality result."""
        obj_x = MockPoseObj("ax", asset_type="cylinder", orientation=_quat_rot_x(20))
        obj_y = MockPoseObj("ay", asset_type="cylinder", orientation=_quat_rot_y(20))
        assert is_vertical(obj_x, max_tilt_deg=15) == is_vertical(obj_y, max_tilt_deg=15)


class TestIsHorizontal:
    """Pose-based is_horizontal tests using mock objects with semantic labels."""

    def test_upright_not_horizontal(self):
        """Identity orientation: up-axis points +Z, far from horizontal plane."""
        obj = MockPoseObj("c", asset_type="cylinder")
        assert is_horizontal(obj, max_tilt_deg=15) is False

    def test_on_side_is_horizontal(self):
        """90° rotation puts up-axis in the horizontal plane → passes."""
        obj = MockPoseObj("c", asset_type="cylinder", orientation=_quat_rot_x(90))
        assert is_horizontal(obj, max_tilt_deg=15) is True

    def test_near_horizontal_10deg_off_passes(self):
        """80° from upright = 10° from horizontal → passes 15° threshold."""
        obj = MockPoseObj("c", asset_type="cylinder", orientation=_quat_rot_x(80))
        assert is_horizontal(obj, max_tilt_deg=15) is True

    def test_off_horizontal_20deg_fails(self):
        """70° from upright = 20° from horizontal → fails 15° threshold."""
        obj = MockPoseObj("c", asset_type="cylinder", orientation=_quat_rot_x(70))
        assert is_horizontal(obj, max_tilt_deg=15) is False

    def test_upside_down_not_horizontal(self):
        """Inverted (180°) is along -Z, still vertical (not horizontal)."""
        obj = MockPoseObj("c", asset_type="cylinder", orientation=_quat_rot_x(180))
        assert is_horizontal(obj, max_tilt_deg=15) is False

    def test_missing_asset_type_returns_false(self):
        """Object without semantic 'type' label returns False (no metadata)."""
        obj = MockPoseObj("unknown")
        assert is_horizontal(obj, max_tilt_deg=15) is False


class TestIsWithinBoxGeometry:
    """Tests for is_within_box_geometry using monkeypatched get_corrected_aabb."""

    def _make_aabb_fn(self, center_x, center_y, z_min, half_x, half_y, half_z):
        """Return an AABB array for monkeypatching."""
        return np.array([
            center_x - half_x, center_y - half_y, z_min,
            center_x + half_x, center_y + half_y, z_min + 2 * half_z,
        ])

    def test_object_inside_box(self, monkeypatch):
        """Object centered inside box should pass."""
        import task_verification
        prim = MockPrim("obj_in_box")
        # Object at box center, z_min = 0.063 (just above floor at 0.06244)
        aabb = self._make_aabb_fn(0.5, 0.3, 0.063, 0.02, 0.02, 0.02)
        monkeypatch.setattr(task_verification, "get_corrected_aabb", lambda obj, bb_cache, obj_scale=None: aabb)

        result = is_within_box_geometry(
            prim,
            box_center_xy=np.array([0.5, 0.3]),
            box_inner_size=np.array([0.20, 0.16]),
            box_floor_z=0.06244,
            box_height=0.08,
        )
        assert result

    def test_object_outside_x(self, monkeypatch):
        """Object outside box in X should fail."""
        import task_verification
        prim = MockPrim("obj_outside_x")
        aabb = self._make_aabb_fn(0.8, 0.3, 0.063, 0.02, 0.02, 0.02)
        monkeypatch.setattr(task_verification, "get_corrected_aabb", lambda obj, bb_cache, obj_scale=None: aabb)

        result = is_within_box_geometry(
            prim,
            box_center_xy=np.array([0.5, 0.3]),
            box_inner_size=np.array([0.20, 0.16]),
            box_floor_z=0.06244,
            box_height=0.08,
        )
        assert not result

    def test_object_outside_y(self, monkeypatch):
        """Object outside box in Y should fail."""
        import task_verification
        prim = MockPrim("obj_outside_y")
        aabb = self._make_aabb_fn(0.5, 0.6, 0.063, 0.02, 0.02, 0.02)
        monkeypatch.setattr(task_verification, "get_corrected_aabb", lambda obj, bb_cache, obj_scale=None: aabb)

        result = is_within_box_geometry(
            prim,
            box_center_xy=np.array([0.5, 0.3]),
            box_inner_size=np.array([0.20, 0.16]),
            box_floor_z=0.06244,
            box_height=0.08,
        )
        assert not result

    def test_object_below_floor(self, monkeypatch):
        """Object below box floor should fail."""
        import task_verification
        prim = MockPrim("obj_below")
        aabb = self._make_aabb_fn(0.5, 0.3, 0.01, 0.02, 0.02, 0.02)
        monkeypatch.setattr(task_verification, "get_corrected_aabb", lambda obj, bb_cache, obj_scale=None: aabb)

        result = is_within_box_geometry(
            prim,
            box_center_xy=np.array([0.5, 0.3]),
            box_inner_size=np.array([0.20, 0.16]),
            box_floor_z=0.06244,
            box_height=0.08,
        )
        assert not result

    def test_object_above_box(self, monkeypatch):
        """Object above box height should fail."""
        import task_verification
        prim = MockPrim("obj_above")
        aabb = self._make_aabb_fn(0.5, 0.3, 0.20, 0.02, 0.02, 0.02)
        monkeypatch.setattr(task_verification, "get_corrected_aabb", lambda obj, bb_cache, obj_scale=None: aabb)

        result = is_within_box_geometry(
            prim,
            box_center_xy=np.array([0.5, 0.3]),
            box_inner_size=np.array([0.20, 0.16]),
            box_floor_z=0.06244,
            box_height=0.08,
        )
        assert not result

    def test_object_near_edge_within_tolerance(self, monkeypatch):
        """Object near box edge but within xy_tol should pass."""
        import task_verification
        prim = MockPrim("obj_edge")
        # Center at x=0.605 → distance from box center 0.5 = 0.105
        # half_w = 0.10 + 0.01 = 0.11 → 0.105 <= 0.11 → passes
        aabb = self._make_aabb_fn(0.605, 0.3, 0.063, 0.02, 0.02, 0.02)
        monkeypatch.setattr(task_verification, "get_corrected_aabb", lambda obj, bb_cache, obj_scale=None: aabb)

        result = is_within_box_geometry(
            prim,
            box_center_xy=np.array([0.5, 0.3]),
            box_inner_size=np.array([0.20, 0.16]),
            box_floor_z=0.06244,
            box_height=0.08,
            xy_tol=0.01,
        )
        assert result

    def test_exception_returns_false(self, monkeypatch):
        """AABB computation failure returns False."""
        import task_verification
        prim = MockPrim("bad_obj")
        monkeypatch.setattr(task_verification, "get_corrected_aabb",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("mock")))
        assert is_within_box_geometry(
            prim,
            box_center_xy=np.array([0.0, 0.0]),
            box_inner_size=np.array([1.0, 1.0]),
            box_floor_z=0.0,
            box_height=1.0,
        ) is False


class TestMultiOccupancyVerifier:
    """Tests for PlacementChecker with allow_multi_occupancy=True."""

    @staticmethod
    def _make_spatial_fn(occupancy_map):
        """Create a spatial_check_fn from {(pick_idx, target_idx): True/False}."""
        def check(pick, target, bb_cache=None, obj_scale=None):
            pi = int(pick.name.split("_")[1])
            ti = int(target.name.split("_")[1])
            return occupancy_map.get((pi, ti), False)
        return check

    @staticmethod
    def _make_picks(n):
        return [MockPrim(f"pick_{i}") for i in range(n)]

    @staticmethod
    def _make_targets(n):
        return [MockPrim(f"target_{i}") for i in range(n)]

    def test_multiple_picks_same_target(self):
        """Multiple picks on the same target all pass with multi-occupancy."""
        picks = self._make_picks(3)
        targets = self._make_targets(1)
        # All 3 picks are on target_0
        spatial = self._make_spatial_fn({(0, 0): True, (1, 0): True, (2, 0): True})

        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial, bb_cache_factory=lambda: None,
            allow_multi_occupancy=True,
        )
        result = verifier.verify()

        assert result.success is True
        assert len(result.failures) == 0
        assert all(c.passed for c in result.checks)

    def test_pick_not_on_valid_target_fails(self):
        """Pick not on any valid target fails even with multi-occupancy."""
        picks = self._make_picks(3)
        targets = self._make_targets(2)
        # pick_0 and pick_1 on target_0, pick_2 not on anything
        spatial = self._make_spatial_fn({(0, 0): True, (1, 0): True})

        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial, bb_cache_factory=lambda: None,
            allow_multi_occupancy=True,
        )
        result = verifier.verify()

        assert result.success is False
        assert len(result.failures) == 1
        assert "pick_2" in result.failures[0]

    def test_mixed_targets(self):
        """Some targets have multiple picks, some have none — all placed picks pass."""
        picks = self._make_picks(4)
        targets = self._make_targets(3)
        # pick_0, pick_1 → target_0; pick_2 → target_1; pick_3 → target_2
        spatial = self._make_spatial_fn({
            (0, 0): True, (1, 0): True, (2, 1): True, (3, 2): True
        })

        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial, bb_cache_factory=lambda: None,
            allow_multi_occupancy=True,
        )
        result = verifier.verify()

        assert result.success is True
        assert len(result.failures) == 0

    def test_check_occupancy_returns_lists(self):
        """check_occupancy with multi-occupancy returns lists of pick indices."""
        picks = self._make_picks(3)
        targets = self._make_targets(2)
        spatial = self._make_spatial_fn({(0, 0): True, (1, 0): True, (2, 1): True})

        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial, bb_cache_factory=lambda: None,
            allow_multi_occupancy=True,
        )
        occ = verifier.check_occupancy()

        assert occ == {0: [0, 1], 1: [2]}

    def test_valid_targets_filtering_with_multi_occupancy(self):
        """Valid targets filtering works with multi-occupancy."""
        picks = self._make_picks(2)
        targets = self._make_targets(2)
        # Both picks on target_0
        spatial = self._make_spatial_fn({(0, 0): True, (1, 0): True})

        # pick_1 can only go to target_1 — so target_0 doesn't count for it
        verifier = PlacementChecker(
            pick_objs=picks, target_objs=targets,
            spatial_check_fn=spatial,
            valid_targets_fn=lambda pi: [0] if pi == 0 else [1],
            bb_cache_factory=lambda: None,
            allow_multi_occupancy=True,
        )
        result = verifier.verify()

        # pick_0 → target_0 OK, pick_1 → target_0 not valid (needs target_1) → FAIL
        assert result.success is False
        assert len(result.failures) == 1
        assert "pick_1" in result.failures[0]
