"""Tests for PrimGeometry dataclass, compute_prim_geometry(), and the new drop Z formula."""

import sys

import numpy as np
import pytest

from asset_data_utils import (
    AssetMetaData,
    AssetSymmetry,
    get_asset_symmetry,
    lookup_prim_geometry,
    ITEMS_MAP,
    PrimGeometry,
)
from asset_utils import compute_prim_geometry

# Keep a reference to the module that compute_prim_geometry belongs to,
# so monkeypatching targets the correct module globals even when other
# test files have already imported asset_utils under a different module object.
_asset_utils_module = sys.modules["asset_utils"]


# ---------------------------------------------------------------------------
# Mock prim helper
# ---------------------------------------------------------------------------


class MockGeomPrim:
    """Mock prim with configurable world pose for geometry tests."""

    def __init__(self, name, position=None, prim_path=None):
        self.name = name
        self.prim_path = prim_path or f"/World/{name}"
        self._position = position if position is not None else np.array([0.5, 0.5, 0.5])

    def get_world_pose(self):
        return self._position.copy(), np.array([1.0, 0.0, 0.0, 0.0])

    @property
    def prim(self):
        return self


# ---------------------------------------------------------------------------
# Tests for PrimGeometry dataclass
# ---------------------------------------------------------------------------


class TestPrimGeometry:
    def test_creation(self):
        geom = PrimGeometry(
            grasp_height=0.05,
            rest_height=0.025,
            top_surface_height=0.025,
            local_half_extents=np.array([0.05, 0.05, 0.05]),
            needs_aabb_scale_correction=False,
        )
        assert geom.grasp_height == 0.05
        assert geom.rest_height == 0.025
        assert geom.top_surface_height == 0.025
        assert geom.needs_aabb_scale_correction is False
        np.testing.assert_array_equal(geom.local_half_extents, [0.05, 0.05, 0.05])


# ---------------------------------------------------------------------------
# Tests for compute_prim_geometry
# ---------------------------------------------------------------------------


class TestComputePrimGeometry:
    """Tests for compute_prim_geometry using configurable AABB stubs."""

    def test_basic_computation(self):
        """Test basic geometry computation from AABB."""
        # AABB: min=(0, 0, 0), max=(1, 1, 1)
        # Prim origin at (0.5, 0.5, 0.5) — center of the box
        prim = MockGeomPrim("test_cube", position=np.array([0.5, 0.5, 0.5]))

        geom = compute_prim_geometry(prim)

        assert geom.grasp_height == pytest.approx(0.5)  # aabb_height/2 = 1.0/2
        assert geom.rest_height == pytest.approx(0.5)  # origin_z - bottom_z = 0.5 - 0
        assert geom.top_surface_height == pytest.approx(
            0.5
        )  # top_z - origin_z = 1.0 - 0.5
        np.testing.assert_array_almost_equal(geom.local_half_extents, [0.5, 0.5, 0.5])
        assert geom.needs_aabb_scale_correction is False

    def test_asymmetric_origin(self):
        """Test when object origin is not at the center of its AABB."""
        # AABB: min=(0, 0, 0), max=(1, 1, 1) but origin at (0.5, 0.5, 0.1)
        # This simulates an object whose origin is near the bottom
        prim = MockGeomPrim("bottom_origin", position=np.array([0.5, 0.5, 0.1]))

        geom = compute_prim_geometry(prim)

        assert geom.grasp_height == pytest.approx(0.5)  # aabb_height/2
        assert geom.rest_height == pytest.approx(0.1)  # 0.1 - 0.0
        assert geom.top_surface_height == pytest.approx(0.9)  # 1.0 - 0.1

    def test_custom_aabb(self):
        """Test with a non-default AABB."""
        custom_aabb = np.array([0.0, 0.0, 0.1, 0.2, 0.2, 0.3], dtype=float)
        # Patch compute_aabb in the actual module that compute_prim_geometry uses
        original = _asset_utils_module.__dict__["compute_aabb"]
        _asset_utils_module.__dict__["compute_aabb"] = lambda *a, **k: custom_aabb
        try:
            prim = MockGeomPrim("small_obj", position=np.array([0.1, 0.1, 0.2]))

            geom = compute_prim_geometry(prim)

            assert geom.grasp_height == pytest.approx(0.1)  # (0.3 - 0.1) / 2
            assert geom.rest_height == pytest.approx(0.1)  # 0.2 - 0.1
            assert geom.top_surface_height == pytest.approx(0.1)  # 0.3 - 0.2
            np.testing.assert_array_almost_equal(
                geom.local_half_extents, [0.1, 0.1, 0.1]
            )
        finally:
            _asset_utils_module.__dict__["compute_aabb"] = original

    def test_metadata_overrides(self):
        """Test that ITEMS_MAP metadata overrides take precedence."""
        test_meta = AssetMetaData(
            asset_type="cube",
            grasp_height=0.03,
            rest_height=0.01,
            top_surface_height=0.04,
        )
        # Directly modify the ITEMS_MAP dict that compute_prim_geometry references
        items_map = _asset_utils_module.__dict__["ITEMS_MAP"]
        items_map["test_override_type"] = test_meta
        try:
            prim = MockGeomPrim("override_test", position=np.array([0.5, 0.5, 0.5]))
            geom = compute_prim_geometry(prim, asset_type="test_override_type")

            assert geom.grasp_height == pytest.approx(0.03)
            assert geom.rest_height == pytest.approx(0.01)
            assert geom.top_surface_height == pytest.approx(0.04)
        finally:
            items_map.pop("test_override_type", None)

    def test_partial_metadata_overrides(self):
        """Test that only non-None metadata fields override computed values."""
        test_meta = AssetMetaData(
            asset_type="cube",
            grasp_height=0.07,
            # rest_height and top_surface_height remain None
        )
        items_map = _asset_utils_module.__dict__["ITEMS_MAP"]
        items_map["partial_override"] = test_meta
        try:
            prim = MockGeomPrim("partial_test", position=np.array([0.5, 0.5, 0.5]))
            geom = compute_prim_geometry(prim, asset_type="partial_override")

            assert geom.grasp_height == pytest.approx(0.07)  # overridden
            assert geom.rest_height == pytest.approx(0.5)  # computed
            assert geom.top_surface_height == pytest.approx(0.5)  # computed
        finally:
            items_map.pop("partial_override", None)


# ---------------------------------------------------------------------------
# Tests for the drop Z formula
# ---------------------------------------------------------------------------


class TestDropZFormula:
    """Test the drop Z formula: target_world_z + target.top_surface_height + pick.rest_height"""

    def test_symmetric_objects(self):
        """Both pick and target are centered in their AABBs."""
        pick_geom = PrimGeometry(
            grasp_height=0.025,
            rest_height=0.025,
            top_surface_height=0.025,
            local_half_extents=np.array([0.025, 0.025, 0.025]),
            needs_aabb_scale_correction=False,
        )
        target_geom = PrimGeometry(
            grasp_height=0.025,
            rest_height=0.025,
            top_surface_height=0.025,
            local_half_extents=np.array([0.025, 0.025, 0.025]),
            needs_aabb_scale_correction=False,
        )
        target_world_z = 0.75  # target sitting on a table at z=0.75

        drop_z = target_world_z + target_geom.top_surface_height + pick_geom.rest_height

        # Expected: 0.75 + 0.025 + 0.025 = 0.8 (pick bottom rests on target top)
        assert drop_z == pytest.approx(0.8)

    def test_asymmetric_pick(self):
        """Pick object with origin near bottom (like a bottle)."""
        pick_geom = PrimGeometry(
            grasp_height=0.05,
            rest_height=0.01,
            top_surface_height=0.09,
            local_half_extents=np.array([0.02, 0.02, 0.05]),
            needs_aabb_scale_correction=False,
        )
        target_geom = PrimGeometry(
            grasp_height=0.025,
            rest_height=0.025,
            top_surface_height=0.025,
            local_half_extents=np.array([0.025, 0.025, 0.025]),
            needs_aabb_scale_correction=False,
        )
        target_world_z = 0.75

        drop_z = target_world_z + target_geom.top_surface_height + pick_geom.rest_height

        # Expected: 0.75 + 0.025 + 0.01 = 0.785
        assert drop_z == pytest.approx(0.785)

    def test_tall_target(self):
        """Target with significant height (like a container)."""
        pick_geom = PrimGeometry(
            grasp_height=0.025,
            rest_height=0.025,
            top_surface_height=0.025,
            local_half_extents=np.array([0.025, 0.025, 0.025]),
            needs_aabb_scale_correction=False,
        )
        target_geom = PrimGeometry(
            grasp_height=0.05,
            rest_height=0.05,
            top_surface_height=0.05,
            local_half_extents=np.array([0.05, 0.05, 0.05]),
            needs_aabb_scale_correction=False,
        )
        target_world_z = 0.80  # target origin at z=0.80

        drop_z = target_world_z + target_geom.top_surface_height + pick_geom.rest_height

        # Expected: 0.80 + 0.05 + 0.025 = 0.875
        assert drop_z == pytest.approx(0.875)


# ---------------------------------------------------------------------------
# Tests for MockTaskContext with PrimGeometry
# These rely on extsMock being on sys.path (handled by conftest.py)
# ---------------------------------------------------------------------------


class TestMockTaskContextWithGeometry:
    """Test that MockTaskContext uses PrimGeometry for placing and EE offset."""

    @pytest.fixture(autouse=True)
    def _import_mock_context(self):
        """Import MockTaskContext lazily so extsMock modules are available."""
        from task_context_mock import MockTaskContext

        self.MockTaskContext = MockTaskContext

    def test_placing_info_with_geometry(self):
        pick_geom = PrimGeometry(
            grasp_height=0.03,
            rest_height=0.02,
            top_surface_height=0.03,
            local_half_extents=np.array([0.025, 0.025, 0.025]),
            needs_aabb_scale_correction=False,
        )
        target_geom = PrimGeometry(
            grasp_height=0.03,
            rest_height=0.02,
            top_surface_height=0.03,
            local_half_extents=np.array([0.025, 0.025, 0.025]),
            needs_aabb_scale_correction=False,
        )

        ctx = self.MockTaskContext(
            pick_names=["pick_0"],
            target_names=["target_0"],
            prim_geometry={"pick_0": pick_geom, "target_0": target_geom},
        )

        target_name, target_pos, target_orient = ctx.get_placing_info("pick_0")

        assert target_name == "target_0"
        # Default target_0 position is [-0.5, 0.0, 0.05]
        # drop_z = 0.05 + 0.03 (target top_surface) + 0.02 (pick rest) = 0.1
        assert target_pos[2] == pytest.approx(0.1)

    def test_placing_info_without_geometry(self):
        """Without geometry, falls back to simple target position."""
        ctx = self.MockTaskContext(
            pick_names=["pick_0"],
            target_names=["target_0"],
        )

        target_name, target_pos, _ = ctx.get_placing_info("pick_0")

        assert target_name == "target_0"
        # Fallback: just the raw target position z
        assert target_pos[2] == pytest.approx(0.05)

    def test_ee_offset_with_geometry(self):
        pick_geom = PrimGeometry(
            grasp_height=0.042,
            rest_height=0.02,
            top_surface_height=0.03,
            local_half_extents=np.array([0.025, 0.025, 0.025]),
            needs_aabb_scale_correction=False,
        )

        ctx = self.MockTaskContext(
            pick_names=["pick_0"],
            target_names=["target_0"],
            prim_geometry={"pick_0": pick_geom},
        )

        offset = ctx.get_end_effector_offset("pick_0")

        np.testing.assert_array_almost_equal(offset, [0.0, 0.0, 0.042])

    def test_ee_offset_without_geometry(self):
        ctx = self.MockTaskContext(
            pick_names=["pick_0"],
            target_names=["target_0"],
        )

        offset = ctx.get_end_effector_offset("pick_0")

        # Unified fallback (TaskContextBase._EE_OFFSET_FALLBACK) when no
        # PrimGeometry is cached for the pick.
        np.testing.assert_array_almost_equal(offset, [0.0, 0.0, 0.02])


# ---------------------------------------------------------------------------
# Tests for lookup_prim_geometry
# ---------------------------------------------------------------------------


class TestLookupPrimGeometry:
    """Tests for lookup_prim_geometry using precomputed JSON data."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Clear the precomputed geometry cache before each test."""
        import asset_data_utils
        asset_data_utils._precomputed_geometry_cache = None

    def test_identity_scale_cube(self):
        """Identity scale returns raw JSON values for cube."""
        geom = lookup_prim_geometry("cube")
        assert geom is not None
        assert geom.grasp_height == pytest.approx(0.5)
        assert geom.rest_height == pytest.approx(0.50274658203125)
        assert geom.top_surface_height == pytest.approx(0.49725341796875)
        np.testing.assert_array_almost_equal(geom.local_half_extents, [0.5, 0.5, 0.5])
        assert geom.needs_aabb_scale_correction is True

    def test_identity_scale_madara_bottle(self):
        """Identity scale returns raw JSON values for madara_bottle."""
        geom = lookup_prim_geometry("madara_bottle")
        assert geom is not None
        assert geom.grasp_height == pytest.approx(0.0675126351416111)
        assert geom.rest_height == pytest.approx(0.06294623762369156)
        assert geom.top_surface_height == pytest.approx(0.07207903265953064)
        assert geom.needs_aabb_scale_correction is False

    def test_uniform_scale_cube(self):
        """Uniform non-identity scale: heights scale by sz, extents scale element-wise."""
        scale = [0.0515, 0.0515, 0.0515]
        geom = lookup_prim_geometry("cube", obj_scale=scale)
        assert geom is not None
        assert geom.grasp_height == pytest.approx(0.5 * 0.0515)
        assert geom.rest_height == pytest.approx(0.50274658203125 * 0.0515)
        assert geom.top_surface_height == pytest.approx(0.49725341796875 * 0.0515)
        np.testing.assert_array_almost_equal(
            geom.local_half_extents,
            [0.5 * 0.0515, 0.5 * 0.0515, 0.5 * 0.0515],
        )

    def test_anisotropic_scale_rect(self):
        """Anisotropic scale: heights scale by sz, extents scale by respective axes."""
        scale = [0.1, 0.1, 0.002]
        geom = lookup_prim_geometry("rect", obj_scale=scale)
        assert geom is not None
        assert geom.grasp_height == pytest.approx(0.5 * 0.002)
        assert geom.rest_height == pytest.approx(0.5 * 0.002)
        assert geom.top_surface_height == pytest.approx(0.5 * 0.002)
        np.testing.assert_array_almost_equal(
            geom.local_half_extents,
            [0.5 * 0.1, 0.5 * 0.1, 0.5 * 0.002],
        )

    def test_none_scale_treated_as_identity(self):
        """obj_scale=None returns unscaled values (same as identity)."""
        geom_none = lookup_prim_geometry("disc", obj_scale=None)
        geom_identity = lookup_prim_geometry("disc", obj_scale=[1.0, 1.0, 1.0])
        assert geom_none is not None
        assert geom_identity is not None
        assert geom_none.grasp_height == pytest.approx(geom_identity.grasp_height)
        assert geom_none.rest_height == pytest.approx(geom_identity.rest_height)
        assert geom_none.top_surface_height == pytest.approx(geom_identity.top_surface_height)
        np.testing.assert_array_almost_equal(
            geom_none.local_half_extents, geom_identity.local_half_extents
        )

    def test_unknown_asset_type_returns_none(self):
        """Unknown asset type returns None."""
        geom = lookup_prim_geometry("nonexistent_asset_xyz")
        assert geom is None

    def test_needs_aabb_scale_correction_preserved(self):
        """needs_aabb_scale_correction flag is preserved across scales."""
        # cube has correction=True
        geom = lookup_prim_geometry("cube", obj_scale=[2.0, 2.0, 2.0])
        assert geom is not None
        assert geom.needs_aabb_scale_correction is True

        # disc has correction=False
        geom = lookup_prim_geometry("disc", obj_scale=[2.0, 2.0, 2.0])
        assert geom is not None
        assert geom.needs_aabb_scale_correction is False


class TestGetAssetSymmetry:
    """Tests for get_asset_symmetry dispatch + scale-preservation check."""

    def test_ball_uniform_scale_returns_full(self):
        sym = get_asset_symmetry("ball", obj_scale=[0.05, 0.05, 0.05])
        assert sym is not None
        assert sym.kind == "full"

    def test_ball_no_scale_returns_full(self):
        # Without scale we cannot verify uniformity; return the tag as-is.
        sym = get_asset_symmetry("ball")
        assert sym is not None
        assert sym.kind == "full"

    def test_ball_non_uniform_scale_downgrades(self):
        # Ellipsoid is not spherical → no rotational symmetry.
        sym = get_asset_symmetry("ball", obj_scale=[0.05, 0.05, 0.10])
        assert sym is None

    def test_cylinder_uniform_cross_section(self):
        # sx == sy, sz free — still cylindrical → axis symmetry preserved.
        sym = get_asset_symmetry("cylinder", obj_scale=[0.04, 0.04, 0.20])
        assert sym is not None
        assert sym.kind == "continuous_axis"
        np.testing.assert_allclose(sym.axis_local, [0.0, 0.0, 1.0])

    def test_cylinder_non_uniform_cross_section_downgrades(self):
        # sx != sy → ellipse cross-section, not rotationally symmetric.
        sym = get_asset_symmetry("cylinder", obj_scale=[0.04, 0.05, 0.20])
        assert sym is None

    def test_soup_can_axis_local_y(self):
        sym = get_asset_symmetry("soup_can")
        assert sym is not None
        assert sym.kind == "continuous_axis"
        np.testing.assert_allclose(sym.axis_local, [0.0, 1.0, 0.0])

    def test_madara_bottle_axis_local_z(self):
        sym = get_asset_symmetry("madara_bottle")
        assert sym is not None
        assert sym.kind == "continuous_axis"
        np.testing.assert_allclose(sym.axis_local, [0.0, 0.0, 1.0])

    def test_cone_and_disc_tagged(self):
        for asset in ("cone", "disc"):
            sym = get_asset_symmetry(asset)
            assert sym is not None, f"{asset} missing symmetry tag"
            assert sym.kind == "continuous_axis"
            np.testing.assert_allclose(sym.axis_local, [0.0, 0.0, 1.0])

    def test_untagged_asset_returns_none(self):
        # cracker_box and mustard_bottle are deliberately NOT tagged in
        # the first pass — correction should remain disabled for them.
        assert get_asset_symmetry("cracker_box") is None
        assert get_asset_symmetry("mustard_bottle") is None
        assert get_asset_symmetry("sugar_box") is None
        assert get_asset_symmetry("cube") is None

    def test_unknown_asset_returns_none(self):
        assert get_asset_symmetry("not_a_real_asset") is None

    def test_lookup_prim_geometry_populates_symmetry(self):
        # The dispatch result lands on PrimGeometry.symmetry.
        geom = lookup_prim_geometry("ball", obj_scale=[0.03, 0.03, 0.03])
        assert geom is not None
        assert geom.symmetry is not None
        assert geom.symmetry.kind == "full"

        geom = lookup_prim_geometry("cylinder", obj_scale=[0.04, 0.04, 0.15])
        assert geom is not None
        assert geom.symmetry is not None
        assert geom.symmetry.kind == "continuous_axis"

    def test_lookup_prim_geometry_non_uniform_scale_downgrades(self):
        # Non-uniform scale on ball → PrimGeometry.symmetry is None.
        geom = lookup_prim_geometry("ball", obj_scale=[0.03, 0.03, 0.05])
        assert geom is not None
        assert geom.symmetry is None


# ---------------------------------------------------------------------------
# Tests for default_grasp_offset (asset-level grasp-point shift)
# ---------------------------------------------------------------------------


class TestDefaultGraspOffset:
    """Tests for the default_grasp_offset field on AssetMetaData and PrimGeometry."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Clear the precomputed geometry cache before each test."""
        import asset_data_utils
        asset_data_utils._precomputed_geometry_cache = None

    def test_primgeometry_defaults_to_zero(self):
        """PrimGeometry without an explicit offset stores zero 3-vector."""
        geom = PrimGeometry(
            grasp_height=0.05,
            rest_height=0.025,
            top_surface_height=0.025,
            local_half_extents=np.array([0.05, 0.05, 0.05]),
            needs_aabb_scale_correction=False,
        )
        np.testing.assert_array_equal(geom.default_grasp_offset, [0.0, 0.0, 0.0])

    def test_assetmetadata_rejects_non_3vec_offset(self):
        with pytest.raises(ValueError):
            AssetMetaData(
                asset_type="cube",
                default_grasp_offset=np.array([0.01, 0.02]),
            )

    def test_assetmetadata_accepts_zero_offset(self):
        meta = AssetMetaData(
            asset_type="cube",
            default_grasp_offset=np.array([0.0, 0.0, 0.0]),
        )
        np.testing.assert_array_equal(meta.default_grasp_offset, [0.0, 0.0, 0.0])

    def test_lookup_default_offset_zero_when_unset(self):
        """An asset with no asset-level default_grasp_offset reports zero."""
        geom = lookup_prim_geometry("cube")
        assert geom is not None
        np.testing.assert_array_almost_equal(geom.default_grasp_offset, [0.0, 0.0, 0.0])

    def test_lookup_picks_up_asset_default(self):
        """When an asset declares default_grasp_offset in ITEMS_MAP, lookup_prim_geometry returns it."""
        import asset_data_utils as adu
        offset = np.array([0.0, 0.0, 0.015])
        new_meta = AssetMetaData(
            asset_type="madara_bottle",
            pick_axis=2,
            usd_path="SimEnvs/assets/bottle_v3.usd",
            color="green",
            is_local_asset=True,
            default_grasp_offset=offset,
            symmetry=adu.AssetSymmetry(
                kind="continuous_axis",
                axis_local=np.array([0.0, 0.0, 1.0]),
            ),
        )
        # Patch ITEMS_MAP so the test asset metadata carries the offset.
        original = adu.ITEMS_MAP["madara_bottle"]
        adu.ITEMS_MAP["madara_bottle"] = new_meta
        try:
            geom = lookup_prim_geometry("madara_bottle")
            assert geom is not None
            np.testing.assert_array_almost_equal(
                geom.default_grasp_offset, [0.0, 0.0, 0.015],
            )
        finally:
            adu.ITEMS_MAP["madara_bottle"] = original

    def test_lookup_scales_asset_default(self):
        """Asset-default default_grasp_offset scales component-wise with obj_scale."""
        import asset_data_utils as adu
        offset = np.array([0.0, 0.0, 0.015])
        new_meta = AssetMetaData(
            asset_type="madara_bottle",
            pick_axis=2,
            usd_path="SimEnvs/assets/bottle_v3.usd",
            color="green",
            is_local_asset=True,
            default_grasp_offset=offset,
            symmetry=adu.AssetSymmetry(
                kind="continuous_axis",
                axis_local=np.array([0.0, 0.0, 1.0]),
            ),
        )
        original = adu.ITEMS_MAP["madara_bottle"]
        adu.ITEMS_MAP["madara_bottle"] = new_meta
        try:
            geom = lookup_prim_geometry("madara_bottle", obj_scale=[2.0, 2.0, 2.0])
            assert geom is not None
            np.testing.assert_array_almost_equal(
                geom.default_grasp_offset, [0.0, 0.0, 0.030],
            )
        finally:
            adu.ITEMS_MAP["madara_bottle"] = original
