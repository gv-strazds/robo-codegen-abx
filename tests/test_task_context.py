"""Tests for TaskContext and MockTaskContext."""

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

from task_context_mock import MockPickObj, MockTaskContext


class TestMockTaskContextInit:
    """Test MockTaskContext initialization and defaults."""

    def test_default_init(self):
        ctx = MockTaskContext()
        assert ctx.get_current_pick_name() == "pick_0"
        assert not ctx.task_finished
        assert not ctx.all_picks_done
        assert not ctx.targets_exhausted

    def test_custom_picks_targets(self):
        ctx = MockTaskContext(
            pick_names=["a", "b"],
            target_names=["t0", "t1", "t2"],
        )
        assert ctx.get_current_pick_name() == "a"
        assert len(ctx.strategy.pick_objs) == 2
        assert len(ctx.strategy.target_objs) == 3

    def test_custom_positions(self):
        pos = np.array([1.0, 2.0, 3.0])
        ctx = MockTaskContext(
            pick_names=["p0"],
            target_names=["t0"],
            pick_positions={"p0": pos},
        )
        result = ctx.get_picking_position("p0")
        np.testing.assert_array_equal(result, pos)


class TestMockTaskContextQueries:
    """Test state query methods."""

    def test_get_picking_position(self):
        ctx = MockTaskContext()
        pos = ctx.get_picking_position("pick_0")
        assert pos is not None
        assert len(pos) == 3

    def test_get_picking_position_unknown(self):
        ctx = MockTaskContext()
        pos = ctx.get_picking_position("nonexistent")
        assert pos is None

    def test_get_placing_info(self):
        ctx = MockTaskContext()
        name, pos, orient = ctx.get_placing_info("pick_0")
        assert name == "target_0"
        assert pos is not None
        assert orient is not None

    def test_get_placing_info_no_target(self):
        ctx = MockTaskContext(
            pick_names=["p0", "p1"],
            target_names=["t0"],
        )
        # p0 has target 0, p1 has no target
        name, pos, orient = ctx.get_placing_info("p1")
        assert name is None
        assert pos is None

    def test_get_joint_positions(self):
        ctx = MockTaskContext()
        joints = ctx.get_joint_positions()
        assert len(joints) == 6

    def test_get_ee_offset(self):
        ctx = MockTaskContext()
        offset = ctx.get_end_effector_offset("pick_0")
        assert len(offset) == 3

    def test_get_ee_orientation(self):
        ctx = MockTaskContext()
        orient = ctx.get_end_effector_orientation("pick_0")
        assert len(orient) == 4

    def test_get_ee_orientation_no_pick_name(self):
        ctx = MockTaskContext()
        orient = ctx.get_end_effector_orientation()
        assert len(orient) == 4

    def test_get_ee_orientation_for_drop_default_none(self):
        ctx = MockTaskContext()
        result = ctx.get_end_effector_orientation_for_drop("pick_0", "target_0")
        assert result is None

    def test_get_ee_offset_for_drop_default_none(self):
        ctx = MockTaskContext()
        # With no drop orientation, offset should be None
        result = ctx.get_end_effector_offset_for_drop("pick_0")
        assert result is None

    def test_get_ee_offset_for_drop_same_orientation_returns_pick_offset(self):
        """When the drop orientation is the same as the pick orientation
        the EE-to-item-center vector hasn't rotated, so the world-frame
        drop offset must equal the pick offset ``[0, 0, grasp_height]``.

        Guards against the regression where callers that resolve a None
        drop orientation to the pick orientation before invoking this
        API would accidentally trigger a spurious lateral X offset."""
        from asset_data_utils import PrimGeometry

        geom = PrimGeometry(
            rest_height=0.05,
            grasp_height=0.04,
            top_surface_height=0.025,
            local_half_extents=np.array([0.01, 0.01, 0.05]),
            needs_aabb_scale_correction=False,
        )
        ctx = MockTaskContext(prim_geometry={"pick_0": geom})
        pick_orient = ctx.get_end_effector_orientation("pick_0")
        # Same-orientation drop → drop offset == pick offset.
        result = ctx.get_end_effector_offset_for_drop("pick_0", pick_orient.copy())
        np.testing.assert_array_almost_equal(result, [0.0, 0.0, 0.04])
        # Sign-flipped quaternion represents the same rotation → also pick offset.
        result_neg = ctx.get_end_effector_offset_for_drop("pick_0", -pick_orient)
        np.testing.assert_array_almost_equal(result_neg, [0.0, 0.0, 0.04])

    def test_get_ee_height_for_move(self):
        ctx = MockTaskContext()
        h = ctx.get_ee_height_for_move()
        assert h == pytest.approx(0.3)


class TestMockTaskContextMutations:
    """Test mutation methods."""

    def test_advance_pick_index(self):
        ctx = MockTaskContext()
        assert ctx.get_current_pick_name() == "pick_0"

        next_name = ctx.advance_pick_index()
        assert next_name == "pick_1"
        assert ctx.get_current_pick_name() == "pick_1"

    def test_advance_past_end(self):
        ctx = MockTaskContext(pick_names=["p0"])
        # advance_pick_index past the only pick returns None
        result = ctx.advance_pick_index()
        assert result is None
        # all_picks_done is defined semantically (every pick completed or
        # permanently unreachable); mark p0 complete to satisfy it.
        ctx.mark_pick_complete("p0")
        assert ctx.all_picks_done

    def test_mark_pick_complete(self):
        ctx = MockTaskContext()
        ctx.mark_pick_complete("pick_0")
        assert "pick_0" in ctx._completed_picks

    def test_task_finished_property(self):
        ctx = MockTaskContext()
        assert not ctx.task_finished
        ctx.task_finished = True
        assert ctx.task_finished

    def test_targets_exhausted_property(self):
        ctx = MockTaskContext()
        assert not ctx.targets_exhausted
        ctx.targets_exhausted = True
        assert ctx.targets_exhausted
        # get_current_pick_name returns None when exhausted
        assert ctx.get_current_pick_name() is None


class TestMockTaskContextReset:
    """Test reset and reorder."""

    def test_reset(self):
        ctx = MockTaskContext()
        ctx.advance_pick_index()
        ctx.task_finished = True
        ctx.targets_exhausted = True
        ctx.mark_pick_complete("pick_0")

        ctx.reset()
        assert ctx.get_current_pick_name() == "pick_0"
        assert not ctx.task_finished
        assert not ctx.targets_exhausted
        assert not ctx.all_picks_done
        assert len(ctx._completed_picks) == 0

    def test_reset_with_new_order(self):
        ctx = MockTaskContext()
        ctx.advance_pick_index()  # now at pick_1

        ctx.reset(picking_order_item_names=["pick_2", "pick_0", "pick_1"])
        assert ctx.get_current_pick_name() == "pick_2"

    def test_reorder_picks(self):
        ctx = MockTaskContext()
        ctx.reorder_picks(["pick_2", "pick_1", "pick_0"])
        assert ctx.get_current_pick_name() == "pick_2"

    def test_reorder_picks_preserve_current(self):
        ctx = MockTaskContext()
        ctx.advance_pick_index()  # now at pick_1
        ctx.reorder_picks(["pick_2", "pick_1", "pick_0"], current_pick_name="pick_1")
        assert ctx.get_current_pick_name() == "pick_1"

    def test_update_pairings(self):
        ctx = MockTaskContext()
        ctx.update_pairings(
            {"pick_0": "target_2", "pick_1": "target_0", "pick_2": "target_1"},
        )
        name, _, _ = ctx.get_placing_info("pick_0")
        assert name == "target_2"


class TestHardwareOwnership:
    """Test robot hardware ownership properties and methods."""

    def test_gripper_default_from_robot(self):
        ctx = MockTaskContext()
        assert ctx.gripper is ctx.robot.gripper

    def test_gripper_via_init(self):
        from robot_controllers.mock_robot import MockGripper

        mock_gripper = MockGripper()
        ctx = MockTaskContext(gripper=mock_gripper)
        assert ctx.gripper is mock_gripper
        assert ctx.gripper is not ctx.robot.gripper

    def test_robot_property(self):
        ctx = MockTaskContext()
        assert ctx.robot is not None
        assert hasattr(ctx.robot, "get_joints_state")

    def test_reset_gripper(self):
        from robot_controllers.mock_robot import MockGripper

        mock_gripper = MockGripper()
        ctx = MockTaskContext(gripper=mock_gripper)
        mock_gripper.close()
        assert mock_gripper.is_closed()
        ctx.reset_gripper()
        np.testing.assert_array_equal(
            mock_gripper.get_joint_positions(),
            mock_gripper.joint_opened_positions,
        )

    def test_arm_commander_available(self):
        ctx = MockTaskContext()
        assert ctx.arm_commander is not None
        assert hasattr(ctx.arm_commander, 'send_motion_command')

    def test_gripper_commander_available(self):
        ctx = MockTaskContext()
        assert ctx.gripper_commander is not None
        assert hasattr(ctx.gripper_commander, 'open')
        assert hasattr(ctx.gripper_commander, 'close')


class TestGraspOffsetWorld:
    """Test ``TaskContextBase._grasp_offset_world`` resolution and caching.

    The helper resolves a per-pick grasp offset in world frame from:
      1. ``self._grasp_offset_local_overrides`` keyed by the pick's
         asset_type (task-level override), or
      2. ``PrimGeometry.default_grasp_offset`` (asset-level default).
    The chosen local-frame vector is rotated by
    ``geom.reference_orientation`` to world frame and cached per pick_name.
    """

    def _make_ctx(
        self,
        *,
        bottle_asset_type="madara_bottle",
        non_bottle_asset_type="cube",
        bottle_default=None,           # asset default for the bottle (or None)
        bottle_reference=None,         # reference_orientation on the bottle's geom
        bottle_scale=None,             # local scale on the bottle pick_obj
        overrides=None,
    ):
        """Build a minimal MockTaskContext with two picks of different asset_types."""
        from asset_data_utils import PrimGeometry

        # Geometry: simple boxy defaults; default_grasp_offset only on bottle.
        def _make_geom(default_offset=None, ref=None):
            kwargs = dict(
                grasp_height=0.05,
                rest_height=0.025,
                top_surface_height=0.025,
                local_half_extents=np.array([0.02, 0.02, 0.07]),
                needs_aabb_scale_correction=False,
            )
            if default_offset is not None:
                kwargs["default_grasp_offset"] = np.asarray(default_offset, dtype=float)
            if ref is not None:
                kwargs["reference_orientation"] = np.asarray(ref, dtype=float)
            return PrimGeometry(**kwargs)

        prim_geometry = {
            "bottle_0": _make_geom(default_offset=bottle_default, ref=bottle_reference),
            "cube_0": _make_geom(default_offset=None, ref=None),
        }
        ctx = MockTaskContext(
            pick_names=["bottle_0", "cube_0"],
            target_names=["t0", "t1"],
            prim_geometry=prim_geometry,
            grasp_offset_local_overrides=overrides,
        )
        # MockTaskContext creates its own MockPickObj instances; attach
        # semantic labels and per-pick scale on those.
        bottle_obj = ctx.strategy.pick_objs_by_name["bottle_0"]
        bottle_obj._semantic_labels["type"] = [bottle_asset_type]
        if bottle_scale is not None:
            bottle_obj.get_local_scale = lambda s=bottle_scale: np.asarray(s, dtype=float)
        cube_obj = ctx.strategy.pick_objs_by_name["cube_0"]
        cube_obj._semantic_labels["type"] = [non_bottle_asset_type]
        return ctx

    def test_no_override_no_default_returns_zero(self):
        ctx = self._make_ctx()
        np.testing.assert_array_equal(
            ctx._grasp_offset_world("bottle_0"), [0.0, 0.0, 0.0],
        )
        np.testing.assert_array_equal(
            ctx._grasp_offset_world("cube_0"), [0.0, 0.0, 0.0],
        )

    def test_asset_default_used_when_no_override(self):
        ctx = self._make_ctx(bottle_default=[0.0, 0.0, 0.020])
        np.testing.assert_array_almost_equal(
            ctx._grasp_offset_world("bottle_0"), [0.0, 0.0, 0.020],
        )
        # Other pick (no default, no override) stays at zero.
        np.testing.assert_array_equal(
            ctx._grasp_offset_world("cube_0"), [0.0, 0.0, 0.0],
        )

    def test_override_wins_over_default(self):
        ctx = self._make_ctx(
            bottle_default=[0.0, 0.0, 0.020],
            overrides={"madara_bottle": np.array([0.0, 0.0, 0.015])},
        )
        np.testing.assert_array_almost_equal(
            ctx._grasp_offset_world("bottle_0"), [0.0, 0.0, 0.015],
        )

    def test_override_only_applies_to_matching_asset_type(self):
        ctx = self._make_ctx(
            overrides={"madara_bottle": np.array([0.0, 0.0, 0.015])},
        )
        np.testing.assert_array_almost_equal(
            ctx._grasp_offset_world("bottle_0"), [0.0, 0.0, 0.015],
        )
        # The cube has no override entry — stays at zero.
        np.testing.assert_array_equal(
            ctx._grasp_offset_world("cube_0"), [0.0, 0.0, 0.0],
        )

    def test_override_scaled_by_pick_local_scale(self):
        ctx = self._make_ctx(
            bottle_scale=[2.0, 2.0, 2.0],
            overrides={"madara_bottle": np.array([0.0, 0.0, 0.015])},
        )
        np.testing.assert_array_almost_equal(
            ctx._grasp_offset_world("bottle_0"), [0.0, 0.0, 0.030],
        )

    def test_reference_orientation_rotates_offset_to_world(self):
        # +90° around X rotates (x, y, z) -> (x, -z, y).
        # Bottle local +Z (0, 0, 0.015) ends up at world (0, -0.015, 0).
        half_x = np.cos(np.pi / 4)
        sin_x = np.sin(np.pi / 4)
        ctx = self._make_ctx(
            bottle_default=[0.0, 0.0, 0.015],
            bottle_reference=np.array([half_x, sin_x, 0.0, 0.0]),
        )
        result = ctx._grasp_offset_world("bottle_0")
        np.testing.assert_allclose(result, [0.0, -0.015, 0.0], atol=1e-9)

    def test_cache_returns_same_array_on_repeat(self):
        ctx = self._make_ctx(
            overrides={"madara_bottle": np.array([0.0, 0.0, 0.015])},
        )
        first = ctx._grasp_offset_world("bottle_0")
        second = ctx._grasp_offset_world("bottle_0")
        # Same array identity (memoized).
        assert first is second

    def test_unknown_pick_name_returns_zero(self):
        ctx = self._make_ctx()
        np.testing.assert_array_equal(
            ctx._grasp_offset_world("nonexistent"), [0.0, 0.0, 0.0],
        )

    def test_empty_pick_name_returns_zero(self):
        ctx = self._make_ctx()
        np.testing.assert_array_equal(
            ctx._grasp_offset_world(""), [0.0, 0.0, 0.0],
        )
        np.testing.assert_array_equal(
            ctx._grasp_offset_world(None), [0.0, 0.0, 0.0],
        )


class TestBottlePickStrategy:
    """Test BottlePickStrategy drop orientation and offset via MockTaskContext."""

    def _make_context(self, prim_geometry=None):
        from multi_pick_strategy import BottlePickStrategy

        pick_objs = [MockPickObj("bottle_0"), MockPickObj("bottle_1")]
        target_objs = [MockPickObj("pad_0"), MockPickObj("pad_1")]
        strategy = BottlePickStrategy(
            pick_objs=pick_objs,
            target_objs=target_objs,
        )
        strategy.initialize_pairings()
        ctx = MockTaskContext(
            pick_names=["bottle_0", "bottle_1"],
            target_names=["pad_0", "pad_1"],
            strategy=strategy,
            prim_geometry=prim_geometry,
        )
        return ctx

    def test_drop_orientation_not_none(self):
        ctx = self._make_context()
        orient = ctx.get_end_effector_orientation_for_drop("bottle_0", "pad_0")
        assert orient is not None
        assert len(orient) == 4

    def test_drop_offset_no_geometry(self):
        """Without geometry, drop offset is None (no geometry available)."""
        ctx = self._make_context()
        drop_orient = ctx.get_end_effector_orientation_for_drop("bottle_0", "pad_0")
        offset = ctx.get_end_effector_offset_for_drop("bottle_0", drop_orient)
        assert offset is None

    def test_drop_offset_from_geometry(self):
        """Bottle case: the EE rotates 90° between pick and drop, so the
        pick-time vertical offset of ``grasp_height`` rotates into a
        horizontal world offset of the same magnitude.  For the default
        UR10 pick orientation (π/2 around Y) and the BottlePickStrategy
        drop orientation (π/2 around X), the relative rotation maps
        ``[0, 0, grasp_height]`` to ``[-grasp_height, 0, 0]``."""
        from asset_data_utils import PrimGeometry

        geom = PrimGeometry(
            rest_height=0.05,
            grasp_height=0.04,
            top_surface_height=0.025,
            local_half_extents=np.array([0.01, 0.01, 0.05]),
            needs_aabb_scale_correction=False,
        )
        ctx = self._make_context(prim_geometry={"bottle_0": geom})
        drop_orient = ctx.get_end_effector_orientation_for_drop("bottle_0", "pad_0")
        offset = ctx.get_end_effector_offset_for_drop("bottle_0", drop_orient)
        np.testing.assert_array_almost_equal(offset, [-0.04, 0.0, 0.0])

    def test_pick_orientation_is_default(self):
        ctx = self._make_context()
        orient = ctx.get_end_effector_orientation("bottle_0")
        assert orient is not None
        assert len(orient) == 4

    def test_via_mock_context(self):
        """BottlePickStrategy works through MockTaskContext delegation."""
        from multi_pick_strategy import BottlePickStrategy

        pick_objs = [MockPickObj("b0")]
        target_objs = [MockPickObj("t0")]
        strategy = BottlePickStrategy(
            pick_objs=pick_objs,
            target_objs=target_objs,
        )
        strategy.initialize_pairings()
        ctx = MockTaskContext(
            pick_names=["b0"],
            target_names=["t0"],
            strategy=strategy,
        )
        # Drop orientation should be non-None (from BottlePickStrategy)
        drop_orient = ctx.get_end_effector_orientation_for_drop("b0", "t0")
        assert drop_orient is not None
        # Drop offset without geometry should be None
        drop_offset = ctx.get_end_effector_offset_for_drop("b0", drop_orient)
        assert drop_offset is None


class TestLayeredStackStrategy:
    """Test LayeredStackStrategy with arbitrary layers and classification."""

    @staticmethod
    def _name_classify(obj):
        """Classify by the part of the name before '_' (e.g. 'A_0' -> 'A')."""
        return obj.name.split("_")[0]

    def _make_objs(self, labels_and_counts):
        """Create MockPickObj list from (label, count) pairs.

        Returns list like [MockPickObj("A_0"), MockPickObj("A_1"), MockPickObj("B_0"), ...]
        """
        objs = []
        for label, count in labels_and_counts:
            for i in range(count):
                objs.append(MockPickObj(f"{label}_{i}"))
        return objs

    def _make_strategy(self, pick_labels, layer_order, max_stacks=3,
                       skip_values=None, num_targets=None):
        from multi_pick_strategy import LayeredStackStrategy

        picks = self._make_objs(pick_labels)
        n_targets = num_targets if num_targets is not None else max_stacks
        targets = [MockPickObj(f"marker_{i}") for i in range(n_targets)]

        strategy = LayeredStackStrategy(
            pick_objs=picks,
            target_objs=targets,
            layer_order=layer_order,
            max_stacks=max_stacks,
            classify_fn=self._name_classify,
            skip_values=skip_values,
        )
        strategy.initialize_pairings()
        return strategy, picks, targets

    def test_two_layer_pairings(self):
        """2-layer stacks (A=bottom, B=top), 2 complete stacks."""
        strategy, picks, targets = self._make_strategy(
            pick_labels=[("A", 2), ("B", 2)],
            layer_order=["A", "B"],
            max_stacks=3,
        )
        base = strategy._base_target_count
        assert strategy._num_complete_stacks == 2

        # Picking order: A_0, B_0, A_1, B_1 (stack-by-stack, bottom-to-top)
        order = strategy.picking_order_item_names
        assert order == ["A_0", "B_0", "A_1", "B_1"]

        # Check pairings: A picks -> markers, B picks -> A-as-target
        by_name = strategy.pairings_by_pick_name
        target_idx_by_name = {t.name: i for i, t in enumerate(targets)}
        assert by_name["A_0"] == "marker_0"
        assert by_name["A_1"] == "marker_1"
        # B targets should be in the extended target region
        assert by_name["B_0"] is not None
        assert by_name["B_1"] is not None
        assert target_idx_by_name[by_name["B_0"]] >= base  # extended targets

    def test_two_layer_valid_targets(self):
        """valid_targets_for_pick correctly separates layers for 2-layer stacks."""
        strategy, picks, targets = self._make_strategy(
            pick_labels=[("A", 2), ("B", 2)],
            layer_order=["A", "B"],
            max_stacks=3,
        )
        base = strategy._base_target_count
        target_idx_by_name = {t.name: i for i, t in enumerate(targets)}

        # A picks should have marker targets
        valid_a = strategy.valid_targets_for_pick("A_0")
        assert all(target_idx_by_name[n] < base for n in valid_a)

        # B picks should have extended targets (A-as-target)
        valid_b = strategy.valid_targets_for_pick("B_0")
        assert all(target_idx_by_name[n] >= base for n in valid_b)

    def test_four_layer_stacks(self):
        """4-layer stacks with 2 complete stacks."""
        strategy, picks, targets = self._make_strategy(
            pick_labels=[("W", 2), ("X", 2), ("Y", 2), ("Z", 2)],
            layer_order=["W", "X", "Y", "Z"],
            max_stacks=3,
        )
        base = strategy._base_target_count
        assert strategy._num_complete_stacks == 2

        order = strategy.picking_order_item_names
        # Stack 0: W_0, X_0, Y_0, Z_0; Stack 1: W_1, X_1, Y_1, Z_1
        assert order == ["W_0", "X_0", "Y_0", "Z_0", "W_1", "X_1", "Y_1", "Z_1"]

        # Verify each layer has correct valid targets
        by_name = strategy.pairings_by_pick_name
        target_idx_by_name = {t.name: i for i, t in enumerate(targets)}
        assert by_name["W_0"] == "marker_0"
        assert by_name["W_1"] == "marker_1"
        # X targets W (extended), Y targets X, Z targets Y
        assert target_idx_by_name[by_name["X_0"]] >= base
        assert target_idx_by_name[by_name["Y_0"]] > target_idx_by_name[by_name["X_0"]]
        assert target_idx_by_name[by_name["Z_0"]] > target_idx_by_name[by_name["Y_0"]]

    def test_skip_values(self):
        """Objects with skip values get None target."""
        strategy, picks, targets = self._make_strategy(
            pick_labels=[("A", 2), ("B", 2), ("skip", 3)],
            layer_order=["A", "B"],
            max_stacks=3,
            skip_values=["skip"],
        )
        by_name = strategy.pairings_by_pick_name
        for name, tgt in by_name.items():
            if name.startswith("skip"):
                assert tgt is None, f"{name} should have no target"

        # A and B picks should have targets
        assert by_name["A_0"] is not None
        assert by_name["B_0"] is not None

    def test_zero_complete_stacks(self):
        """When a layer has no objects, all picks get None target."""
        strategy, picks, targets = self._make_strategy(
            pick_labels=[("A", 3)],  # no B objects
            layer_order=["A", "B"],
            max_stacks=3,
        )
        assert strategy._num_complete_stacks == 0

        by_name = strategy.pairings_by_pick_name
        for name, tgt in by_name.items():
            assert tgt is None, f"{name} should have no target"

        assert strategy.picking_order_item_names == []

    def test_max_stacks_limits_pairings(self):
        """max_stacks caps the number of complete stacks formed."""
        strategy, picks, targets = self._make_strategy(
            pick_labels=[("A", 5), ("B", 5)],
            layer_order=["A", "B"],
            max_stacks=2,
            num_targets=2,
        )
        assert strategy._num_complete_stacks == 2
        order = strategy.picking_order_item_names
        assert len(order) == 4  # 2 stacks * 2 layers

        # Excess A and B should have None
        by_name = strategy.pairings_by_pick_name
        none_count = sum(1 for v in by_name.values() if v is None)
        assert none_count == 6  # 5+5 - 4 = 6

    def test_color_stack_backward_compat(self):
        """ColorStackStrategy produces the same pairings as LayeredStackStrategy
        with the same layer_order for a known input."""
        from multi_pick_strategy import LayeredStackStrategy
        from tasks_mock.mock_task_utils import setup_mock_modules
        setup_mock_modules()
        from tasks.table_task_conveyor_color_stacks import ColorStackStrategy

        # Custom has_color_fn that classifies by name prefix (avoids asset_utils import)
        def _has_color(obj, color_name):
            return obj.name.startswith(color_name + "_")

        pick_names = ["blue_0", "green_0", "red_0", "blue_1", "green_1", "red_1", "yellow_0"]
        all_colors = ["blue", "green", "red", "yellow"]

        picks = [MockPickObj(name) for name in pick_names]
        markers = [MockPickObj(f"marker_{i}") for i in range(3)]

        cs = ColorStackStrategy(
            pick_objs=list(picks), target_objs=list(markers), max_stacks=3,
            has_color_fn=_has_color, color_palette=all_colors,
        )
        cs.initialize_pairings()

        # Build an equivalent LayeredStackStrategy with a classify_fn
        def _classify(obj):
            for c in all_colors:
                if obj.name.startswith(c + "_"):
                    return c
            return None

        picks2 = [MockPickObj(name) for name in pick_names]
        markers2 = [MockPickObj(f"marker_{i}") for i in range(3)]

        ls = LayeredStackStrategy(
            pick_objs=list(picks2), target_objs=list(markers2),
            layer_order=["blue", "green", "red"],
            max_stacks=3,
            classify_fn=_classify,
            skip_values=["yellow"],
        )
        ls.initialize_pairings()

        assert cs._num_complete_stacks == ls._num_complete_stacks
        assert cs.picking_order_item_names == ls.picking_order_item_names

        # Compare pairing structure (pick_name -> has target or not)
        for name in pick_names:
            cs_has = cs.pairings_by_pick_name.get(name) is not None
            ls_has = ls.pairings_by_pick_name.get(name) is not None
            assert cs_has == ls_has, f"Mismatch for {name}: CS={cs_has}, LS={ls_has}"


class TestBaseCheckFnLayeredStack:
    """Test base_check_fn parameter for LayeredStackStrategy."""

    @classmethod
    def setup_class(cls):
        from tasks_mock.mock_task_utils import setup_mock_modules
        setup_mock_modules()

    @staticmethod
    def _name_classify(obj):
        return obj.name.split("_")[0]

    def _make_objs(self, labels_and_counts):
        objs = []
        for label, count in labels_and_counts:
            for i in range(count):
                objs.append(MockPickObj(f"{label}_{i}"))
        return objs

    def test_base_check_fn_overrides_bin_geometry(self):
        """Custom base_check_fn takes priority over bin_geometry."""
        from multi_pick_strategy import LayeredStackStrategy

        picks = self._make_objs([("A", 2), ("B", 2)])
        markers = [MockPickObj("marker_0"), MockPickObj("marker_1")]

        calls = []

        def custom_check(pick_obj, target_obj=None, bb_cache=None, obj_scale=None):
            calls.append(pick_obj.name)
            return True

        strategy = LayeredStackStrategy(
            pick_objs=picks, target_objs=markers,
            layer_order=["A", "B"], max_stacks=2,
            classify_fn=self._name_classify,
            bin_geometry={"center_xy": np.array([0, 0]), "inner_size": np.array([1, 1]),
                          "floor_z": 0.0, "height": 0.2},
            base_check_fn=custom_check,
        )
        strategy.initialize_pairings()

        check_fn = strategy.get_spatial_check_fn()
        assert check_fn is not None

        # Call for a base target — should use custom_check, not is_within_box_geometry
        result = check_fn(picks[0], markers[0])
        assert result is True
        assert "A_0" in calls

    def test_base_check_fn_without_bin_geometry(self):
        """base_check_fn works without bin_geometry; dispatches correctly per layer."""
        from multi_pick_strategy import LayeredStackStrategy

        picks = self._make_objs([("A", 1), ("B", 1)])
        markers = [MockPickObj("marker_0")]

        base_calls = []

        def custom_check(pick_obj, target_obj=None, bb_cache=None, obj_scale=None):
            base_calls.append(pick_obj.name)
            return True

        strategy = LayeredStackStrategy(
            pick_objs=picks, target_objs=markers,
            layer_order=["A", "B"], max_stacks=1,
            classify_fn=self._name_classify,
            base_check_fn=custom_check,
        )
        strategy.initialize_pairings()

        check_fn = strategy.get_spatial_check_fn()
        assert check_fn is not None

        # Base target -> custom_check is called
        result = check_fn(picks[0], markers[0])
        assert result is True
        assert base_calls == ["A_0"]

        # Upper-layer target -> should NOT call custom_check (dispatches to is_on_top)
        base_calls.clear()
        upper_target = strategy._target_objs[strategy._base_target_count]
        # is_on_top may fail on mock objects, but the key assertion is
        # that the custom base check is NOT invoked for upper-layer targets
        try:
            check_fn(picks[1], upper_target)
        except Exception:
            pass
        assert len(base_calls) == 0, "base_check_fn should not be called for upper-layer targets"

    def test_bin_geometry_backward_compat(self):
        """bin_geometry alone produces a working spatial check (backward compat).

        Patches is_within_box_geometry to avoid AABB errors on mock objects,
        but verifies the correct geometry args are forwarded.
        """
        from unittest.mock import patch
        from multi_pick_strategy import LayeredStackStrategy

        picks = self._make_objs([("A", 1), ("B", 1)])
        markers = [MockPickObj("marker_0")]
        bg = {"center_xy": np.array([0, 0]), "inner_size": np.array([1, 1]),
              "floor_z": 0.0, "height": 0.5, "z_tol": 0.05}

        strategy = LayeredStackStrategy(
            pick_objs=picks, target_objs=markers,
            layer_order=["A", "B"], max_stacks=1,
            classify_fn=self._name_classify,
            bin_geometry=bg,
        )
        strategy.initialize_pairings()

        check_fn = strategy.get_spatial_check_fn()
        assert check_fn is not None

        # Patch is_within_box_geometry where it was imported (in multi_pick_strategy)
        with patch("task_verification.is_within_box_geometry", return_value=True) as mock_iwbg:
            result = check_fn(picks[0], markers[0])
            assert result is True
            mock_iwbg.assert_called_once()
            call_kwargs = mock_iwbg.call_args
            np.testing.assert_array_equal(call_kwargs.kwargs["box_center_xy"], bg["center_xy"])
            assert call_kwargs.kwargs["z_tol"] == 0.05

    def test_no_check_fn_returns_none(self):
        """get_spatial_check_fn() returns None when neither base_check_fn nor bin_geometry."""
        from multi_pick_strategy import LayeredStackStrategy

        picks = self._make_objs([("A", 1), ("B", 1)])
        markers = [MockPickObj("marker_0")]

        strategy = LayeredStackStrategy(
            pick_objs=picks, target_objs=markers,
            layer_order=["A", "B"], max_stacks=1,
            classify_fn=self._name_classify,
        )
        strategy.initialize_pairings()

        assert strategy.get_spatial_check_fn() is None


class TestBaseCheckFnSingleStack:
    """Test base_check_fn parameter for SingleStackStrategy."""

    @classmethod
    def setup_class(cls):
        from tasks_mock.mock_task_utils import setup_mock_modules
        setup_mock_modules()

    def test_base_check_fn_overrides_bin_geometry(self):
        """Custom base_check_fn takes priority over bin_geometry."""
        from multi_pick_strategy import SingleStackStrategy

        picks = [MockPickObj("p0"), MockPickObj("p1")]
        markers = [MockPickObj("marker_0")]

        calls = []

        def custom_check(pick_obj, target_obj=None, bb_cache=None, obj_scale=None):
            calls.append(pick_obj.name)
            return True

        strategy = SingleStackStrategy(
            pick_objs=picks, target_objs=markers,
            bin_geometry={"center_xy": np.array([0, 0]), "inner_size": np.array([1, 1]),
                          "floor_z": 0.0, "height": 0.2},
            base_check_fn=custom_check,
        )
        strategy.initialize_pairings()

        check_fn = strategy.get_spatial_check_fn()
        assert check_fn is not None

        result = check_fn(picks[0], markers[0])
        assert result is True
        assert "p0" in calls

    def test_base_check_fn_without_bin_geometry(self):
        """base_check_fn works without bin_geometry."""
        from multi_pick_strategy import SingleStackStrategy

        picks = [MockPickObj("p0"), MockPickObj("p1")]
        markers = [MockPickObj("marker_0")]

        def custom_check(pick_obj, target_obj=None, bb_cache=None, obj_scale=None):
            return True

        strategy = SingleStackStrategy(
            pick_objs=picks, target_objs=markers,
            base_check_fn=custom_check,
        )
        strategy.initialize_pairings()

        check_fn = strategy.get_spatial_check_fn()
        assert check_fn is not None

        # Base target -> custom_check
        result = check_fn(picks[0], markers[0])
        assert result is True

    def test_bin_geometry_backward_compat(self):
        """bin_geometry alone still works for SingleStackStrategy."""
        from unittest.mock import patch
        from multi_pick_strategy import SingleStackStrategy

        picks = [MockPickObj("p0"), MockPickObj("p1")]
        markers = [MockPickObj("marker_0")]
        bg = {"center_xy": np.array([0, 0]), "inner_size": np.array([1, 1]),
              "floor_z": 0.0, "height": 0.5, "z_tol": 0.05}

        strategy = SingleStackStrategy(
            pick_objs=picks, target_objs=markers,
            bin_geometry=bg,
        )
        strategy.initialize_pairings()

        check_fn = strategy.get_spatial_check_fn()
        assert check_fn is not None

        with patch("task_verification.is_within_box_geometry", return_value=True) as mock_iwbg:
            result = check_fn(picks[0], markers[0])
            assert result is True
            mock_iwbg.assert_called_once()

    def test_no_check_fn_returns_none(self):
        """get_spatial_check_fn() returns None when neither provided."""
        from multi_pick_strategy import SingleStackStrategy

        picks = [MockPickObj("p0")]
        markers = [MockPickObj("marker_0")]

        strategy = SingleStackStrategy(
            pick_objs=picks, target_objs=markers,
        )
        strategy.initialize_pairings()

        assert strategy.get_spatial_check_fn() is None


class TestBuildBinGeometryCheck:
    """Test the build_bin_geometry_check() helper function."""

    @classmethod
    def setup_class(cls):
        from tasks_mock.mock_task_utils import setup_mock_modules
        setup_mock_modules()

    def test_produces_callable(self):
        from multi_pick_strategy import build_bin_geometry_check

        bg = {"center_xy": np.array([0, 0]), "inner_size": np.array([1, 1]),
              "floor_z": 0.0, "height": 0.5}
        fn = build_bin_geometry_check(bg)
        assert callable(fn)

    def test_passes_z_tol(self):
        from unittest.mock import patch
        from multi_pick_strategy import build_bin_geometry_check

        bg = {"center_xy": np.array([0, 0]), "inner_size": np.array([1, 1]),
              "floor_z": 0.0, "height": 0.5, "z_tol": 0.05}
        fn = build_bin_geometry_check(bg)

        with patch("task_verification.is_within_box_geometry", return_value=True) as mock_iwbg:
            result = fn(MockPickObj("test"))
            assert result is True
            mock_iwbg.assert_called_once()
            assert mock_iwbg.call_args.kwargs["z_tol"] == 0.05


class TestStackingConstraints:
    """Test compute_stacking_map and stacking constraint enforcement in MultiPickStrategy."""

    def test_compute_stacking_map_basic(self):
        """Two items at same XY, different Z → correct map."""
        from multi_pick_strategy import compute_stacking_map

        bottom = MockPickObj("bottom", position=np.array([0.0, 0.0, 0.1]))
        top = MockPickObj("top", position=np.array([0.0, 0.0, 0.2]))
        smap = compute_stacking_map([bottom, top])
        assert smap == {"bottom": ["top"]}

    def test_compute_stacking_map_no_stacking(self):
        """Items at different XY positions → empty map."""
        from multi_pick_strategy import compute_stacking_map

        a = MockPickObj("a", position=np.array([0.0, 0.0, 0.1]))
        b = MockPickObj("b", position=np.array([1.0, 0.0, 0.1]))
        smap = compute_stacking_map([a, b])
        assert smap == {}

    def test_compute_stacking_map_three_layers(self):
        """Three items in a column → correct chain."""
        from multi_pick_strategy import compute_stacking_map

        low = MockPickObj("low", position=np.array([0.5, 0.5, 0.0]))
        mid = MockPickObj("mid", position=np.array([0.5, 0.5, 0.1]))
        high = MockPickObj("high", position=np.array([0.5, 0.5, 0.2]))
        smap = compute_stacking_map([low, mid, high])
        assert smap == {"low": ["mid"], "mid": ["high"]}

    def test_compute_stacking_map_multiple_columns(self):
        """Separate stack columns are computed independently."""
        from multi_pick_strategy import compute_stacking_map

        a_low = MockPickObj("a_low", position=np.array([0.0, 0.0, 0.0]))
        a_high = MockPickObj("a_high", position=np.array([0.0, 0.0, 0.1]))
        b_low = MockPickObj("b_low", position=np.array([1.0, 1.0, 0.0]))
        b_high = MockPickObj("b_high", position=np.array([1.0, 1.0, 0.1]))
        smap = compute_stacking_map([a_low, a_high, b_low, b_high])
        assert smap == {"a_low": ["a_high"], "b_low": ["b_high"]}

    def test_stacking_reorders_top_down(self):
        """initialize_pairings() with stacking_map reorders picking order top-down."""
        from multi_pick_strategy import MultiPickStrategy, compute_stacking_map

        # 3 items in a single column, generated bottom-up
        objs = [
            MockPickObj("item_0", position=np.array([0.0, 0.0, 0.0])),
            MockPickObj("item_1", position=np.array([0.0, 0.0, 0.1])),
            MockPickObj("item_2", position=np.array([0.0, 0.0, 0.2])),
        ]
        targets = [MockPickObj(f"t_{i}") for i in range(3)]
        smap = compute_stacking_map(objs)

        strategy = MultiPickStrategy(
            pick_objs=objs, target_objs=targets, stacking_map=smap,
        )
        strategy.initialize_pairings()

        # Top item first, then middle, then bottom
        assert strategy.picking_order_item_names == ["item_2", "item_1", "item_0"]

    def test_stacking_advance_skips_blocked(self):
        """advance_pick_index() skips blocked items."""
        from multi_pick_strategy import MultiPickStrategy

        # Two columns: (a_low, a_high) and (b_low, b_high)
        objs = [
            MockPickObj("a_low", position=np.array([0.0, 0.0, 0.0])),
            MockPickObj("a_high", position=np.array([0.0, 0.0, 0.1])),
            MockPickObj("b_low", position=np.array([1.0, 0.0, 0.0])),
            MockPickObj("b_high", position=np.array([1.0, 0.0, 0.1])),
        ]
        targets = [MockPickObj(f"t_{i}") for i in range(4)]
        smap = {"a_low": ["a_high"], "b_low": ["b_high"]}

        strategy = MultiPickStrategy(
            pick_objs=objs, target_objs=targets, stacking_map=smap,
        )
        strategy.initialize_pairings()

        # After reorder, top items (depth 0) come first, then bottom (depth 1)
        order = strategy.picking_order_item_names
        top_names = {"a_high", "b_high"}
        bottom_names = {"a_low", "b_low"}
        assert set(order[:2]) == top_names
        assert set(order[2:]) == bottom_names

        # Pick first top item
        first = strategy.get_current_pick_name()
        assert first in top_names
        strategy.mark_pick_complete(first)

        # Advance — should get second top item (not a bottom item)
        second = strategy.advance_pick_index()
        assert second in top_names
        assert second != first

    def test_stacking_availability_after_completion(self):
        """Blocked item becomes available after item above is completed."""
        from multi_pick_strategy import MultiPickStrategy

        objs = [
            MockPickObj("bottom", position=np.array([0.0, 0.0, 0.0])),
            MockPickObj("top", position=np.array([0.0, 0.0, 0.1])),
        ]
        targets = [MockPickObj("t_0"), MockPickObj("t_1")]
        smap = {"bottom": ["top"]}

        strategy = MultiPickStrategy(
            pick_objs=objs, target_objs=targets, stacking_map=smap,
        )
        strategy.initialize_pairings()

        # First pick should be top (only available one)
        assert strategy.get_current_pick_name() == "top"

        # bottom is blocked
        assert not strategy._is_pick_available("bottom")

        # Complete top
        strategy.mark_pick_complete("top")

        # Now bottom should be available
        assert strategy._is_pick_available("bottom")

        # Advance to bottom
        next_name = strategy.advance_pick_index()
        assert next_name == "bottom"

    def test_stacking_reassigns_targets_to_top_items(self):
        """With fewer targets than picks, top items (picked first) get targets."""
        from multi_pick_strategy import MultiPickStrategy

        # 2 columns, 2 layers: 4 items, only 2 targets
        objs = [
            MockPickObj("a_low", position=np.array([0.0, 0.0, 0.0])),
            MockPickObj("a_high", position=np.array([0.0, 0.0, 0.1])),
            MockPickObj("b_low", position=np.array([1.0, 0.0, 0.0])),
            MockPickObj("b_high", position=np.array([1.0, 0.0, 0.1])),
        ]
        targets = [MockPickObj("t_0"), MockPickObj("t_1")]
        smap = {"a_low": ["a_high"], "b_low": ["b_high"]}

        strategy = MultiPickStrategy(
            pick_objs=objs, target_objs=targets, stacking_map=smap,
        )
        strategy.initialize_pairings()

        by_name = strategy.pairings_by_pick_name
        # Top items (picked first due to stacking order) should have targets
        assert by_name["a_high"] is not None
        assert by_name["b_high"] is not None
        # Bottom items should have None targets
        assert by_name["a_low"] is None
        assert by_name["b_low"] is None

    def test_stacking_rescan_finds_previously_blocked(self):
        """After completing top items, bottom items are found via wrap-around rescan."""
        from multi_pick_strategy import MultiPickStrategy

        # 2 columns, 2 layers: 4 items, only 2 targets
        objs = [
            MockPickObj("a_low", position=np.array([0.0, 0.0, 0.0])),
            MockPickObj("a_high", position=np.array([0.0, 0.0, 0.1])),
            MockPickObj("b_low", position=np.array([1.0, 0.0, 0.0])),
            MockPickObj("b_high", position=np.array([1.0, 0.0, 0.1])),
        ]
        targets = [MockPickObj("t_0"), MockPickObj("t_1")]
        smap = {"a_low": ["a_high"], "b_low": ["b_high"]}

        strategy = MultiPickStrategy(
            pick_objs=objs, target_objs=targets, stacking_map=smap,
        )
        strategy.initialize_pairings()

        # Pick and complete both top items
        first = strategy.get_current_pick_name()
        assert first in ("a_high", "b_high")
        strategy.mark_pick_complete(first)
        second = strategy.advance_pick_index()
        assert second in ("a_high", "b_high")
        assert second != first
        strategy.mark_pick_complete(second)

        # Now advance — should wrap around and find nothing (bottom items have no targets)
        third = strategy.advance_pick_index()
        # Bottom items have no targets, so should return None
        assert third is None

    def test_stacking_fewer_targets_completes_correct_count(self):
        """With 4 items in 2x2 stacking and 2 targets, exactly 2 picks complete."""
        from multi_pick_strategy import MultiPickStrategy

        objs = [
            MockPickObj("a_low", position=np.array([0.0, 0.0, 0.0])),
            MockPickObj("a_high", position=np.array([0.0, 0.0, 0.1])),
            MockPickObj("b_low", position=np.array([1.0, 0.0, 0.0])),
            MockPickObj("b_high", position=np.array([1.0, 0.0, 0.1])),
        ]
        targets = [MockPickObj("t_0"), MockPickObj("t_1")]
        smap = {"a_low": ["a_high"], "b_low": ["b_high"]}

        strategy = MultiPickStrategy(
            pick_objs=objs, target_objs=targets, stacking_map=smap,
        )
        strategy.initialize_pairings()

        completed = []
        for _ in range(10):  # safety limit
            name = strategy.get_current_pick_name()
            if name is None:
                break
            strategy.mark_pick_complete(name)
            completed.append(name)
            next_name = strategy.advance_pick_index()
            if next_name is None:
                break

        assert len(completed) == 2
        assert set(completed) == {"a_high", "b_high"}

    def test_stacking_no_map_unchanged(self):
        """Default behavior unchanged when no stacking_map provided."""
        from multi_pick_strategy import MultiPickStrategy

        objs = [MockPickObj(f"p_{i}") for i in range(3)]
        targets = [MockPickObj(f"t_{i}") for i in range(3)]

        strategy = MultiPickStrategy(pick_objs=objs, target_objs=targets)
        strategy.initialize_pairings()

        # Original order preserved
        assert strategy.picking_order_item_names == ["p_0", "p_1", "p_2"]
        assert strategy.get_current_pick_name() == "p_0"
        assert strategy.advance_pick_index() == "p_1"
        assert strategy.advance_pick_index() == "p_2"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
