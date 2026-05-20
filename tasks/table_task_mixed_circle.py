import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from task_verification import is_vertical

# Asset types that have an elongated vertical axis and must remain upright for
# success (USD bottles/boxes/cans plus the cylinder/cone primitives, which the
# task's MixedScaleStrategy elongates along Z).
_VERTICAL_ASSET_TYPES = {
    "cracker_box", "sugar_box", "soup_can", "mustard_bottle",
    "cylinder", "cone",
}

logger = logging.getLogger(__name__)


class TableTaskMixedCircle(UR10MultiPickPlaceTask):
    """Pick mixed items (cube, cone, cylinder, bottle) from a circle on the conveyor (drop zone)
    and place them into the pick bin on the cart.
    """

    DEFAULT_TASK_NAME = "table_task_mixed_circle"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        # Lazily import Isaac utilities to avoid import-order issues
        from isaacsim.core.utils.stage import get_stage_units
        from asset_data_utils import lookup_prim_geometry
        from item_generation import (
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
            MixedOrientationStrategy,
            MixedScaleStrategy,
            PositionGenerator,
            RandomChoice,
            SequentialChoice,
        )
        from table_setup import (
            BIN_SIZE,
            BIN_X_COORD,
            BIN_Y_COORD,
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            setup_two_tables,
            ITEM_SPAWN_REFERENCE_Z,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        # --- Generation Strategies ---
        stage_units = get_stage_units()
        # Typical object scale is ~5cm
        expected_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units

        # CLI overrides for pick/target counts; default is 8 items for both.
        import random as py_random
        num_items = 8
        pick_count = kwargs.pop("pick_count", None) or num_items
        target_count = kwargs.pop("target_count", None) or num_items

        # Pre-sample types to ensure scale strategy is stable relative to type strategy
        # (RandomChoice with None seed is unstable between attribute calls)
        type_options = ["cube", "ball", "cone", "cylinder", "soup_can", "sugar_box", "mustard_bottle"]
        sampled_types = [py_random.choice(type_options) for _ in range(num_items)]

        asset_type_strategy = SequentialChoice(sampled_types, loop=True)
        orientation_strategy = MixedOrientationStrategy(sampled_types)
        scale_strategy = MixedScaleStrategy(sampled_types, expected_scale)

        # Per-item spawn Z: place each item with its bottom ~5 mm above the
        # dropzone surface, derived from the asset's rest_height under the
        # same scale and orientation the ItemGenerator will apply.  Items
        # then settle into contact under gravity rather than free-falling
        # from a uniform 15 cm hover.
        spawn_clearance = 0.005  # 5 mm gap to avoid surface intersection at spawn
        per_index_z = []
        for i in range(num_items):
            t = sampled_types[i]
            scale = scale_strategy.get_value(i, num_items)
            orient = orientation_strategy.get_value(i, num_items)
            geom = lookup_prim_geometry(t, obj_scale=scale, orientation=orient)
            rest_h = geom.rest_height if geom is not None else 0.05
            per_index_z.append(DROPZONE_Z + rest_h + spawn_clearance)

        class _CircularPerItemZGenerator(PositionGenerator):
            """Like CircularPositionGenerator(randomize=False) but with per-slot Z."""

            def __init__(self, center_xy, radius, count, z_per_index):
                self._cx = float(center_xy[0])
                self._cy = float(center_xy[1])
                self._radius = float(radius)
                self._count = int(count)
                self._z = z_per_index

            @property
            def capacity(self):
                return self._count

            def get_positions(self, count, seed=None):
                n = min(count, self._count)
                positions = []
                for i in range(n):
                    angle = 2 * np.pi * i / self._count
                    x = self._cx + self._radius * np.cos(angle)
                    y = self._cy + self._radius * np.sin(angle)
                    positions.append(np.array([x, y, self._z[i]]))
                return positions

        pick_pos_gen = _CircularPerItemZGenerator(
                center_xy=DROPZONE_CENTER_POINT[:2]+0.04,
            radius=0.22,
            count=num_items,
            z_per_index=per_index_z,
        )

        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=asset_type_strategy,
            orientation_strategy=orientation_strategy,
            scale_strategy=scale_strategy,
            color_strategy=RandomChoice(["red", "green", "blue", "yellow"]),
        )

        # Target Strategy: Grid in the Pick Bin on the cart
        marker_scale = np.array([0.0515, 0.0515, 0.001]) / stage_units
        target_z = 0.0573 + 0.005 + marker_scale[2] / 2

        rows = 4
        cols = 2
        spacing_x = 0.10
        spacing_y = 0.08

        target_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, target_z]),
            rows=rows,
            cols=cols,
            spacing_x=spacing_x,
            spacing_y=spacing_y,
            randomize=False
        )

        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("marker"),
            color_strategy=FixedValue("white"),
            scale_strategy=FixedValue(marker_scale),
            hidden_strategy=FixedValue(True),
        )

        # Bin geometry for verification via box_verification_info.
        # NOTE: BIN_SIZE and bin_floor_z are approximate (see table_setup.py).
        # Use a generous height to accommodate physics settling.
        # The bin prim spawns at ITEM_SPAWN_REFERENCE_Z + 0.05 and may settle
        # downward during physics; prim_path + spawn_position enable
        # automatic adjustment at verification time.
        bin_spawn_pos = np.array([BIN_X_COORD, BIN_Y_COORD, ITEM_SPAWN_REFERENCE_Z + 0.05])
        bin_floor_z = 0.0573 + 0.005  # cart surface + small lift (approximate)
        box_specs = [
            {
                "name": "pick_bin",
                "center_xy": np.array([BIN_X_COORD, BIN_Y_COORD]),
                "inner_size": np.array(BIN_SIZE[:2]),
                "floor_z": bin_floor_z,
                "height": 0.15,  # generous wall height for containment check
                "z_tol": 0.03,  # generous Z tolerance for balls settling below estimated floor
                "prim_path": "/KLT_Bin",
                "spawn_position": bin_spawn_pos,
            },
        ]

        # Build placement constraints closure that will use self._pick_objs
        # and self._get_bb_cache at verification time.
        # Only asset types with a clearly identifiable vertical axis need the
        # verticality check (USD bottles/boxes/cans plus the elongated
        # cylinder/cone primitives).
        task_ref = self

        def _check_verticality(pick_index, target_index):
            pick_obj = task_ref._pick_objs[pick_index]
            if any(pick_obj.name.startswith(t) for t in _VERTICAL_ASSET_TYPES):
                if not is_vertical(pick_obj, max_tilt_deg=15):
                    return (False, "item is not vertical (upright orientation required)")
            return (True, "")

        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick mixed items (cubes, cones, cylinders, bottles) arranged in a circle on the conveyor and place them into the bin on the cart.",
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            target_count=target_count,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            placement_constraints_fn=_check_verticality,
            box_verification_info={"box_specs": box_specs},
            containment_check=True,
            scenario={
                "source": "conveyor",
                "destination": "bin",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cube", "ball", "cone", "cylinder", "soup_can", "cracker_box", "sugar_box", "mustard_bottle"],
                "count": 8,
                "arrangement": "circle (r=0.18m, 8 positions) on conveyor/dropzone",
                "colors": "RandomChoice(['red', 'green', 'blue', 'yellow'])",
                "orientation": "mixed (MixedOrientationStrategy: upright for USD assets, default for primitives)",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "4x2 grid in bin on cart",
                "count": 8,
                "virtual": True,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                # Clear an upright cracker_box (0.213 m tall) on the dropzone
                # while carrying another cracker_box (rest_height 0.107 m):
                # 0.213 + 0.107 + 0.08 margin = 0.40 m.
                ee_height_for_move=0.40 / stage_units,
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                },
            ),
            verification_description={
                "spatial_check": "box_verification_info (bin containment via build_box_verification_hooks)",
                "placement_constraints": "is_vertical (for USD asset types only)",
                "containment_check": True,
            },
            rationale={
                "create_strategy": "Default sequential pairing — mixed items placed into bin slots without type or color matching",
                "box_verification_info": "Single bin container — uses centralized box containment verification (1 box target) instead of per-marker spatial check",
                "placement_constraints_fn": (
                    "Items with an elongated vertical axis (USD bottles/boxes/cans and "
                    "the elongated cylinder/cone primitives) must remain upright after "
                    "placement; isotropic primitives (cubes, balls) are exempt"
                ),
                "containment_check": "Items placed inside the bin require boundary verification to confirm correct placement",
                "virtual_target_generation_strategy": "Hidden markers in bin generated at pairing time to match actual pick count",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
