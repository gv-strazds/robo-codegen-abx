import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTaskLayeredCircle(UR10MultiPickPlaceTask):
    """Pick sugar boxes from a circle stacked 2 layers high and stack them in the bin.

    - Sources (pick area): LayeredPositionGenerator wrapping a CircularPositionGenerator
      with 5 slots, 2 layers at layer_height=0.045 (horizontal sugar box height).
      Sugar boxes lie flat in native orientation (no rotation).
    - Targets (bin): Single hidden marker at the bin center (base of destination stack).
    - Strategy: SingleStackStrategy with source stacking constraints (top-down pick
      ordering) and destination stacking (single growing stack in the bin).
    """

    DEFAULT_TASK_NAME = "table_task_layered_circle"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import (
            CircularPositionGenerator,
            FixedValue,
            ItemGenerator,
            ItemSpec,
            LayeredPositionGenerator,
        )
        from table_setup import (
            BIN_SIZE,
            BIN_X_COORD,
            BIN_Y_COORD,
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            setup_two_tables,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        # Remove target_count from kwargs (always 1 for single stack destination)
        kwargs.pop("target_count", None)

        # Default pick_count to 10 (full capacity) unless overridden by caller
        pick_count = kwargs.pop("pick_count", None)
        if pick_count is None:
            pick_count = 8  # stacks of more than 8 are too unstable

        stage_units = get_stage_units()
        # Horizontal sugar box: native orientation (no rotation), Z height ≈ 0.045m
        horizontal_height = 0.045

        # === PICK STRATEGY ===
        # 5 sugar boxes in a circle, stacked 2 high, lying flat
        pick_z = DROPZONE_Z + 0.001 + horizontal_height / 2
        base_pick_gen = CircularPositionGenerator(
            center=np.array(DROPZONE_CENTER_POINT) + np.array([0, 0, pick_z]),
            radius=0.18,
            count=5,
            randomize=False,
        )
        pick_pos_gen = LayeredPositionGenerator(
            base_generator=base_pick_gen,
            num_layers=2,
            layer_height=horizontal_height,
        )

        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("sugar_box"),
            scale_strategy=FixedValue(np.array([1.0, 1.0, 1.0])),
        )

        # === TARGET STRATEGY ===
        # Single hidden marker in the bin (base of destination stack).
        # The KLT bin spawns at ITEM_SPAWN_REFERENCE_Z + 0.05 and settles to the
        # cart surface (~0.0573).  Use the settled floor Z for marker placement.
        bin_floor_z = 0.0573 + 0.005  # cart surface + small lift (approximate)
        marker_scale = np.array([0.05, 0.05, 0.001]) / stage_units
        target_pos = np.array([BIN_X_COORD, BIN_Y_COORD, bin_floor_z])

        target_items = [ItemSpec(
            asset_type="marker",
            position=target_pos,
            color="white",
            scale=marker_scale,
            hidden=True,
        )]

        class FixedListGenerator:
            def __init__(self, items):
                self.items = items
            def generate(self, count_range=(1, 1), seed=None):
                return self.items

        target_strategy = FixedListGenerator(target_items)

        # Bin geometry for spatial verification (strategy-level, not box_verification_info).
        # floor_z must be the settled value since the strategy uses it directly
        # without runtime settling adjustment.
        bin_geometry = {
            "name": "pick_bin",
            "center_xy": np.array([BIN_X_COORD, BIN_Y_COORD]),
            "inner_size": np.array(BIN_SIZE[:2]),
            "floor_z": bin_floor_z,
            "height": 0.15,  # generous wall height for containment check
            "z_tol": 0.03,   # generous Z tolerance for physics settling
        }


        def _strategy_factory(picks, targets):
            from multi_pick_strategy import SingleStackStrategy, compute_stacking_map
            stacking_map = compute_stacking_map(picks)
            return SingleStackStrategy(
                pick_objs=picks, target_objs=targets,
                stacking_map=stacking_map, bin_geometry=bin_geometry,
            )

        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick horizontal sugar boxes from a circle stacked 2 layers high (10 total) and stack them in the pick bin on the cart.",
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            target_count=1,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root, standard_objs=False, add_bin=True
            ),
            stacking_enabled=True,
            scenario={
                "source": "dropzone",
                "destination": "bin",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["sugar_box"],
                "count": 10,
                "arrangement": "circle (r=0.18m, 5 positions) on dropzone, stacked 2 layers high",
                "colors": "USD asset default",
                "orientation": "horizontal (native, no rotation)",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "single marker at bin center (base of destination stack)",
                "count": 1,
                "virtual": True,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                create_strategy=_strategy_factory,
                ee_height_for_move=0.27 / stage_units,
                strategy_description={
                    "class": "SingleStackStrategy",
                    "pairing": "stacking",
                    "details": "stacking_map computed from source positions for top-down pick ordering; single growing stack at destination",
                },
                # in teleport mode, the first item can get teleported *under* the bin before the bin
                # has had time to drop onto the cart surface. Add a small delay to allow settling.
                startup_delay_seconds=1.0,
            ),
            rationale={
                "create_strategy": "Items are stacked at source (2 layers) and must be collected into a single growing stack in the bin — SingleStackStrategy handles both source unstacking and destination stacking",
                "stacking_enabled": "Source items are stacked 2 layers high — stacking constraints enforce top-down pick order",
                "ee_height_for_move": "Transport height must clear bin walls (~0.20m) plus carried item rest height (~0.023m) plus margin",
                "virtual_target_generation_strategy": "Single destination stack — marker generated at pairing time since all items go to one location",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
