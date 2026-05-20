import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTaskLayeredCubes(UR10MultiPickPlaceTask):
    """Pick cubes from a 2x3 grid stacked 3 layers high (18 cubes total).

    - Sources (pick bin): LayeredPositionGenerator wrapping a 2x3 grid, 3 layers
      at layer_height=0.0515. Colors cycle red(6), green(6), blue(6) bottom-up.
    - Targets (dropzone): Flat 6x3 grid of 18 hidden markers.
    - Strategy: MultiPickStrategy with stacking constraints (top-down pick ordering).
    """

    DEFAULT_TASK_NAME = "table_task_layered_cubes"

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
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
            LayeredPositionGenerator,
            SequentialChoice,
        )
        from table_setup import (
            BIN_X_COORD,
            BIN_Y_COORD,
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            setup_two_tables,
            ITEM_SPAWN_REFERENCE_Z,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        # Default pick_count and target_count to 18 (full capacity) unless
        # overridden by caller (e.g. via --pick-count or --target-count CLI args).
        # Counts are independent — if picks > targets, excess picks are skipped
        # when no target is available; the task finishes when pairings run out.
        if "pick_count" not in kwargs or kwargs["pick_count"] is None:
            kwargs["pick_count"] = 18
        if "target_count" not in kwargs or kwargs["target_count"] is None:
            kwargs["target_count"] = 18

        stage_units = get_stage_units()
        cube_size = 0.0515 / stage_units
        expected_scale = np.array([cube_size, cube_size, cube_size])

        # === PICK STRATEGY ===
        # 2x3 grid in bin, stacked 3 layers high
        pick_z = ITEM_SPAWN_REFERENCE_Z + cube_size / 2 + 0.025
        base_pick_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=3, cols=2,
            spacing_x=0.08, spacing_y=0.08,
            randomize=False,
        )
        pick_pos_gen = LayeredPositionGenerator(
            base_generator=base_pick_gen,
            num_layers=3,
            layer_height=cube_size,
        )

        # Colors: 6 red (layer 0), 6 green (layer 1), 6 blue (layer 2)
        layer_colors = (["red"] * 6) + (["green"] * 6) + (["blue"] * 6)

        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("cube"),
            scale_strategy=FixedValue(expected_scale),
            color_strategy=SequentialChoice(layer_colors, loop=False),
        )

        # === TARGET STRATEGY ===
        # Flat 6x3 grid of 18 hidden markers on the dropzone
        marker_scale = np.array([cube_size, cube_size, 0.001 / stage_units])
        marker_z = DROPZONE_Z + 0.001 + marker_scale[2] / 2
        target_pos_gen = GridPositionGenerator(
            center=np.array(DROPZONE_CENTER_POINT) + np.array([0, 0, marker_z]),
            rows=6, cols=3,
            spacing_x=-0.10, spacing_y=0.10,
            randomize=False,
        )
        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("marker"),
            color_strategy=FixedValue("white"),
            scale_strategy=FixedValue(marker_scale),
            hidden_strategy=FixedValue(True),
        )

        def _strategy_factory(picks, targets):
            from multi_pick_strategy import MultiPickStrategy, compute_stacking_map
            stacking_map = compute_stacking_map(picks)
            return MultiPickStrategy(
                pick_objs=picks, target_objs=targets, stacking_map=stacking_map,
            )

        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick cubes from a 2x3 grid stacked 3 layers high (18 total) and place them onto markers in the dropzone.",
            pick_generation_strategy=pick_strategy,
            pick_count=kwargs.pop("pick_count", 18),
            target_count=kwargs.pop("target_count", 18),
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            stacking_enabled=True,
            scenario={
                "source": "bin",
                "destination": "dropzone_grid",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cube"],
                "count": 18,
                "arrangement": "2x3 grid in bin, stacked 3 layers high (layer_height=0.0515m)",
                "colors": "SequentialChoice: 6 red (layer 0), 6 green (layer 1), 6 blue (layer 2)",
                "orientation": "default",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "6x3 flat grid on dropzone",
                "count": 18,
                "virtual": True,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                create_strategy=_strategy_factory,
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                    "details": "stacking_map computed from pick positions to enforce top-down pick ordering",
                },
            ),
            rationale={
                "create_strategy": "Source cubes are stacked — MultiPickStrategy with stacking_map ensures upper layers are picked before lower layers",
                "stacking_enabled": "Cubes are stacked 3 layers high at source — stacking constraints enforce top-down pick order",
                "virtual_target_generation_strategy": "Hidden markers generated at pairing time; count may be reduced by CLI overrides",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
