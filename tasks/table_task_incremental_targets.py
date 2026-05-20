import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTaskIncrementalTargets(UR10MultiPickPlaceTask):
    """Place cubes on disc targets, with both sides streaming in over time.

    Pick cubes (4x3, 12 total) are added to the bin incrementally at a
    1.0 s interval; disc targets (3x3, 9 total) appear on the dropzone at
    a 1.5 s interval.  Exercises both incremental schedulers running
    concurrently — ``SelectNextPick`` idles when the pick queue is momentarily
    empty, and ``CheckTargetAvailable`` idles when the next target has not
    yet spawned.
    """

    DEFAULT_TASK_NAME = "table_task_incremental_targets"

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
            IncrementalGenerationConfig,
            ItemGenerator,
            SequentialChoice,
        )
        from table_setup import (
            BIN_X_COORD,
            BIN_Y_COORD,
            DROPZONE_X,
            DROPZONE_Y,
            DROPZONE_Z,
            setup_two_tables,
            ITEM_SPAWN_REFERENCE_Z,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()
        expected_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units

        # Picks: 4x3 grid of cubes in the bin (12 total), streamed in.
        pick_z = ITEM_SPAWN_REFERENCE_Z + expected_scale[2] / 2 + 0.025
        pick_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=4,
            cols=3,
            spacing_x=0.08,
            spacing_y=0.08,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("cube"),
            scale_strategy=FixedValue(expected_scale),
            color_strategy=SequentialChoice(
                ["red", "green", "blue", "yellow", "cyan", "magenta"], loop=True,
            ),
        )

        # Targets: 3x3 grid of discs on the dropzone, streamed in.
        dx = -0.15
        dy = 0.15
        grid_w = 3
        grid_l = 3
        center_grid_x = DROPZONE_X + (grid_w - 1) * dx / 2
        center_grid_y = DROPZONE_Y + (grid_l - 1) * dy / 2
        center_grid_z = DROPZONE_Z + 0.001 + expected_scale[2] / 2

        target_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, center_grid_z]),
            rows=grid_l,
            cols=grid_w,
            spacing_x=dx,
            spacing_y=dy,
        )
        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("disc"),
            color_strategy=FixedValue("purple"),
            scale_strategy=FixedValue(expected_scale),
        )

        # Stream picks every 1.0 s and targets every 1.5 s; BT starts
        # immediately so the robot can begin picking while later picks
        # and targets are still arriving.  CLI flags
        # ``--dynamic-pick-interval`` / ``--dynamic-target-interval`` can
        # override these batch intervals at runtime.
        pick_inc_config = IncrementalGenerationConfig(
            items_per_batch=1,
            batch_interval=1.0,
            bt_start_threshold=1,
        )
        target_inc_config = IncrementalGenerationConfig(
            items_per_batch=1,
            batch_interval=1.5,
            bt_start_threshold=0,
        )

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Place 12 cubes from the bin onto 9 disc targets; both sides"
                " are streamed in — picks arrive in the bin every 1.0 s, and"
                " targets appear on the dropzone every 1.5 s.  The BT must"
                " idle waiting for either side when the other has not yet"
                " spawned the next item."
            ),
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            pick_count=12,
            target_count=9,
            pick_incremental_config=pick_inc_config,
            target_incremental_config=target_inc_config,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            scenario={
                "source": "bin",
                "destination": "dropzone_grid",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cube"],
                "count": 12,
                "arrangement": "4x3 grid in pick bin",
                "colors": "SequentialChoice(['red','green','blue','yellow','cyan','magenta'])",
                "incremental": "1 pick every 1.0 s; BT starts after first arrives",
            },
            target_description={
                "type": "visible_discs",
                "arrangement": "3x3 grid on dropzone",
                "count": 9,
                "incremental": "1 target every 1.5 s; BT starts immediately",
            },
            implementation=TaskImplementationSpec(
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                },
            ),
            verification_description={"spatial_check": "is_on_top (default)"},
            rationale={
                "pick_incremental_config": (
                    "Exercises concurrent pick + target streaming — picks"
                    " arrive slightly faster than targets so the bottleneck"
                    " alternates between the two queues."
                ),
                "target_incremental_config": (
                    "Demonstrates dynamic target spawning — placement slots"
                    " appear on the dropzone while the robot works through"
                    " an also-growing pick queue."
                ),
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
