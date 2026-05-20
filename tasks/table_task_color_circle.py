import logging
import random as py_random
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTaskColorCircle(UR10MultiPickPlaceTask):
    """Pick randomly colored cubes from the bin and place them in a circle on the drop zone."""

    DEFAULT_TASK_NAME = "table_task_color_circle"

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
        from item_generation import (
            CircularPositionGenerator,
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
            RandomChoice,
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

        # --- Generation Strategies ---
        stage_units = get_stage_units()
        expected_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units

        # Default: random count (2-16) for both picks and targets.
        # CLI overrides (--pick-count, --target-count, --target-count-max) are
        # respected when provided.
        rng = py_random.Random(kwargs.get("seed"))
        num_items = rng.randint(2, 16)
        pick_count = kwargs.pop("pick_count", None) or num_items
        target_count = kwargs.pop("target_count", None) or num_items

        # === PICK STRATEGY ===
        # 4x4 grid in the pick bin (capacity=16), randomize slot selection
        pick_z = ITEM_SPAWN_REFERENCE_Z + expected_scale[2] / 2 + 0.02
        pick_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=4,
            cols=4,
            spacing_x=0.08,
            spacing_y=0.08,
            randomize=True,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("cube"),
            scale_strategy=FixedValue(expected_scale),
            color_strategy=RandomChoice(["red", "green", "blue", "magenta"]),
        )

        # === TARGET STRATEGY ===
        # Circle of hidden markers on the drop zone
        marker_scale = np.array([0.0515, 0.0515, 0.001]) / stage_units
        marker_z = DROPZONE_Z + 0.001 + marker_scale[2] / 2
        target_pos_gen = CircularPositionGenerator(
            center=np.array(DROPZONE_CENTER_POINT) + np.array([0, 0, marker_z]),
            radius=0.15,
            count=16,
            randomize=False,
        )
        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("marker"),
            color_strategy=FixedValue("white"),
            scale_strategy=FixedValue(marker_scale),
            hidden_strategy=FixedValue(True),
        )

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick randomly colored cubes from the bin and place them in a circle "
                "on the drop zone."
            ),
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            target_count=target_count,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            scenario={
                "source": "bin",
                "destination": "dropzone_circle",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cube"],
                "count": "random (2-16)",
                "arrangement": "4x4 grid in pick bin (randomize=True for slot selection)",
                "colors": "RandomChoice(['red', 'green', 'blue', 'magenta'])",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "circle (r=0.15m, 16 positions) on dropzone",
                "count": "matches pick count",
                "virtual": True,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                },
            ),
            rationale={
                "create_strategy": "Default sequential pairing — cubes placed on circle markers without matching",
                "virtual_target_generation_strategy": "Hidden markers generated at pairing time to match the actual (randomized) pick count",
                "pick_count": "Randomized count (2-16) tests the system with varying workloads each run",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
