import logging
import random
from typing import Optional

import numpy as np
from asset_utils import add_asset

from multi_pick_strategy import BottleGridColumnFillStrategy
from multi_pickplace_task import UR10MultiPickPlaceTask
from task_verification import is_vertical, is_within

logger = logging.getLogger(__name__)


class TableTaskBottles1(UR10MultiPickPlaceTask):
    """Pick bottles from the pick bin and place into pads.

    Similar to TableTask3, but source objects are Madara bottles added
    via ``asset_utils.add_asset`` (USD path) rather than primitive shapes.
    """

    DEFAULT_TASK_NAME = "table_task_bottles_1"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        from isaacsim.core.utils import rotations

        # Lazily import Isaac utilities to avoid import-order issues
        from isaacsim.core.utils.stage import get_stage_units
        from isaacsim.cortex.framework.cortex_utils import get_assets_root_path_or_die
        from item_generation import FixedValue, GridPositionGenerator, ItemGenerator
        from pxr import Gf
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

        seed = kwargs.pop("seed", None)
        rng = random.Random(seed)

        # Default object size used for controller logic and targets
        stage_units = get_stage_units()
        expected_scale = np.array([1.0, 1.0, 1.0]) / stage_units

        # --- Strategies ---
        # Pick Strategy: 4x2 Grid (rows=2, cols=4). Bottle V3.
        # Orientation: -90 deg X.
        default_orientation = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        )

        pick_z = ITEM_SPAWN_REFERENCE_Z + (0.03 / stage_units) / 2 + 0.025

        pick_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=2,
            cols=4,
            spacing_x=0.08,
            spacing_y=0.15,  # Increased further to avoid collision
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("madara_bottle"),
            orientation_strategy=FixedValue(default_orientation),
            scale_strategy=FixedValue(np.array([1.0, 1.0, 1.0])),
            color_strategy=None,
        )

        # Target Strategy: Nx4 Grid (N cols random 1-3). Madara Pad.
        TARGET_HEIGHT = 0.002
        dx = -0.15
        dy = 0.15
        start_grid_x = DROPZONE_X
        start_grid_y = DROPZONE_Y
        grid_w = rng.randint(1, 3)
        grid_l = 4
        center_grid_x = start_grid_x + (grid_w - 1) * dx / 2
        center_grid_y = start_grid_y + (grid_l - 1) * dy / 2
        center_grid_z = DROPZONE_Z + 0.001 + TARGET_HEIGHT / 2

        target_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, center_grid_z]),
            rows=grid_l,
            cols=grid_w,
            spacing_x=dx,
            spacing_y=dy,
        )
        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("madara_pad"),
            scale_strategy=FixedValue(None),  # Default
        )

        def _bottle_spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None):
            return (
                is_within(pick_obj, target_obj, bb_cache, obj_scale)
                and is_vertical(pick_obj, obj_scale=obj_scale, max_tilt_deg=15)
            )

        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick bottles from the bin and place them into carrier pads in an Nx4 grid in the dropzone (N columns, 1-3).",
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            spatial_check_fn=_bottle_spatial_check,
            scenario={"source": "bin", "destination": "dropzone_grid", "workspace": "two_tables"},
            pick_description={
                "asset_types": ["madara_bottle"],
                "count": 8,
                "arrangement": "4x2 grid in pick bin",
                "colors": "USD asset default",
                "orientation": "upright (-90° X)",
            },
            target_description={
                "type": "carrier_pads",
                "arrangement": f"{grid_w}x4 grid on dropzone (cols random 1-3)",
                "count": grid_w * grid_l,
            },
            implementation=TaskImplementationSpec(
                create_strategy=lambda picks, targets: BottleGridColumnFillStrategy(
                    pick_objs=picks, target_objs=targets
                ),
                strategy_description={
                    "class": "BottleGridColumnFillStrategy",
                    "pairing": "column-fill (highest-x first, y ascending within column)",
                    "details": (
                        "Fills the target grid one column at a time, starting from "
                        "the highest-x column, to avoid the arm reaching over "
                        "previously placed bottles. Inherits BottlePickStrategy "
                        "drop orientation (pi/2 X) and EE offset from pick geometry."
                    ),
                },
            ),
            verification_description={
                "spatial_check": "is_within + is_vertical",
            },
            rationale={
                "create_strategy": (
                    "Bottles require specialized gripper orientation during drop "
                    "and custom EE offset from bottle geometry; column-fill "
                    "pairing avoids the arm reaching over already-placed bottles "
                    "at the same y but lower x."
                ),
                "spatial_check_fn": "Bottles must be within their carrier pad (is_within) and remain upright (is_vertical) after placement",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
