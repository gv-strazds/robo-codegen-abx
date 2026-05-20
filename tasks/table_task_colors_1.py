import logging
from typing import Optional

import numpy as np
from multi_pick_strategy import ColorMatchStrategy

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTaskColors1(UR10MultiPickPlaceTask):
    """Pick colored cubes and place onto markers that match in color.

    - Sources (pick bin): 4x3 grid of cubes colored red/green/blue.
    - Targets (dropzone): 3x4 grid of thin rectangles colored red/yellow/green/blue.
    """

    DEFAULT_TASK_NAME = "table_task_colors_1"

    #   Success criterion (color-matching with availability):
    #      - For each pick object, either
    #         (a) it is currently located on top of a target object whose color matches the pick's color, or
    #         (b) there is no available target of that color. A target is available if no other pick object is
    #            currently placed on top of it.
    #      Returns True if all picks satisfy the criterion; otherwise logs a warning and returns False.

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
        from isaacsim.cortex.framework.cortex_utils import get_assets_root_path_or_die
        from item_generation import (
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
            RandomChoice,
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

        # --- Strategies ---
        stage_units = get_stage_units()
        expected_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units

        # Pick Strategy: 4x3 Grid in Bin (cols=4, rows=3)
        # Colors: Red/Green/Blue (Random)
        pick_z = ITEM_SPAWN_REFERENCE_Z + expected_scale[2] / 2 + 0.025
        pick_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=3,
            cols=4,
            spacing_x=0.08,  # Adjust as needed to fit
            spacing_y=0.08,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("cube"),
            scale_strategy=FixedValue(expected_scale),
            color_strategy=RandomChoice(["red", "green", "blue"]),
        )

        # Target Strategy: 3x4 Grid in Dropzone
        target_colors = ["red", "cyan", "yellow", "green", "blue", "magenta"]
        RECT_HEIGHT = 0.002

        dx = -0.15
        dy = 0.15
        start_grid_x = DROPZONE_X
        start_grid_y = DROPZONE_Y
        grid_w = 3
        grid_l = 4
        center_grid_x = start_grid_x + (grid_w - 1) * dx / 2
        center_grid_y = start_grid_y + (grid_l - 1) * dy / 2
        center_grid_z = DROPZONE_Z + 0.001 + RECT_HEIGHT / 2

        target_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, center_grid_z]),
            rows=grid_l,
            cols=grid_w,
            spacing_x=dx,
            spacing_y=dy,
        )

        target_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units

        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("cube"),
            color_strategy=SequentialChoice(
                target_colors, loop=True
            ),  # Repeating to fill grid
            scale_strategy=FixedValue(target_scale),
        )

        # Color palette used for matching — includes all recognizable pick colors
        color_palette = ["red", "green", "blue", "yellow"]

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick colored cubes from the bin (4x3 grid of red/green/blue) and place them onto"
                " matching colored markers (3x4 grid including cyan/magenta) in the dropzone."
            ),
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
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
                "colors": "RandomChoice(['red', 'green', 'blue'])",
            },
            target_description={
                "type": "visible_markers",
                "arrangement": "3x4 grid on dropzone",
                "count": 12,
            },
            implementation=TaskImplementationSpec(
                create_strategy=lambda picks, targets: ColorMatchStrategy(
                    picks, targets, color_palette=color_palette
                ),
                strategy_description={
                    "class": "ColorMatchStrategy",
                    "pairing": "color_match",
                    "details": "color_palette=['red', 'green', 'blue', 'yellow']; target colors cycle through red/cyan/yellow/green/blue/magenta — only matching colors receive picks",
                },
            ),
            rationale={
                "create_strategy": "Cubes must be placed on same-color targets; ColorMatchStrategy pairs picks to targets by color and skips unmatched items",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
