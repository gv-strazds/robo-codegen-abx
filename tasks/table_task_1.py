import logging
import random
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from multi_pick_strategy import ColorMatchStrategy

logger = logging.getLogger(__name__)


class TableTask1(UR10MultiPickPlaceTask):
    """Pick 16 colored cubes from the bin and arrange them in a 4x4 grid on the conveyor with horizontal color stripes."""

    DEFAULT_TASK_NAME = "table_task_1"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        from isaacsim.core.utils.stage import get_stage_units
        from task_spec import TaskImplementationSpec, TaskSpec
        from item_generation import (
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
            ItemSpec,
            SequentialChoice,
            resolve_count,
        )
        from table_setup import (
            BIN_X_COORD,
            BIN_Y_COORD,
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            setup_two_tables,
            ITEM_SPAWN_REFERENCE_Z,
        )

        stage_units = get_stage_units()

        # Cube dimensions
        cube_size = 0.0515 / stage_units
        expected_scale = np.array([cube_size, cube_size, cube_size])

        color_palette = ["red", "green", "blue", "orange"]

        # === PICK STRATEGY ===
        # 4x4 grid in bin, with colors randomly shuffled
        pick_z = ITEM_SPAWN_REFERENCE_Z + cube_size / 2 + 0.02

        class ShuffledColorPickGenerator:
            """Generate 16 cubes with 4 of each color, randomly shuffled."""
            def __init__(self, center, scale, colors):
                self.center = center
                self.scale = scale
                self.colors = colors
                # Pack cubes tightly in the bin so edge cubes don't get displaced
                bin_spacing = cube_size + 0.005
                self.pos_gen = GridPositionGenerator(
                    center=center, rows=4, cols=4,
                    spacing_x=bin_spacing, spacing_y=bin_spacing, randomize=False,
                )

            def generate(self, count_range=None, seed=None):
                rng = random.Random(seed)
                # 4 of each color, shuffled
                color_list = self.colors * 4
                rng.shuffle(color_list)
                positions = self.pos_gen.get_positions(16, seed)
                count = resolve_count(count_range, capacity=16, seed=seed)
                items = []
                for i in range(min(count or 16, 16)):
                    items.append(ItemSpec(
                        asset_type="cube",
                        position=positions[i],
                        scale=self.scale,
                        color=color_list[i],
                    ))
                return items

        pick_strategy = ShuffledColorPickGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            scale=expected_scale,
            colors=color_palette,
        )

        # === TARGET STRATEGY ===
        # 4x4 tightly packed grid of hidden markers on the dropzone
        # Spacing = cube_size + small gap for tight packing
        spacing = cube_size + 0.0025 / stage_units
        marker_scale = np.array([cube_size, cube_size, 0.001 / stage_units])
        marker_z = DROPZONE_Z + 0.001 + marker_scale[2] / 2

        target_center = np.array(DROPZONE_CENTER_POINT) + np.array([0, 0, marker_z])

        # Each row of 4 markers gets one color: red, green, blue, orange
        # GridPositionGenerator iterates row-major: (r0,c0),(r0,c1),...,(r1,c0),...
        # So 4 consecutive targets share the same row (same color)
        target_colors = []
        for color in color_palette:
            target_colors.extend([color] * 4)

        target_pos_gen = GridPositionGenerator(
            center=target_center,
            rows=4, cols=4,
            spacing_x=spacing, spacing_y=spacing,
            randomize=False,
        )

        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("marker"),
            scale_strategy=FixedValue(marker_scale),
            color_strategy=SequentialChoice(target_colors, loop=False),
            hidden_strategy=FixedValue(True),
        )

        # === BUILD TASK SPEC ===
        pick_count = kwargs.pop("pick_count", 16)
        target_count = kwargs.pop("target_count", 16)

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick 16 colored cubes (4 each of red, green, blue, orange) from the bin "
                "and arrange them in a tightly packed 4x4 grid on the conveyor with horizontal color stripes."
            ),
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            target_count=target_count,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                create_strategy=lambda picks, targets: ColorMatchStrategy(
                    picks, targets, color_palette=color_palette,
                ),
                strategy_description={
                    "class": "ColorMatchStrategy",
                    "pairing": "color_match",
                    "details": "color_palette=['red', 'green', 'blue', 'orange']; cubes routed to same-color row",
                },
            ),
            scenario={
                "source": "bin",
                "destination": "dropzone_grid",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cube"],
                "count": 16,
                "arrangement": "4x4 grid in pick bin, randomly shuffled colors",
                "colors": "4 each of red, green, blue, orange (shuffled)",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "4x4 tightly packed grid on dropzone",
                "count": 16,
                "virtual": True,
            },
            rationale={
                "create_strategy": "Cubes must be sorted by color into specific rows — ColorMatchStrategy pairs each cube to a target with the matching color label",
                "virtual_target_generation_strategy": "Hidden markers used as placement targets; never spawned as USD prims to avoid scene clutter",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
