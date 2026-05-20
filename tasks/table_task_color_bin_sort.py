import random
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from multi_pick_strategy import ColorMatchStrategy
from colors import color_to_rgb
from item_generation import ItemSpec, ConveyorPositionGenerator
import logging

logger = logging.getLogger(__name__)




class TableTaskColorBinSort(UR10MultiPickPlaceTask):
    """Sort randomly colored cubes and balls into the matching red, green, and blue collection boxes."""

    DEFAULT_TASK_NAME = "table_task_color_bin_sort"

    MIN_OBJECTS_PER_SHAPE = 1
    MAX_OBJECTS_PER_SHAPE = 5

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        from isaacsim.core.utils.stage import get_stage_units
        from table_setup import (
            setup_two_tables,
            spawn_open_box,
            TABLETOP_CENTER_POINT,
            ITEM_SPAWN_REFERENCE_Z,
            DROPZONE_CENTER_POINT,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        table_center = TABLETOP_CENTER_POINT
        dropzone_center = DROPZONE_CENTER_POINT
        color_palette = ["red", "green", "blue"]

        # Primitive scales
        cube_scale = np.array([0.045, 0.045, 0.045]) / stage_units
        ball_radius = 0.025 / stage_units
        ball_scale = np.array([ball_radius, ball_radius, ball_radius])

        # Box dimensions
        box_inner_size = np.array([0.20, 0.16]) / stage_units
        box_wall = 0.012 / stage_units
        box_height = 0.08 / stage_units
        box_base_thickness = 0.06244 / stage_units
        box_base_center_z = 0.06244 / stage_units
        box_floor_z = box_base_center_z + box_base_thickness / 2

        marker_scale = np.array([0.04, 0.04, 0.002]) / stage_units

        # Build box specs
        box_offsets = {
            "red": np.array([0.25, -0.08, 0.0]),
            "green": np.array([0.02, 0.16, 0.0]),
            "blue": np.array([-0.20, -0.05, 0.0]),
        }
        box_specs = []
        for color, boff in box_offsets.items():
            offset_scaled = boff / stage_units
            center = table_center + offset_scaled
            box_specs.append({
                "name": f"{color}_collection_box",
                "color": color,
                "center_xy": np.array([center[0], center[1]]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "match_labels": {"color": color},
            })

        # Build marker slots
        offset_templates = [
            np.array([-0.06, -0.045]),
            np.array([0.06, -0.045]),
            np.array([-0.06, 0.045]),
            np.array([0.06, 0.045]),
            np.array([0.0, -0.07]),
            np.array([0.0, 0.07]),
        ]
        offset_templates = [vec / stage_units for vec in offset_templates]
        lift = marker_scale[2] / 2
        marker_slots = []
        for bspec in box_specs:
            for moff in offset_templates:
                marker_slots.append({
                    "position": np.array([
                        bspec["center_xy"][0] + moff[0],
                        bspec["center_xy"][1] + moff[1],
                        bspec["floor_z"] + lift,
                    ]),
                    "color": bspec["color"],
                })

        # --- Strategies ---
        class ColorBinSortGenerator:
            def __init__(self, position_generator, min_obj, max_obj, colors, cube_scale, ball_scale):
                self.position_generator = position_generator
                self.min_obj = min_obj
                self.max_obj = max_obj
                self.colors = colors
                self.cube_scale = cube_scale
                self.ball_scale = ball_scale

            def generate(self, count_range=(1,1), seed=None):
                rng = random.Random(seed)
                num_cubes = rng.randint(self.min_obj, self.max_obj)
                num_balls = rng.randint(self.min_obj, self.max_obj)
                total = num_cubes + num_balls

                entries = []
                for _ in range(num_cubes):
                    entries.append({"type": "cube", "color": rng.choice(self.colors)})
                for _ in range(num_balls):
                    entries.append({"type": "ball", "color": rng.choice(self.colors)})
                rng.shuffle(entries)

                positions = self.position_generator.get_positions(total, seed)

                items = []
                for i, entry in enumerate(entries):
                    if i >= len(positions): break
                    scale = self.cube_scale if entry["type"] == "cube" else self.ball_scale
                    items.append(ItemSpec(
                        asset_type=entry["type"],
                        position=positions[i],
                        color=entry["color"],
                        scale=scale
                    ))
                return items

        spacing = 0.12 / stage_units
        x_pos = dropzone_center[0]
        pick_z = dropzone_center[2] + 0.035 / stage_units
        jitter = 0.01 / stage_units

        pick_pos_gen = ConveyorPositionGenerator(
            center_x=x_pos,
            center_y=dropzone_center[1],
            z=pick_z,
            spacing=spacing,
            jitter_x=jitter,
            jitter_y=jitter
        )

        pick_strategy = ColorBinSortGenerator(
            position_generator=pick_pos_gen,
            min_obj=self.MIN_OBJECTS_PER_SHAPE,
            max_obj=self.MAX_OBJECTS_PER_SHAPE,
            colors=color_palette,
            cube_scale=cube_scale,
            ball_scale=ball_scale
        )

        target_positions = [slot["position"] for slot in marker_slots]
        target_colors = [slot["color"] for slot in marker_slots]

        class FixedListGenerator:
            def __init__(self, items):
                self.items = items
            def generate(self, count_range=(1,1), seed=None):
                return self.items

        target_items = [
             ItemSpec(
                 asset_type="rect",
                 position=p,
                 color=c,
                 scale=marker_scale
             ) for p, c in zip(target_positions, target_colors)
        ]
        target_strategy = FixedListGenerator(target_items)

        def _workspace_setup(scene, assets_root):
            setup_two_tables(scene, assets_root, standard_objs=False, add_bin=False)
            for bspec in box_specs:
                center = np.array([
                    bspec["center_xy"][0], bspec["center_xy"][1],
                    box_floor_z + box_height / 2,
                ])
                spawn_open_box(
                    scene, name=bspec["name"], center=center,
                    inner_size=box_inner_size, wall_height=box_height,
                    wall_thickness=box_wall, base_thickness=box_base_thickness,
                    color=color_to_rgb(bspec["color"]),
                    base_center_z=box_base_center_z,
                )

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick between one and five cubes and balls (each tinted red, green, or blue) from the conveyor and drop"
                " them into the color-matching collection boxes."
            ),
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=_workspace_setup,
            box_verification_info={"box_specs": box_specs},
            containment_check=True,
            scenario={
                "source": "conveyor",
                "destination": "boxes_on_table",
                "workspace": "two_tables_custom_boxes",
            },
            pick_description={
                "asset_types": ["cube", "ball"],
                "count": "random (1-5 cubes + 1-5 balls)",
                "arrangement": "conveyor line, shuffled order",
                "colors": "RandomChoice(['red', 'green', 'blue'])",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "6 marker slots per box (2x3 pattern)",
                "count": 18,
                "containers": {
                    "count": 3,
                    "layout": "3 color-coded boxes on table (red, green, blue)",
                    "capacity_per_box": 6,
                },
            },
            implementation=TaskImplementationSpec(
                create_strategy=lambda picks, targets: ColorMatchStrategy(
                    picks, targets, color_palette=color_palette,
                ),
                strategy_description={
                    "class": "ColorMatchStrategy",
                    "pairing": "color_match",
                    "details": "color_palette=['red', 'green', 'blue']; each item routed to the box matching its color",
                },
            ),
            verification_description={
                "containment_check": True,
            },
            rationale={
                "create_strategy": "Items must be sorted by color into matching collection boxes — ColorMatchStrategy pairs each pick to a target in its same-color box",
                "containment_check": "Items placed inside boxes require box-boundary verification to confirm correct placement",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
