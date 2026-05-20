import random
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from multi_pick_strategy import ColorMatchStrategy
from colors import color_to_rgb
from item_generation import ItemSpec, ConveyorPositionGenerator
import logging

logger = logging.getLogger(__name__)


class TableTaskColorShapes(UR10MultiPickPlaceTask):
    """Sort randomly colored cubes, cylinders, cones, and balls from the conveyor into matching color boxes."""

    DEFAULT_TASK_NAME = "table_task_color_shapes"

    MIN_OBJECTS_PER_TYPE = 1
    MAX_OBJECTS_PER_TYPE = 2

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
        cyl_radius = 0.03 / stage_units
        cyl_height = 0.08 / stage_units
        cylinder_scale = np.array([cyl_radius, cyl_radius, cyl_height])
        cone_radius = 0.03 / stage_units
        cone_height = 0.09 / stage_units
        cone_scale = np.array([cone_radius, cone_radius, cone_height])
        marker_scale = np.array([0.05, 0.04, 0.002]) / stage_units
        ball_scale = np.array([0.0315, 0.0315, 0.0315]) / stage_units

        # Box dimensions
        box_inner_size = np.array([0.20, 0.16]) / stage_units
        box_height = 0.08 / stage_units
        box_wall = 0.012 / stage_units
        box_base_thickness = 0.06244 / stage_units
        box_base_center_z = 0.06244 / stage_units
        box_floor_z = box_base_center_z + box_base_thickness / 2

        # Build box specs
        box_offsets = {
            "red": np.array([0.25, -0.08, 0.0]),
            "green": np.array([0.02, 0.16, 0.0]),
            "blue": np.array([-0.20, -0.05, 0.0]),
        }
        box_specs = []
        for color, boff in box_offsets.items():
            offset_scaled = boff / stage_units
            center_xy = table_center + offset_scaled
            box_specs.append({
                "name": f"{color}_collection_box",
                "color": color,
                "center_xy": np.array([center_xy[0], center_xy[1]]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "match_labels": {"color": color},
            })

        # Compute marker targets
        cols, rows = 3, 2
        spacing_x = box_inner_size[0] / (cols + 1)
        spacing_y = box_inner_size[1] / (rows + 1)
        target_markers = []
        for bspec in box_specs:
            for row in range(rows):
                for col in range(cols):
                    offset_x = -box_inner_size[0] / 2 + (col + 1) * spacing_x
                    offset_y = -box_inner_size[1] / 2 + (row + 1) * spacing_y
                    target_markers.append({
                        "position": np.array([
                            bspec["center_xy"][0] + offset_x,
                            bspec["center_xy"][1] + offset_y,
                            bspec["floor_z"] + 0.001 / stage_units,
                        ]),
                        "color": bspec["color"],
                    })

        # --- Strategies ---
        class ColorShapesGenerator:
            def __init__(self, position_generator, min_obj, max_obj, colors, cube_scale, cyl_scale, cone_scale, ball_scale):
                self.position_generator = position_generator
                self.min_obj = min_obj
                self.max_obj = max_obj
                self.colors = colors
                self.cube_scale = cube_scale
                self.cyl_scale = cyl_scale
                self.cone_scale = cone_scale
                self.ball_scale = ball_scale

            def generate(self, count_range=(1,1), seed=None):
                rng = random.Random(seed)
                num_cubes = rng.randint(self.min_obj, self.max_obj)
                num_cyls = rng.randint(self.min_obj, self.max_obj)
                num_cones = rng.randint(self.min_obj, self.max_obj)
                num_balls = rng.randint(self.min_obj, self.max_obj)
                total = num_cubes + num_cyls + num_cones + num_balls

                entries = []
                for _ in range(num_cubes):
                    entries.append({"type": "cube", "color": rng.choice(self.colors)})
                for _ in range(num_cyls):
                    entries.append({"type": "cylinder", "color": rng.choice(self.colors)})
                for _ in range(num_cones):
                    entries.append({"type": "cone", "color": rng.choice(self.colors)})
                for _ in range(num_balls):
                    entries.append({"type": "ball", "color": rng.choice(self.colors)})
                rng.shuffle(entries)

                positions = self.position_generator.get_positions(total, seed)

                items = []
                for i, entry in enumerate(entries):
                    if i >= len(positions): break
                    ctype = entry["type"]
                    s = self.cube_scale
                    if ctype == "cylinder": s = self.cyl_scale
                    elif ctype == "cone": s = self.cone_scale
                    elif ctype == "ball": s = self.ball_scale

                    items.append(ItemSpec(
                        asset_type=ctype,
                        position=positions[i],
                        color=entry["color"],
                        scale=s
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

        pick_strategy = ColorShapesGenerator(
            position_generator=pick_pos_gen,
            min_obj=self.MIN_OBJECTS_PER_TYPE,
            max_obj=self.MAX_OBJECTS_PER_TYPE,
            colors=color_palette,
            cube_scale=cube_scale,
            cyl_scale=cylinder_scale,
            cone_scale=cone_scale,
            ball_scale=ball_scale
        )

        target_positions = [m["position"] for m in target_markers]
        target_colors = [m["color"] for m in target_markers]

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
                "Pick cubes, cylinders, cones, and balls spawned on the conveyor (each tinted red, green, or blue) and place"
                " them into the matching colored boxes on the table."
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
                "asset_types": ["cube", "cylinder", "cone", "ball"],
                "count": "random (1-2 per shape type, 4-8 total)",
                "arrangement": "conveyor line, shuffled order",
                "colors": "RandomChoice(['red', 'green', 'blue'])",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "3x2 grid per box",
                "count": 18,
                "containers": {"count": 3, "layout": "3 color-coded boxes on table (red, green, blue)", "capacity_per_box": 6},
            },
            implementation=TaskImplementationSpec(
                create_strategy=lambda picks, targets: ColorMatchStrategy(
                    picks, targets, color_palette=color_palette,
                ),
                strategy_description={
                    "class": "ColorMatchStrategy",
                    "pairing": "color_match",
                    "details": "color_palette=['red', 'green', 'blue']; multiple shape types sorted purely by color",
                },
            ),
            verification_description={
                "containment_check": True,
                "containment_check_rationale": "Items placed inside boxes require box-boundary verification to confirm correct placement",
            },
            rationale={
                "create_strategy": "Mixed shapes sorted by color — ColorMatchStrategy routes items to matching-color boxes regardless of shape type",
                "containment_check": "Items placed inside boxes require box-boundary verification to confirm correct placement",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)

