import numpy as np
from typing import Optional

from multi_pickplace_task import UR10MultiPickPlaceTask
from multi_pick_strategy import TypeBasedStrategy
from item_generation import ItemSpec, ConveyorPositionGenerator
import logging

logger = logging.getLogger(__name__)


class TableTaskConveyorCubesBalls(UR10MultiPickPlaceTask):
    """Pick cubes and balls from the conveyor, sorting them into two separate boxes on the table."""

    def __init__(
        self,
        task_name: str = "table_task_conveyor_cubes_balls",
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        from isaacsim.core.utils.stage import get_stage_units
        from table_setup import (
            setup_two_tables,
            spawn_open_box,
            TABLETOP_CENTER_POINT,
            ITEM_SPAWN_REFERENCE_Z,
            DROPZONE_CENTER_POINT,
        )
        from task_spec import TaskSpec

        stage_units = get_stage_units()

        num_cubes = 3
        num_balls = 3
        cube_scale = np.array([0.045, 0.045, 0.045]) / stage_units
        ball_diameter = 0.035
        ball_scale = np.array([ball_diameter, ball_diameter, ball_diameter]) / stage_units
        table_surface_z = ITEM_SPAWN_REFERENCE_Z
        table_center = TABLETOP_CENTER_POINT

        box_base_offset = -0.01
        target_box_specs = [
            {
                "name": "cube_box",
                "offset": np.array([0.18, -0.02, 0.0]),
                "inner_size": np.array([0.20, 0.14]),
                "height": 0.08,
                "wall_thickness": 0.012,
                "item_type": "cube",
                "marker_color": "purple",
                "color": np.array([0.35, 0.3, 0.4]),
                "capacity": num_cubes,
            },
            {
                "name": "ball_box",
                "offset": np.array([-0.02, 0.20, 0.0]),
                "inner_size": np.array([0.18, 0.16]),
                "height": 0.08,
                "wall_thickness": 0.012,
                "item_type": "ball",
                "marker_color": "yellow",
                "color": np.array([0.22, 0.22, 0.22]),
                "capacity": num_balls,
            },
        ]

        # --- Strategies ---
        class ConveyorCubesBallsGenerator:
            def __init__(self, position_generator, num_cubes, num_balls, cube_scale, ball_scale):
                self.position_generator = position_generator
                self.num_cubes = num_cubes
                self.num_balls = num_balls
                self.cube_scale = cube_scale
                self.ball_scale = ball_scale

            def generate(self, count_range=(1,1), seed=None):
                total = self.num_cubes + self.num_balls
                positions = self.position_generator.get_positions(total, seed)
                items = []
                for i in range(total):
                    if i < self.num_cubes:
                        items.append(ItemSpec(asset_type="cube", position=positions[i], color="purple", scale=self.cube_scale))
                    else:
                        items.append(ItemSpec(asset_type="ball", position=positions[i], color="yellow", scale=self.ball_scale))
                return items

        spacing = 0.08 / stage_units
        x_position = DROPZONE_CENTER_POINT[0] - 0.06
        z_height = DROPZONE_CENTER_POINT[2] + 0.035

        pick_pos_gen = ConveyorPositionGenerator(
            center_x=x_position,
            center_y=DROPZONE_CENTER_POINT[1],
            z=z_height,
            spacing=spacing,
            jitter_x=0.01 / stage_units,
            jitter_y=0.01 / stage_units
        )

        pick_strategy = ConveyorCubesBallsGenerator(
            position_generator=pick_pos_gen,
            num_cubes=num_cubes,
            num_balls=num_balls,
            cube_scale=cube_scale,
            ball_scale=ball_scale
        )

        # Precompute markers
        target_marker_colors = []
        target_positions = []
        cube_target_indices = []
        ball_target_indices = []

        def generate_box_slots(spec, tc, tz):
            capacity = spec["capacity"]
            inner_size = spec["inner_size"]
            margin_y = 0.02
            usable_y = max(inner_size[1] - 2 * margin_y, margin_y)
            if capacity <= 1:
                offsets_y = [0.0]
            else:
                offsets_y = np.linspace(-usable_y / 2, usable_y / 2, capacity)
            slot_z = spec.get("slot_z", tz + 0.01)
            center = spec.get("center_world")
            if center is None:
                bo = spec.get("offset", np.zeros(3))
                center = tc + bo
            return [[center[0], center[1] + dy, slot_z] for dy in offsets_y]

        running_index = 0

        for spec in target_box_specs:
            boff = spec["offset"]
            height = spec["height"]
            wall = spec["wall_thickness"]
            center_xy = table_center[:2] + boff[:2]
            base_plane_z = table_surface_z + box_base_offset
            box_center = np.array([center_xy[0], center_xy[1], base_plane_z + height / 2])

            base_thickness = spec.get("base_thickness", wall)
            spec["slot_z"] = base_plane_z + base_thickness + 0.004
            spec["center_world"] = box_center

            slot_positions = generate_box_slots(spec, table_center, table_surface_z)

            indices = list(range(running_index, running_index + len(slot_positions)))
            running_index += len(slot_positions)

            target_positions.extend(slot_positions)
            target_marker_colors.extend([spec["marker_color"]] * len(slot_positions))

            if spec["item_type"] == "cube":
                cube_target_indices = indices
            else:
                ball_target_indices = indices

        marker_scale = np.array([0.04, 0.04, 0.002]) / stage_units

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
             ) for p, c in zip(target_positions, target_marker_colors)
        ]
        target_strategy = FixedListGenerator(target_items)

        source_types = ["cube"] * num_cubes + ["ball"] * num_balls

        def _strategy_factory(picks, targets):
            return TypeBasedStrategy(
                pick_objs=picks, target_objs=targets,
                target_indices_by_type={
                    "cube": cube_target_indices,
                    "ball": ball_target_indices,
                },
                source_types=source_types,
            )

        def _workspace_setup(scene, assets_root):
            setup_two_tables(scene, assets_root, standard_objs=False, add_bin=False)
            _spawn_target_boxes(scene)

        def _spawn_target_boxes(scene):
            """Create the two box receptacles on the table."""
            for bspec in target_box_specs:
                boff = bspec["offset"]
                height = bspec["height"]
                wall = bspec["wall_thickness"]
                cxy = table_center[:2] + boff[:2]
                bpz = table_surface_z + box_base_offset
                bt = bspec.get("base_thickness", wall)
                center = np.array([cxy[0], cxy[1], bpz + height / 2])
                spawn_open_box(
                    scene, name=bspec["name"], center=center,
                    inner_size=bspec["inner_size"], wall_height=height,
                    wall_thickness=wall, base_thickness=bt,
                    color=bspec["color"],
                )

        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick three cubes followed by three balls from the conveyor and place them into two boxes on the table.",
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            create_strategy=_strategy_factory,
            setup_workspace=_workspace_setup,
            scenario={
                "source": "conveyor",
                "destination": "boxes_on_table",
                "workspace": "two_tables_custom_boxes",
            },
            pick_description={
                "asset_types": ["cube", "ball"],
                "count": "3 cubes + 3 balls",
                "arrangement": "conveyor line (cubes first, then balls)",
                "colors": "cubes purple, balls yellow",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "3 marker slots per box (along Y axis)",
                "count": 6,
                "containers": {"count": 2, "layout": "2 boxes on table", "capacity_per_box": 3},
            },
            strategy_description={
                "class": "TypeBasedStrategy",
                "pairing": "type_based",
                "details": "Routes cubes to cube_box markers (indices 0-2) and balls to ball_box markers (indices 3-5)",
            },
            rationale={
                "create_strategy": "Two distinct object types must be sorted into separate boxes — TypeBasedStrategy routes each type to its designated box",
            },
        )

        super().__init__(task_spec=spec, offset=offset, **kwargs)
