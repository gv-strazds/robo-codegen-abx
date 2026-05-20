import logging
import random
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from multi_pick_strategy import TypeBasedStrategy

logger = logging.getLogger(__name__)


class TableTaskShapeSortBoxes(UR10MultiPickPlaceTask):
    """Pick randomly colored cubes and balls from the conveyor and sort them by shape into two boxes on the cart."""

    DEFAULT_TASK_NAME = "table_task_shape_sort_boxes"

    MIN_PER_TYPE = 2
    MAX_PER_TYPE = 4

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import ItemSpec, ConveyorPositionGenerator, resolve_count
        from table_setup import (
            CART_SURFACE_CENTER,
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            setup_two_tables,
            spawn_open_box,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        # --- Cart / Box Constants ---
        cart_surface_center = CART_SURFACE_CENTER
        cart_z = cart_surface_center[2]
        cx, cy = cart_surface_center[0], cart_surface_center[1]

        # Box dimensions (inner space sized for 2x2 grid of primitives)
        # Primitives ~0.05m; 2x2 grid at 0.07m spacing needs ~0.14 x 0.14 inner
        box_inner_size = np.array([0.16, 0.16]) / stage_units
        box_height = 0.10 / stage_units  # taller walls to contain rolling balls
        box_wall = 0.01 / stage_units
        box_base_thickness = 0.01 / stage_units
        box_floor_z = cart_z + box_base_thickness + 0.001

        # Two boxes side-by-side on the cart (offset along Y)
        box_y_offset = 0.13
        box_specs = [
            {
                "name": "cube_box",
                "center": np.array([cx, cy - box_y_offset,
                                    cart_z + box_height / 2]),
                "center_xy": np.array([cx, cy - box_y_offset]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.65, 0.50, 0.35]),  # light brown
                "match_labels": {"type": "cube"},
                "z_tol": 0.03,
            },
            {
                "name": "ball_box",
                "center": np.array([cx, cy + box_y_offset,
                                    cart_z + box_height / 2]),
                "center_xy": np.array([cx, cy + box_y_offset]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.40, 0.30, 0.25]),  # dark brown
                "match_labels": {"type": "ball"},
                "z_tol": 0.03,  # generous for round objects
            },
        ]

        # --- Pick Strategy: cubes + balls shuffled on conveyor ---
        color_palette = ["red", "green", "blue", "yellow"]
        cube_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units
        ball_scale = np.array([0.035, 0.035, 0.035]) / stage_units  # smaller so 4 fit in a box
        dropzone_center = DROPZONE_CENTER_POINT
        pick_z = DROPZONE_Z + 0.035 / stage_units

        conv_pos_gen = ConveyorPositionGenerator(
            center_x=dropzone_center[0],
            center_y=dropzone_center[1],
            z=pick_z,
            spacing=0.12 / stage_units,
            jitter_x=0.015 / stage_units,
            jitter_y=0.005 / stage_units,
        )

        class ShapeSortPickGenerator:
            """Generate cubes and balls in random interleaved order on conveyor."""
            def __init__(self, pos_gen, min_per_type, max_per_type, colors,
                         cube_scale, ball_scale):
                self.pos_gen = pos_gen
                self.min_per_type = min_per_type
                self.max_per_type = max_per_type
                self.colors = colors
                self.cube_scale = cube_scale
                self.ball_scale = ball_scale

            def generate(self, count_range=None, seed=None):
                rng = random.Random(seed)
                num_cubes = rng.randint(self.min_per_type, self.max_per_type)
                num_balls = rng.randint(self.min_per_type, self.max_per_type)

                entries = []
                for _ in range(num_cubes):
                    entries.append({"type": "cube", "color": rng.choice(self.colors)})
                for _ in range(num_balls):
                    entries.append({"type": "ball", "color": rng.choice(self.colors)})
                rng.shuffle(entries)

                total = len(entries)
                count = resolve_count(count_range, capacity=total, seed=seed)
                if count is not None and count < total:
                    entries = entries[:count]
                    total = count

                positions = self.pos_gen.get_positions(total, seed)

                items = []
                for i, entry in enumerate(entries):
                    if i >= len(positions):
                        break
                    scale = self.cube_scale if entry["type"] == "cube" else self.ball_scale
                    items.append(ItemSpec(
                        asset_type=entry["type"],
                        position=positions[i],
                        color=entry["color"],
                        scale=scale,
                        name=f"{entry['type']}_{i}",
                    ))
                return items

        pick_strategy = ShapeSortPickGenerator(
            pos_gen=conv_pos_gen,
            min_per_type=self.MIN_PER_TYPE,
            max_per_type=self.MAX_PER_TYPE,
            colors=color_palette,
            cube_scale=cube_scale,
            ball_scale=ball_scale,
        )

        # --- Target Strategy: virtual hidden markers in boxes ---
        marker_scale = np.array([0.04, 0.04, 0.001]) / stage_units

        class BoxMarkerGenerator:
            """Generate hidden markers inside each box (2x2 grid)."""
            def __init__(self, box_specs, z_floor, scale):
                self.box_specs = box_specs
                self.z_floor = z_floor
                self.scale = scale
                self.grid_spacing = 0.06  # 6cm between marker centers

            def generate(self, count_range=None, seed=None):
                targets = []
                for bspec in self.box_specs:
                    for row in range(2):
                        for col in range(2):
                            ox = (col - 0.5) * self.grid_spacing
                            oy = (row - 0.5) * self.grid_spacing
                            targets.append(ItemSpec(
                                asset_type="marker",
                                position=np.array([
                                    bspec["center_xy"][0] + ox,
                                    bspec["center_xy"][1] + oy,
                                    self.z_floor,
                                ]),
                                scale=self.scale,
                                hidden=True,
                            ))
                return targets

        target_strategy = BoxMarkerGenerator(
            box_specs=box_specs,
            z_floor=box_floor_z,
            scale=marker_scale,
        )

        # --- Strategy factory ---
        # We need to build source_types and target index ranges at runtime
        # from the generated pick items, so use a factory lambda.
        def _create_strategy(picks, targets):
            source_types = []
            for p in picks:
                # Determine type from name prefix
                if p.name.startswith("cube"):
                    source_types.append("cube")
                elif p.name.startswith("ball"):
                    source_types.append("ball")
                else:
                    source_types.append("cube")  # fallback

            # First half of targets belong to cube_box, second half to ball_box
            markers_per_box = len(targets) // 2
            cube_target_indices = list(range(markers_per_box))
            ball_target_indices = list(range(markers_per_box, len(targets)))

            return TypeBasedStrategy(
                picks, targets,
                target_indices_by_type={
                    "cube": cube_target_indices,
                    "ball": ball_target_indices,
                },
                source_types=source_types,
            )

        # --- Workspace setup ---
        def _workspace_setup(scene, assets_root):
            setup_two_tables(scene, assets_root, standard_objs=False, add_bin=False)
            for bspec in box_specs:
                spawn_open_box(
                    scene, name=bspec["name"], center=bspec["center"],
                    inner_size=box_inner_size, wall_height=box_height,
                    wall_thickness=box_wall, base_thickness=box_base_thickness,
                    color=bspec["color"],
                )

        # --- TaskSpec ---
        pick_count = kwargs.pop("pick_count", None)
        kwargs.pop("target_count", None)

        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick randomly colored cubes and balls from the conveyor and sort them by shape into two boxes on the cart.",
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            setup_workspace=_workspace_setup,
            box_verification_info={"box_specs": box_specs},
            containment_check=True,
            scenario={
                "source": "conveyor",
                "destination": "boxes_on_cart",
                "workspace": "two_tables_custom_boxes",
            },
            pick_description={
                "asset_types": ["cube", "ball"],
                "count": "random (2-4 per type, 4-8 total)",
                "arrangement": "shuffled row on conveyor with X jitter",
                "colors": "RandomChoice(['red', 'green', 'blue', 'yellow'])",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "2x2 grid per box",
                "count": 8,
                "containers": {"count": 2, "layout": "side-by-side on cart", "capacity_per_box": 4},
                "virtual": True,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                create_strategy=_create_strategy,
                strategy_description={
                    "class": "TypeBasedStrategy",
                    "pairing": "type_based",
                    "details": "Cubes routed to cube_box, balls routed to ball_box",
                },
            ),
            verification_description={
                "containment_check": True,
                "match_labels": "type-based: cube_box accepts only cubes, ball_box accepts only balls",
            },
            rationale={
                "create_strategy": "TypeBasedStrategy routes items by asset type — cubes to one box, balls to another",
                "containment_check": "Items placed inside boxes — box-boundary verification confirms each item is within its target box",
                "virtual_target_generation_strategy": "Hidden markers inside boxes generated at pairing time — avoids scene clutter",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
