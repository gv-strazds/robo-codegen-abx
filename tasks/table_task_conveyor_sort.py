import logging
import random
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from multi_pick_strategy import TypeBasedStrategy

logger = logging.getLogger(__name__)


class TableTaskConveyorSort(UR10MultiPickPlaceTask):
    """Pick cubes and balls from the conveyor and sort them into separate boxes on the cart."""

    DEFAULT_TASK_NAME = "table_task_conveyor_sort"

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

        # --- Constants ---
        min_per_type = self.MIN_PER_TYPE
        max_per_type = self.MAX_PER_TYPE
        cube_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units
        ball_scale = np.array([0.035, 0.035, 0.035]) / stage_units  # smaller to fit in box without bouncing out

        # --- Box specs on the cart ---
        cart_z = CART_SURFACE_CENTER[2]
        box_inner_size = np.array([0.14, 0.14]) / stage_units
        box_height = 0.04 / stage_units
        box_wall = 0.01 / stage_units
        box_base_thickness = 0.01 / stage_units
        box_floor_z = cart_z + box_base_thickness + 0.001

        cx, cy = CART_SURFACE_CENTER[0], CART_SURFACE_CENTER[1]
        box_x_offset = 0.10

        box_specs = [
            {
                "name": "cube_box",
                "center": np.array([cx - box_x_offset, cy,
                                    cart_z + box_height / 2]),
                "center_xy": np.array([cx - box_x_offset, cy]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.65, 0.45, 0.25]),
                "match_labels": {"type": "cube"},
            },
            {
                "name": "ball_box",
                "center": np.array([cx + box_x_offset, cy,
                                    cart_z + box_height / 2]),
                "center_xy": np.array([cx + box_x_offset, cy]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.45, 0.55, 0.65]),
                "match_labels": {"type": "ball"},
                "z_tol": 0.03,  # generous for round objects settling
            },
        ]

        # --- Pick generator: cubes and balls randomly interleaved on conveyor ---
        class ConveyorCubeBallGenerator:
            """Generate cubes and balls in random interleaved order on the conveyor."""

            def __init__(self, center_x, center_y, z, spacing,
                         min_per_type, max_per_type, cube_scale, ball_scale):
                self.pos_gen = ConveyorPositionGenerator(
                    center_x=center_x,
                    center_y=center_y,
                    z=z,
                    spacing=spacing,
                    jitter_x=0.01 / stage_units,
                    jitter_y=0.01 / stage_units,
                )
                self.min_per_type = min_per_type
                self.max_per_type = max_per_type
                self.cube_scale = cube_scale
                self.ball_scale = ball_scale
                self.source_types = []

            def generate(self, count_range=None, seed=None):
                rng = random.Random(seed)
                n_cubes = rng.randint(self.min_per_type, self.max_per_type)
                n_balls = rng.randint(self.min_per_type, self.max_per_type)
                entries = (
                    [{"type": "cube"} for _ in range(n_cubes)]
                    + [{"type": "ball"} for _ in range(n_balls)]
                )
                rng.shuffle(entries)

                total = len(entries)
                count = resolve_count(count_range, capacity=total, seed=seed)
                if count is not None and count < total:
                    entries = entries[:count]

                positions = self.pos_gen.get_positions(len(entries), seed)

                self.source_types = [e["type"] for e in entries]
                items = []
                for i, entry in enumerate(entries):
                    s = self.cube_scale if entry["type"] == "cube" else self.ball_scale
                    items.append(ItemSpec(
                        asset_type=entry["type"],
                        position=positions[i],
                        color=None,
                        scale=s,
                        name=f"{entry['type']}_{i}",
                    ))
                return items

        pick_z = DROPZONE_Z + cube_scale[2] / 2 + 0.005
        pick_strategy = ConveyorCubeBallGenerator(
            center_x=DROPZONE_CENTER_POINT[0],
            center_y=DROPZONE_CENTER_POINT[1],
            z=pick_z,
            spacing=0.10 / stage_units,
            min_per_type=min_per_type,
            max_per_type=max_per_type,
            cube_scale=cube_scale,
            ball_scale=ball_scale,
        )

        # --- Target generator: virtual hidden markers inside each box ---
        grid_spacing = 0.065

        class BoxMarkerGenerator:
            """Generate hidden markers in a 2x2 grid inside each box.

            When count_range limits the total, markers are distributed
            evenly across boxes (remainder goes to earlier boxes).
            """

            def __init__(self, box_specs, z_floor):
                self.box_specs = box_specs
                self.z_floor = z_floor
                self.marker_scale = np.array([0.04, 0.04, 0.001])
                self.markers_per_box = []  # populated by generate()

            def generate(self, count_range=None, seed=None):
                # Build full marker grid for each box
                all_box_markers = []
                for bspec in self.box_specs:
                    center = bspec["center_xy"]
                    box_markers = []
                    for row in range(2):
                        for col in range(2):
                            ox = (col - 0.5) * grid_spacing
                            oy = (row - 0.5) * grid_spacing
                            box_markers.append(ItemSpec(
                                asset_type="marker",
                                position=np.array([
                                    center[0] + ox,
                                    center[1] + oy,
                                    self.z_floor,
                                ]),
                                scale=self.marker_scale,
                                hidden=True,
                            ))
                    all_box_markers.append(box_markers)

                total_capacity = sum(len(m) for m in all_box_markers)
                count = resolve_count(count_range, capacity=total_capacity, seed=seed)

                # Distribute evenly across boxes
                if count is not None and count < total_capacity:
                    n_boxes = len(all_box_markers)
                    per_box = count // n_boxes
                    remainder = count % n_boxes
                    self.markers_per_box = [
                        per_box + (1 if i < remainder else 0)
                        for i in range(n_boxes)
                    ]
                else:
                    self.markers_per_box = [len(m) for m in all_box_markers]

                targets = []
                for i, box_markers in enumerate(all_box_markers):
                    targets.extend(box_markers[:self.markers_per_box[i]])
                return targets

        target_strategy = BoxMarkerGenerator(
            box_specs=box_specs,
            z_floor=box_floor_z,
        )

        # --- Strategy factory ---
        def _create_strategy(picks, targets):
            source_types = pick_strategy.source_types[:len(picks)]
            n_cubes = sum(1 for t in source_types if t == "cube")
            n_balls = sum(1 for t in source_types if t == "ball")
            cube_markers = target_strategy.markers_per_box[0]
            ball_markers = target_strategy.markers_per_box[1]
            cube_target_indices = list(range(0, min(n_cubes, cube_markers)))
            ball_start = cube_markers  # ball targets follow cube targets
            ball_target_indices = list(range(ball_start, ball_start + min(n_balls, ball_markers)))
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

        # CLI overrides
        pick_count = kwargs.pop("pick_count", None)
        target_count = kwargs.pop("target_count", None)

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick cubes and balls from the conveyor and sort them"
                " into separate boxes on the cart."
            ),
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            target_count=target_count,
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
                "count": "random 2-4 cubes + 2-4 balls (4-8 total)",
                "arrangement": "randomly interleaved line on conveyor",
                "colors": "default (color=None)",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "2x2 grid per box",
                "count": 8,
                "containers": {
                    "count": 2,
                    "layout": "side by side on cart",
                    "capacity_per_box": 4,
                },
                "virtual": True,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                create_strategy=_create_strategy,
                strategy_description={
                    "class": "TypeBasedStrategy",
                    "pairing": "type_based",
                    "details": "cubes → cube_box, balls → ball_box",
                },
            ),
            verification_description={
                "containment_check": True,
                "match_labels": "type-based (cube/ball)",
            },
            rationale={
                "create_strategy": (
                    "TypeBasedStrategy routes items by asset type — cubes to one box,"
                    " balls to another, regardless of pick order"
                ),
                "containment_check": (
                    "Items placed inside boxes — box-boundary verification"
                    " confirms each item is in the correct box"
                ),
                "virtual_target_generation_strategy": (
                    "Hidden markers inside boxes generated at pairing time"
                    " — avoids cluttering the scene"
                ),
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
