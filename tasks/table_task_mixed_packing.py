import numpy as np
from typing import Optional

from multi_pickplace_task import UR10MultiPickPlaceTask
from task_verification import is_vertical

# Asset types for which upright orientation is required for success
_VERTICAL_ASSET_TYPES = {"madara_bottle", "cracker_box", "soup_can", "mustard_bottle"}

import logging

logger = logging.getLogger(__name__)


class TableTaskMixedPacking(UR10MultiPickPlaceTask):
    """
    Pick Cracker Boxes and Soup Cans from the conveyor area (5 rows: 1 Box, 4 Cans)
    and place four Soup Cans and one Cracker Box into each of two boxes on the cart.
    """

    DEFAULT_TASK_NAME = "table_task_mixed_packing"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        from isaacsim.core.utils import rotations
        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import ItemSpec, resolve_count
        from pxr import Gf
        from table_setup import CART_SURFACE_CENTER, DROPZONE_CENTER_POINT, setup_two_tables, spawn_open_box
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        # --- Constants ---
        cart_surface_center = CART_SURFACE_CENTER
        cart_z = cart_surface_center[2]

        # Large enough for box + 2x2 cans
        box_inner_size = np.array([0.25, 0.35]) / stage_units
        box_height = 0.10 / stage_units
        box_wall = 0.01 / stage_units
        box_base_thickness = 0.01 / stage_units

        # Box locations on cart (spaced along Y; cart Y center is ~0.36)
        box_floor_z = cart_z + box_base_thickness + 0.001
        box_specs = [
            {
                "name": "cart_box_1",
                "center": np.array([cart_surface_center[0], cart_surface_center[1]-0.295, cart_z + box_height / 2]),
                "center_xy": np.array([cart_surface_center[0], cart_surface_center[1]-0.295]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.5, 0.4, 0.3]),  # cardboard-ish
                "targets": []
            },
            {
                "name": "cart_box_2",
                "center": np.array([cart_surface_center[0], cart_surface_center[1]+0.105, cart_z + box_height / 2]),
                "center_xy": np.array([cart_surface_center[0], cart_surface_center[1]+0.105]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.5, 0.4, 0.3]),
                "targets": []
            }
        ]

        # --- Pick Strategy: Interleaved Columns (along Y) ---
        class MixedRowGenerator:
            def __init__(self, start_point, box_count, can_count_per_col):
                self.start_point = start_point
                self.box_count = box_count
                self.can_count_per_col = can_count_per_col
                self.can_col_spacing = 0.08   # 8cm between can columns
                self.box_can_gap = 0.20      # gap between box column and first can column
                self.box_y_spacing = 0.15    # 15cm Y spacing for boxes
                self.can_y_spacing = 0.10    # 10cm Y spacing for cans (compact)
                self.box_ori = rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90))
                self.can_ori = rotations.gf_rotation_to_np_array(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90))

            def generate(self, count_range=None, seed=None):
                # 1 box column + 3 can columns
                total_width = self.box_can_gap + 2 * self.can_col_spacing
                start_x = (self.start_point[0] - total_width / 2) + 0.10
                center_y = self.start_point[1] + 0.15
                start_z = self.start_point[2] + 0.10

                boxes = []
                box_col_height = (self.box_count - 1) * self.box_y_spacing
                box_start_y = center_y - box_col_height / 2
                for i in range(self.box_count):
                    pos = np.array([start_x, box_start_y + i * self.box_y_spacing, start_z])
                    boxes.append(ItemSpec(asset_type="cracker_box", position=pos, orientation=self.box_ori, name=f"cracker_box_{i}"))

                cans = []
                can_col_height = (self.can_count_per_col - 1) * self.can_y_spacing
                can_start_y = center_y - can_col_height / 2
                for i in range(self.can_count_per_col):
                    for col_idx in range(3):  # 3 columns of cans
                        can_x = start_x + self.box_can_gap + col_idx * self.can_col_spacing
                        pos = np.array([can_x, can_start_y + i * self.can_y_spacing, start_z])
                        cans.append(ItemSpec(asset_type="soup_can", position=pos, orientation=self.can_ori, name=f"soup_can_r{i}_c{col_idx}"))

                # Interleave: 1 box, then 4 cans per set
                items = []
                can_idx = 0
                max_sets = max(self.box_count, (len(cans) + 3) // 4)
                for i in range(max_sets):
                    if i < len(boxes):
                        items.append(boxes[i])
                    for _ in range(4):
                        if can_idx < len(cans):
                            items.append(cans[can_idx])
                            can_idx += 1
                count = resolve_count(count_range, capacity=len(items), seed=seed)
                if count is not None and count < len(items):
                    items = items[:count]
                return items

        pick_strategy = MixedRowGenerator(
            start_point=DROPZONE_CENTER_POINT,
            box_count=7,
            can_count_per_col=9
        )

        # --- Target Strategy ---
        class CartTargetGenerator:
            def __init__(self, box_specs, box_inner_size, z_floor):
                self.box_specs = box_specs
                self.box_inner_size = box_inner_size
                self.z_floor = z_floor
                self.marker_scale = np.array([0.08, 0.08, 0.002])

            def generate(self, count_range=(1, 1), seed=None):
                targets = []
                # Per box: 1 cracker box target (Y- side) + 4 soup can targets (2x2 grid, Y+ side)
                offset_box_y = -0.10       # cracker box offset (Y- side)
                center_cans_y = 0.08       # cans grid center (Y+ side)
                grid_spacing = 0.08

                for spec in self.box_specs:
                    center = spec["center"]
                    # Cracker box target (green)
                    t_box_pos = np.array([center[0], center[1] + offset_box_y, self.z_floor])
                    targets.append(ItemSpec(asset_type="rect", position=t_box_pos, color="green", scale=self.marker_scale, hidden=True))

                    # Soup can targets (red) - 2x2 grid
                    d = grid_spacing / 2
                    offsets = [(-d, -d), (d, -d), (-d, d), (d, d)]
                    for dx, dy in offsets:
                        t_can_pos = np.array([
                            center[0] + dx,
                            center[1] + center_cans_y + dy,
                            self.z_floor
                        ])
                        targets.append(ItemSpec(asset_type="rect", position=t_can_pos, color="red", scale=self.marker_scale, hidden=True))

                return targets

        target_strategy = CartTargetGenerator(
            box_specs=box_specs,
            box_inner_size=box_inner_size,
            z_floor=box_floor_z
        )

        task_ref = self

        def _check_mixed_verticality(pick_index, target_index):
            pick_obj = task_ref._pick_objs[pick_index]
            if any(pick_obj.name.startswith(t) for t in _VERTICAL_ASSET_TYPES):
                if not is_vertical(pick_obj, max_tilt_deg=15):
                    return (False, "item is not vertical (upright orientation required)")
            return (True, "")

        def _workspace_setup(scene, assets_root):
            setup_two_tables(scene, assets_root, standard_objs=False, add_bin=False)
            for bspec in box_specs:
                spawn_open_box(
                    scene, name=bspec["name"], center=bspec["center"],
                    inner_size=box_inner_size, wall_height=box_height,
                    wall_thickness=box_wall, base_thickness=box_base_thickness,
                    color=bspec["color"],
                )

        # CLI override for pick count (target count tied to box geometry)
        pick_count = kwargs.pop("pick_count", None)
        kwargs.pop("target_count", None)  # not meaningful for box-packing targets

        spec = TaskSpec(
            task_name=task_name,
            task_description="Pick Cracker Boxes and Soup Cans from 5 rows on the conveyor (1 row Boxes, 4 rows Cans) and place four Soup Cans and one Cracker Box into each of two boxes on the cart. The Cans should form a 2x2 grid.",
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            setup_workspace=_workspace_setup,
            placement_constraints_fn=_check_mixed_verticality,
            box_verification_info={"box_specs": box_specs},
            containment_check=True,
            scenario={
                "source": "conveyor",
                "destination": "boxes_on_cart",
                "workspace": "two_tables_custom_boxes",
            },
            pick_description={
                "asset_types": ["cracker_box", "soup_can"],
                "count": "7 cracker boxes + 27 soup cans (interleaved: 1 box then 4 cans per set)",
                "arrangement": "interleaved columns on conveyor — 1 box column, 3 can columns",
                "colors": "USD asset default",
                "orientation": "upright (-90° X)",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "per box: 1 green marker (cracker box) + 4 red markers (soup cans, 2x2 grid)",
                "count": 10,
                "containers": {"count": 2, "layout": "2 boxes on cart (spaced along Y)", "capacity_per_box": 5},
                "virtual": True,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                    "details": "Pick order is interleaved (box, can, can, can, can, box, ...) and markers ordered per-box (box_marker, can_markers x4), so sequential pairing fills each box with 1 box + 4 cans",
                },
            ),
            verification_description={
                "spatial_check": "is_on_top (default)",
                "placement_constraints": "is_vertical (for USD asset types)",
                "containment_check": True,
            },
            rationale={
                "create_strategy": "Sequential pairing sufficient — interleaved pick order pre-arranged to fill each box correctly without explicit type routing",
                "placement_constraints_fn": "Cracker boxes and soup cans must remain upright after placement for stable packing",
                "containment_check": "Items placed inside boxes — box-boundary verification confirms each item is within its target box",
                "virtual_target_generation_strategy": "Hidden markers inside boxes generated at pairing time — target count tied to box geometry",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)

