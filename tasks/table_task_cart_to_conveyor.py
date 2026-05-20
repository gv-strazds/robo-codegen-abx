import logging
import random
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)

# USD asset types used in this task
_ITEM_TYPES = ["cracker_box", "soup_can", "mustard_bottle", "sugar_box"]


class TableTaskCartToConveyor(UR10MultiPickPlaceTask):
    """Pick cracker boxes, soup cans, mustard bottles, and sugar boxes from the cart
    and place one of each vertically into boxes on the conveyor."""

    DEFAULT_TASK_NAME = "table_task_cart_to_conveyor"

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
        from table_setup import (
            CART_SURFACE_CENTER,
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            setup_two_tables,
            spawn_open_box,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        # Raise move height so carried items clear tall objects on the cart.
        # Default 0.3m is too low — upright cracker_box tops reach ~0.30m on cart.
        # (Applied via ee_height_for_move in TaskSpec below.)

        # --- Random counts ---
        seed = kwargs.pop("seed", None)
        rng = random.Random(seed)
        num_boxes = rng.randint(1, 6)
        item_counts = {t: rng.randint(1, 8) for t in _ITEM_TYPES}
        num_fillable = min(num_boxes, *item_counts.values())

        logger.info(
            f"CartToConveyor: {num_boxes} boxes, counts={item_counts}, "
            f"fillable={num_fillable}"
        )

        # --- Upright orientation for USD assets ---
        default_orientation = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        )

        # --- Cart layout constants ---
        cart_center = CART_SURFACE_CENTER
        cart_z = cart_center[2]

        # Approximate half-heights for z calculation (USD assets, local Y is tall axis)
        _HALF_HEIGHTS = {
            "cracker_box": 0.107,
            "soup_can": 0.051,
            "mustard_bottle": 0.10,
            "sugar_box": 0.09,
        }

        # Arrange items in 4 columns (one per type) on the cart.
        # Columns along X, items within column along Y.
        # Spacing accounts for actual item widths (cracker_box is widest at 0.164m).
        type_x_offsets = {
            "cracker_box": -0.18,
            "soup_can": -0.04,
            "mustard_bottle": 0.06,
            "sugar_box": 0.18,
        }
        y_spacing = 0.10
        max_count = max(item_counts.values())
        y_start = -(max_count - 1) * y_spacing / 2

        # --- Build pick items ---
        # Order: fillable items interleaved (cracker_0, soup_0, mustard_0, sugar_0,
        # cracker_1, ...) then extras at end.
        class CartPickGenerator:
            def __init__(
                self, cart_center, cart_z, item_counts, num_fillable,
                orientation, type_x_offsets, y_spacing, y_start, half_heights,
            ):
                self.cart_center = cart_center
                self.cart_z = cart_z
                self.item_counts = item_counts
                self.num_fillable = num_fillable
                self.orientation = orientation
                self.type_x_offsets = type_x_offsets
                self.y_spacing = y_spacing
                self.y_start = y_start
                self.half_heights = half_heights

            def generate(self, count_range=None, seed=None):
                items = []
                type_counters = {t: 0 for t in _ITEM_TYPES}

                def _make_item(item_type, index_in_type):
                    dx = self.type_x_offsets[item_type]
                    dy = self.y_start + index_in_type * self.y_spacing
                    pick_z = self.cart_z + 0.025 + self.half_heights[item_type]
                    pos = np.array([
                        self.cart_center[0] + dx,
                        self.cart_center[1] + dy,
                        pick_z,
                    ])
                    return ItemSpec(
                        asset_type=item_type, position=pos,
                        orientation=self.orientation,
                        scale=np.array([1.0, 1.0, 1.0]),
                        name=f"{item_type}_{index_in_type}",
                    )

                # First: interleaved fillable items
                for box_idx in range(self.num_fillable):
                    for item_type in _ITEM_TYPES:
                        items.append(_make_item(item_type, type_counters[item_type]))
                        type_counters[item_type] += 1

                # Then: extras (won't get targets via sequential pairing)
                for item_type in _ITEM_TYPES:
                    remaining = self.item_counts[item_type] - type_counters[item_type]
                    for _ in range(remaining):
                        items.append(_make_item(item_type, type_counters[item_type]))
                        type_counters[item_type] += 1

                count = resolve_count(count_range, capacity=len(items), seed=seed)
                if count is not None and count < len(items):
                    items = items[:count]
                return items

        pick_strategy = CartPickGenerator(
            cart_center=cart_center, cart_z=cart_z,
            item_counts=item_counts, num_fillable=num_fillable,
            orientation=default_orientation, type_x_offsets=type_x_offsets,
            y_spacing=y_spacing, y_start=y_start, half_heights=_HALF_HEIGHTS,
        )

        # --- Conveyor boxes ---
        # Boxes in a row along Y on the dropzone table.
        # Box must fit 2x2 grid of items; cracker_box is widest at 0.164m footprint.
        # Two items side-by-side in X need ~0.36m + margin.
        # Tallest item upright is cracker_box at 0.213m.
        box_inner_size = np.array([0.38, 0.20]) / stage_units
        box_height = 0.15 / stage_units
        box_wall = 0.01 / stage_units
        box_base_thickness = 0.01 / stage_units

        dropzone_x = DROPZONE_CENTER_POINT[0]
        dropzone_y = DROPZONE_CENTER_POINT[1]
        box_outer_y = box_inner_size[1] + 2 * box_wall
        box_spacing_y = box_outer_y + 0.03  # 3cm gap between boxes
        box_z_center = DROPZONE_Z + box_height / 2
        box_floor_z = DROPZONE_Z + box_base_thickness + 0.001

        # Center the box row on the dropzone, but ensure the first box y >= 0.20.
        box_row_y_start = dropzone_y - (num_boxes - 1) / 2 * box_spacing_y
        box_row_y_start = max(box_row_y_start, 0.50)

        conveyor_box_specs = []
        for i in range(num_boxes):
            cy = box_row_y_start + i * box_spacing_y
            conveyor_box_specs.append({
                "name": f"conveyor_box_{i}",
                "center": np.array([dropzone_x, cy, box_z_center]),
                "center_xy": np.array([dropzone_x, cy]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.45, 0.35, 0.25]),
            })

        # --- Target markers: 4 per fillable box (2×2 grid) ---
        class BoxMarkerGenerator:
            def __init__(self, cbox_specs, nfill, bfloor_z, binner_size):
                self.box_specs = cbox_specs
                self.num_fillable = nfill
                self.box_floor_z = bfloor_z
                self.box_inner_size = binner_size
                self.marker_scale = np.array([0.04, 0.04, 0.001])

            def generate(self, count_range=(1, 1), seed=None):
                targets = []
                # 2x2 grid offsets inside each box
                sx = self.box_inner_size[0] / 4  # quarter-width
                sy = self.box_inner_size[1] / 4  # quarter-height
                offsets = [(-sx, -sy), (sx, -sy), (-sx, sy), (sx, sy)]
                for i in range(self.num_fillable):
                    spec = self.box_specs[i]
                    for ox, oy in offsets:
                        pos = np.array([
                            spec["center_xy"][0] + ox,
                            spec["center_xy"][1] + oy,
                            self.box_floor_z,
                        ])
                        targets.append(ItemSpec(
                            asset_type="marker", position=pos,
                            scale=self.marker_scale, hidden=True,
                        ))
                return targets

        target_strategy = BoxMarkerGenerator(
            cbox_specs=conveyor_box_specs,
            nfill=num_fillable,
            bfloor_z=box_floor_z,
            binner_size=box_inner_size,
        )

        task_ref = self

        def _check_verticality(pick_index, target_index):
            from task_verification import is_vertical
            pick_obj = task_ref._pick_objs[pick_index]
            if not is_vertical(pick_obj, max_tilt_deg=15):
                return (False, "item is not vertical (upright orientation required)")
            return (True, "")

        def _workspace_setup(scene, assets_root):
            setup_two_tables(scene, assets_root, standard_objs=False, add_bin=False)
            for bspec in conveyor_box_specs:
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
            task_description="Pick cracker boxes, soup cans, mustard bottles, and sugar boxes from the cart and place one of each vertically into boxes on the conveyor.",
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            setup_workspace=_workspace_setup,
            placement_constraints_fn=_check_verticality,
            box_verification_info={"box_specs": conveyor_box_specs},
            containment_check=True,
            scenario={
                "source": "cart",
                "destination": "boxes_on_conveyor",
                "workspace": "two_tables_custom_boxes",
            },
            pick_description={
                "asset_types": ["cracker_box", "soup_can", "mustard_bottle", "sugar_box"],
                "count": "random (1-8 per type, interleaved for fillable boxes, extras at end)",
                "arrangement": "4 columns on cart (one per type), items along Y",
                "colors": "USD asset default",
                "orientation": "upright (-90° X)",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "2x2 grid per fillable box",
                "count": "4 per fillable box",
                "containers": {"count": "random (1-7)", "layout": "row along Y on conveyor", "capacity_per_box": 4},
                "virtual": True,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                ee_height_for_move=0.45 / stage_units,
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                    "details": "Items interleaved by type (cracker, soup, mustard, sugar per box); sequential pairing fills each box with one of each type",
                },
            ),
            verification_description={
                "spatial_check": "is_on_top (default)",
                "placement_constraints": "is_vertical",
                "containment_check": True,
            },
            rationale={
                "create_strategy": "Sequential pairing sufficient — pick order is pre-interleaved so each group of 4 fills one box with one of each item type",
                "placement_constraints_fn": "All USD asset types must remain upright after placement for stable packing",
                "containment_check": "Items placed inside boxes — box-boundary verification confirms each item is within its target box",
                "virtual_target_generation_strategy": "Hidden markers inside boxes generated at pairing time — number of fillable boxes depends on random item counts",
                "ee_height_for_move": "Default 0.3m too low — upright cracker_box tops reach ~0.30m on cart; 0.45m clears all items with margin",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)

