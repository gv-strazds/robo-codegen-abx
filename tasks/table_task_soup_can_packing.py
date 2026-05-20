import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from task_verification import is_vertical

logger = logging.getLogger(__name__)


class TableTaskSoupCanPacking(UR10MultiPickPlaceTask):
    """Pick soup cans from the conveyor and place 6 into each of 4 boxes on the cart."""

    DEFAULT_TASK_NAME = "table_task_soup_can_packing"

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

        # --- Cart / Box Constants ---
        cart_surface_center = CART_SURFACE_CENTER
        cart_z = cart_surface_center[2]

        # Box dimensions (inner space sized for 2x3 grid of soup cans)
        # Soup can diameter ~0.068m; 2x3 grid at 0.07m spacing needs ~0.14 x 0.21 inner
        box_inner_size = np.array([0.15, 0.22]) / stage_units
        box_height = 0.12 / stage_units
        box_wall = 0.01 / stage_units
        box_base_thickness = 0.01 / stage_units

        # 2x2 grid of boxes on the cart
        box_x_offset = 0.105
        box_y_offset = 0.14
        cx, cy = cart_surface_center[0], cart_surface_center[1]
        box_floor_z = cart_z + box_base_thickness + 0.001
        box_specs = [
            {
                "name": "cart_box_1",
                "center": np.array([cx - box_x_offset, cy - box_y_offset,
                                    cart_z + box_height / 2]),
                "center_xy": np.array([cx - box_x_offset, cy - box_y_offset]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.55, 0.43, 0.33]),
            },
            {
                "name": "cart_box_2",
                "center": np.array([cx + box_x_offset, cy - box_y_offset,
                                    cart_z + box_height / 2]),
                "center_xy": np.array([cx + box_x_offset, cy - box_y_offset]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.55, 0.43, 0.33]),
            },
            {
                "name": "cart_box_3",
                "center": np.array([cx - box_x_offset, cy + box_y_offset,
                                    cart_z + box_height / 2]),
                "center_xy": np.array([cx - box_x_offset, cy + box_y_offset]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.50, 0.40, 0.30]),
            },
            {
                "name": "cart_box_4",
                "center": np.array([cx + box_x_offset, cy + box_y_offset,
                                    cart_z + box_height / 2]),
                "center_xy": np.array([cx + box_x_offset, cy + box_y_offset]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.50, 0.40, 0.30]),
            },
        ]

        # --- Pick Strategy: 4 rows x 6 positions on the conveyor ---
        # Rows extend along Y (increasing y).  Pick order: one can from each
        # row at the first Y position, then advance to the next Y position.
        default_orientation = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        )
        pick_z = DROPZONE_Z + 0.12  # safe height above ground

        num_rows = 4      # distinct X positions (one per box)
        cans_per_row = 6  # positions along Y per row

        class ConveyorRowsGenerator:
            """Generate soup cans in interleaved row order.

            Produces items column-by-column: for each Y index, emit one can
            from every row (X position) before advancing to the next Y index.
            """
            def __init__(self, center, num_rows, cans_per_row, spacing_x, spacing_y,
                         orientation):
                self.center = np.array(center)
                self.num_rows = num_rows
                self.cans_per_row = cans_per_row
                self.spacing_x = spacing_x
                self.spacing_y = spacing_y
                self.orientation = orientation

            def generate(self, count_range=None, seed=None):
                items = []
                x_offsets = [(col - (self.num_rows - 1) / 2) * self.spacing_x
                             for col in range(self.num_rows)]
                y_offsets = [(row - (self.cans_per_row - 1) / 2) * self.spacing_y
                             for row in range(self.cans_per_row)]

                for y_idx, dy in enumerate(y_offsets):
                    for x_idx, dx in enumerate(x_offsets):
                        pos = self.center + np.array([dx, dy, 0.0])
                        items.append(ItemSpec(
                            asset_type="soup_can",
                            position=pos,
                            orientation=self.orientation,
                            scale=np.array([1.0, 1.0, 1.0]),
                            name=f"soup_can_r{x_idx}_y{y_idx}",
                        ))
                count = resolve_count(count_range, capacity=len(items), seed=seed)
                if count is not None and count < len(items):
                    items = items[:count]
                return items

        pick_strategy = ConveyorRowsGenerator(
            center=DROPZONE_CENTER_POINT + np.array([0, 0, pick_z]),
            num_rows=num_rows,
            cans_per_row=cans_per_row,
            spacing_x=0.08,
            spacing_y=0.10,
            orientation=default_orientation,
        )

        # --- Target Strategy: 6 hidden markers per box (2x3 grid) ---
        class BoxMarkerGenerator:
            def __init__(self, box_specs, z_floor):
                self.box_specs = box_specs
                self.z_floor = z_floor
                self.marker_scale = np.array([0.05, 0.05, 0.001])
                self.grid_spacing = 0.07  # 7cm between can centers

            def generate(self, count_range=(1, 1), seed=None):
                targets = []
                # Offsets for 2x3 grid (2 cols along X, 3 rows along Y)
                offsets = []
                for row in range(3):
                    for col in range(2):
                        ox = (col - 0.5) * self.grid_spacing
                        oy = (row - 1.0) * self.grid_spacing
                        offsets.append((ox, oy))

                for spec in self.box_specs:
                    center = spec["center"]
                    for ox, oy in offsets:
                        pos = np.array([
                            center[0] + ox,
                            center[1] + oy,
                            self.z_floor,
                        ])
                        targets.append(ItemSpec(
                            asset_type="marker",
                            position=pos,
                            scale=self.marker_scale,
                            hidden=True,
                        ))
                return targets

        target_strategy = BoxMarkerGenerator(
            box_specs=box_specs,
            z_floor=box_floor_z,
        )

        task_ref = self

        def _check_verticality(pick_index, target_index):
            pick_obj = task_ref._pick_objs[pick_index]
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
            task_description="Pick soup cans from the conveyor and place 6 into each of 4 boxes on the cart.",
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            setup_workspace=_workspace_setup,
            placement_constraints_fn=_check_verticality,
            box_verification_info={"box_specs": box_specs},
            containment_check=True,
            scenario={
                "source": "conveyor",
                "destination": "boxes_on_cart",
                "workspace": "two_tables_custom_boxes",
            },
            pick_description={
                "asset_types": ["soup_can"],
                "count": 24,
                "arrangement": "4 rows x 6 positions on conveyor, interleaved column-by-column order",
                "colors": "USD asset default",
                "orientation": "upright (-90° X)",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "2x3 grid per box (2 columns X, 3 rows Y)",
                "count": 24,
                "containers": {"count": 4, "layout": "2x2 grid on cart", "capacity_per_box": 6},
                "virtual": True,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                    "details": "Interleaved pick order fills boxes sequentially: picks 0-5 → box 1, picks 6-11 → box 2, etc.",
                },
            ),
            verification_description={
                "spatial_check": "is_on_top (default)",
                "placement_constraints": "is_vertical",
                "containment_check": True,
            },
            rationale={
                "create_strategy": "Sequential pairing sufficient — pick order is pre-arranged to fill each box in turn without explicit routing",
                "placement_constraints_fn": "Soup cans must remain upright after placement to fit properly in the 2x3 grid",
                "containment_check": "Cans placed inside boxes — box-boundary verification confirms each can is within its target box",
                "virtual_target_generation_strategy": "Hidden markers inside boxes generated at pairing time — target count tied to box geometry, not CLI",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
