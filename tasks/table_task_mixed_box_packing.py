import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from multi_pick_strategy import TypeBasedStrategy
from task_verification import is_vertical

logger = logging.getLogger(__name__)

N_CRACKER = 3
N_SOUP = 6
N_BOTTLE = 6
ITEMS_PER_ROW = 3
N_BOXES = 3
ITEMS_PER_BOX = 5  # 1 cracker + 2 soup + 2 bottle


class TableTaskMixedBoxPacking(UR10MultiPickPlaceTask):
    """Pack cracker boxes, soup cans, and mustard bottles from conveyor rows into
    three boxes on the cart (1 cracker + 2 soups + 2 bottles per box, all upright).

    Items are arranged in 6 alternating rows on the stationary conveyor:
    cracker, soup, bottle (repeated twice). Each row has 3 items (18 total).
    Only 15 are picked; 3 extra cracker boxes remain on the conveyor.
    """

    DEFAULT_TASK_NAME = "table_task_mixed_box_packing"

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

        default_orientation = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        )

        cracker_box_height = 0.213
        soup_can_height = 0.102
        mustard_bottle_height = 0.20

        # ------------------------------------------------------------------
        # Box specs on the cart — 3 boxes along Y
        # ------------------------------------------------------------------
        cart_z = CART_SURFACE_CENTER[2]
        box_inner_size = np.array([0.26, 0.32]) / stage_units
        box_height = 0.10 / stage_units
        box_wall = 0.01 / stage_units
        box_base_thickness = 0.01 / stage_units
        box_floor_z = cart_z + box_base_thickness + 0.001
        wall_center_z = cart_z + box_base_thickness / 2 + box_height / 2

        cx, cy = CART_SURFACE_CENTER[0], CART_SURFACE_CENTER[1]
        box_x_offset = 0.18
        box_y_offset = 0.35

        box_specs = [
            {
                "name": f"packing_box_{i}",
                "center": np.array([cx + box_x_offset, cy + dy, wall_center_z]),
                "center_xy": np.array([cx + box_x_offset, cy + dy]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.55, 0.43, 0.33]),
                "z_tol": 0.03,
            }
            for i, dy in enumerate([-box_y_offset, 0.0, box_y_offset])
        ]

        # ------------------------------------------------------------------
        # Pick generator: 6 rows × 3 items on conveyor (alternating types)
        # ------------------------------------------------------------------
        class AlternatingRowsGenerator:
            """Generate items in 6 alternating rows on the stationary conveyor.

            Row order: cracker, soup, bottle, cracker, soup, bottle.
            Returns items ordered so paired items (3 cracker + 6 soup + 6 bottle)
            come before the 3 extra crackers.
            """

            def __init__(self, center_x, center_y, orientation,
                         row_spacing, item_spacing,
                         cracker_z, soup_z, bottle_z):
                self.center_x = center_x
                self.center_y = center_y
                self.orientation = orientation
                self.row_spacing = row_spacing
                self.item_spacing = item_spacing
                self.cracker_z = cracker_z
                self.soup_z = soup_z
                self.bottle_z = bottle_z

            def generate(self, count_range=None, seed=None):
                row_types = [
                    "cracker_box", "soup_can", "mustard_bottle",
                    "cracker_box", "soup_can", "mustard_bottle",
                ]
                type_z = {
                    "cracker_box": self.cracker_z,
                    "soup_can": self.soup_z,
                    "mustard_bottle": self.bottle_z,
                }

                n_rows = len(row_types)
                # Rows separated along Y, items within each row along X
                row_y_start = self.center_y - (n_rows - 1) * self.row_spacing / 2
                item_x_start = self.center_x - (ITEMS_PER_ROW - 1) * self.item_spacing / 2

                rows_items = []
                type_counters = {"cracker_box": 0, "soup_can": 0, "mustard_bottle": 0}
                for row_idx, asset_type in enumerate(row_types):
                    row_y = row_y_start + row_idx * self.row_spacing
                    z = type_z[asset_type]
                    row = []
                    for col_idx in range(ITEMS_PER_ROW):
                        x = item_x_start + col_idx * self.item_spacing
                        idx = type_counters[asset_type]
                        type_counters[asset_type] += 1
                        row.append(ItemSpec(
                            asset_type=asset_type,
                            position=np.array([x, row_y, z]),
                            orientation=self.orientation,
                            name=f"{asset_type}_{idx}",
                        ))
                    rows_items.append((asset_type, row))

                # Order: paired crackers (row 0) first, then all soups,
                # then all bottles, then extra crackers (row 3) last.
                paired_crackers = rows_items[0][1]   # row 0: 3 crackers
                soups_r1 = rows_items[1][1]           # row 1: 3 soups
                bottles_r2 = rows_items[2][1]         # row 2: 3 bottles
                extra_crackers = rows_items[3][1]     # row 3: 3 crackers (extras)
                soups_r4 = rows_items[4][1]           # row 4: 3 soups
                bottles_r5 = rows_items[5][1]         # row 5: 3 bottles

                items = (
                    paired_crackers
                    + soups_r1 + soups_r4
                    + bottles_r2 + bottles_r5
                    + extra_crackers
                )

                count = resolve_count(count_range, capacity=len(items), seed=seed)
                if count is not None and count < len(items):
                    items = items[:count]
                return items

        hover = 0.005
        cracker_spawn_z = DROPZONE_Z + cracker_box_height / 2 + hover
        soup_spawn_z = DROPZONE_Z + soup_can_height / 2 + hover
        bottle_spawn_z = DROPZONE_Z + mustard_bottle_height / 2 + hover

        pick_strategy = AlternatingRowsGenerator(
            center_x=DROPZONE_CENTER_POINT[0],
            center_y=DROPZONE_CENTER_POINT[1],
            orientation=default_orientation,
            row_spacing=0.14 / stage_units,
            item_spacing=0.18 / stage_units,
            cracker_z=cracker_spawn_z,
            soup_z=soup_spawn_z,
            bottle_z=bottle_spawn_z,
        )

        # ------------------------------------------------------------------
        # Virtual targets: 5 hidden markers per box (2D layout)
        # ------------------------------------------------------------------
        class BoxGridMarkerGenerator:
            """Generate 5 hidden markers per box in a 2D layout.

            Per box:
              Y=-0.07: cracker (center X)
              Y=+0.03: soup_0 (X=-0.05), bottle_0 (X=+0.05)
              Y=+0.12: soup_1 (X=-0.05), bottle_1 (X=+0.05)
            """

            MARKER_OFFSETS = [
                (0.0, -0.10),      # cracker
                (-0.06, -0.01),    # soup_0
                (-0.06, 0.09),     # soup_1
                (0.06, -0.01),     # bottle_0
                (0.06, 0.09),      # bottle_1
            ]

            def __init__(self, box_specs):
                self.box_specs = box_specs
                self.marker_scale = np.array([0.05, 0.05, 0.001])

            def generate(self, count_range=None, seed=None):
                targets = []
                for bspec in self.box_specs:
                    bx, by = bspec["center_xy"]
                    z = bspec["floor_z"]
                    for dx, dy in self.MARKER_OFFSETS:
                        targets.append(ItemSpec(
                            asset_type="marker",
                            position=np.array([bx + dx, by + dy, z]),
                            scale=self.marker_scale,
                            hidden=True,
                        ))
                return targets

        target_strategy = BoxGridMarkerGenerator(box_specs=box_specs)

        # ------------------------------------------------------------------
        # Strategy factory: TypeBasedStrategy routing by asset type
        # ------------------------------------------------------------------
        def _create_strategy(picks, targets):
            return TypeBasedStrategy(
                picks,
                targets,
                target_indices_by_type={
                    "cracker_box": [0, 5, 10],
                    "soup_can": [1, 2, 6, 7, 11, 12],
                    "mustard_bottle": [3, 4, 8, 9, 13, 14],
                },
            )

        # ------------------------------------------------------------------
        # Verification: every placed item must be upright
        # ------------------------------------------------------------------
        task_ref = self

        def _placement_constraints_vertical(pick_index, target_index):
            pick_obj = task_ref._pick_objs[pick_index]
            if not is_vertical(pick_obj, max_tilt_deg=15):
                return (False, f"{pick_obj.name} is not vertical (tilt > 15°)")
            return (True, "")

        # ------------------------------------------------------------------
        # Workspace setup
        # ------------------------------------------------------------------
        def _workspace_setup(scene, assets_root):
            setup_two_tables(scene, assets_root, standard_objs=False, add_bin=False)
            for bspec in box_specs:
                spawn_open_box(
                    scene,
                    name=bspec["name"],
                    center=bspec["center"],
                    inner_size=bspec["inner_size"],
                    wall_height=bspec["height"],
                    wall_thickness=box_wall,
                    base_thickness=box_base_thickness,
                    color=bspec["color"],
                )

        # CLI overrides
        pick_count = kwargs.pop("pick_count", None)
        kwargs.pop("target_count", None)

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick cracker boxes, soup cans, and mustard bottles from 6"
                " alternating rows on the stationary conveyor and pack them into"
                " 3 open-top boxes on the cart. Each box receives 1 cracker box,"
                " 2 soup cans, and 2 mustard bottles. All items must remain"
                " upright after placement."
            ),
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            target_count=None,
            setup_workspace=_workspace_setup,
            containment_check=True,
            box_verification_info={"box_specs": box_specs},
            placement_constraints_fn=_placement_constraints_vertical,
            scenario={
                "source": "conveyor",
                "destination": "three_boxes_on_cart",
                "workspace": "two_tables_custom_boxes",
            },
            pick_description={
                "asset_types": ["cracker_box", "soup_can", "mustard_bottle"],
                "count": (
                    f"{N_CRACKER} cracker_box + {N_SOUP} soup_can +"
                    f" {N_BOTTLE} mustard_bottle ({N_CRACKER + N_SOUP + N_BOTTLE}"
                    f" picked of 18 total)"
                ),
                "arrangement": (
                    "6 alternating rows on stationary conveyor"
                    " (cracker, soup, bottle ×2), 3 items per row"
                ),
                "colors": "USD asset default",
                "orientation": "upright (-90° X)",
            },
            target_description={
                "type": "hidden_markers_in_open_boxes",
                "arrangement": "5 markers per box (2D grid: 1 cracker + 2 soup + 2 bottle slots)",
                "count": N_BOXES * ITEMS_PER_BOX,
                "containers": {
                    "count": N_BOXES,
                    "layout": "3 boxes along Y on the cart",
                    "capacity_per_box": ITEMS_PER_BOX,
                },
                "virtual": True,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                create_strategy=_create_strategy,
                ee_height_for_move=0.44 / stage_units,
                strategy_description={
                    "class": "TypeBasedStrategy",
                    "pairing": "type_based_name_prefix",
                    "details": (
                        "cracker_box_* → cracker markers [0,5,10];"
                        " soup_can_* → soup markers [1,2,6,7,11,12];"
                        " mustard_bottle_* → bottle markers [3,4,8,9,13,14]"
                    ),
                },
            ),
            verification_description={
                "containment_check": True,
                "placement_constraints": "is_vertical(max_tilt_deg=15) on every placed item",
            },
            rationale={
                "create_strategy": (
                    "TypeBasedStrategy routes items by name-prefix-derived type to"
                    " per-type marker slots across the 3 boxes"
                ),
                "containment_check": (
                    "Items placed inside open-top boxes — containment verification"
                    " confirms each item is within a box"
                ),
                "placement_constraints_fn": (
                    "Cracker boxes, soup cans, and mustard bottles must remain"
                    " upright after placement"
                ),
                "virtual_target_generation_strategy": (
                    "Hidden markers inside boxes in a 2D layout prevent item"
                    " collisions; one marker per placement slot"
                ),
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
