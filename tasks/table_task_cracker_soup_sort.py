import logging
import random
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask
from multi_pick_strategy import TypeBasedStrategy
from task_verification import is_vertical

logger = logging.getLogger(__name__)


class TableTaskCrackerSoupSort(UR10MultiPickPlaceTask):
    """Sort cracker boxes and soup cans by type into two same-size boxes on the cart.

    4 cracker_box + 4 soup_can items are randomly interleaved in a single line on
    the stationary conveyor. Cracker boxes are placed into the LEFT box on the
    cart, soup cans into the RIGHT box. Both items must remain upright (vertical)
    after placement; both boxes are identical in size.
    """

    DEFAULT_TASK_NAME = "table_task_cracker_soup_sort"

    N_CRACKER = 4
    N_SOUP = 4

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
        from item_generation import ItemSpec, ConveyorPositionGenerator, resolve_count
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

        n_cracker = self.N_CRACKER
        n_soup = self.N_SOUP

        # Default upright orientation for YCB cracker_box and soup_can — -90° X
        # rotation puts each asset's local up-axis at world +Z.
        default_orientation = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        )

        # World-frame upright heights (post -90° X rotation).
        cracker_box_height = 0.213
        soup_can_height = 0.102

        # --- Box specs on the cart (BOTH SAME SIZE per user requirement) ---
        cart_z = CART_SURFACE_CENTER[2]
        box_inner_size = np.array([0.20, 0.40]) / stage_units
        box_height = 0.10 / stage_units
        box_wall = 0.01 / stage_units
        box_base_thickness = 0.01 / stage_units
        box_floor_z = cart_z + box_base_thickness + 0.001
        wall_center_z = cart_z + box_base_thickness / 2 + box_height / 2

        cx, cy = CART_SURFACE_CENTER[0], CART_SURFACE_CENTER[1]
        box_x_offset = 0.18

        box_specs = [
            {
                "name": "cracker_box_target",
                "center": np.array([cx - box_x_offset, cy, wall_center_z]),
                "center_xy": np.array([cx - box_x_offset, cy]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.50, 0.40, 0.30]),  # brown
                "match_labels": {"type": "cracker_box"},
                "z_tol": 0.03,
            },
            {
                "name": "soup_can_target",
                "center": np.array([cx + box_x_offset, cy, wall_center_z]),
                "center_xy": np.array([cx + box_x_offset, cy]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": np.array([0.65, 0.55, 0.40]),  # tan
                "match_labels": {"type": "soup_can"},
                "z_tol": 0.03,
            },
        ]

        # --- Pick generator: 4 cracker_box + 4 soup_can, shuffled, single conveyor line ---
        class InterleavedConveyorGenerator:
            """Generate 4 cracker_box + 4 soup_can items in random interleaved order on the conveyor."""

            def __init__(
                self,
                n_cracker,
                n_soup,
                center_x,
                center_y,
                spacing,
                jitter_x,
                jitter_y,
                orientation,
                cracker_z,
                soup_z,
            ):
                self.n_cracker = n_cracker
                self.n_soup = n_soup
                self.pos_gen = ConveyorPositionGenerator(
                    center_x=center_x,
                    center_y=center_y,
                    z=0.0,  # placeholder; replaced per item by type-specific Z below
                    spacing=spacing,
                    jitter_x=jitter_x,
                    jitter_y=jitter_y,
                )
                self.orientation = orientation
                self.cracker_z = cracker_z
                self.soup_z = soup_z

            def generate(self, count_range=None, seed=None):
                rng = random.Random(seed)
                entries = (
                    [{"type": "cracker_box"} for _ in range(self.n_cracker)]
                    + [{"type": "soup_can"} for _ in range(self.n_soup)]
                )
                rng.shuffle(entries)

                count = resolve_count(count_range, capacity=len(entries), seed=seed)
                if count is not None and count < len(entries):
                    entries = entries[:count]

                positions = self.pos_gen.get_positions(len(entries), seed)

                items = []
                cracker_i = 0
                soup_i = 0
                for i, entry in enumerate(entries):
                    p = positions[i].copy()
                    if entry["type"] == "cracker_box":
                        p[2] = self.cracker_z
                        name = f"cracker_box_{cracker_i}"
                        cracker_i += 1
                    else:
                        p[2] = self.soup_z
                        name = f"soup_can_{soup_i}"
                        soup_i += 1
                    items.append(
                        ItemSpec(
                            asset_type=entry["type"],
                            position=p,
                            orientation=self.orientation,
                            color=None,
                            name=name,
                        )
                    )
                return items

        # Per-type spawn Z: item center at half-height above the conveyor surface,
        # plus a small hover so items settle cleanly under gravity.
        hover = 0.005
        cracker_spawn_z = DROPZONE_Z + cracker_box_height / 2 + hover
        soup_spawn_z = DROPZONE_Z + soup_can_height / 2 + hover

        pick_strategy = InterleavedConveyorGenerator(
            n_cracker=n_cracker,
            n_soup=n_soup,
            center_x=DROPZONE_CENTER_POINT[0],
            center_y=DROPZONE_CENTER_POINT[1],
            spacing=0.10 / stage_units,
            jitter_x=0.01 / stage_units,
            jitter_y=0.005 / stage_units,
            orientation=default_orientation,
            cracker_z=cracker_spawn_z,
            soup_z=soup_spawn_z,
        )

        # --- Virtual targets: 4 hidden markers per box, single line along Y ---
        class BoxLineMarkerGenerator:
            """Generate `count_per_box` hidden markers per box, in a line along Y."""

            def __init__(self, box_specs, count_per_box, spacing):
                self.box_specs = box_specs
                self.count_per_box = count_per_box
                self.spacing = spacing
                self.marker_scale = np.array([0.05, 0.05, 0.001])

            def generate(self, count_range=None, seed=None):
                all_box_markers = []
                for bspec in self.box_specs:
                    cx_box, cy_box = bspec["center_xy"]
                    z = bspec["floor_z"]
                    n = self.count_per_box
                    start_dy = -(n - 1) * self.spacing / 2
                    markers = []
                    for i in range(n):
                        markers.append(
                            ItemSpec(
                                asset_type="marker",
                                position=np.array(
                                    [cx_box, cy_box + start_dy + i * self.spacing, z]
                                ),
                                scale=self.marker_scale,
                                hidden=True,
                            )
                        )
                    all_box_markers.append(markers)

                total_capacity = sum(len(m) for m in all_box_markers)
                count = resolve_count(count_range, capacity=total_capacity, seed=seed)

                if count is not None and count < total_capacity:
                    n_boxes = len(all_box_markers)
                    per_box = count // n_boxes
                    remainder = count % n_boxes
                    per_box_counts = [
                        per_box + (1 if i < remainder else 0) for i in range(n_boxes)
                    ]
                else:
                    per_box_counts = [len(m) for m in all_box_markers]

                targets = []
                for i, markers in enumerate(all_box_markers):
                    targets.extend(markers[: per_box_counts[i]])
                return targets

        target_strategy = BoxLineMarkerGenerator(
            box_specs=box_specs,
            count_per_box=n_cracker,  # equals n_soup; both boxes have same slot count
            spacing=0.09,
        )

        # --- Strategy factory: TypeBasedStrategy with name-prefix auto-detection ---
        def _create_strategy(picks, targets):
            return TypeBasedStrategy(
                picks,
                targets,
                target_indices_by_type={
                    "cracker_box": list(range(0, n_cracker)),
                    "soup_can": list(range(n_cracker, n_cracker + n_soup)),
                },
            )

        # --- Placement constraint: every placed item must be upright ---
        task_ref = self

        def _placement_constraints_vertical(pick_index, target_index):
            pick_obj = task_ref._pick_objs[pick_index]
            if not is_vertical(pick_obj, max_tilt_deg=15):
                return (False, f"{pick_obj.name} is not vertical (tilt > 15°)")
            return (True, "")

        # --- Workspace setup ---
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
        kwargs.pop("target_count", None)  # not meaningful for box-packing targets

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick 4 cracker boxes and 4 soup cans that are randomly interleaved in a"
                " single line on the stationary conveyor, and sort them by type into two"
                " same-size open-top boxes on the cart: cracker boxes into the left box,"
                " soup cans into the right box. Every item must remain upright after"
                " placement."
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
                "destination": "two_boxes_on_cart",
                "workspace": "two_tables_custom_boxes",
            },
            pick_description={
                "asset_types": ["cracker_box", "soup_can"],
                "count": f"{n_cracker} cracker_box + {n_soup} soup_can ({n_cracker + n_soup} total)",
                "arrangement": "randomly interleaved single line on the stationary conveyor",
                "colors": "USD asset default",
                "orientation": "upright (-90° X)",
            },
            target_description={
                "type": "hidden_markers_in_open_boxes",
                "arrangement": f"1x{n_cracker} line per box",
                "count": n_cracker + n_soup,
                "containers": {
                    "count": 2,
                    "layout": "two same-size open-top boxes side-by-side on the cart (left/right along X)",
                    "capacity_per_box": n_cracker,
                    "same_size": True,
                },
                "virtual": True,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                create_strategy=_create_strategy,
                strategy_description={
                    "class": "TypeBasedStrategy",
                    "pairing": "type_based_name_prefix",
                    "details": "cracker_box_* → left-box markers [0..3]; soup_can_* → right-box markers [4..7]",
                },
            ),
            verification_description={
                "containment_check": True,
                "match_labels": "type-based (cracker_box → left box; soup_can → right box)",
                "placement_constraints": "is_vertical(max_tilt_deg=15) on every placed item",
            },
            rationale={
                "create_strategy": (
                    "TypeBasedStrategy routes items by name-prefix-derived type regardless"
                    " of the random interleaved pick order on the conveyor"
                ),
                "containment_check": (
                    "Items placed inside open-top boxes — match_labels enforces that the"
                    " correct type ends up in each box"
                ),
                "placement_constraints_fn": (
                    "Both cracker boxes and soup cans must remain upright after placement"
                    " (vertical orientation required by the task)"
                ),
                "virtual_target_generation_strategy": (
                    "One hidden marker per placement slot keeps placements spread across"
                    " distinct positions in each box (see learnings Issue 8)"
                ),
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
