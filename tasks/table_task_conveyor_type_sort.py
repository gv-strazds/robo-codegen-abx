import logging
import random
from typing import List, Optional

import numpy as np

from multi_pick_strategy import TypeBasedStrategy
from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


# Order matters: indices into box_specs / box_target stand-ins.
_TYPE_ORDER = ["cracker_box", "sugar_box", "mustard_bottle"]


class TableTaskConveyorTypeSort(UR10MultiPickPlaceTask):
    """Sort items arriving on a moving conveyor into type-specific cart boxes.

    Items of three types (cracker_box, sugar_box, mustard_bottle) are spawned
    one at a time at the far end of the drop zone.  The conveyor runs at 1.5x
    the default belt speed and the spawn interval is 4.0 s so successive items
    are ~8 cm apart.  The robot must place each item into its type-matching
    open-top box on the cart; verification uses per-box containment plus a
    verticality check.
    """

    DEFAULT_TASK_NAME = "table_task_conveyor_type_sort"

    MIN_TOTAL = 5
    MAX_TOTAL = 15
    MAX_PER_TYPE = 5

    def pre_step(self, time_step_index: int, simulation_time: float) -> None:
        super().pre_step(time_step_index=time_step_index, simulation_time=simulation_time)
        # Kick-start each newly-spawned pick with a linear velocity matching
        # the belt.  Items (notably the YCB mustard_bottle with its concave,
        # rim-only contact when upright) can otherwise settle into a
        # marginal-contact state on the thin kinematic conveyor surface
        # where PhysX's surface velocity never couples through friction.
        # Seeding the velocity avoids that transient.
        try:
            from isaacsim.core.prims import SingleRigidPrim
        except Exception:
            return
        belt_v = self._task_spec.conveyor_speed if self._task_spec else None
        if not belt_v:
            return
        if not hasattr(self, "_belt_velocity_initialized"):
            self._belt_velocity_initialized: set = set()
        picks = getattr(self, "_pick_objs", None) or []
        velocity = np.array([0.0, belt_v, 0.0])
        for prim in picks:
            name = getattr(prim, "name", None)
            if name is None or name in self._belt_velocity_initialized:
                continue
            try:
                SingleRigidPrim(prim_path=prim.prim_path).set_linear_velocity(velocity)
                self._belt_velocity_initialized.add(name)
            except Exception:
                # Rigid body view may not be ready on the very first tick for
                # items spawned during set_up_scene — we'll retry next tick.
                pass

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
        from pxr import Gf

        from asset_data_utils import lookup_prim_geometry
        from env_config_values import (
            CART_SURFACE_CENTER,
            DEFAULT_CONVEYOR_SPEED,
            DROPZONE_CENTER_POINT,
            DROPZONE_Y,
            DROPZONE_Z,
        )
        from item_generation import (
            IncrementalGenerationConfig,
            ItemSpec,
            resolve_count,
        )
        from table_setup import setup_two_tables, spawn_open_box
        from task_spec import TaskImplementationSpec, TaskSpec
        from task_verification import is_vertical

        stage_units = get_stage_units()

        # Increase the default belt speed so 4 s spawn interval ≈ 8 cm spacing.
        conveyor_speed = 1.0 * DEFAULT_CONVEYOR_SPEED

        # Upright orientation for all USD assets (Y-up in local frame → Z-up in world).
        upright_orientation = rotations.gf_rotation_to_np_array(
            Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        )

        # --- Box specs on the cart ------------------------------------------------
        # Each box's long (0.40 m) axis runs along cart Y so items dropped in a
        # row along Y fit naturally; boxes are laid out side-by-side along
        # cart X (the cart's short 0.70 m axis).
        cart_z = CART_SURFACE_CENTER[2]
        box_inner_size = np.array([0.18, 0.40]) / stage_units
        box_height = 0.05 / stage_units
        box_wall = 0.01 / stage_units
        box_base_thickness = 0.01 / stage_units
        box_floor_z = cart_z + box_base_thickness + 0.001

        cx, cy = CART_SURFACE_CENTER[0], CART_SURFACE_CENTER[1]
        # Pull the boxes 12 cm toward the robot along Y. Without the shift, the 5th
        # marker slot of the cracker box (at cx-0.22, cy+0.16) sits ~1.31 m from the
        # robot base in 3D — outside the 1.25 m UR10 working radius once the EE is at
        # ee_height_for_move (z≈0.57). The shift drops that to ~1.23 m (22 mm margin)
        # while leaving the boxes inside the cart's 1.09 m Y footprint.
        cy = cy - 0.12
        box_x_offsets = [-0.22, 0.0, 0.22]
        # Cardboard-ish tones per type: tan / pale buff / pale yellow.
        box_colors = [
            np.array([0.78, 0.62, 0.40]),   # cracker_box
            np.array([0.85, 0.78, 0.60]),   # sugar_box
            np.array([0.90, 0.82, 0.35]),   # mustard_bottle
        ]

        box_specs = []
        for asset_type, dx, color in zip(_TYPE_ORDER, box_x_offsets, box_colors):
            box_specs.append({
                "name": f"box_{asset_type}",
                "center": np.array([cx + dx, cy, cart_z + box_height / 2]),
                "center_xy": np.array([cx + dx, cy]),
                "floor_z": box_floor_z,
                "inner_size": box_inner_size,
                "height": box_height,
                "color": color,
                "match_labels": {"type": asset_type},
            })

        # --- Pick generator: random sequence of items arriving on the conveyor ----
        # Spawn far down the conveyor (high-Y). User requested the spawn point
        # be roughly twice as far from the drop zone centerline (along Y) as
        # the drop zone's own half-depth: 2 × (centerline → DROPZONE_Y far edge).
        dropzone_half_depth = DROPZONE_CENTER_POINT[1] - DROPZONE_Y
        far_y = DROPZONE_CENTER_POINT[1] + 2.0 * dropzone_half_depth
        centerline_x = DROPZONE_CENTER_POINT[0]
        jitter_x = 0.02  # ±2 cm per user spec
        min_total = self.MIN_TOTAL
        max_total = self.MAX_TOTAL
        max_per_type = self.MAX_PER_TYPE

        class ConveyorTypeSpawnGenerator:
            """Generate the full sequence of items ahead-of-time for the
            ``IncrementalItemScheduler`` to release one by one.

            Constraint: no type exceeds ``max_per_type`` so every item fits
            inside its corresponding box.
            """

            def __init__(self):
                self.source_types: List[str] = []

            def generate(self, count_range=None, seed=None):
                rng = random.Random(seed)
                capacity = max_per_type * len(_TYPE_ORDER)
                if count_range is None:
                    total = rng.randint(min_total, min(max_total, capacity))
                else:
                    total = resolve_count(count_range, capacity=capacity, seed=seed)
                    if total is None:
                        total = rng.randint(min_total, min(max_total, capacity))
                total = max(1, min(total, capacity))

                # Uniform-per-type draw with hard cap of max_per_type each.
                pool: List[str] = []
                for t in _TYPE_ORDER:
                    pool.extend([t] * max_per_type)
                rng.shuffle(pool)
                picked = pool[:total]

                self.source_types = list(picked)

                # Per-type rest height for correct spawn Z (upright orientation).
                rest_by_type = {}
                for t in _TYPE_ORDER:
                    geom = lookup_prim_geometry(t, orientation=upright_orientation)
                    rest_by_type[t] = geom.rest_height if geom is not None else 0.05

                items = []
                for i, t in enumerate(picked):
                    dx = rng.uniform(-jitter_x, jitter_x)
                    # Spawn with bottom at DROPZONE_Z (no hover). Avoids an
                    # initial drop that can leave curved-base items like the
                    # YCB mustard_bottle balanced on an outer rim with near-
                    # zero belt contact, preventing PhysX surface velocity
                    # from transferring.
                    pos = np.array([
                        (centerline_x + dx) / stage_units,
                        far_y / stage_units,
                        (DROPZONE_Z + rest_by_type[t]) / stage_units,
                    ])
                    items.append(ItemSpec(
                        asset_type=t,
                        position=pos,
                        orientation=upright_orientation,
                        scale=None,
                        color=None,
                        name=f"{t}_{i}",
                    ))
                return items

        pick_strategy = ConveyorTypeSpawnGenerator()

        # --- Virtual target generator: fixed 5-slot rows inside each box ---------
        # Always emit MAX_PER_TYPE markers per box in a row along the box's
        # long (Y) axis, sized for the widest item (cracker_box narrow face,
        # 0.072 m).  Per-item spawn counts are not consulted: the strategy
        # uses only as many slots as there are picks of each type.
        marker_scale = np.array([0.03, 0.03, 0.001])
        slots_per_box = self.MAX_PER_TYPE

        class BoxRowMarkerGenerator:
            """Emit a fixed row of hidden markers inside each type's box.

            Five evenly-spaced slots per box regardless of actual item
            counts, sized to fit five cracker boxes (the widest item).
            Markers are ordered cracker → sugar → mustard to align with
            the strategy's per-type iteration.
            """

            def generate(self, count_range=None, seed=None):
                specs = []
                for i, atype in enumerate(_TYPE_ORDER):
                    box_spec = box_specs[i]
                    cx_b, cy_b = box_spec["center_xy"]
                    inner_y = box_spec["inner_size"][1]
                    step = inner_y / slots_per_box
                    start = cy_b - inner_y / 2 + step / 2
                    for j in range(slots_per_box):
                        y = start + j * step
                        specs.append(ItemSpec(
                            asset_type="marker",
                            position=np.array([cx_b, y, box_spec["floor_z"]]),
                            scale=marker_scale,
                            hidden=True,
                            name=f"marker_{atype}_{j}",
                        ))
                return specs

        target_strategy = BoxRowMarkerGenerator()

        # --- Strategy factory -----------------------------------------------------
        def _create_strategy(picks, targets):
            # Group target indices by type using the marker name prefix.
            # Markers were generated in type order; names are "marker_<type>_<j>".
            indices_by_type: dict = {t: [] for t in _TYPE_ORDER}
            for j, tgt in enumerate(targets):
                name = getattr(tgt, "name", "")
                for t in _TYPE_ORDER:
                    if name.startswith(f"marker_{t}_"):
                        indices_by_type[t].append(j)
                        break
            return TypeBasedStrategy(
                picks, targets,
                target_indices_by_type=indices_by_type,
            )

        # --- Workspace setup ------------------------------------------------------
        def _workspace_setup(scene, assets_root):
            setup_two_tables(
                scene, assets_root,
                standard_objs=False, add_bin=False,
                conveyor_speed=conveyor_speed,
            )
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

        # --- Verification: vertical check composed with box containment -----------
        def _vertical_check(pick_obj, bb_cache=None, obj_scale=None):
            return is_vertical(pick_obj, obj_scale=obj_scale, max_tilt_deg=15)

        # --- Incremental spawn config --------------------------------------------
        inc_config = IncrementalGenerationConfig(
            items_per_batch=1,
            batch_interval=6.0, #4.0,
            bt_start_threshold=1,
        )

        # CLI overrides
        pick_count = kwargs.pop("pick_count", None)
        target_count = kwargs.pop("target_count", None)

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Sort items arriving one by one on a moving conveyor into the"
                " cart-top open-top box that matches each item's type"
                " (cracker_box, sugar_box, or mustard_bottle)."
            ),
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            target_count=target_count,
            setup_workspace=_workspace_setup,
            conveyor_speed=conveyor_speed,
            pick_incremental_config=inc_config,
            containment_check=True,
            box_verification_info={
                "box_specs": box_specs,
                "extra_pick_check": _vertical_check,
            },
            scenario={
                "source": "conveyor",
                "destination": "type_sorted_boxes_on_cart",
                "workspace": "two_tables_custom_boxes",
            },
            pick_description={
                "asset_types": list(_TYPE_ORDER),
                "count": f"random {min_total}-{max_total} total, ≤ {max_per_type} per type",
                "arrangement": "sequential spawn at far-Y end of conveyor drop zone",
                "colors": "USD asset default",
                "orientation": "upright (-90° X)",
                "incremental": (
                    "1 item every 4.0 s at 1.5 default belt speed; BT starts"
                    " once the first item is released"
                ),
            },
            target_description={
                "type": "open_boxes_on_cart_with_row_markers",
                "arrangement": (
                    "three boxes in a row along cart-X; each box has a fixed"
                    " 5-slot row of hidden markers evenly spaced along its"
                    " long Y axis (sized for the widest item, cracker_box)"
                ),
                "count": 15,
                "containers": {
                    "count": 3,
                    "layout": "side by side along cart X",
                    "capacity_per_box": 5,
                    "inner_size": [0.18, 0.40],
                    "wall_height": 0.05,
                },
                "virtual": True,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                create_strategy=_create_strategy,
                # Reachability gate (consumed by CheckPickReachable in the
                # cortex BT): defer items that have dropped 10 cm below the
                # conveyor surface; rely on the default UR10_WORKING_RADIUS
                # for the radial check so the BT idles instead of chasing
                # items spawned at the far end of the conveyor.
                pick_min_reachable_z=DROPZONE_Z - 0.10,
                ee_height_for_move=0.45,
                strategy_description={
                    "class": "TypeBasedStrategy",
                    "pairing": "type_based",
                    "details": (
                        "Each pick is routed to the stand-in target corresponding"
                        " to its asset type; multi-pick pairing to the same target"
                        " is supported via containment_check=True"
                    ),
                },
            ),
            verification_description={
                "containment_check": True,
                "match_labels": "type-based (cracker_box / sugar_box / mustard_bottle)",
                "extra_pick_check": "is_vertical (max_tilt_deg=15)",
            },
            rationale={
                "create_strategy": (
                    "TypeBasedStrategy routes items by type label without"
                    " needing per-item marker positions — containment mode"
                    " handles multi-occupancy per box"
                ),
                "containment_check": (
                    "Items are sorted into boxes — box-geometry containment"
                    " verifies each pick landed in its type-matched box"
                ),
                "box_verification_info.extra_pick_check": (
                    "User requirement: placed items must remain upright"
                    " (≤ 15° tilt) inside the box"
                ),
                "pick_incremental_config": (
                    "Simulates items arriving on a production line one at a"
                    " time; 4 s interval × 1.5× default belt speed gives the"
                    " requested ~8 spacing along Y"
                ),
                "conveyor_speed": (
                    "Increased from DEFAULT_CONVEYOR_SPEED so the belt carries"
                    " items toward the robot fast enough to keep spacing"
                    " reasonable at the 4 s spawn interval"
                ),
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
