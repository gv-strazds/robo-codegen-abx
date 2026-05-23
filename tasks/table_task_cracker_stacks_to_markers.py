import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


# Source layout: 2 cols X × 3 rows Y × 3 layers, 18 cracker boxes total.
# Each source layer holds 6 picks (indices [layer*6 .. layer*6 + 5]); within
# a layer the 6 picks are split into two row-pairs that get mapped to two
# adjacent destination layers (3 picks each, one per stack).
_SOURCE_LAYERS = 3
_PICKS_PER_LAYER = 6
_NUM_STACKS = 3
_LAYERS_PER_STACK = (_SOURCE_LAYERS * _PICKS_PER_LAYER) // _NUM_STACKS  # 6


def _classify_pick_to_dest_layer(obj) -> Optional[str]:
    """Map a pick obj (named "cracker_box_<idx>") to a destination layer label.

    The mapping ensures stack-by-stack pick order (which LayeredStackStrategy
    enforces in initialize_pairings) is also top-of-source first, so the
    pre-stacked source constraint is satisfied at every step:

      source layer 2 (top, idx 12..17)  → dest lvl_0, lvl_1
      source layer 1 (mid, idx  6..11)  → dest lvl_2, lvl_3
      source layer 0 (bot, idx  0.. 5)  → dest lvl_4, lvl_5
    """
    try:
        idx = int(obj.name.rsplit("_", 1)[-1])
    except (ValueError, AttributeError):
        return None
    source_layer = idx // _PICKS_PER_LAYER
    within = idx % _PICKS_PER_LAYER
    sub_layer = within // _NUM_STACKS
    dest_layer = (_SOURCE_LAYERS - 1 - source_layer) * 2 + sub_layer
    return f"lvl_{dest_layer}"


def _cracker_horizontal_on_top(
    pick_obj, target_obj, bb_cache=None, obj_scale=None, log_failure=False,
):
    """Verify a horizontally-lying cracker box rests on its paired target.

    Used for both base-layer picks (cracker box on green rect marker) and
    upper-layer picks (cracker box on cracker box). Tolerances are generous
    because tall stacks of YCB cracker boxes lean ~10-15° per layer (the
    asset's surfaces aren't perfectly flat), which inflates each box's AABB
    Z-extent. The inflated AABB tops/bottoms make a strict ``is_on_top``
    z-check spuriously fail even when the stack is physically intact.
    """
    from task_verification import is_horizontal, is_on_top

    on_top = is_on_top(
        pick_obj, target_obj, bb_cache=bb_cache, obj_scale=obj_scale,
        z_tol=0.08, log_failure=log_failure,
    )
    if not on_top:
        return False
    return is_horizontal(
        pick_obj, obj_scale=obj_scale, max_tilt_deg=30.0, log_failure=log_failure,
    )


class TableTaskCrackerStacksToMarkers(UR10MultiPickPlaceTask):
    """Unstack 18 horizontal cracker boxes from a 3-layer 2x3 dropzone footprint
    and restack them as three 6-high horizontal stacks on three green-square
    markers in a row on the cart.

    - Sources (dropzone): LayeredPositionGenerator wrapping a 2x3 grid,
      3 layers at layer_height=0.074. Cracker boxes lie flat (identity orientation).
    - Targets (cart): 3 visible green rect markers in a row along Y.
    - Strategy: LayeredStackStrategy with max_stacks=3 and a spawn-index
      classify_fn that maps source-top → destination-bottom, so the
      stack-by-stack picking order also satisfies the source-stacking
      constraint enforced by stacking_enabled=True.
    """

    DEFAULT_TASK_NAME = "table_task_cracker_stacks_to_markers"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME

        from isaacsim.core.utils.stage import get_stage_units
        from env_config_values import CART_SURFACE_CENTER
        from item_generation import (
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
            LayeredPositionGenerator,
        )
        from table_setup import (
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            setup_two_tables,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        # Pick / target counts are fixed for this task: 18 boxes, 3 markers.
        kwargs["pick_count"] = 18
        kwargs["target_count"] = 3

        stage_units = get_stage_units()

        # --- Cracker box geometry (lying face up, identity orientation) ---
        # Asset AABB (from asset_prim_geometry.json): half_extents
        # = [0.082, 0.107, 0.0359] → full dims 0.164 (X) × 0.213 (Y) × 0.072 (Z).
        box_z_thickness = 0.0718  # 2 * 0.0359
        layer_height = box_z_thickness + 0.002  # 2 mm margin to avoid spawn interpenetration

        # === PICK STRATEGY ===
        # 2x3 grid (2 cols X × 3 rows Y) on the dropzone, stacked 3 layers high.
        # spacing_y is tight (5 mm gap between adjacent boxes' edges); the row
        # span (0.649 m) slightly exceeds the dropzone friction region (0.62 m),
        # which is acceptable — items rest on the picking table surface that
        # extends past the dropzone.
        base_pick_z = DROPZONE_Z + 0.002 + box_z_thickness / 2
        base_pick_gen = GridPositionGenerator(
            center=np.array([DROPZONE_CENTER_POINT[0], DROPZONE_CENTER_POINT[1], base_pick_z]),
            rows=3,
            cols=2,
            spacing_x=0.20,
            spacing_y=0.218,
            randomize=False,
        )
        pick_pos_gen = LayeredPositionGenerator(
            base_generator=base_pick_gen,
            num_layers=_SOURCE_LAYERS,
            layer_height=layer_height,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("cracker_box"),
            scale_strategy=FixedValue(np.array([1.0, 1.0, 1.0])),
        )

        # === TARGET STRATEGY ===
        # 3 visible green rect markers in a row along Y on the cart surface.
        # rect → FixedCuboid (static, no jitter — per Issue 16). Marker is
        # slightly larger than the horizontal cracker box footprint so the
        # stack site is clearly visible from above.
        #
        # Position the row near the cart's +X edge (the side closest to the
        # robot) so the UR10 can reach all three stacks comfortably — the
        # default centered placement put the far stack near the kinematic
        # envelope and full-sim runs struggled to reach it.
        #
        # Negative ``spacing_y`` reverses the marker order so target_0 sits
        # at the largest +Y, target_1 in the middle, target_2 at the smallest
        # +Y. LayeredStackStrategy fills stacks in target order, so the
        # furthest stack is built first (when the dropzone is densest and
        # the robot's path to it is least obstructed) and the nearest stack
        # last.
        cart_x, cart_y, cart_z = CART_SURFACE_CENTER
        marker_scale = np.array([0.22, 0.17, 0.005]) / stage_units
        marker_z = cart_z + 0.003 + marker_scale[2] / 2
        marker_x_offset = 0.22  # shift toward +X edge of cart (cart half-width ≈ 0.35 m)
        marker_spacing_y = -0.30  # reverse: target_0 → largest +Y
        target_pos_gen = GridPositionGenerator(
            center=np.array([cart_x + marker_x_offset, cart_y, marker_z]),
            rows=3,
            cols=1,
            spacing_x=0.0,
            spacing_y=marker_spacing_y,
            randomize=False,
        )
        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("rect"),
            color_strategy=FixedValue("green"),
            scale_strategy=FixedValue(marker_scale),
        )

        # === STRATEGY FACTORY ===
        def _strategy_factory(picks, targets):
            from multi_pick_strategy import LayeredStackStrategy
            layer_order = [f"lvl_{k}" for k in range(_LAYERS_PER_STACK)]
            return LayeredStackStrategy(
                pick_objs=picks,
                target_objs=targets,
                layer_order=layer_order,
                max_stacks=_NUM_STACKS,
                classify_fn=_classify_pick_to_dest_layer,
            )

        # EE transport height: clear the tallest stack (6 boxes ≈ 0.43 m above
        # marker top ≈ 0.49 m world Z) + carried box height (≈ 0.07 m) +
        # safety margin. Reference: TableTaskLayeredCircle uses 0.27 m for a
        # 2-stack of 0.045 m sugar boxes — this task's stacks are ~3× taller.
        ee_height_for_move = 0.65 / stage_units

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Unstack 18 cracker boxes from a 3-layer 2x3 footprint on the dropzone "
                "and restack them (lying face up) as three 6-high stacks on three green "
                "square markers in a row on the cart."
            ),
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            pick_count=18,
            target_count=3,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root, standard_objs=False, add_bin=False,
            ),
            stacking_enabled=True,
            spatial_check_fn=_cracker_horizontal_on_top,
            scenario={
                "source": "dropzone",
                "destination": "cart_markers",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cracker_box"],
                "count": 18,
                "arrangement": "2x3 grid on dropzone, stacked 3 layers high (layer_height=0.074m)",
                "colors": "USD asset default (red)",
                "orientation": "horizontal (face up, identity rotation)",
            },
            target_description={
                "type": "visible_markers",
                "asset_types": ["rect"],
                "arrangement": (
                    "3 green squares in a row along Y near the +X edge of the cart "
                    "(0.30 m Y spacing); ordered target_0 → largest +Y, target_2 → "
                    "smallest +Y, so the furthest stack is built first."
                ),
                "count": 3,
                "colors": "green",
            },
            verification_description={
                "spatial_check": "is_on_top (z_tol=0.08) AND is_horizontal (max_tilt=30deg) — generous to absorb stack lean from YCB cracker box geometry",
            },
            implementation=TaskImplementationSpec(
                create_strategy=_strategy_factory,
                ee_height_for_move=ee_height_for_move,
                startup_delay_seconds=1.0,
                strategy_description={
                    "class": "LayeredStackStrategy",
                    "pairing": "stacking",
                    "details": (
                        "layer_order=['lvl_0'..'lvl_5']; max_stacks=3; classify by spawn "
                        "index so top-of-source maps to bottom-of-destination, satisfying "
                        "both source unstacking and destination bottom-up stacking."
                    ),
                },
            ),
            rationale={
                "stacking_enabled": (
                    "Source items pre-stacked 3 layers high — top-down pick order required "
                    "to avoid colliding with overlying boxes."
                ),
                "create_strategy": (
                    "Distributes 18 picks across 3 destination stacks (6 layers each) via "
                    "LayeredStackStrategy with a spawn-index-derived classify_fn that aligns "
                    "destination layer ordering with source unstacking order."
                ),
                "ee_height_for_move": (
                    "0.65 m must clear the tallest stack (~0.49 m world Z when full) plus "
                    "carried box (~0.07 m) plus margin."
                ),
                "target_generation_strategy": (
                    "User explicitly named visible appearance (green-square markers), so "
                    "spawn visible rect targets rather than hidden virtual markers (Issue 15)."
                ),
                "setup_workspace": (
                    "standard_objs=False, add_bin=False keep the cart empty so the default "
                    "YCB props and KLT bin don't collide with our 3-marker stack layout "
                    "(Issue 14)."
                ),
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
