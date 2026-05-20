import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTask3c(UR10MultiPickPlaceTask):
    """Pick balls into disc gaps, then place red cart balls into gaps between placed balls.

    Extends TableTask3b: after placing 3-6 balls from the bin into disc-gap pockets
    in a tight 3x4 disc grid, 0-2 red balls from the cart surface are placed into
    gaps between the placed balls. Red ball count depends on bin ball count:
    4+ bin balls → 1 red ball, 6 bin balls → 2 red balls.
    """

    DEFAULT_TASK_NAME = "table_task_3c"

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        from isaacsim.core.utils.stage import get_stage_units
        from item_generation import (
            GridPositionGenerator,
            ItemSpec,
            resolve_count,
        )
        from table_setup import (
            BIN_X_COORD,
            BIN_Y_COORD,
            DROPZONE_X,
            DROPZONE_Y,
            DROPZONE_Z,
            setup_two_tables,
            ITEM_SPAWN_REFERENCE_Z,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()
        expected_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units

        # --- Object geometry (same as 3b) ---
        ball_radius = expected_scale[2]  # 0.0515m
        disc_radius = expected_scale[0]  # 0.0515m
        disc_half_height = expected_scale[2] * 0.5  # 0.02575m

        # --- Disc grid layout (3x4, almost touching) ---
        disc_gap = 0.002  # 2mm gap between disc edges
        disc_spacing = 2 * disc_radius + disc_gap  # ~0.105m center-to-center
        grid_w = 3  # columns (along X)
        grid_l = 4  # rows (along Y)

        start_grid_x = DROPZONE_X
        start_grid_y = DROPZONE_Y
        dx = -disc_spacing  # negative = right-to-left
        dy = disc_spacing

        center_grid_x = start_grid_x + (grid_w - 1) * dx / 2
        center_grid_y = start_grid_y + (grid_l - 1) * dy / 2
        disc_center_z = DROPZONE_Z + 0.001 + disc_half_height
        disc_top_z = disc_center_z + disc_half_height

        # --- Pocket geometry: ball resting between 4 discs ---
        d_contact = disc_spacing * np.sqrt(2) / 2 - disc_radius
        pocket_height = np.sqrt(ball_radius**2 - d_contact**2)
        ball_center_z_pocket = disc_top_z + pocket_height

        # --- Pocket geometry: red ball resting on 4 placed balls ---
        # Distance from ball-gap center to each supporting ball center (diagonal half)
        d_horiz = disc_spacing * np.sqrt(2) / 2
        # Vertical offset: sphere-sphere contact distance
        dz_ball_on_balls = np.sqrt((2 * ball_radius)**2 - d_horiz**2)
        red_ball_center_z = ball_center_z_pocket + dz_ball_on_balls

        # --- Bin ball positions (3x2 grid in bin) ---
        pick_z = ITEM_SPAWN_REFERENCE_Z + expected_scale[2] / 2 + 0.02
        bin_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=3,
            cols=2,
            spacing_x=0.08,
            spacing_y=0.08,
        )

        # --- Red ball positions (on cart surface, outside bin) ---
        cart_ball_z = ITEM_SPAWN_REFERENCE_Z + ball_radius + 0.001
        cart_ball_positions = [
            np.array([BIN_X_COORD + 0.07, BIN_Y_COORD - 0.35, cart_ball_z]),
            np.array([BIN_X_COORD - 0.08, BIN_Y_COORD - 0.35, cart_ball_z]),
        ]

        # --- Disc-gap positions (2x3 grid between disc centers, same as 3b) ---
        gap_positions = []
        for row_gap in range(grid_l - 1):  # 3 gap rows between 4 disc rows
            for col_gap in range(grid_w - 1):  # 2 gap cols between 3 disc cols
                gap_x = center_grid_x + (col_gap + 0.5 - (grid_w - 1) / 2) * dx
                gap_y = center_grid_y + (row_gap + 0.5 - (grid_l - 1) / 2) * dy
                gap_positions.append(np.array([gap_x, gap_y]))

        # --- Ball-gap positions (center of 2x2 placed-ball groups) ---
        # Group A: balls 0,1,2,3 → center at disc (col=1, row=1)
        # Group B: balls 2,3,4,5 → center at disc (col=1, row=2)
        ball_gap_positions = [
            np.array([center_grid_x, center_grid_y - 0.5 * dy]),
            np.array([center_grid_x, center_grid_y + 0.5 * dy]),
        ]

        # --- Marker Z computations ---
        marker_scale = np.array([0.04, 0.04, 0.001])
        marker_top_surface = 0.5 * marker_scale[2]
        ball_rest_height_approx = 1.003 * expected_scale[2]
        disc_gap_marker_z = ball_center_z_pocket - marker_top_surface - ball_rest_height_approx
        ball_gap_marker_z = red_ball_center_z - marker_top_surface - ball_rest_height_approx

        # --- Custom pick generator: bin balls + always 2 red cart balls ---
        _expected_scale = expected_scale
        _cart_ball_positions = cart_ball_positions

        class _PickGenerator:
            """Generate bin balls (3-6) + always 2 red cart balls.

            Red balls are always spawned on the cart surface. The target
            generator creates ball-gap targets only for the red balls that
            should be picked (0, 1, or 2 based on bin ball count). Unpaired
            red balls remain on the cart — the BT stops when it encounters
            a pick with no target.
            """

            def __init__(self, pos_gen):
                self._bin_pos_gen = pos_gen
                self.n_bin_balls = None
                self.n_red_balls = None

            def generate(self, count_range=None, seed=None):
                count = resolve_count(count_range, capacity=6, seed=seed)
                if count is None:
                    count = 6
                self.n_bin_balls = min(count, 6)

                if self.n_bin_balls >= 6:
                    self.n_red_balls = 2
                elif self.n_bin_balls >= 4:
                    self.n_red_balls = 1
                else:
                    self.n_red_balls = 0

                bin_positions = self._bin_pos_gen.get_positions(6, seed=seed)
                items = []
                for i in range(self.n_bin_balls):
                    items.append(ItemSpec(
                        asset_type="ball",
                        position=bin_positions[i],
                        scale=_expected_scale,
                        name=f"bin_ball_{i}",
                    ))
                # Always spawn both red balls on the cart surface
                for j in range(2):
                    items.append(ItemSpec(
                        asset_type="ball",
                        position=_cart_ball_positions[j],
                        scale=_expected_scale,
                        color="red",
                        name=f"red_ball_{j}",
                    ))
                return items

        pick_gen = _PickGenerator(bin_pos_gen)

        # --- Custom virtual target generator: disc gaps + ball gaps ---
        _gap_positions = gap_positions
        _ball_gap_positions = ball_gap_positions
        _disc_gap_marker_z = disc_gap_marker_z
        _ball_gap_marker_z = ball_gap_marker_z
        _marker_scale = marker_scale

        class _TargetGenerator:
            """Generate disc-gap markers + ball-gap markers based on pick generator state."""

            def __init__(self, pick_gen_ref):
                self._pick_gen = pick_gen_ref

            def generate(self, count_range=None, seed=None):
                items = []
                for i in range(self._pick_gen.n_bin_balls):
                    pos = _gap_positions[i]
                    items.append(ItemSpec(
                        asset_type="marker",
                        position=np.array([pos[0], pos[1], _disc_gap_marker_z]),
                        scale=_marker_scale,
                        hidden=True,
                        name=f"gap_marker_{i}",
                    ))
                for j in range(self._pick_gen.n_red_balls):
                    pos = _ball_gap_positions[j]
                    items.append(ItemSpec(
                        asset_type="marker",
                        position=np.array([pos[0], pos[1], _ball_gap_marker_z]),
                        scale=_marker_scale,
                        hidden=True,
                        name=f"ball_gap_marker_{j}",
                    ))
                return items

        target_gen = _TargetGenerator(pick_gen)

        # --- Custom spatial check (two Z levels) ---
        _ball_center_z_pocket = ball_center_z_pocket
        _red_ball_center_z = red_ball_center_z
        _xy_tol = disc_spacing * 0.4  # ~42mm — generous but won't match wrong gap
        _z_tol = ball_radius  # 51.5mm — ball center within one radius of expected

        def _spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None):
            """Check ball center proximity to target gap position."""
            pick_pos, _ = pick_obj.get_world_pose()
            target_pos, _ = target_obj.get_world_pose()
            target_name = getattr(target_obj, 'name', '')
            xy_dist = np.linalg.norm(pick_pos[:2] - target_pos[:2])
            if 'ball_gap_marker' in target_name:
                expected_z = _red_ball_center_z
            else:
                expected_z = _ball_center_z_pocket
            z_diff = abs(pick_pos[2] - expected_z)
            return xy_dist < _xy_tol and z_diff < _z_tol

        # --- Disc grid positions for workspace setup ---
        disc_colors = ["purple", "cyan", "black", "yellow"]
        disc_positions = []
        for row in range(grid_l):
            for col in range(grid_w):
                disc_x = center_grid_x + (col - (grid_w - 1) / 2) * dx
                disc_y = center_grid_y + (row - (grid_l - 1) / 2) * dy
                disc_positions.append({
                    "x": disc_x,
                    "y": disc_y,
                    "color": disc_colors[(row * grid_w + col) % len(disc_colors)],
                })

        _disc_center_z = disc_center_z

        def _workspace_setup(scene, assets_root):
            setup_two_tables(scene, assets_root, standard_objs=False)
            from asset_utils import add_prim_asset
            for i, dp in enumerate(disc_positions):
                add_prim_asset(
                    scene,
                    asset_type="disc",
                    obj_name=f"disc_{i}",
                    position=np.array([dp["x"], dp["y"], _disc_center_z]),
                    scale=_expected_scale,
                    color=dp["color"],
                )

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick balls from the bin into disc-gap pockets, then place red balls "
                "from the cart into gaps between the placed balls."
            ),
            pick_generation_strategy=pick_gen,
            pick_count=(3, 6),
            setup_workspace=_workspace_setup,
            spatial_check_fn=_spatial_check,
            scenario={
                "source": "bin_and_cart",
                "destination": "dropzone_grid_gaps",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["ball"],
                "count": "3-6 bin balls (random) + 0-2 red cart balls (conditional)",
                "arrangement": "bin: 3x2 grid; cart: 2 positions outside bin",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "disc-gap positions (2x3) + ball-gap positions (up to 2)",
                "count": "matches total pick count (3-8)",
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_gen,
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                    "details": "bin balls paired to disc gaps, red balls paired to ball gaps",
                },
            ),
            verification_description={
                "spatial_check": (
                    "position proximity — XY near gap center, Z near expected "
                    "pocket height (disc-pocket or ball-pocket level)"
                ),
            },
            rationale={
                "create_strategy": (
                    "Default sequential pairing — bin balls fill disc gaps first, "
                    "then red balls fill gaps between placed balls"
                ),
                "spatial_check_fn": (
                    "Custom proximity check with two Z levels: disc-pocket for bin balls, "
                    "ball-pocket for red balls on top of placed balls"
                ),
                "pick_count": (
                    "3-6 bin balls randomly chosen; 0-2 red balls conditionally added "
                    "based on whether 4+ or 6 bin balls were generated"
                ),
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
