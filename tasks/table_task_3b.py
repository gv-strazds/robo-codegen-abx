import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTask3b(UR10MultiPickPlaceTask):
    """Pick balls from a bin and place them into gaps between discs in a tight grid on the dropzone.

    The discs are arranged in a 3x4 grid with minimal spacing (almost touching).
    Balls are placed at the midpoint of each 2x2 group of adjacent discs, where
    they nestle stably in the pocket formed by 4 disc rims.
    """

    DEFAULT_TASK_NAME = "table_task_3b"

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
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
            ItemSpec,
            SequentialChoice,
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

        # --- Object geometry ---
        # Ball: DynamicSphere, half_extent = 1.0 * scale in all axes
        ball_radius = expected_scale[2]  # 0.0515m
        # Disc: DynamicCylinder, XY half_extent = 1.0 * scale, Z half_extent = 0.5 * scale
        disc_radius = expected_scale[0]  # 0.0515m
        disc_half_height = expected_scale[2] * 0.5  # 0.02575m

        # --- Disc grid layout (3x4, almost touching) ---
        disc_gap = 0.002  # 2mm gap between disc edges
        disc_spacing = 2 * disc_radius + disc_gap  # ~0.105m center-to-center
        grid_w = 3  # columns (along X)
        grid_l = 4  # rows (along Y)

        # Grid center computation (same convention as TableTask3)
        start_grid_x = DROPZONE_X
        start_grid_y = DROPZONE_Y
        dx = -disc_spacing  # negative = right-to-left
        dy = disc_spacing

        center_grid_x = start_grid_x + (grid_w - 1) * dx / 2
        center_grid_y = start_grid_y + (grid_l - 1) * dy / 2
        disc_center_z = DROPZONE_Z + 0.001 + disc_half_height
        disc_top_z = disc_center_z + disc_half_height

        # --- Pocket geometry (ball resting between 4 discs) ---
        # Distance from gap center to nearest disc rim (XY plane)
        d_contact = disc_spacing * np.sqrt(2) / 2 - disc_radius
        # Height of ball center above disc top surface
        pocket_height = np.sqrt(ball_radius**2 - d_contact**2)
        ball_center_z_pocket = disc_top_z + pocket_height

        # --- Pick items: 3-6 balls from a 3x2 grid in the bin ---
        pick_z = ITEM_SPAWN_REFERENCE_Z + expected_scale[2] / 2 + 0.02
        pick_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=3,
            cols=2,
            spacing_x=0.08,
            spacing_y=0.08,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("ball"),
            scale_strategy=FixedValue(expected_scale),
            color_strategy=None,  # Random
        )

        # --- Compute gap positions (2x3 grid between disc centers) ---
        gap_positions = []
        for row_gap in range(grid_l - 1):  # 3 gap rows between 4 disc rows
            for col_gap in range(grid_w - 1):  # 2 gap cols between 3 disc cols
                gap_x = center_grid_x + (col_gap + 0.5 - (grid_w - 1) / 2) * dx
                gap_y = center_grid_y + (row_gap + 0.5 - (grid_l - 1) / 2) * dy
                gap_positions.append(np.array([gap_x, gap_y]))

        # --- Target markers at gap positions (virtual, hidden) ---
        # Marker Z is set so the computed drop height puts the ball at the pocket.
        # drop_z = marker_z + marker_top_surface + ball_rest_height
        # We want drop_z ~= ball_center_z_pocket
        marker_scale = np.array([0.04, 0.04, 0.001])
        marker_top_surface = 0.5 * marker_scale[2]  # 0.0005m
        # Ball rest_height from precomputed geometry: ~1.003 * scale[2]
        ball_rest_height_approx = 1.003 * expected_scale[2]
        marker_z = ball_center_z_pocket - marker_top_surface - ball_rest_height_approx

        class GapMarkerGenerator:
            """Generate hidden markers at gap positions between disc grid."""

            def __init__(self, positions, z, scale):
                self.positions = positions
                self.z = z
                self.scale = scale

            def generate(self, count_range=None, seed=None):
                items = []
                for i, pos in enumerate(self.positions):
                    items.append(ItemSpec(
                        asset_type="marker",
                        position=np.array([pos[0], pos[1], self.z]),
                        scale=self.scale,
                        hidden=True,
                        name=f"gap_marker_{i}",
                    ))
                return items

        target_strategy = GapMarkerGenerator(gap_positions, marker_z, marker_scale)

        # --- Custom spatial check (position proximity) ---
        # The default is_on_top check is too strict for pocket geometry because
        # physics settling shifts the ball Z relative to the thin marker AABB.
        # Instead, check that the ball center is near the gap center in XY and
        # at approximately the expected pocket height in Z.
        _ball_center_z_pocket = ball_center_z_pocket
        _xy_tol = disc_spacing * 0.4  # ~42mm — generous but won't match wrong gap
        _z_tol = ball_radius  # 51.5mm — ball center within one radius of pocket

        def _spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None):
            """Check if ball center is near the target gap position."""
            pick_pos, _ = pick_obj.get_world_pose()
            target_pos, _ = target_obj.get_world_pose()
            xy_dist = np.linalg.norm(pick_pos[:2] - target_pos[:2])
            z_diff = abs(pick_pos[2] - _ball_center_z_pocket)
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
        _expected_scale = expected_scale

        def _workspace_setup(scene, assets_root):
            setup_two_tables(scene, assets_root)
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
                "Pick balls from the bin and place them into gaps between disc targets "
                "arranged in a tight 3x4 grid on the dropzone table."
            ),
            pick_generation_strategy=pick_strategy,
            pick_count=(3, 6),
            setup_workspace=_workspace_setup,
            spatial_check_fn=_spatial_check,
            scenario={
                "source": "bin",
                "destination": "dropzone_grid_gaps",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["ball"],
                "count": "random(3,6)",
                "arrangement": "3x2 grid in pick bin",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "2x3 grid of gap positions between discs on dropzone",
                "count": 6,
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                },
            ),
            verification_description={
                "spatial_check": "position proximity — ball center near gap center XY and pocket height Z",
            },
            rationale={
                "create_strategy": (
                    "Default sequential pairing — balls placed in gaps between disc targets"
                ),
                "spatial_check_fn": (
                    "Custom proximity check because is_on_top z_tol is too strict "
                    "for pocket geometry where physics settling shifts ball Z"
                ),
                "pick_count": "3-6 balls from a 6-capacity grid for variety across runs",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
