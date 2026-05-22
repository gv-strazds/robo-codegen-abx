import logging
from typing import Optional

import numpy as np
from multi_pick_strategy import ColorMatchStrategy

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTaskColoredBallsToDiscGrid(UR10MultiPickPlaceTask):
    """Pick 6 small balls (randomly red/green/blue) from the bin and place
    each onto a matching-color disc in a 3x3 grid of red/green/blue disc
    markers on the dropzone."""

    DEFAULT_TASK_NAME = "table_task_colored_balls_to_disc_grid"

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
            RandomChoice,
            SequentialChoice,
        )
        from table_setup import (
            BIN_X_COORD,
            BIN_Y_COORD,
            DROPZONE_X,
            DROPZONE_Y,
            DROPZONE_Z,
            ITEM_SPAWN_REFERENCE_Z,
            setup_two_tables,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        # --- Pick strategy: 3x2 grid of small balls in the bin ---
        # scale_xy/z acts as the sphere radius (DynamicSphere default r=1.0);
        # see asset_utils.compute_prim_geometry ball branch (half_ext = scale).
        ball_radius = 0.025
        ball_scale = np.array([ball_radius, ball_radius, ball_radius]) / stage_units
        # Minimal drop margin: 5 cm balls roll more freely than cubes/cans, so
        # we tighten the usual +0.025 hover to +0.005 to limit pile-up at spawn.
        pick_z = ITEM_SPAWN_REFERENCE_Z + ball_radius + 0.005
        pick_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=3,
            cols=2,
            spacing_x=0.09,
            spacing_y=0.10,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("ball"),
            scale_strategy=FixedValue(ball_scale),
            color_strategy=RandomChoice(["red", "green", "blue"]),
        )

        # --- Target strategy: 3x3 grid of static disc markers, cycling R/G/B ---
        # Use "fixed_disc" (FixedCylinder, static) so balls resting on the disc
        # don't trigger contact-resolution jitter (Issue 16).  Per Issue 13,
        # FixedCylinder's scale_xy acts as the radius, not diameter.
        DISC_THICKNESS = 0.03
        DISC_RADIUS = 0.045  # → 9 cm diameter
        dx = -0.12
        dy = 0.13
        grid_rows = 3  # along Y
        grid_cols = 3  # along X
        center_grid_x = DROPZONE_X + (grid_cols - 1) * dx / 2
        center_grid_y = DROPZONE_Y + (grid_rows - 1) * dy / 2
        center_grid_z = DROPZONE_Z + 0.001 + DISC_THICKNESS / 2

        target_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, center_grid_z]),
            rows=grid_rows,
            cols=grid_cols,
            spacing_x=dx,
            spacing_y=dy,
            randomize=False,
        )
        target_scale = (
            np.array([DISC_RADIUS, DISC_RADIUS, DISC_THICKNESS]) / stage_units
        )
        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("fixed_disc"),
            color_strategy=SequentialChoice(
                ["red", "green", "blue"], loop=True
            ),
            scale_strategy=FixedValue(target_scale),
        )

        color_palette = ["red", "green", "blue"]

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick 6 small balls (randomly red/green/blue) from the bin "
                "and place each one onto the matching-color disc in a 3x3 "
                "grid of red/green/blue disc markers on the dropzone."
            ),
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root
            ),
            scenario={
                "source": "bin",
                "destination": "dropzone_grid",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["ball"],
                "count": 6,
                "arrangement": "3x2 grid in the pick bin",
                "colors": "RandomChoice(['red', 'green', 'blue'])",
                "size": "5 cm diameter (small)",
            },
            target_description={
                "type": "visible_markers",
                "asset_type": "fixed_disc",
                "arrangement": (
                    "3x3 grid on the dropzone, colors cycled R/G/B across "
                    "the 9 positions (3 of each color, interleaved)"
                ),
                "count": 9,
                "colors": "red, green, blue (3 of each)",
                "scale": "9 cm diameter, 3 cm thick",
            },
            verification_description={"spatial_check": "is_on_top (default)"},
            rationale={
                "create_strategy": (
                    "Balls must be placed on same-color discs; "
                    "ColorMatchStrategy pairs each ball to a same-color "
                    "disc and filters surplus-color picks out of the "
                    "picking order via initialize_pairings()."
                ),
                "fixed_disc_targets": (
                    "fixed_disc (FixedCylinder, static) avoids the "
                    "contact-resolution jitter of dynamic disc markers "
                    "squeezed between a placed ball and the kinematic "
                    "dropzone (Issue 16)."
                ),
                "disc_scale_as_radius": (
                    "FixedCylinder default radius is 1.0; scale_xy is the "
                    "radius (Issue 13).  scale_xy=0.045 → 9 cm diameter, "
                    "~2 cm margin around the 5 cm ball footprint."
                ),
                "ball_drop_margin": (
                    "Drop margin tightened to +0.005 from the usual "
                    "+0.025 so balls don't roll into a corner at spawn "
                    "(spheres roll more freely than cubes/cans)."
                ),
                "interleaved_disc_colors": (
                    "R/G/B cycled across grid positions (not row-blocked) "
                    "so ColorMatchStrategy must actually search by color "
                    "rather than match by index."
                ),
            },
            implementation=TaskImplementationSpec(
                create_strategy=lambda picks, targets: ColorMatchStrategy(
                    picks, targets, color_palette=color_palette
                ),
                strategy_description={
                    "class": "ColorMatchStrategy",
                    "pairing": "color_match",
                    "details": (
                        "color_palette=['red','green','blue']; each ball "
                        "is placed on the next available same-color "
                        "fixed_disc.  Surplus-color picks are filtered "
                        "from the picking order."
                    ),
                },
            ),
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
