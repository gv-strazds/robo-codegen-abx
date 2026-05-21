import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTaskDiscsGridToYellowCircle(UR10MultiPickPlaceTask):
    """Pick discs from a 2x4 grid in the bin and place them onto a circle
    of 8 thin yellow rectangular markers on the dropzone."""

    DEFAULT_TASK_NAME = "table_task_discs_grid_to_yellow_circle"

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
            CircularPositionGenerator,
            FixedValue,
            GridPositionGenerator,
            ItemGenerator,
        )
        from table_setup import (
            BIN_X_COORD,
            BIN_Y_COORD,
            DROPZONE_CENTER_POINT,
            DROPZONE_Z,
            ITEM_SPAWN_REFERENCE_Z,
            setup_two_tables,
        )
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        # --- Pick strategy: 2x4 grid of dynamic discs in the bin ---
        # Disc native local_half_extents=[1.0, 1.0, 0.5]; uniform scale 0.045 →
        # diameter ≈ 0.09 m, thickness ≈ 0.045 m. Discs spawn Z-up natively, no
        # orientation override needed.
        disc_scale = np.array([0.045, 0.045, 0.045]) / stage_units
        pick_z = ITEM_SPAWN_REFERENCE_Z + disc_scale[2] / 2 + 0.025

        pick_pos_gen = GridPositionGenerator(
            center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
            rows=4,
            cols=2,
            spacing_x=0.10,
            spacing_y=0.095,
        )
        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("disc"),
            scale_strategy=FixedValue(disc_scale),
            color_strategy=None,  # default per-instance coloring (varies per disc)
        )

        # --- Target strategy: 8 thin yellow square markers in a circle ---
        # Square is slightly larger than the ~9 cm disc diameter so each disc
        # fits within the marker footprint with a small margin.
        RECT_THICKNESS = 0.002
        marker_scale = np.array([0.10, 0.10, RECT_THICKNESS]) / stage_units
        marker_z = DROPZONE_Z + 0.001 + RECT_THICKNESS / 2

        target_pos_gen = CircularPositionGenerator(
            center=np.array(
                [DROPZONE_CENTER_POINT[0], DROPZONE_CENTER_POINT[1], marker_z]
            ),
            radius=0.20,
            count=8,
            randomize=False,
        )
        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("marker"),
            color_strategy=FixedValue("yellow"),
            scale_strategy=FixedValue(marker_scale),
        )

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick discs from a 2x4 grid in the bin and place them onto "
                "a circle of 8 thin yellow square markers on the dropzone."
            ),
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root
            ),
            scenario={
                "source": "bin",
                "destination": "dropzone_circle",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["disc"],
                "count": 8,
                "arrangement": "2x4 grid (2 cols x 4 rows along Y) in pick bin",
                "colors": "default per-instance coloring (color_strategy=None; not specified by task)",
                "orientation": "native (Z-up, flat)",
            },
            target_description={
                "type": "visible_markers",
                "asset_type": "marker",
                "arrangement": "circle (r=0.20 m, 8 positions, evenly spaced) on dropzone",
                "count": 8,
                "colors": "yellow",
                "scale": "10 cm x 10 cm x 2 mm (square, slightly larger than disc diameter)",
            },
            verification_description={"spatial_check": "is_on_top (default)"},
            rationale={
                "create_strategy": (
                    "Default sequential pairing — picks and targets are all "
                    "identical and counts match 1:1, so no custom pairing is needed."
                ),
                "spatial_check_fn": (
                    "Default is_on_top is sufficient. Discs have continuous "
                    "Z-axis symmetry (see ITEMS_MAP['disc']) so no orientation "
                    "constraint applies after placement."
                ),
                "pick_arrangement": (
                    "2x4 fits in the bin's 0.270 x 0.393 m inner cavity with "
                    "margin (X footprint ~0.19 m, Y footprint ~0.375 m)."
                ),
                "target_arrangement": (
                    "Radius 0.20 m gives ~0.157 m center-to-center spacing for "
                    "8 markers — plenty of room for a 9 cm disc on each."
                ),
            },
            implementation=TaskImplementationSpec(
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                },
            ),
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
