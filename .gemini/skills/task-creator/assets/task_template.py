import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTaskTemplate(UR10MultiPickPlaceTask):
    """Template for a new UR10 pick-and-place task."""

    def __init__(
        self,
        task_name: str = "table_task_template",
        task_description: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        # Lazily import Isaac utilities to avoid import-order issues
        from isaacsim.core.utils.stage import get_stage_units
        from task_spec import TaskImplementationSpec, TaskSpec
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
            setup_two_tables,
            ITEM_SPAWN_REFERENCE_Z,
        )

        # Default description
        if task_description is None:
            task_description = "Pick objects from the bin and place them onto target markers in the dropzone."

        # --- Generation Strategies ---
        stage_units = get_stage_units()
        # Typical cube scale is ~5cm
        expected_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units

        # Pick Strategy: 2x3 Grid in Bin
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
            asset_type_strategy=FixedValue("cube"),
            scale_strategy=FixedValue(expected_scale),
            color_strategy=RandomChoice(["red", "green", "blue", "yellow"]),
        )

        # Target Strategy: 2x4 Grid in Dropzone
        dx = -0.15
        dy = 0.15
        grid_w = 2
        grid_l = 4

        # Calculate center for grid relative to DROPZONE_X/Y
        center_grid_x = DROPZONE_X + (grid_w - 1) * dx / 2
        center_grid_y = DROPZONE_Y + (grid_l - 1) * dy / 2
        center_grid_z = DROPZONE_Z + 0.001 + expected_scale[2] / 2

        target_pos_gen = GridPositionGenerator(
            center=np.array([center_grid_x, center_grid_y, center_grid_z]),
            rows=grid_l,
            cols=grid_w,
            spacing_x=dx,
            spacing_y=dy,
        )

        target_strategy = ItemGenerator(
            position_generator=target_pos_gen,
            asset_type_strategy=FixedValue("disc"),
            color_strategy=FixedValue("white"),
            scale_strategy=FixedValue(expected_scale),
        )

        # === BUILD TASK SPEC ===
        # Scene-side fields go on TaskSpec; execution-policy fields (strategy
        # factory, BT tree, virtual targets, postures, timeouts, etc.) go on
        # TaskImplementationSpec, assigned via implementation=.
        # For simple marker-based tasks (items onto visible markers/rects/discs):
        spec = TaskSpec(
            task_name=task_name,
            task_description=task_description,
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            # Scene-side metadata
            scenario={
                "source": "bin",               # "bin" | "conveyor" | "cart" | "dropzone"
                "destination": "dropzone_grid", # "dropzone_grid" | "dropzone_circle" | "boxes_on_cart" | etc.
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cube"],
                "count": 6,
                "arrangement": "3x2 grid in pick bin",
                "colors": "RandomChoice(['red', 'green', 'blue', 'yellow'])",
            },
            target_description={
                "type": "visible_markers",
                "arrangement": "2x4 grid on dropzone",
                "count": 8,
            },
            rationale={
                "create_strategy": "TODO: explain why this strategy was chosen",
            },
            implementation=TaskImplementationSpec(
                strategy_description={
                    "class": "MultiPickStrategy",
                    "pairing": "sequential",
                },
            ),
        )

        # For box-packing tasks (items into box containers), use virtual targets
        # (on the implementation spec — they're policy helpers) and centralized
        # box containment verification (outer TaskSpec — verification semantics):
        #
        # spec = TaskSpec(
        #     task_name=task_name,
        #     task_description=task_description,
        #     pick_generation_strategy=pick_strategy,
        #     setup_workspace=_workspace_setup,
        #     box_verification_info={"box_specs": box_specs},
        #     containment_check=True,
        #     # Optional: placement_constraints_fn=_check_verticality,
        #     implementation=TaskImplementationSpec(
        #         virtual_target_generation_strategy=target_strategy,  # hidden markers
        #         # create_strategy=lambda picks, targets: TypeBasedStrategy(...),
        #     ),
        # )
        #
        # Each box_spec needs: name, center_xy, floor_z, inner_size, height,
        # and optionally match_labels (e.g., {"color": "red"}).
        # See TableTaskSoupCanPacking for a complete example.

        super().__init__(task_spec=spec, offset=offset, **kwargs)

    # To use a non-default pairing strategy, pass create_strategy inside the
    # TaskImplementationSpec block. For example:
    #
    # spec = TaskSpec(
    #     ...
    #     implementation=TaskImplementationSpec(
    #         create_strategy=lambda picks, targets: ColorMatchStrategy(
    #             picks, targets, color_palette=["red", "green", "blue"],
    #         ),
    #     ),
    # )
