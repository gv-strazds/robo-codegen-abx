import logging
from typing import Optional

import numpy as np

from multi_pickplace_task import UR10MultiPickPlaceTask

logger = logging.getLogger(__name__)


class TableTaskTemplate(UR10MultiPickPlaceTask):
    """<One-line description of the task.>"""

    def __init__(
        self,
        task_name: str = "table_task_template",
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

        # --- Generation Strategies ---
        stage_units = get_stage_units()

        # === PICK STRATEGY ===
        # For primitives (cube, ball, cylinder, cone):
        #   expected_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units
        #   pick_z = ITEM_SPAWN_REFERENCE_Z + expected_scale[2] / 2 + 0.02
        #
        # For USD assets (soup_can, cracker_box, madara_bottle, etc.):
        #   Use -90 deg X orientation so the item stands upright:
        #   from isaacsim.core.utils import rotations
        #   from pxr import Gf
        #   default_orientation = rotations.gf_rotation_to_np_array(
        #       Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
        #   )
        #   pick_z = ITEM_SPAWN_REFERENCE_Z + 0.025 + <half_height>

        expected_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units
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
            color_strategy=RandomChoice(["red", "green", "blue"]),
        )

        # === TARGET STRATEGY ===
        # Targets can be markers, rects, discs, pads, or custom boxes.
        # For grid targets in the dropzone:
        dx = -0.15
        dy = 0.15
        grid_w = 2
        grid_l = 4
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
        # Scene/description-side fields live on TaskSpec.  Execution-policy
        # fields (strategy factory, BT tree, virtual targets, postures,
        # timeouts, cuRobo flags, ...) live on TaskImplementationSpec, nested
        # under `implementation=`.  Default sequential pairing doesn't need
        # an implementation spec at all — omit it entirely (or just set
        # ``strategy_description`` if you want metadata).
        spec = TaskSpec(
            task_name=task_name,
            task_description="TODO: Replace with concise task description.",
            pick_generation_strategy=pick_strategy,
            target_generation_strategy=target_strategy,
            setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
            # Human-readable metadata (documentation, not executable config)
            scenario={
                "source": "bin",               # "bin" | "conveyor" | "cart" | "dropzone"
                "destination": "dropzone_grid", # "dropzone_grid" | "dropzone_circle" | "boxes_on_cart" | etc.
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cube"],
                "count": 6,
                "arrangement": "3x2 grid in pick bin",
                "colors": "RandomChoice(['red', 'green', 'blue'])",
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

        super().__init__(task_spec=spec, offset=offset, **kwargs)

    # --- Optional overrides ---

    # To use a non-default pairing strategy, pass create_strategy inside
    # the TaskImplementationSpec block. For example:
    #
    # spec = TaskSpec(
    #     ...
    #     implementation=TaskImplementationSpec(
    #         create_strategy=lambda picks, targets: ColorMatchStrategy(
    #             picks, targets, color_palette=["red", "green", "blue"],
    #         ),
    #         strategy_description={"class": "ColorMatchStrategy", "pairing": "color_match"},
    #     ),
    # )

    # --- Custom spatial verification ---
    # Use spatial_check_fn directly on the TaskSpec (verification semantics
    # are description-side, not policy):
    #
    # from task_verification import is_on_top, is_vertical
    #
    # def _spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None):
    #     return (
    #         is_on_top(pick_obj, target_obj, bb_cache=bb_cache, obj_scale=obj_scale)
    #         and is_vertical(pick_obj, bb_cache=bb_cache, obj_scale=obj_scale)
    #     )
    #
    # spec = TaskSpec(
    #     ...
    #     spatial_check_fn=_spatial_check,
    # )

    # --- Box containment verification (use for tasks placing items into boxes/bins) ---
    # This is the DEFAULT for container tasks. No overrides needed — just set
    # ``box_verification_info`` and ``containment_check=True`` on the TaskSpec
    # itself. The base class uses build_box_verification_hooks() automatically.
    #
    # Each box spec needs: name, center_xy, floor_z, inner_size, height,
    # and optionally match_labels (e.g., {"color": "red"}).
    #
    # For box-packing tasks, use virtual_target_generation_strategy (on the
    # implementation spec) instead of target_generation_strategy.  Virtual
    # targets are policy helpers — generated at pairing time, never spawned
    # as USD prims — so they live inside TaskImplementationSpec.
    #
    # Example TaskSpec for a box-packing task:
    #
    # box_specs = [
    #     {
    #         "name": "red_box",
    #         "center_xy": np.array([x, y]),
    #         "floor_z": box_floor_z,
    #         "inner_size": box_inner_size,
    #         "height": box_height,
    #         "match_labels": {"color": "red"},  # optional
    #     },
    # ]
    #
    # spec = TaskSpec(
    #     ...
    #     pick_generation_strategy=pick_strategy,
    #     box_verification_info={"box_specs": box_specs},
    #     containment_check=True,
    #     # Optional: placement_constraints_fn=_check_verticality,
    #     implementation=TaskImplementationSpec(
    #         virtual_target_generation_strategy=target_strategy,  # hidden markers
    #         # create_strategy=lambda picks, targets: TypeBasedStrategy(...),
    #     ),
    # )
    #
    # For upright orientation constraints, add a placement_constraints_fn:
    #
    # def _check_verticality(pick_index, target_index):
    #     pick_obj = task_ref._pick_objs[pick_index]
    #     if not is_vertical(pick_obj, bb_cache=task_ref._get_bb_cache()):
    #         return (False, "item is not vertical")
    #     return (True, "")
    #
    # See TableTaskSoupCanPacking for a complete box-packing example.


# ===========================================================================
# Alternative: Custom Generator Pattern (for complex layouts)
# ===========================================================================
# For tasks with complex pick layouts (interleaved orders, multi-type mixes,
# per-box marker grids), define a custom generator class instead of using
# ItemGenerator. Example:
#
# from item_generation import ItemSpec, resolve_count
#
# class MyCustomPickGenerator:
#     """Generate items with custom layout logic."""
#     def __init__(self, center, orientation, ...):
#         self.center = center
#         self.orientation = orientation
#
#     def generate(self, count_range=None, seed=None):
#         items = []
#         # ... custom position/type/color logic ...
#         items.append(ItemSpec(
#             asset_type="soup_can",
#             position=np.array([x, y, z]),
#             orientation=self.orientation,
#             scale=np.array([1.0, 1.0, 1.0]),
#             name=f"soup_can_{i}",
#         ))
#         # Support CLI --pick-count override:
#         count = resolve_count(count_range, capacity=len(items), seed=seed)
#         if count is not None and count < len(items):
#             items = items[:count]
#         return items
#
# See TableTaskSoupCanPacking, TableTaskColorShapes, TableTaskMixedPacking
# for complete examples of custom generators.
