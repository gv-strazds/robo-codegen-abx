import numpy as np
from typing import List, Optional

from multi_pickplace_task import UR10MultiPickPlaceTask
from multi_pick_strategy import LayeredStackStrategy
from item_generation import ItemSpec, ConveyorPositionGenerator, ItemGenerator, FixedValue, RandomChoice
import logging

logger = logging.getLogger(__name__)


class ColorStackStrategy(LayeredStackStrategy):
    """Stack cubes by color in triplets: blue (bottom), green (middle), red (top).

    Thin subclass of ``LayeredStackStrategy`` that builds a color-based
    classify_fn from a ``has_color_fn`` and ``color_palette``.

    Args:
        pick_objs: List of pick objects.
        target_objs: List of target objects (markers for bottom layer).
        max_stacks: Maximum number of stack positions available.
        color_palette: All colors to recognize (stack colors + skip colors).
        has_color_fn: Callable(obj, color_name) -> bool.
        bin_geometry: Optional bin geometry dict for bottom-layer containment checks.
        base_check_fn: Optional callable for bottom-layer spatial verification.
            When provided, takes priority over *bin_geometry*.
    """

    STACK_COLORS = ["blue", "green", "red"]
    SKIP_COLORS = ["yellow"]
    LAYERS_PER_STACK = 3

    def __init__(
        self,
        pick_objs: list,
        target_objs: list,
        max_stacks: int = 3,
        color_palette: Optional[List[str]] = None,
        has_color_fn=None,
        bin_geometry: Optional[dict] = None,
        base_check_fn=None,
    ) -> None:
        # Build a classify_fn from has_color_fn + palette
        palette = color_palette or (self.STACK_COLORS + self.SKIP_COLORS)
        if has_color_fn is None:
            from asset_utils import has_color
            has_color_fn = has_color
        _hc = has_color_fn
        _all = list(palette)

        def _color_classify(obj, _colors=_all, _has=_hc):
            for cname in _colors:
                if _has(obj, cname):
                    return cname
            return None

        super().__init__(
            pick_objs=pick_objs,
            target_objs=target_objs,
            layer_order=self.STACK_COLORS,
            max_stacks=max_stacks,
            classify_fn=_color_classify,
            bin_geometry=bin_geometry,
            base_check_fn=base_check_fn,
        )


class TableTaskConveyorColorStacks(UR10MultiPickPlaceTask):
    """Pick colored cubes from the conveyor and stack them in the bin as (blue, green, red) triplets.

    5-10 cubes spawn on the conveyor in a random mix of red, green, blue, and yellow.
    The robot forms stacks of 3 in the pick bin: blue (bottom), green (middle), red (top).
    Yellow cubes and cubes that cannot form a complete triplet are skipped.
    """

    DEFAULT_TASK_NAME = "table_task_conveyor_color_stacks"

    MAX_STACKS = 3
    COLOR_PALETTE = ["red", "green", "blue", "yellow"]

    def __init__(
        self,
        task_name: Optional[str] = None,
        offset: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        if task_name is None:
            task_name = self.DEFAULT_TASK_NAME
        from isaacsim.core.utils.stage import get_stage_units
        from table_setup import (
            setup_two_tables,
            BIN_X_COORD,
            BIN_Y_COORD,
            BIN_SIZE,
            ITEM_SPAWN_REFERENCE_Z,
            DROPZONE_CENTER_POINT,
        )

        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()

        # --- Cube specifications ---
        cube_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units

        # --- Source: 5-10 cubes on conveyor, randomly R/G/B/Y ---
        spacing = 0.08 / stage_units
        x_position = DROPZONE_CENTER_POINT[0] - 0.06
        z_height = DROPZONE_CENTER_POINT[2] + 0.035

        pick_pos_gen = ConveyorPositionGenerator(
            center_x=x_position,
            center_y=DROPZONE_CENTER_POINT[1],
            z=z_height,
            spacing=spacing,
            jitter_x=0.01 / stage_units,
            jitter_y=0.01 / stage_units,
        )

        pick_strategy = ItemGenerator(
            position_generator=pick_pos_gen,
            asset_type_strategy=FixedValue("cube"),
            scale_strategy=FixedValue(cube_scale),
            color_strategy=RandomChoice(self.COLOR_PALETTE),
        )

        # --- Targets: bottom-layer markers only (upper layers target actual cubes) ---
        max_stacks = self.MAX_STACKS
        color_palette = list(self.COLOR_PALETTE)
        marker_scale = np.array([0.04, 0.04, 0.002]) / stage_units

        stack_spacing_y = 0.10
        stack_center_x = BIN_X_COORD
        stack_center_y = BIN_Y_COORD
        # The bin spawns at ITEM_SPAWN_REFERENCE_Z + 0.05 and settles down to the
        # cart surface (~0.0573) during physics.  Use the settled floor Z
        # with generous tolerances (matching TableTaskMixedCircle).
        bin_floor_z = 0.0573 + 0.005  # cart surface + small lift (approximate)

        bin_geometry = {
            "center_xy": np.array([stack_center_x, stack_center_y]),
            "inner_size": np.array([BIN_SIZE[0], BIN_SIZE[1]]),
            "floor_z": bin_floor_z,
            "height": 0.15,  # generous wall height for containment check
            "z_tol": 0.03,  # generous Z tolerance for physics settling
        }

        # Bottom-layer markers only — upper layers target actual cubes below
        target_items = []
        for stack_idx in range(max_stacks):
            stack_y = stack_center_y + (stack_idx - 1) * stack_spacing_y
            target_pos = np.array([stack_center_x, stack_y, bin_floor_z])
            target_items.append(ItemSpec(
                asset_type="marker",
                position=target_pos,
                color="blue",
                scale=marker_scale,
                hidden=True,
            ))

        class FixedListGenerator:
            def __init__(self, items):
                self.items = items
            def generate(self, count_range=(1, 1), seed=None):
                return self.items

        target_strategy = FixedListGenerator(target_items)

        # Default pick_count to (5, 10) unless overridden by caller
        pick_count = kwargs.pop("pick_count", None)
        if pick_count is None:
            pick_count = (5, 10)

        def _strategy_factory(picks, targets):
            return ColorStackStrategy(
                pick_objs=picks, target_objs=targets,
                max_stacks=max_stacks, color_palette=color_palette,
                bin_geometry=bin_geometry,
            )

        spec = TaskSpec(
            task_name=task_name,
            task_description=(
                "Pick colored cubes from the conveyor and stack them in the bin "
                "in triplets: blue (bottom), green (middle), red (top). "
                "Skip yellow cubes and excess cubes."
            ),
            pick_generation_strategy=pick_strategy,
            pick_count=pick_count,
            setup_workspace=lambda scene, assets_root: setup_two_tables(
                scene, assets_root, standard_objs=False, add_bin=True
            ),
            scenario={
                "source": "conveyor",
                "destination": "bin",
                "workspace": "two_tables",
            },
            pick_description={
                "asset_types": ["cube"],
                "count": "random (5-10)",
                "count_rationale": "Randomized count (5-10) tests variable workload; yellow cubes are intentional distractors",
                "arrangement": "conveyor line",
                "colors": "RandomChoice(['red', 'green', 'blue', 'yellow'])",
            },
            target_description={
                "type": "hidden_markers",
                "arrangement": "3 bottom-layer markers in bin (one per stack position)",
                "count": 3,
                "virtual": True,
                "virtual_rationale": "Only bottom-layer markers are pre-placed; upper layer targets are the actual cubes already stacked below",
            },
            implementation=TaskImplementationSpec(
                virtual_target_generation_strategy=target_strategy,
                create_strategy=_strategy_factory,
                strategy_description={
                    "class": "ColorStackStrategy",
                    "pairing": "stacking",
                    "details": "layer_order=['blue', 'green', 'red']; max_stacks=3; yellow cubes skipped",
                },
            ),
            rationale={
                "create_strategy": "Cubes must be stacked in color triplets (blue bottom, green middle, red top) — ColorStackStrategy classifies by color and assigns layer positions within each stack",
                "virtual_target_generation_strategy": "Only bottom-layer markers are pre-placed; upper layer targets are the actual cubes already stacked below",
                "pick_count": "Randomized count (5-10) tests variable workload; yellow cubes are intentional distractors",
            },
        )

        spec = self._customize_spec(spec)

        super().__init__(task_spec=spec, offset=offset, **kwargs)
