"""Experiment variant of TableTaskBottlesToConveyor2.

Identical to the parent task except that the cortex tree's descent node
(``CortexDownToInsert``) is replaced by ``LowerToPlaceFromFKBehaviour``
(time-interpolated, FK-anchored).  Kept as a sibling class so the
unmodified baseline (``TableTaskBottlesToConveyor2``) remains runnable
for side-by-side comparison.

See ``docs/lower-to-place-in-cortex-tree.md`` for context.
"""
from dataclasses import replace

from tasks2.table_task_bottles_to_conveyor_2 import TableTaskBottlesToConveyor2
from robot_controllers.pt_cortex_lowertoplace_experiment import (
    make_cortex_with_lowertoplace_tree,
)


class TableTaskBottlesToConveyor2x(TableTaskBottlesToConveyor2):
    """Cortex BT with the descent swapped for LowerToPlaceFromFKBehaviour."""

    def __init__(
        self,
        task_name: str = "table_task_bottles_to_conveyor_2x",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(task_name=task_name, *args, **kwargs)
        # tree_factory is consumed in UR10MultiPickPlaceTask.post_reset(),
        # which runs after __init__, so swapping it here takes effect.
        self._task_spec = replace(
            self._task_spec.with_impl(tree_factory=make_cortex_with_lowertoplace_tree),
            task_description="Pick bottles from 2 stacked layers in the bin and place them into carrier pads in a row on the conveyor (cortex BT with FK-anchored LowerToPlace descent experiment).",
        )
