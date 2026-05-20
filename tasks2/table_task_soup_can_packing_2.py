"""Cortex-tree variant of TableTaskSoupCanPacking.

Identical to :class:`TableTaskSoupCanPacking` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_soup_can_packing.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_soup_can_packing import TableTaskSoupCanPacking


class TableTaskSoupCanPacking2(TableTaskSoupCanPacking):
    DEFAULT_TASK_NAME = "table_task_soup_can_packing_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick soup cans from the conveyor and place 6 into each of 4 boxes on the cart (cortex-style BT).",
        )
