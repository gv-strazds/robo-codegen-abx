"""Cortex-tree variant of TableTaskShapeSortBoxes.

Identical to :class:`TableTaskShapeSortBoxes` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_shape_sort_boxes.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_shape_sort_boxes import TableTaskShapeSortBoxes


class TableTaskShapeSortBoxes2(TableTaskShapeSortBoxes):
    DEFAULT_TASK_NAME = "table_task_shape_sort_boxes_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick randomly colored cubes and balls from the conveyor and sort them by shape into two boxes on the cart (cortex-style BT).",
        )
