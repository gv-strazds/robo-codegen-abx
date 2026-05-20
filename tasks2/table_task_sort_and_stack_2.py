"""Cortex-tree variant of TableTaskSortAndStack.

Identical to :class:`TableTaskSortAndStack` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_sort_and_stack.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_sort_and_stack import TableTaskSortAndStack


class TableTaskSortAndStack2(TableTaskSortAndStack):
    DEFAULT_TASK_NAME = "table_task_sort_and_stack_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick red, green, and blue cubes from a 6x5x3 stacked grid and sort them into matching color-coded boxes on the cart, stacking on previously placed cubes; yellow cubes are relocated to 6 stacks on the dropzone floor (cortex-style BT).",
        )
