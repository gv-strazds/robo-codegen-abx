"""Cortex-tree variant of TableTask2.

Identical to :class:`TableTask2` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_2.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_2 import TableTask2


class TableTask2v2(TableTask2):
    DEFAULT_TASK_NAME = "table_task_2_v2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick multiple cubes and place them on blue cubes arranged in a 3x4 grid (cortex-style BT).",
        )
