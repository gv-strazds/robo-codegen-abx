"""Cortex-tree variant of TableTaskBottles1.

Identical to :class:`TableTaskBottles1` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_bottles_1.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_bottles_1 import TableTaskBottles1


class TableTaskBottles2(TableTaskBottles1):
    DEFAULT_TASK_NAME = "table_task_bottles_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick bottles from the bin and place them into carrier pads in a 3x4 grid in the dropzone (cortex-style BT).",
        )
