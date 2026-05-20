"""Cortex-tree variant of TableTask4.

Identical to :class:`TableTask4` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_4.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_4 import TableTask4


class TableTask4v2(TableTask4):
    DEFAULT_TASK_NAME = "table_task_4_v2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick cubes from the bin and place them onto yellow rectangles arranged in a circle on the dropzone table (cortex-style BT).",
        )
