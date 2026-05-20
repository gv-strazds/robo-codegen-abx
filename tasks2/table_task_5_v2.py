"""Cortex-tree variant of TableTask5.

Identical to :class:`TableTask5` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_5.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_5 import TableTask5


class TableTask5v2(TableTask5):
    DEFAULT_TASK_NAME = "table_task_5_v2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick green cubes from the dropzone table and place them onto red rectangles arranged in a circle (cortex-style BT).",
        )
