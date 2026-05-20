"""Cortex-tree variant of TableTaskColors1.

Identical to :class:`TableTaskColors1` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_colors_1.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_colors_1 import TableTaskColors1


class TableTaskColors2(TableTaskColors1):
    DEFAULT_TASK_NAME = "table_task_colors_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick colored cubes from the bin and place them onto matching colored markers in the dropzone (cortex-style BT).",
        )
