"""Cortex-tree variant of TableTask3.

Identical to :class:`TableTask3` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_3.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_3 import TableTask3


class TableTask3v2(TableTask3):
    DEFAULT_TASK_NAME = "table_task_3_v2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick balls from the bin and place them onto disc targets arranged in a 3x4 grid on the dropzone table (cortex-style BT).",
        )
