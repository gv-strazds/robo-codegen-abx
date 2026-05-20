"""Cortex-tree variant of TableTask3c.

Identical to :class:`TableTask3c` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_3c.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_3c import TableTask3c


class TableTask3c2(TableTask3c):
    DEFAULT_TASK_NAME = "table_task_3c_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick balls from the bin into disc-gap pockets, then place red cart balls into gaps between the placed balls (cortex-style BT).",
        )
