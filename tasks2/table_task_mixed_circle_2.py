"""Cortex-tree variant of TableTaskMixedCircle.

Identical to :class:`TableTaskMixedCircle` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_mixed_circle.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_mixed_circle import TableTaskMixedCircle


class TableTaskMixedCircle2(TableTaskMixedCircle):
    DEFAULT_TASK_NAME = "table_task_mixed_circle_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick a random mix of primitives and USD assets from a circle on the conveyor and place them into the bin on the cart (cortex-style BT).",
        )
