"""Cortex-tree variant of TableTaskLayeredCircle.

Identical to :class:`TableTaskLayeredCircle` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_layered_circle.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_layered_circle import TableTaskLayeredCircle


class TableTaskLayeredCircle2(TableTaskLayeredCircle):
    DEFAULT_TASK_NAME = "table_task_layered_circle_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick sugar boxes lying flat from a layered circle on the dropzone and stack them into a single column in the bin (cortex-style BT).",
        )
