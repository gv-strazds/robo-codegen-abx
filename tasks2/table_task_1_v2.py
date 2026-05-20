"""Cortex-tree variant of TableTask1.

Identical to :class:`TableTask1` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_1.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_1 import TableTask1


class TableTask1v2(TableTask1):
    DEFAULT_TASK_NAME = "table_task_1_v2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Sort 16 randomly shuffled colored cubes from the bin into a 4x4 color-striped grid on the dropzone (cortex-style BT).",
        )
