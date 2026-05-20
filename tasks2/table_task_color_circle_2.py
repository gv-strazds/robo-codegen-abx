"""Cortex-tree variant of TableTaskColorCircle.

Identical to :class:`TableTaskColorCircle` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_color_circle.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_color_circle import TableTaskColorCircle


class TableTaskColorCircle2(TableTaskColorCircle):
    DEFAULT_TASK_NAME = "table_task_color_circle_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick randomly colored cubes from the bin and place them in a circle on the drop zone (cortex-style BT).",
        )
