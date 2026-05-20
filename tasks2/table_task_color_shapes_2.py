"""Cortex-tree variant of TableTaskColorShapes.

Identical to :class:`TableTaskColorShapes` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_color_shapes.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_color_shapes import TableTaskColorShapes


class TableTaskColorShapes2(TableTaskColorShapes):
    DEFAULT_TASK_NAME = "table_task_color_shapes_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick cubes, cylinders, cones, and balls from the conveyor and place them into the matching colored boxes on the table (cortex-style BT).",
        )
