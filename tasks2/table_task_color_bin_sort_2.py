"""Cortex-tree variant of TableTaskColorBinSort.

Identical to :class:`TableTaskColorBinSort` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_color_bin_sort.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_color_bin_sort import TableTaskColorBinSort


class TableTaskColorBinSort2(TableTaskColorBinSort):
    DEFAULT_TASK_NAME = "table_task_color_bin_sort_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick between one and five cubes and balls from the conveyor and drop them into the color-matching collection boxes (cortex-style BT).",
        )
