"""Cortex-tree variant of TableTaskConveyorColorStacks.

Identical to :class:`TableTaskConveyorColorStacks` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_conveyor_color_stacks.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_conveyor_color_stacks import TableTaskConveyorColorStacks


class TableTaskConveyorColorStacks2(TableTaskConveyorColorStacks):
    DEFAULT_TASK_NAME = "table_task_conveyor_color_stacks_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick colored cubes from the conveyor and stack them in the bin in triplets: blue (bottom), green (middle), red (top); yellow and excess cubes are skipped (cortex-style BT).",
        )
