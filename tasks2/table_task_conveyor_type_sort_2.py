"""Cortex-tree variant of TableTaskConveyorTypeSort.

Identical to :class:`TableTaskConveyorTypeSort` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_conveyor_type_sort.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_conveyor_type_sort import TableTaskConveyorTypeSort


class TableTaskConveyorTypeSort2(TableTaskConveyorTypeSort):
    DEFAULT_TASK_NAME = "table_task_conveyor_type_sort_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(
                tree_factory=make_cortex_task_controller_tree,
                # 7 mm vs the 5 mm default: RMPFlow has a small steady-state lag tracking the
                # moving belt, so CortexExecuteApproach asymptotes at the threshold and dwells
                # for seconds before wait_for_grip closes the gripper — by which time items
                # have drifted out of reach.
                pick_approach_p_thresh=0.007,
            ),
            task_description="Pick items arriving one at a time on a moving conveyor and sort them into the cart-top box that corresponds to each item's type (cortex-style BT).",
        )
