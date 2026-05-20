"""Cortex-tree variant of TableTaskCrackerCircle.

Identical to :class:`TableTaskCrackerCircle` except that:
- The behaviour tree is the cortex-style ``MotionCommand``-based tree.
- ``place_hover_above_z`` and ``place_approach_distance`` are tightened
  to 0.08 m so the cortex tree's CortexMoveToPlace can reach the top of
  the growing destination stack (the default 0.20 m hover + funnel
  pushes the wrist beyond UR10 reach for the upper items).
- ``startup_delay_seconds`` lets the bin settle in teleport mode before
  the first item is teleported.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_cracker_circle import TableTaskCrackerCircle


class TableTaskCrackerCircle2(TableTaskCrackerCircle):
    DEFAULT_TASK_NAME = "table_task_cracker_circle_2"

    def _customize_spec(self, spec):
        from isaacsim.core.utils.stage import get_stage_units
        return replace(
            spec.with_impl(
                tree_factory=make_cortex_task_controller_tree,
                place_hover_above_z=0.08 / get_stage_units(),
                place_approach_distance=0.08 / get_stage_units(),
                startup_delay_seconds=1.0,
            ),
            task_description="Pick cracker boxes lying flat (longest dimension along X) from a layered circle on the dropzone and stack them into a single column in the bin (cortex-style BT).",
        )
