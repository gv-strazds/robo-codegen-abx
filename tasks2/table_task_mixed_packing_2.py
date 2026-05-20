"""Cortex-tree variant of TableTaskMixedPacking.

Identical to :class:`TableTaskMixedPacking` except that:
- The behaviour tree is the cortex-style ``MotionCommand``-based tree.
- ``ee_height_for_move`` is raised to 0.35 m so the carried tall items
  (mustard bottles, cracker boxes) clear the loaded box walls during
  transport between picks.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_mixed_packing import TableTaskMixedPacking


class TableTaskMixedPacking2(TableTaskMixedPacking):
    DEFAULT_TASK_NAME = "table_task_mixed_packing_2"

    def _customize_spec(self, spec):
        from isaacsim.core.utils.stage import get_stage_units
        return replace(
            spec.with_impl(
                tree_factory=make_cortex_task_controller_tree,
                ee_height_for_move=0.35 / get_stage_units(),
            ),
            task_description="Pick Cracker Boxes and Soup Cans from the conveyor and place one Cracker Box and four Soup Cans into each of two boxes on the cart (cortex-style BT).",
        )
