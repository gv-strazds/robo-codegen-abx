"""Cortex-tree variant of TableTaskCartToConveyor.

Identical to :class:`TableTaskCartToConveyor` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_cart_to_conveyor.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_cart_to_conveyor import TableTaskCartToConveyor


class TableTaskCartToConveyor2(TableTaskCartToConveyor):
    DEFAULT_TASK_NAME = "table_task_cart_to_conveyor_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Pick cracker boxes, soup cans, mustard bottles, and sugar boxes from the cart and place one of each vertically into boxes on the conveyor (cortex-style BT).",
        )
