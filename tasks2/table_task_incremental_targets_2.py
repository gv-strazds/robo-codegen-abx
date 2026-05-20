"""Cortex-tree variant of TableTaskIncrementalTargets.

Identical to :class:`TableTaskIncrementalTargets` except that the behaviour tree is the
cortex-style ``MotionCommand``-based tree rather than the default
9-phase time-interpolated tree.  See ``tasks/table_task_incremental_targets.py`` for the
full task description, generation strategies, and verification config.
"""
from dataclasses import replace

from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree
from tasks.table_task_incremental_targets import TableTaskIncrementalTargets


class TableTaskIncrementalTargets2(TableTaskIncrementalTargets):
    DEFAULT_TASK_NAME = "table_task_incremental_targets_2"

    def _customize_spec(self, spec):
        return replace(
            spec.with_impl(tree_factory=make_cortex_task_controller_tree),
            task_description="Cortex-style BT variant of TableTaskIncrementalTargets: pick items from the bin and place them onto incrementally-spawned targets.",
        )
