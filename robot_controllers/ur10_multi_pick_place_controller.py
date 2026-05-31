"""UR10MultiPickPlaceController: task-level controller wrapping a py_trees BehaviourTree.

Orchestrates multi-object pick-and-place via a py_trees tree that includes:
- ContextMonitorBehaviour: refreshes blackboard from TaskContext each tick
- Task-level behaviours: SelectNextPick, CheckTargetAvailable, etc.
- 9-phase pick_then_place sequence (from pt_pick_place_behaviours)

The controller's forward() method ticks the tree.  PickPlace behaviours
send commands directly to an IArmCommander / IGripperCommander (Cortex-aligned).

Hardware ownership lives in TaskContext; this controller is a thin
orchestrator that sources commander refs from it.
"""

import logging
from typing import List, Optional

import py_trees

from robot_controllers.pt_task_tree import make_task_controller_tree

logger = logging.getLogger(__name__)


class RunningTransitionVisitor(py_trees.visitors.VisitorBase):
    """Logs INFO when a behaviour first transitions to RUNNING.

    Only logs once per transition — not on every tick while RUNNING.
    Useful for observing which behaviour is active without the verbosity
    of per-tick logging.
    """

    def __init__(self):
        super().__init__(full=False)
        self._prev_status = {}
        self._curr_status = {}

    def initialise(self):
        self._prev_status = self._curr_status
        self._curr_status = {}

    def run(self, behaviour):
        self._curr_status[behaviour.id] = behaviour.status
        prev = self._prev_status.get(behaviour.id)
        if behaviour.status == py_trees.common.Status.RUNNING and prev != py_trees.common.Status.RUNNING:
            logger.info(f"[BT] {behaviour.name} [{type(behaviour).__name__}] -> RUNNING")


class UR10MultiPickPlaceController:
    """Task-level controller wrapping a py_trees BehaviourTree.

    Manages multi-pick orchestration via a behaviour tree that cycles through
    pick objects, executing the 9-phase pick-place sequence for each.

    Args:
        name: Controller name.
        task_context: TaskContext (or MockTaskContext) holding simulation state,
            robot hardware references, and commander objects.
        arm_commander: IArmCommander for end-effector motion.  Resolution:
            1. Explicit param  2. task_context.arm_commander
        gripper_commander: IGripperCommander for gripper control.  Resolution:
            1. Explicit param  2. task_context.gripper_commander
        fake_fast: Use fast completion times for testing.
    """

    def __init__(
        self,
        name: str,
        task_context=None,
        arm_commander=None,
        gripper_commander=None,
        fake_fast: bool = False,
        tree_factory=None,
        show_status: bool = False,
    ) -> None:
        self._name = name
        self._task_context = task_context

        # Build the py_trees behaviour tree
        factory = tree_factory if tree_factory is not None else make_task_controller_tree
        self._tree_root = factory(fake_fast=fake_fast)
        self._tree = py_trees.trees.BehaviourTree(root=self._tree_root)

        # Add visitor that logs RUNNING transitions (only with --show-status)
        if show_status:
            self._tree.add_visitor(RunningTransitionVisitor())

        # Resolve commanders via cascade:
        #   1. Explicit param  2. task_context attribute
        if arm_commander is None and self._task_context is not None:
            arm_commander = self._task_context.arm_commander
        if gripper_commander is None and self._task_context is not None:
            gripper_commander = self._task_context.gripper_commander

        # Setup the tree with context and commander interfaces
        self._tree.setup(
            timeout=15,
            context=self._task_context,
            arm_commander=arm_commander,
            gripper_commander=gripper_commander,
        )

        # Reset gripper to open position
        if self._task_context is not None:
            self._task_context.reset_gripper()

    def attach_visitor(self, visitor) -> None:
        """Register a py_trees VisitorBase on the underlying BehaviourTree.

        Public hook used by external callers (e.g. ``run_task.py``'s
        snapshot-capture wiring) so the tree object itself stays
        encapsulated.  ``RunningTransitionVisitor`` is wired internally
        via ``__init__(show_status=True)``; further visitors layer on top.
        """
        self._tree.add_visitor(visitor)

    @property
    def tree_root(self):
        """Read-only access to the root py_trees Behaviour.

        Lets external callers walk the tree (e.g. to install
        snapshot-event hooks on ``SimTimeout`` watchdogs) without
        reaching for the private ``_tree_root`` attribute.
        """
        return self._tree_root

    def forward(self) -> None:
        """Tick the behaviour tree.

        The ContextMonitorBehaviour refreshes blackboard data from the
        TaskContext each tick, then the task orchestration tree decides
        which phase to execute.  PickPlace behaviours send motion commands
        directly to the arm/gripper commanders.
        """
        self._tree.tick()

    def is_done(self) -> bool:
        """Return True if the task is finished or all picks are exhausted.

        When incremental generation is active (``more_items_expected``),
        exhaustion of current picks is not treated as done — more items
        will arrive and be added to the strategy.

        Both terminal branches (``targets_exhausted`` and the
        ``all_picks_done`` + no-more-expected combination) latch
        ``context.task_finished`` to ``True`` before returning so that
        readers of the context see a consistent "task complete" signal
        regardless of whether the BT's ``SetTaskFinished`` ran or this
        short-circuit fired first.  Important in real sim, where
        ``multi_pickplace_task.pre_step`` exits as soon as ``is_done()``
        is true and may not tick the BT again.
        """
        if self._task_context is not None:
            if self._task_context.task_finished:
                return True
            if self._task_context.targets_exhausted:
                self._task_context.task_finished = True
                return True
            if self._task_context.all_picks_done:
                # If more items are expected from incremental generation,
                # don't report done — the strategy will grow.
                if not self._task_context.strategy.more_items_expected:
                    self._task_context.task_finished = True
                    return True
                return False
        return self._tree_root.status == py_trees.common.Status.SUCCESS

    def reset(self, picking_order_item_names: Optional[List[str]] = None) -> None:
        """Reset controller and tree state."""
        if self._task_context is not None:
            self._task_context.reset_gripper()
            self._task_context.reset(picking_order_item_names)

        # Reset the tree to INVALID so all nodes re-initialise on next tick
        if self._tree_root.status != py_trees.common.Status.INVALID:
            self._tree_root.stop(py_trees.common.Status.INVALID)

    def get_current_pick_name(self) -> Optional[str]:
        """Return the current pick name or None if done/exhausted."""
        if self._task_context is not None:
            return self._task_context.get_current_pick_name()
        return None

    def reorder_picks(
        self, new_order_names: List[str], current_pick_name: Optional[str] = None
    ) -> None:
        """Reorder picking sequence. Delegates to TaskContext."""
        if self._task_context is not None:
            self._task_context.reorder_picks(new_order_names, current_pick_name)
