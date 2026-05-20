"""Factory function for building the full task-level py_trees behaviour tree.

This is the main entry point used by UR10MultiPickPlaceController to construct
the behaviour tree that orchestrates multi-object pick-and-place tasks.
"""
import py_trees

from robot_controllers.pt_pick_place_behaviours import create_pick_place_sequence
from robot_controllers.pt_context_monitor import ContextMonitorBehaviour
from robot_controllers.pt_task_behaviours import (
    CheckAllDone,
    SelectNextPick,
    CheckTargetAvailable,
    ResetPickPlaceTree,
    MarkPickComplete,
    SetTaskFinished,
    WaitForCycleTime,
)


def make_task_controller_tree(fake_fast: bool = False,
                              track_picked_item_during_lift: bool = False):
    """Build the full task-level behaviour tree for UR10MultiPickPlaceController.

    Structure:
        Parallel("TaskRoot", SuccessOnOne)
        |-- ContextMonitorBehaviour        [always RUNNING]
        |-- Selector("task_orchestration", memory=False)
            |-- CheckAllDone               [SUCCESS if done]
            |-- Sequence("finish_or_fail", memory=True)
                |-- Repeat("repeat_picks", num_success=-1)
                |   |-- Sequence("do_one_pick_place", memory=True)
                |       |-- WaitForCycleTime    [no-op when min_cycle_time_s=0]
                |       |-- SelectNextPick
                |       |-- CheckTargetAvailable
                |       |-- ResetPickPlaceTree
                |       |-- pick_then_place (9 phases)
                |       |-- MarkPickComplete
                |-- SetTaskFinished

    Args:
        fake_fast: Use fast completion times for testing.
        track_picked_item_during_lift: Off by default — the post-grasp lift
            holds the latched pick XY so the EE rises straight up regardless
            of whether the original item location is being moved (e.g. by a
            conveyor surface dragging the held item).  A task that needs the
            legacy "follow the live picked item during lift" behaviour can
            opt in by supplying its own ``tree_factory`` in its TaskSpec, e.g.
            ``tree_factory=lambda fake_fast=False: make_task_controller_tree(
            fake_fast=fake_fast, track_picked_item_during_lift=True)``.

    Returns:
        The root behaviour (Parallel).
    """
    # Context monitor (first child of parallel, ticked every time)
    context_monitor = ContextMonitorBehaviour(name="ContextMonitor")

    # Task-level behaviours
    check_all_done = CheckAllDone(name="check_all_done")
    wait_cycle = WaitForCycleTime(name="WaitForCycleTime")
    select_next_pick = SelectNextPick(name="SelectNextPick")
    check_target = CheckTargetAvailable(name="CheckTargetAvailable")
    reset_subtree = ResetPickPlaceTree(name="ResetPickPlaceTree")
    mark_complete = MarkPickComplete(name="MarkPickComplete")
    set_finished = SetTaskFinished(name="SetTaskFinished")

    # The 9-phase pick-then-place sequence
    pick_then_place = create_pick_place_sequence(
        fake_fast=fake_fast,
        track_picked_item_during_lift=track_picked_item_during_lift,
    )

    # Wire the reset node to the pick_then_place subtree
    reset_subtree.set_subtree(pick_then_place)

    # do_one_pick_place: cooldown -> select -> check -> reset -> execute -> mark
    do_one_pick_place = py_trees.composites.Sequence(
        name="do_one_pick_place", memory=True,
        children=[wait_cycle, select_next_pick, check_target, reset_subtree,
                  pick_then_place, mark_complete],
    )

    # Repeat indefinitely (exits via FAILURE from SelectNextPick or CheckTargetAvailable)
    repeat_picks = py_trees.decorators.Repeat(
        name="repeat_picks",
        child=do_one_pick_place,
        num_success=-1,
    )

    # Wrap Repeat: when it fails (all picks done / targets exhausted),
    # convert to SUCCESS so the Sequence proceeds to SetTaskFinished.
    repeat_until_exhausted = py_trees.decorators.FailureIsSuccess(
        name="repeat_until_exhausted",
        child=repeat_picks,
    )

    # After the repeat loop ends, set finished
    finish_or_fail = py_trees.composites.Sequence(
        name="finish_or_fail", memory=True,
        children=[repeat_until_exhausted, set_finished],
    )

    # Task orchestration: check done first, then run the loop
    task_orchestration = py_trees.composites.Selector(
        name="task_orchestration", memory=False,
        children=[check_all_done, finish_or_fail],
    )

    # Root: parallel with context monitor + task orchestration
    root = py_trees.composites.Parallel(
        name="TaskRoot",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
        children=[context_monitor, task_orchestration],
    )

    return root
