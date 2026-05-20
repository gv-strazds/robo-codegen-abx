"""Experimental cortex tree variant that uses time-interpolated descent.

Swaps ``CortexDownToInsert`` for a ``LowerToPlaceBehaviour`` whose
z-interpolator is anchored at the live FK z (instead of the transport
altitude ``ee_height_for_move``), so the descent starts from where the
EE physically is at the end of ``CortexMoveToPlace``.

Currently consumed only by ``TableTaskBottlesToConveyor2x`` for
side-by-side comparison against ``TableTaskBottlesToConveyor2``.
"""
import numpy as np
import py_trees

from robot_controllers.pt_context_monitor import ContextMonitorBehaviour
from robot_controllers.pt_cortex_behaviours import (
    CortexCloseGripper,
    CortexExecuteApproach,
    CortexMoveRelative,
    CortexMoveToPlace,
    CortexMoveToPreGrasp,
    CortexOpenGripper,
)
from robot_controllers.pt_cortex_perception_behaviours import (
    CheckPickReachable,
    DeferPickAndRelease,
    HaveItemInGripper,
    PrepareGrasp,
    PreparePlacement,
    VerifyGrasp,
)
from robot_controllers.pt_cortex_tree import (
    PICK_RETRY_BUDGET,
    POST_PICK_LIFT_Z,
    POST_PLACE_LIFT_Z,
    RECOVERY_LIFT_Z,
    VERIFY_GRASP_SLIP_THRESHOLD,
)
from robot_controllers.pt_sim_time_decorators import (
    SimTimer,
    sim_timeout_to_success,
)
from task_context_base import (
    DEFAULT_APPROACH_TIMEOUT_S,
    DEFAULT_MOVE_TIMEOUT_S,
)
from robot_controllers.pt_pick_place_behaviours import (
    LowerToPlaceBehaviour,
    PickPlaceBehaviour,
    SinusoidalInterpolator,
)
from robot_controllers.pt_task_behaviours import (
    CheckAllDone,
    CheckTargetAvailable,
    LatchCurrentPick,
    LatchPlacementTarget,
    MarkPickComplete,
    SelectNextPick,
    SetTaskFinished,
    WaitForCycleTime,
)


class LowerToPlaceFromFKBehaviour(LowerToPlaceBehaviour):
    """LowerToPlaceBehaviour anchored at the live FK z, not ee_height_for_move.

    The default ``LowerToPlaceBehaviour.initialise()`` builds its
    z-interpolator from ``bb.ee_height_for_move`` — the transport
    altitude.  Inside the cortex tree the prior ``CortexMoveToPlace``
    parks the EE at ``target_z + place_hover_above_z``, which is
    typically not the transport altitude, so the original anchor would
    cause a Z-jump on the first commanded pose of the descent.

    This subclass reads the actual EE z from
    ``self._arm_commander.get_fk_p()[2]`` at ``initialise()`` time and
    uses that as the interpolator's start.
    """

    def initialise(self) -> None:
        # Skip LowerToPlaceBehaviour.initialise (which reads
        # bb.ee_height_for_move) and call the grandparent reset directly.
        PickPlaceBehaviour.initialise(self)
        fk_z = float(self._arm_commander.get_fk_p()[2])
        self._h_interp = SinusoidalInterpolator(fk_z)


def make_cortex_with_lowertoplace_tree(fake_fast: bool = False):
    """Cortex tree with ``CortexDownToInsert`` replaced by ``LowerToPlaceFromFKBehaviour``.

    Mirrors :func:`robot_controllers.pt_cortex_tree.make_cortex_task_controller_tree`
    structurally; the only behavioural difference is the descent node.
    """
    context_monitor = ContextMonitorBehaviour(name="ContextMonitor")

    check_all_done = CheckAllDone(name="check_all_done")
    wait_cycle = WaitForCycleTime(name="WaitForCycleTime")
    select_next_pick = SelectNextPick(name="SelectNextPick")
    check_target = CheckTargetAvailable(name="CheckTargetAvailable")
    mark_complete = MarkPickComplete(name="MarkPickComplete")
    set_finished = SetTaskFinished(name="SetTaskFinished")

    grip_wait = 0.001 if fake_fast else 0.5
    release_wait = 0.001 if fake_fast else 0.3

    # Sim-time watchdog wrappers — see make_cortex_task_controller_tree
    # for the rationale.  This experimental variant uses the same per-task
    # tunable surface so swapping LowerToPlaceFromFK in/out doesn't change
    # how timeouts are configured.
    def _move_timeout(ctx):
        return ctx.get_move_timeout_s() if ctx is not None else DEFAULT_MOVE_TIMEOUT_S

    def _approach_timeout(ctx):
        return ctx.get_approach_timeout_s() if ctx is not None else DEFAULT_APPROACH_TIMEOUT_S

    def _watchdog(child, duration_resolver):
        return sim_timeout_to_success(
            name=child.name,
            child=child,
            duration=duration_resolver,
            on_timeout=child._timeout_diagnostic,
        )

    pick_attempt = py_trees.composites.Sequence(
        name="pick_attempt", memory=True,
        children=[
            CheckPickReachable(),
            PrepareGrasp(),
            _watchdog(CortexMoveToPreGrasp(fake_fast=fake_fast), _move_timeout),
            _watchdog(CortexExecuteApproach(fake_fast=fake_fast), _approach_timeout),
            CortexCloseGripper(name="CortexCloseGripper"),
            SimTimer(name="wait_for_grip", duration=grip_wait),
            LatchCurrentPick(),
            _watchdog(
                CortexMoveRelative(
                    name="lift_after_pick",
                    offset=np.array([0.0, 0.0, POST_PICK_LIFT_Z]),
                    fake_fast=fake_fast,
                    cap_to_ee_height_for_move=True,
                ),
                _move_timeout,
            ),
            VerifyGrasp(slip_threshold=VERIFY_GRASP_SLIP_THRESHOLD),
        ],
    )
    pick_with_retry = py_trees.decorators.Retry(
        name="pick_with_retry",
        child=pick_attempt,
        num_failures=PICK_RETRY_BUDGET,
    )
    defer_fallback = py_trees.decorators.FailureIsSuccess(
        name="defer_on_exhausted",
        child=DeferPickAndRelease(),
    )
    pick_or_defer = py_trees.composites.Selector(
        name="pick_or_defer", memory=True,
        children=[pick_with_retry, defer_fallback],
    )

    # The descent node — the one structural difference vs. the canonical
    # cortex tree.  num_steps=200 matches the default tree's LowerToPlace
    # phase (pt_pick_place_behaviours.PICKPLACE_PHASE_DTs['LowerToPlace']).
    lower_to_place = LowerToPlaceFromFKBehaviour(
        name="LowerToPlace", num_steps=200, fake_fast=fake_fast,
    )

    # Note: lower_to_place is a LowerToPlaceBehaviour (phase-duration
    # variant), not a CortexMove subclass — it has no _timeout_diagnostic
    # and its duration drives interpolation, so it is left unwrapped.
    place_item = py_trees.composites.Sequence(
        name="place_item", memory=True,
        children=[
            HaveItemInGripper(),
            PreparePlacement(),
            LatchPlacementTarget(),
            _watchdog(CortexMoveToPlace(fake_fast=fake_fast), _move_timeout),
            lower_to_place,
            CortexOpenGripper(name="CortexOpenGripper"),
            SimTimer(name="wait_for_release", duration=release_wait),
            _watchdog(
                CortexMoveRelative(
                    name="lift_after_place",
                    offset=np.array([0.0, 0.0, POST_PLACE_LIFT_Z]),
                    fake_fast=fake_fast,
                    cap_to_ee_height_for_move=True,
                    cap_max_to_ee_height_for_move=True,
                ),
                _move_timeout,
            ),
        ],
    )

    release_and_skip = py_trees.composites.Sequence(
        name="release_and_skip", memory=True,
        children=[
            CortexOpenGripper(name="recovery_open_gripper"),
            SimTimer(name="recovery_wait", duration=release_wait),
            _watchdog(
                CortexMoveRelative(
                    name="recovery_lift",
                    offset=np.array([0.0, 0.0, RECOVERY_LIFT_Z]),
                    fake_fast=fake_fast,
                ),
                _move_timeout,
            ),
        ],
    )

    place_or_recover = py_trees.composites.Selector(
        name="place_or_recover", memory=True,
        children=[place_item, release_and_skip],
    )

    do_one_pick_place = py_trees.composites.Sequence(
        name="do_one_pick_place", memory=True,
        children=[wait_cycle, select_next_pick, check_target,
                  pick_or_defer, place_or_recover, mark_complete],
    )

    repeat_picks = py_trees.decorators.Repeat(
        name="repeat_picks",
        child=do_one_pick_place,
        num_success=-1,
    )

    repeat_until_exhausted = py_trees.decorators.FailureIsSuccess(
        name="repeat_until_exhausted",
        child=repeat_picks,
    )

    finish_or_fail = py_trees.composites.Sequence(
        name="finish_or_fail", memory=True,
        children=[repeat_until_exhausted, set_finished],
    )

    task_orchestration = py_trees.composites.Selector(
        name="task_orchestration", memory=False,
        children=[check_all_done, finish_or_fail],
    )

    root = py_trees.composites.Parallel(
        name="TaskRoot",
        policy=py_trees.common.ParallelPolicy.SuccessOnOne(),
        children=[context_monitor, task_orchestration],
    )

    return root
