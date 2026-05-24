# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import csv
import logging
import os
import time

import sys

from task_cli import (
    discover_task_catalog,
    resolve_task_class,
    add_common_task_arguments,
    handle_list_request,
    resolve_task_selection,
    setup_seed,
    resolve_counts,
    resolve_dynamic_intervals,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


PHYSICS_DT = 1/60.0    # 1/300.0
# Number of physics steps per visual render in the main loop.  Literal int —
# does NOT auto-recompute when PHYSICS_DT changes, so if you raise/lower
# PHYSICS_DT and want to keep ~FRAME_RATE fps, also adjust this (or pass
# --psteps-per-render at runtime).
PHYSICS_STEPS_PER_RENDER_STEP = 1  # 10
assert isinstance(PHYSICS_STEPS_PER_RENDER_STEP, int) and PHYSICS_STEPS_PER_RENDER_STEP >= 1, \
    "PHYSICS_STEPS_PER_RENDER_STEP must be a positive integer"
# Visual frame rate (Hz) implied by PHYSICS_DT × PHYSICS_STEPS_PER_RENDER_STEP.
# Derived for logging; not consumed elsewhere.
FRAME_RATE = 1.0 / (PHYSICS_DT * PHYSICS_STEPS_PER_RENDER_STEP)


def _log_dt_diagnostics(my_world, my_task, log):
    try:
        log.warning("=== Physics/Render configuration ===")
        log.warning(f"  world.get_physics_dt():    {my_world.get_physics_dt()}")
        log.warning(f"  world.get_rendering_dt():  {my_world.get_rendering_dt()}")
        robot = getattr(my_task, "_robot", None) or getattr(my_task, "robot", None)
        if robot is not None and hasattr(robot, "commanders_step_dt"):
            log.warning(f"  robot.commanders_step_dt:  {robot.commanders_step_dt}")
        if robot is not None and hasattr(robot, "arm"):
            arm = robot.arm
            if hasattr(arm, "amp"):
                log.warning(f"  arm.amp._default_physics_dt: {arm.amp._default_physics_dt}")
                mp = getattr(arm.amp, "motion_policy", None)
                if mp is not None and hasattr(mp, "maximum_substep_size"):
                    log.warning(f"  rmpflow.maximum_substep_size: {mp.maximum_substep_size}")
                if mp is not None and hasattr(mp, "ignore_robot_state_updates"):
                    log.warning(f"  rmpflow.ignore_robot_state_updates: {mp.ignore_robot_state_updates}")
        if robot is not None:
            try:
                ac = robot.get_articulation_controller()
                gains = ac.get_gains() if hasattr(ac, "get_gains") else (None, None)
                log.warning(f"  drive kps: {gains[0]}")
                log.warning(f"  drive kds: {gains[1]}")
            except Exception as e:
                log.warning(f"  drive gains: <unable to read: {e}>")
        log.warning("====================================")
    except Exception as e:
        log.warning(f"_log_dt_diagnostics failed: {e}")


def _telemetry_row(my_world, my_task, step_num, on_render_boundary, wall_clock):
    """Build one telemetry row as a list of scalars (None -> empty string in CSV)."""
    sim_time = None
    try:
        sim_time = my_world.current_time
    except Exception:
        pass

    ee_x = ee_y = ee_z = None
    joint_pos = [None] * 6
    joint_vel = [None] * 6
    cmd_pos = [None] * 6
    cmd_vel = [None] * 6

    robot = getattr(my_task, "_robot", None) or getattr(my_task, "robot", None)
    if robot is not None:
        try:
            jp = robot.get_joint_positions()
            if jp is not None:
                joint_pos = [float(v) for v in jp[:6]] + [None] * max(0, 6 - len(jp))
        except Exception:
            pass
        try:
            jv = robot.get_joint_velocities()
            if jv is not None:
                joint_vel = [float(v) for v in jv[:6]] + [None] * max(0, 6 - len(jv))
        except Exception:
            pass
        arm = getattr(robot, "arm", None)
        if arm is not None:
            try:
                if hasattr(arm, "get_fk_p"):
                    p = arm.get_fk_p()
                    if p is not None and len(p) >= 3:
                        ee_x, ee_y, ee_z = float(p[0]), float(p[1]), float(p[2])
            except Exception:
                pass
            try:
                subset = getattr(arm, "articulation_subset", None)
                if subset is not None and hasattr(subset, "get_applied_action"):
                    action = subset.get_applied_action()
                    jp_cmd = getattr(action, "joint_positions", None)
                    if jp_cmd is not None:
                        cmd_pos = [
                            float(v) if v is not None else None
                            for v in list(jp_cmd)[:6]
                        ] + [None] * max(0, 6 - len(list(jp_cmd)))
                    jv_cmd = getattr(action, "joint_velocities", None)
                    if jv_cmd is not None:
                        cmd_vel = [
                            float(v) if v is not None else None
                            for v in list(jv_cmd)[:6]
                        ] + [None] * max(0, 6 - len(list(jv_cmd)))
            except Exception:
                pass

    return [
        sim_time, step_num, int(bool(on_render_boundary)), wall_clock,
        ee_x, ee_y, ee_z,
        *joint_pos, *joint_vel,
        *cmd_pos, *cmd_vel,
    ]


_TELEMETRY_HEADER = (
    ["sim_time", "step_num", "on_render_boundary", "wall_clock"]
    + ["ee_x", "ee_y", "ee_z"]
    + [f"j{i}_pos" for i in range(6)]
    + [f"j{i}_vel" for i in range(6)]
    + [f"cmd_j{i}_pos" for i in range(6)]
    + [f"cmd_j{i}_vel" for i in range(6)]
)


def main() -> None:
    # Parse command-line arguments
    flawed = "--flawed" in sys.argv
    tasks2_only = "--tasks2" in sys.argv
    tasks1_only = "--tasks1" in sys.argv
    catalog = discover_task_catalog(flawed=flawed, tasks2_only=tasks2_only, tasks1_only=tasks1_only)
    if not catalog:
        raise SystemExit("No task classes found.")

#    default_task = "TableTask4" if "TableTask4" in catalog else sorted(catalog.modules)[0]
    default_task = None
    available_task_names = ", ".join(catalog.ordered_names)

    parser = argparse.ArgumentParser(description="Choose the task to run.")
    add_common_task_arguments(parser, available_task_names)
    parser.add_argument("--auto-exit", "-x", action="store_true", help="exit the simulator when the task has finished running")
    parser.add_argument("--headless", action="store_true", help="Don't display the IsaacSim GUI. Also sets auto-exit=True")
    parser.add_argument("--show-status", action="store_true",
                        help="Log INFO messages when behaviours transition to RUNNING state")
    parser.add_argument("--physics-dt", type=float, default=None,
                        help="Override PHYSICS_DT (seconds per physics step). Default: module-level PHYSICS_DT.")
    parser.add_argument("--rendering-dt", type=float, default=None,
                        help="Override the rendering_dt argument passed to CortexWorld. "
                             "If not set, defaults to the resolved physics_dt (substeps=1).")
    parser.add_argument("--psteps-per-render", type=int, default=None,
                        help="Override PHYSICS_STEPS_PER_RENDER_STEP (number of physics steps "
                             "per visual render in the main loop). Default: module-level constant.")
    parser.add_argument("--telemetry-csv", type=str, default=None,
                        help="If set, write per-physics-step robot telemetry to this CSV path.")
    parser.add_argument("--max-sim-time", type=float, default=None,
                        help="If set, force exit after this many seconds of sim time. Safety net for autonomous runs.")

    args = parser.parse_args()

    resolved_physics_dt = args.physics_dt if args.physics_dt is not None else PHYSICS_DT
    resolved_rendering_dt_for_cortex = (
        args.rendering_dt if args.rendering_dt is not None else resolved_physics_dt
    )
    resolved_psteps_per_render = (
        args.psteps_per_render if args.psteps_per_render is not None else PHYSICS_STEPS_PER_RENDER_STEP
    )
    assert isinstance(resolved_psteps_per_render, int) and resolved_psteps_per_render >= 1, \
        f"psteps-per-render must be a positive integer, got {resolved_psteps_per_render!r}"
    resolved_frame_rate = 1.0 / (resolved_physics_dt * resolved_psteps_per_render)

    if handle_list_request(args, catalog):
        return

    resolve_task_selection(args, catalog, default_task)
    setup_seed(args)

    if args.task not in catalog:
        available = ", ".join(catalog.ordered_names)
        message = (
            f"Task '{args.task}' not found. Available task classes: {available}"
            if available
            else "No task classes found."
        )
        raise SystemExit(message)

    # Defer Isaac imports until after SimulationApp is created
    from isaacsim import SimulationApp

    simulation_app = SimulationApp({"headless": args.headless})

    # import carb
    # import omni.log

    from isaacsim.cortex.framework.cortex_utils import get_assets_root_path_or_die
    from isaacsim.cortex.framework.cortex_world import CortexWorld

    from isaacsim.core.api.scenes.scene import Scene
    from isaacsim.core.utils.stage import add_reference_to_stage, get_stage_units
    import isaacsim.robot.manipulators.controllers as manipulators_controllers
    from isaacsim.robot.manipulators.grippers import SurfaceGripper
    from isaacsim.core.prims import SingleArticulation

    # Default: rendering_dt = physics_dt so PhysX substeps=1 and each
    # world.step() advances exactly one physics_dt. This lets us tick the cortex
    # commander once per physics step (every world.step() call invokes the cortex
    # pipeline's robot.pre_step() -> commander.step()) and still control the visual
    # render cadence from our loop via PHYSICS_STEPS_PER_RENDER_STEP. Passing
    # --rendering-dt larger than --physics-dt makes Isaac Sim batch substeps
    # internally during step(render=True), causing time to advance inconsistently
    # and freezing the commander's target across each batch — useful for diagnosis,
    # not for normal operation.
    my_world = CortexWorld(
        stage_units_in_meters=1.0,
        physics_dt=resolved_physics_dt,
        rendering_dt=resolved_rendering_dt_for_cortex,
    )

    task_cls = resolve_task_class(args.task, catalog)

    pick_count, target_count = resolve_counts(args)
    dynamic_pick_interval, dynamic_target_interval = resolve_dynamic_intervals(args)

    my_task = task_cls(
        pick_count=pick_count,
        target_count=target_count,
        seed=args.seed,
        randomize=False if args.no_randomize else None,
        teleport_mode=args.teleport,
        incremental_checks=not args.no_incremental_checks,
        pause_after_cycle=args.pause and not args.headless,
        dynamic_pick_interval=dynamic_pick_interval,
        dynamic_target_interval=dynamic_target_interval,
    )

    my_task._show_status = args.show_status
    my_task._min_cycle_time_s = float(args.min_cycle_time)
    logger.warning(f"Running Task {type(my_task).__name__} with seed:{args.seed}.")
    logger.warning(
        f"Loop config: physics_dt={resolved_physics_dt:.6f}s "
        f"({1.0/resolved_physics_dt:.1f} Hz physics), "
        f"cortex rendering_dt={resolved_rendering_dt_for_cortex:.6f}s "
        f"(substeps={max(int(round(resolved_rendering_dt_for_cortex/resolved_physics_dt)), 1)}), "
        f"psteps_per_render={resolved_psteps_per_render}, "
        f"FRAME_RATE={resolved_frame_rate:.2f} fps."
    )

    my_world.add_task(my_task)

    my_world.reset()

    _log_dt_diagnostics(my_world, my_task, logger)

    telemetry_file = None
    telemetry_writer = None
    telemetry_rows_since_flush = 0
    if args.telemetry_csv:
        telemetry_file = open(args.telemetry_csv, "w", newline="")
        telemetry_writer = csv.writer(telemetry_file)
        telemetry_writer.writerow(_TELEMETRY_HEADER)
        logger.warning(f"Writing per-step telemetry to {args.telemetry_csv}")

    def _close_telemetry():
        nonlocal telemetry_file
        if telemetry_file is not None:
            try:
                telemetry_file.flush()
                telemetry_file.close()
            except Exception:
                pass
            telemetry_file = None

    reset_needed = False
    success_checked = False
    step_num = 0
    wall_clock_start = time.time()
    while simulation_app.is_running():
        step_num += 1
        on_render_boundary = (step_num % resolved_psteps_per_render == 0)
        render_this_step = on_render_boundary and not args.headless
        my_world.step(render=render_this_step)
        if telemetry_writer is not None and my_world.is_playing():
            wall_clock = time.time() - wall_clock_start
            row = _telemetry_row(my_world, my_task, step_num, on_render_boundary, wall_clock)
            telemetry_writer.writerow(row)
            telemetry_rows_since_flush += 1
            if telemetry_rows_since_flush >= 1000:
                telemetry_file.flush()
                telemetry_rows_since_flush = 0
        if args.max_sim_time is not None:
            try:
                if my_world.current_time >= args.max_sim_time:
                    logger.warning(
                        f"--max-sim-time ({args.max_sim_time}s) reached at sim_time={my_world.current_time:.3f}s. Exiting."
                    )
                    _close_telemetry()
                    my_world.clear()
                    simulation_app.close()
                    return
            except Exception:
                pass
        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                # my_controller.reset()
                reset_needed = False
            if not on_render_boundary:
                continue
            current_tasks = my_world.get_current_tasks()
            for task_name in current_tasks:
                task = current_tasks[task_name]
                if hasattr(task,"task_step"):
                    task.task_step()
                if hasattr(task,"can_exit") and task.can_exit():
                    if not success_checked:
                        task_successful, failures = task.check_groundtruth_task_success()
                        success_checked = True
                        if task_successful:
                            logger.warning(f"Task {task_name} Completed successfully (seed: {args.seed}).")
                        else:
                            for msg in failures:
                                logger.warning(msg)
                            logger.warning(f"Task {task_name} Verification checks reported UNSUCCESSFUL completion (seed: {args.seed}).")
                        if hasattr(task, "stop_conveyor"):
                            task.stop_conveyor()
                    elif args.auto_exit or args.headless:
                        logger.warning(f"Task {task_name} signaled exit. Resetting the world.")
                        _close_telemetry()
                        my_world.clear()
                        simulation_app.close()
                        exit()
            # The following has been moved into UR10MultiPickPlaceTask.task_step()
            # observations = my_world.get_observations()  #merges observations from all currently running tasks
            # actions = my_controller.forward(
            #     observations=observations, end_effector_offset=np.array([0.0, 0.0, 0.02])
            # )
            # articulation_controller.apply_action(actions)

    _close_telemetry()
    simulation_app.close()


if __name__ == "__main__":
    main()
