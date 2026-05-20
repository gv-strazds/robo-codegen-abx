#!/usr/bin/env python
"""Run any task configuration through the mock py_trees BT without Isaac Sim.

Usage:
    python run_mock_task.py --task TableTaskColors1 --seed 42
    python run_mock_task.py --list
    python run_mock_task.py --task TableTask3 --render
"""
import argparse
import logging
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# extsMock must be on path before any isaacsim imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "extsMock"))
sys.path.insert(0, os.path.dirname(__file__))

from tasks_mock.mock_task_utils import setup_mock_modules

# Install mock modules before importing tasks
setup_mock_modules()

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


def main():
    flawed = "--flawed" in sys.argv
    tasks2_only = "--tasks2" in sys.argv
    tasks1_only = "--tasks1" in sys.argv
    catalog = discover_task_catalog(flawed=flawed, tasks2_only=tasks2_only, tasks1_only=tasks1_only)
    if not catalog:
        raise SystemExit("No task classes found.")

#    default_task = "TableTask3" if "TableTask3" in catalog else sorted(catalog.modules)[0]
    default_task = None  # randomly select from available tasks if no task is specified
    available_task_names = ", ".join(catalog.ordered_names)

    parser = argparse.ArgumentParser(
        description="Run a task through the mock py_trees BT without Isaac Sim.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_task_arguments(parser, available_task_names)
    parser.add_argument(
        "--render", action="store_true",
        help="Render dot tree to file and exit",
    )
    parser.add_argument(
        "--max-ticks", type=int, default=20000,
        help="Maximum BT tick iterations (default: 20000)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output",
    )
    parser.add_argument(
        "--show-status", action="store_true",
        help="Display py_trees behaviour tree status after each tick",
    )
    parser.add_argument(
        "--tick", type=float, default=None,
        help="Per tick sleep delay (defaults to none, or 0.05 if show-status is true)",
    )
    parser.add_argument(
        "--auto-exit", "-x", action="store_true",
        help="Ignored (accepted for compatibility with run_task.py)",
    )
    # Visual-capture flags accepted (and silently ignored) for compatibility with
    # run_task.py — there is nothing to capture in mock mode.
    for _flag in ("--video", "--snapshots", "--snapshot-errors", "--headless"):
        parser.add_argument(
            _flag, action="store_true",
            help=argparse.SUPPRESS,
        )

    args = parser.parse_args()

    if handle_list_request(args, catalog):
        return

    if args.verbose:
        args.show_status = True

    resolve_task_selection(args, catalog, default_task)
    setup_seed(args)

    pick_count, target_count = resolve_counts(args)
    dynamic_pick_interval, dynamic_target_interval = resolve_dynamic_intervals(args)
    randomize = False if args.no_randomize else None

    task_class = resolve_task_class(args.task, catalog)
    tick_interval = -1 if args.tick is None else args.tick

    from tasks_mock.mock_task_utils import run_mock_task
    context, task_successful = run_mock_task(
        task_class,
        seed=args.seed,
        render=args.render,
        max_ticks=args.max_ticks,
        pick_count=pick_count,
        target_count=target_count,
        randomize=randomize,
        verbose=not args.quiet,
        show_status=args.show_status,
        tick_interval=tick_interval,
        incremental_checks=not args.no_incremental_checks,
        dynamic_pick_interval=dynamic_pick_interval,
        dynamic_target_interval=dynamic_target_interval,
        min_cycle_time_s=args.min_cycle_time,
    )

    # Exit with non-zero if task didn't finish or verification failed
    if not args.render:
        if not context.task_finished:
            sys.exit(1)
        elif task_successful is False:
            sys.exit(2)  # task finished but verification failed


if __name__ == "__main__":
    main()
