#!/usr/bin/env bash
# Run every simulation task (or a specified subset) with --auto-exit.
#
# Usage:
#   bash run_all_simulation_tasks.sh              # run all tasks, skipping SKIP_TASKS
#   bash run_all_simulation_tasks.sh 1 3 5        # run only task indices 1, 3, 5 (skip list ignored)
#
# Environment variable overrides:
#   NUM_RUNS    - number of runs per task           (default: 1)
#   TASK_SET    - --tasks1, --tasks2, or ALL        (default: --tasks2; ALL = both tiers)
#   SKIP_TASKS  - space-separated task names        (default: see below; set to "" to skip nothing)
#   SEEDS_FILE  - JSON or text seeds map            (default: unset → random seeds)
#   EXTRA_ARGS  - extra args appended to run_task.py
#   SIM_LOGS_DIR - directory for per-run logs and results file (default: ./logs)
#
# Examples:
#   NUM_RUNS=3 bash run_all_simulation_tasks.sh
#   SEEDS_FILE=seeds.json bash run_all_simulation_tasks.sh 1 3 5
#   EXTRA_ARGS="--video --snapshots" bash run_all_simulation_tasks.sh
#   SIM_LOGS_DIR=/tmp/run42 bash run_all_simulation_tasks.sh

DEFAULT_NUM_RUNS=1
DEFAULT_TASK_SET="--tasks2"
DEFAULT_SKIP_TASKS=("TableTask3Curobo" "TableTaskBottlesToConveyor2x" "TableTaskSortAndStack" "TableTaskSortAndStack2")

MODE_LABEL="sim"
MODE_TASK_ARGS=""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_run_all_tasks_lib.sh
source "$SCRIPT_DIR/_run_all_tasks_lib.sh"
run_all_tasks "$@"
