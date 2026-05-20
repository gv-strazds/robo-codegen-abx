#!/usr/bin/env bash
# Shared implementation for run_all_simulation_tasks.sh, run_all_teleport_tasks.sh,
# and run_all_mock_tasks.sh.
# Not meant to be executed directly — source it from a wrapper script that defines
# DEFAULT_NUM_RUNS, DEFAULT_TASK_SET, DEFAULT_SKIP_TASKS (array), MODE_LABEL, and
# MODE_TASK_ARGS (and optionally MODE_RUNNER, default run_task.py), then calls
# `run_all_tasks "$@"`.
#
# Environment variable overrides honoured by the wrapper scripts:
#   NUM_RUNS    - number of runs per task (overrides DEFAULT_NUM_RUNS)
#   TASK_SET    - --tasks1, --tasks2, or ALL (overrides DEFAULT_TASK_SET).
#                 ALL (or the empty string) passes no --tasks* arg, which makes
#                 run_task.py discover tasks across both tiers (tasks/ + tasks2/).
#   SKIP_TASKS  - space-separated task names to skip (overrides DEFAULT_SKIP_TASKS).
#                 Set to the empty string to skip nothing.
#   SEEDS_FILE  - path to a JSON or text seed map (see _load_seeds_file below).
#                 When set, the j-th run of TASK_NAME is invoked with --seed
#                 SEEDS_MAP[TASK_NAME][j-1] (when present); otherwise no --seed
#                 arg is passed and run_task.py picks a random seed (default).
#   EXTRA_ARGS  - extra args appended verbatim to every run_task.py invocation.
#   SIM_LOGS_DIR - directory for per-run logs and the results file (default: ./logs).
#                 Created with `mkdir -p` if it does not already exist.

declare -gA SEEDS_MAP=()

# Load seeds from $1 (JSON or whitespace text) into SEEDS_MAP keyed by
# "${task_name}__${run_index_1based}".
#
# JSON format:    {"TableTask3": [42, 1234, 5678], "TableTaskColors1": [100]}
# Text format:    one task per line, "TaskName seed1 seed2 seed3"
#                 (lines starting with '#' and blank lines are ignored)
# Detection is by file extension (.json) with a fallback to JSON-then-text.
_load_seeds_file() {
    local file="$1"
    [ -z "$file" ] && return 0
    if [ ! -f "$file" ]; then
        echo "WARNING: SEEDS_FILE '$file' not found — proceeding without seeds." >&2
        return 0
    fi
    local count=0
    while IFS=$'\t' read -r task run_idx seed; do
        [ -z "$task" ] && continue
        SEEDS_MAP["${task}__${run_idx}"]="$seed"
        (( count++ ))
    done < <(python3 - "$file" <<'PY'
import json, sys, os

path = sys.argv[1]
with open(path) as f:
    text = f.read()

data = None
ext = os.path.splitext(path)[1].lower()
if ext == ".json":
    data = json.loads(text)
else:
    try:
        data = json.loads(text)
    except Exception:
        data = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            data[parts[0]] = [int(x) for x in parts[1:]]

for task, seeds in data.items():
    for i, s in enumerate(seeds, 1):
        print(f"{task}\t{i}\t{int(s)}")
PY
)
    echo "Loaded $count seed entries from $file"
}

_lookup_seed() {
    local task="$1" run_idx="$2"
    echo "${SEEDS_MAP["${task}__${run_idx}"]:-}"
}

run_all_tasks() {
    : "${MODE_LABEL:?MODE_LABEL must be set by the calling wrapper}"
    : "${MODE_TASK_ARGS=}"
    : "${MODE_RUNNER:=run_task.py}"
    : "${DEFAULT_NUM_RUNS:=1}"
    : "${DEFAULT_TASK_SET:=--tasks2}"

    local num_runs="${NUM_RUNS:-$DEFAULT_NUM_RUNS}"
    # Use ${VAR-default} (no colon) so an explicit TASK_SET="" overrides the default.
    local task_set="${TASK_SET-$DEFAULT_TASK_SET}"
    # ALL (case-insensitive) is a sentinel meaning "no --tasks* arg" — run_task.py
    # then enumerates tasks from both tasks/ and tasks2/.
    if [[ "${task_set^^}" == "ALL" ]]; then
        task_set=""
    fi

    # SKIP_TASKS env var: if set (even to ""), it overrides DEFAULT_SKIP_TASKS.
    local -a skip_tasks
    if [ -n "${SKIP_TASKS+x}" ]; then
        read -r -a skip_tasks <<< "$SKIP_TASKS"
    else
        skip_tasks=("${DEFAULT_SKIP_TASKS[@]+"${DEFAULT_SKIP_TASKS[@]}"}")
    fi

    _load_seeds_file "${SEEDS_FILE:-}"

    local logs_dir="${SIM_LOGS_DIR:-./logs}"
    mkdir -p "$logs_dir"
    local results_file="$logs_dir/${MODE_LABEL}_results.out"

    # Determine task indices to run.
    local -a task_indices
    local use_skip_list
    if [ $# -gt 0 ]; then
        task_indices=("$@")
        use_skip_list=false
        echo "Running ${#task_indices[@]} specified task(s) (mode=$MODE_LABEL, skip list ignored) ..."
    else
        local n
        n=$(python "$MODE_RUNNER" $task_set --list 2>&1 | grep -c '^\s*[0-9]')
        task_indices=($(seq 1 "$n"))
        use_skip_list=true
        echo "Running $n tasks (mode=$MODE_LABEL) ..."
    fi

    echo "----FAILURE MESSAGES-----" > "$results_file"

    local loop_index=0 i j task_info task_name skip_name skip seed seed_arg
    for i in "${task_indices[@]}"; do
        (( loop_index++ ))
        task_info=$(python "$MODE_RUNNER" $task_set "$i" --list --verbose 2>&1)
        task_name=$(echo "$task_info" | grep -oP '^\s*\d+\.\K\S+' | head -1)

        echo "$task_info" | tee -a "$results_file"

        if $use_skip_list; then
            skip=false
            for skip_name in "${skip_tasks[@]+"${skip_tasks[@]}"}"; do
                if [[ "$task_name" == "$skip_name" ]]; then
                    skip=true
                    break
                fi
            done
            if $skip; then
                echo "  ** task $task_name skipped **" | tee -a "$results_file"
                echo "----------------------" | tee -a "$results_file"
                continue
            fi
        fi

        for j in $(seq 1 "$num_runs"); do
            local out="$logs_dir/$i-$j-$MODE_LABEL.out"
            local err="$logs_dir/$i-$j-$MODE_LABEL.stderr"
            seed=$(_lookup_seed "$task_name" "$j")
            if [ -n "$seed" ]; then
                seed_arg="--seed $seed"
                echo "--- Task $i.$j ($loop_index.$j of ${#task_indices[@]}.$num_runs) [seed=$seed] ---" | tee -a "$results_file"
            else
                seed_arg=""
                echo "--- Task $i.$j ($loop_index.$j of ${#task_indices[@]}.$num_runs) ---" | tee -a "$results_file"
            fi
            python "$MODE_RUNNER" $task_set $EXTRA_ARGS $MODE_TASK_ARGS $seed_arg "$i" --show-status --auto-exit \
                > "$out" 2> "$err"
            grep -h -e "Completed successfully" -e "UNSUCCESSFUL" "$out" "$err" | head -1 | tee -a "$results_file"
        done
        echo "----------------------" | tee -a "$results_file"
    done
}
