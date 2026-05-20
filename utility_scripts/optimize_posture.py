#!/usr/bin/env python3
"""Bayesian optimization of PLACE_POSTURE_CONFIG for TableTaskBottlesToConveyor2.

Uses optuna (TPE sampler) to search for wrist joint angles that minimize
placement failures when the robot lowers bottles onto the conveyor.

The first 3 joints (shoulder_pan, shoulder_lift, elbow) are searched within
a narrow band around their defaults.  The 3 wrist joints are searched more
broadly, subject to the constraint that the net wrist orientation keeps the
bottle vertical.

Each trial launches ``run_task.py --headless`` as a subprocess with the
candidate posture_config passed via the PLACE_POSTURE_CONFIG_OVERRIDE env
var, captures log output, and scores the run based on:
  - forced DownToInsert timeouts (arm couldn't reach target)
  - incremental verification failures (bottle not placed correctly)
  - residual distance at timeout (how far the arm was from target)

When no fixed --seed is given, each trial runs the same posture config
multiple times (default 3) with different random seeds and the score is
the mean across runs.  This reduces noise from seed-dependent variation.

Results are persisted to an SQLite database so the study can be resumed.

Usage:
    # Activate the Isaac Sim conda env first:
    mamba activate env_isaacsim51

    # Run with defaults (100 trials, 4 targets, 3 runs per trial):
    python optimize_posture.py

    # Faster exploration with fewer targets and fewer repeats:
    python optimize_posture.py --target-count 2 --runs-per-trial 2

    # Single run per trial with a fixed seed:
    python optimize_posture.py --seed 42

    # Resume a previous study:
    python optimize_posture.py --resume

    # Show best result from a previous study:
    python optimize_posture.py --show-best
"""
import argparse
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

try:
    import optuna
except ImportError:
    print("optuna is required: pip install optuna")
    sys.exit(1)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Baseline posture_config [-1.503686, -2.273453, -1.591299, -2.352803, 0.064269, 3.073532]

PROJECT_DIR = Path(__file__).resolve().parent
RUN_TASK_SCRIPT = PROJECT_DIR / "run_task.py"
LOG_DIR = PROJECT_DIR / "optuna_logs"
DB_PATH = PROJECT_DIR / "optuna_posture.db"

# Default posture (from task_context_base.PLACE_POSTURE_CONFIG)
# DEFAULT_SHOULDER_PAN = -math.pi / 2
# DEFAULT_SHOULDER_LIFT = -math.pi / 2
# DEFAULT_ELBOW = -math.pi / 2
# DEFAULT_WRIST_1 = math.pi / 2
# DEFAULT_WRIST_2 = -math.pi / 2
# DEFAULT_WRIST_3 = math.pi

DEFAULT_SHOULDER_PAN = -1.503686
DEFAULT_SHOULDER_LIFT = -2.273453
DEFAULT_ELBOW = -1.591299
DEFAULT_WRIST_1 = -2.352803
DEFAULT_WRIST_2 = 0.064269
DEFAULT_WRIST_3 = 3.073532

# Search ranges
# First 3 joints: narrow band around defaults (+-0.2 rad ~ +-11 deg)
SHOULDER_RANGE = 0.2
# Wrist joints: wider search
WRIST_1_RANGE = (-math.pi, math.pi)            # centered on 0
WRIST_2_RANGE = (-math.pi, math.pi/2)           #
WRIST_3_RANGE = (-math.pi / 4, math.pi)  #

# Worst-case score (used for crashed/timed-out runs)
WORST_SCORE = 20.0

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Pattern: "forcing SUCCESS after 15.0s ... dist=0.1009"
RE_FORCE_TIMEOUT = re.compile(
    r"(CortexDownToInsert|CortexMoveToPlace):\s*forcing SUCCESS.*?"
    r"dist=([0-9.]+)"
)

# Pattern: "still RUNNING after 10.0s ... dist=0.1022"
RE_TIMEOUT_WARNING = re.compile(
    r"(CortexDownToInsert|CortexMoveToPlace):\s*still RUNNING.*?"
    r"dist=([0-9.]+)"
)

# Pattern: "CortexDownToInsert: SUCCESS joint_angles=[j0, j1, j2, j3, j4, j5] pick=name"
RE_INSERT_JOINT_ANGLES = re.compile(
    r"CortexDownToInsert: SUCCESS joint_angles=\[([^\]]+)\]\s+pick=(\S+)"
)

# Pattern: "Incremental check FAIL: 'madara_bottle_9': not placed on any valid target"
RE_INCREMENTAL_FAIL = re.compile(r"Incremental check FAIL")

# Pattern: "Task ... Completed successfully"
RE_TASK_SUCCESS = re.compile(r"Task \S+ Completed successfully")

# Pattern: "Verification checks reported UNSUCCESSFUL"
RE_TASK_FAIL = re.compile(r"Verification checks reported UNSUCCESSFUL")


def parse_log(log_text: str) -> dict:
    """Parse a run_task.py log and extract metrics."""
    forced_timeouts = RE_FORCE_TIMEOUT.findall(log_text)
    timeout_warnings = RE_TIMEOUT_WARNING.findall(log_text)
    incremental_fails = RE_INCREMENTAL_FAIL.findall(log_text)
    task_success = bool(RE_TASK_SUCCESS.search(log_text))
    task_fail = bool(RE_TASK_FAIL.search(log_text))

    # Extract distances from forced timeouts
    forced_dists = [float(d) for _, d in forced_timeouts]
    warning_dists = [float(d) for _, d in timeout_warnings]

    # Count by behaviour type
    down_to_insert_forces = sum(1 for name, _ in forced_timeouts if "DownToInsert" in name)
    move_to_place_forces = sum(1 for name, _ in forced_timeouts if "MoveToPlace" in name)

    # Extract actual joint angles at each DownToInsert SUCCESS
    insert_joints = []
    for angles_str, pick_name in RE_INSERT_JOINT_ANGLES.findall(log_text):
        angles = [float(x.strip()) for x in angles_str.split(",")]
        insert_joints.append({"pick": pick_name, "joints": angles})

    return {
        "forced_timeouts": len(forced_timeouts),
        "down_to_insert_forces": down_to_insert_forces,
        "move_to_place_forces": move_to_place_forces,
        "timeout_warnings": len(timeout_warnings),
        "incremental_fails": len(incremental_fails),
        "task_success": task_success,
        "task_fail": task_fail,
        "forced_dists": forced_dists,
        "warning_dists": warning_dists,
        "avg_forced_dist": float(np.mean(forced_dists)) if forced_dists else 0.0,
        "insert_joints": insert_joints,
    }


def compute_score(metrics: dict, target_count: int) -> float:
    """Compute the optimization objective (lower is better).

    Scoring:
      - Each incremental verification failure: +3.0
      - Each forced DownToInsert timeout: +2.0
      - Each forced MoveToPlace timeout: +1.0
      - Average residual distance at timeout: +5.0 * avg_dist
      - Overall task failure (even partial): +2.0
      - Perfect run bonus: -1.0
    """
    score = 0.0

    # Verification failures are the primary metric
    score += 3.0 * metrics["incremental_fails"]

    # Forced timeouts indicate the arm can't reach the target
    score += 2.0 * metrics["down_to_insert_forces"]
    score += 1.0 * metrics["move_to_place_forces"]

    # Residual distance: how far off the arm was (continuous signal)
    if metrics["forced_dists"]:
        score += 5.0 * metrics["avg_forced_dist"]

    # Overall task result
    if metrics["task_fail"]:
        score += 2.0
    elif metrics["task_success"]:
        score -= 1.0  # bonus for clean success

    return score


# ---------------------------------------------------------------------------
# Trial execution
# ---------------------------------------------------------------------------

def run_single(posture_config: np.ndarray, target_count: int, seed: int,
               trial_num: int, run_idx: int, timeout_seconds: int) -> dict:
    """Run a single Isaac Sim invocation and return a result dict.

    Returns:
        dict with keys: metrics, score, log_path, seed
    """
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"trial_{trial_num:04d}_run_{run_idx}.log"

    # Serialize posture config as JSON list for the env var
    config_json = json.dumps(posture_config.tolist())

    env = os.environ.copy()
    env["PLACE_POSTURE_CONFIG_OVERRIDE"] = config_json

    cmd = [
        sys.executable, str(RUN_TASK_SCRIPT),
        "--task", "TableTaskBottlesToConveyor2",
        "--target-count", str(target_count),
        "--headless",
        "--seed", str(seed),
    ]

    logger.info(f"Trial {trial_num} run {run_idx}: seed={seed}, "
                f"posture_config={posture_config.tolist()}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_DIR),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.time() - start_time
        log_text = result.stdout + "\n" + result.stderr

        # Write log file
        with open(log_path, "w") as f:
            f.write(f"# Trial {trial_num}, run {run_idx}\n")
            f.write(f"# seed: {seed}\n")
            f.write(f"# posture_config: {config_json}\n")
            f.write(f"# elapsed: {elapsed:.1f}s\n")
            f.write(f"# return_code: {result.returncode}\n\n")
            f.write(log_text)

        if result.returncode != 0:
            logger.warning(f"Trial {trial_num} run {run_idx}: "
                           f"run_task.py exited with code {result.returncode}")

        metrics = parse_log(log_text)
        score = compute_score(metrics, target_count)

        logger.info(
            f"Trial {trial_num} run {run_idx}: score={score:.2f}, "
            f"seed={seed}, "
            f"forced_timeouts={metrics['forced_timeouts']}, "
            f"incremental_fails={metrics['incremental_fails']}, "
            f"task_success={metrics['task_success']}, "
            f"elapsed={elapsed:.1f}s"
        )

        return {
            "seed": seed,
            "score": score,
            "metrics": metrics,
            "log_path": str(log_path),
            "elapsed": elapsed,
        }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.warning(f"Trial {trial_num} run {run_idx}: TIMED OUT after {elapsed:.1f}s")
        with open(log_path, "w") as f:
            f.write(f"# Trial {trial_num}, run {run_idx}\n")
            f.write(f"# seed: {seed}\n")
            f.write(f"# posture_config: {config_json}\n")
            f.write(f"# TIMED OUT after {elapsed:.1f}s\n")
        metrics = {
            "forced_timeouts": target_count,
            "down_to_insert_forces": target_count,
            "move_to_place_forces": 0,
            "timeout_warnings": 0,
            "incremental_fails": target_count,
            "task_success": False,
            "task_fail": True,
            "forced_dists": [],
            "warning_dists": [],
            "avg_forced_dist": 0.5,
            "insert_joints": [],
        }
        return {
            "seed": seed,
            "score": WORST_SCORE,
            "metrics": metrics,
            "log_path": str(log_path),
            "elapsed": elapsed,
        }

    except Exception as e:
        logger.error(f"Trial {trial_num} run {run_idx}: unexpected error: {e}")
        return {
            "seed": seed,
            "score": WORST_SCORE,
            "metrics": {},
            "log_path": str(log_path),
            "elapsed": 0.0,
        }


def run_trial(posture_config: np.ndarray, target_count: int, fixed_seed: int,
              trial_num: int, runs_per_trial: int,
              timeout_seconds: int) -> tuple[list, float]:
    """Run one or more Isaac Sim invocations for a posture config.

    Args:
        fixed_seed: If not None, use this seed (single run regardless of
            runs_per_trial).
        runs_per_trial: Number of runs with different random seeds.

    Returns:
        (run_results, aggregated_score) where run_results is a list of dicts
        from run_single().
    """
    if fixed_seed is not None:
        seeds = [fixed_seed]
    else:
        seeds = [random.randint(1, 99999) for _ in range(runs_per_trial)]

    run_results = []
    for run_idx, seed in enumerate(seeds):
        result = run_single(
            posture_config, target_count, seed,
            trial_num=trial_num, run_idx=run_idx,
            timeout_seconds=timeout_seconds,
        )
        run_results.append(result)

    scores = [r["score"] for r in run_results]
    agg_score = float(np.mean(scores))

    n_success = sum(1 for r in run_results if r["metrics"].get("task_success"))
    seeds_str = ", ".join(str(r["seed"]) for r in run_results)
    scores_str = ", ".join(f"{s:.2f}" for s in scores)
    logger.info(
        f"Trial {trial_num} AGGREGATE: score={agg_score:.2f} "
        f"(runs: [{scores_str}]), "
        f"successes={n_success}/{len(run_results)}, "
        f"seeds=[{seeds_str}]"
    )

    return run_results, agg_score


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(target_count: int, seed: int, runs_per_trial: int,
                   timeout_seconds: int, fix_shoulders: bool):
    """Create an optuna objective function."""

    def objective(trial: optuna.Trial) -> float:
        # --- Sample joint angles ---
        if fix_shoulders:
            shoulder_pan = DEFAULT_SHOULDER_PAN
            shoulder_lift = DEFAULT_SHOULDER_LIFT
            elbow = DEFAULT_ELBOW
        else:
            shoulder_pan = trial.suggest_float(
                "shoulder_pan",
                DEFAULT_SHOULDER_PAN - SHOULDER_RANGE,
                DEFAULT_SHOULDER_PAN + SHOULDER_RANGE,
            )
            shoulder_lift = trial.suggest_float(
                "shoulder_lift",
                DEFAULT_SHOULDER_LIFT - SHOULDER_RANGE,
                DEFAULT_SHOULDER_LIFT + SHOULDER_RANGE,
            )
            elbow = trial.suggest_float(
                "elbow",
                DEFAULT_ELBOW - SHOULDER_RANGE,
                DEFAULT_ELBOW + SHOULDER_RANGE,
            )

        wrist_1 = trial.suggest_float("wrist_1", *WRIST_1_RANGE)
        wrist_2 = trial.suggest_float("wrist_2", *WRIST_2_RANGE)
        wrist_3 = trial.suggest_float("wrist_3", *WRIST_3_RANGE)

        posture_config = np.array([
            shoulder_pan, shoulder_lift, elbow,
            wrist_1, wrist_2, wrist_3,
        ])

        run_results, agg_score = run_trial(
            posture_config, target_count, fixed_seed=seed,
            trial_num=trial.number, runs_per_trial=runs_per_trial,
            timeout_seconds=timeout_seconds,
        )

        # Store per-run data as a list of summary dicts
        runs_summary = []
        all_insert_joints = []
        for i, r in enumerate(run_results):
            m = r["metrics"]
            run_info = {
                "run_idx": i,
                "seed": r["seed"],
                "score": r["score"],
                "elapsed": r["elapsed"],
                "log_path": r["log_path"],
                "forced_timeouts": m.get("forced_timeouts", 0),
                "down_to_insert_forces": m.get("down_to_insert_forces", 0),
                "move_to_place_forces": m.get("move_to_place_forces", 0),
                "incremental_fails": m.get("incremental_fails", 0),
                "task_success": m.get("task_success", False),
                "avg_forced_dist": m.get("avg_forced_dist", 0.0),
                "insert_joints": m.get("insert_joints", []),
            }
            runs_summary.append(run_info)
            # Collect insert_joints across all runs, tagging with seed
            for ij in m.get("insert_joints", []):
                all_insert_joints.append({**ij, "seed": r["seed"], "run_idx": i})

        trial.set_user_attr("runs", runs_summary)
        trial.set_user_attr("insert_joints", all_insert_joints)

        # Store convenient aggregates for querying
        trial.set_user_attr("num_runs", len(run_results))
        trial.set_user_attr("seeds", [r["seed"] for r in run_results])
        trial.set_user_attr("per_run_scores", [r["score"] for r in run_results])
        trial.set_user_attr("num_successes",
                            sum(1 for r in run_results
                                if r["metrics"].get("task_success")))
        trial.set_user_attr("all_succeeded",
                            all(r["metrics"].get("task_success")
                                for r in run_results))

        return agg_score

    return objective


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_best(study: optuna.Study):
    """Print the best trial found so far."""
    if len(study.trials) == 0:
        print("No completed trials yet.")
        return

    best = study.best_trial
    print("\n" + "=" * 70)
    print("BEST TRIAL")
    print("=" * 70)
    print(f"  Trial number: {best.number}")
    print(f"  Score:        {best.value:.4f}")
    print(f"  Params:")
    for name, val in sorted(best.params.items()):
        print(f"    {name:16s} = {val:+.6f}  ({math.degrees(val):+.2f} deg)")

    # Reconstruct full config
    config = np.array([
        best.params.get("shoulder_pan", DEFAULT_SHOULDER_PAN),
        best.params.get("shoulder_lift", DEFAULT_SHOULDER_LIFT),
        best.params.get("elbow", DEFAULT_ELBOW),
        best.params["wrist_1"],
        best.params["wrist_2"],
        best.params["wrist_3"],
    ])
    print(f"\n  Full posture_config (for pt_cortex_tree.py):")
    print(f"    np.array({config.tolist()})")
    print(f"\n  As env var:")
    print(f"    PLACE_POSTURE_CONFIG_OVERRIDE='{json.dumps(config.tolist())}'")

    # Per-run details
    num_runs = best.user_attrs.get("num_runs", 1)
    seeds = best.user_attrs.get("seeds", ["N/A"])
    per_run_scores = best.user_attrs.get("per_run_scores", [best.value])
    num_successes = best.user_attrs.get("num_successes", "N/A")

    print(f"\n  Runs:         {num_runs}")
    print(f"  Seeds:        {seeds}")
    print(f"  Per-run scores: {[f'{s:.2f}' for s in per_run_scores]}")
    print(f"  Successes:    {num_successes}/{num_runs}")

    # Per-run breakdown
    runs = best.user_attrs.get("runs", [])
    if runs:
        print(f"\n  Per-run details:")
        for r in runs:
            status = "OK" if r.get("task_success") else "FAIL"
            print(f"    run {r.get('run_idx', '?')}: seed={r.get('seed', '?'):>5}, "
                  f"score={r.get('score', 0):.2f}, {status}, "
                  f"timeouts={r.get('forced_timeouts', 0)}, "
                  f"fails={r.get('incremental_fails', 0)}")

    print("=" * 70)


def print_summary(study: optuna.Study):
    """Print a summary of all trials."""
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print("No completed trials.")
        return

    scores = [t.value for t in completed]
    all_succeeded = sum(1 for t in completed if t.user_attrs.get("all_succeeded", False))
    # Backward compat: old single-run trials have task_success but not all_succeeded
    if all_succeeded == 0:
        all_succeeded = sum(1 for t in completed if t.user_attrs.get("task_success", False))

    print(f"\n--- Study Summary ---")
    print(f"  Completed trials: {len(completed)}")
    print(f"  All-runs-succeeded: {all_succeeded}/{len(completed)} "
          f"({100*all_succeeded/len(completed):.0f}%)")
    print(f"  Score range:      [{min(scores):.2f}, {max(scores):.2f}]")
    print(f"  Score mean:       {np.mean(scores):.2f} +/- {np.std(scores):.2f}")

    # Top 5 trials
    top = sorted(completed, key=lambda t: t.value)[:5]
    print(f"\n  Top 5 trials:")
    for t in top:
        config_str = ", ".join(f"{v:+.3f}" for v in [
            t.params.get("shoulder_pan", DEFAULT_SHOULDER_PAN),
            t.params.get("shoulder_lift", DEFAULT_SHOULDER_LIFT),
            t.params.get("elbow", DEFAULT_ELBOW),
            t.params["wrist_1"],
            t.params["wrist_2"],
            t.params["wrist_3"],
        ])
        n_runs = t.user_attrs.get("num_runs", 1)
        n_succ = t.user_attrs.get("num_successes",
                                   1 if t.user_attrs.get("task_success") else 0)
        print(f"    #{t.number:4d}  score={t.value:6.2f}  "
              f"{n_succ}/{n_runs} OK  [{config_str}]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Bayesian optimization of PLACE_POSTURE_CONFIG",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--num-trials", "-n", type=int, default=100,
                        help="Number of optimization trials (default: 100)")
    parser.add_argument("--target-count", "-t", type=int, default=4,
                        help="Number of bottles per trial (default: 4)")
    parser.add_argument("--seed", "-s", type=int, default=None,
                        help="Fixed random seed for run_task.py (default: None = random). "
                             "When set, --runs-per-trial is forced to 1.")
    parser.add_argument("--runs-per-trial", "-r", type=int, default=3,
                        help="Number of runs (different seeds) per trial (default: 3). "
                             "Ignored when --seed is set.")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Timeout in seconds per single run (default: 300)")
    parser.add_argument("--fix-shoulders", action="store_true", default=True,
                        help="Fix shoulder/elbow joints at defaults (default: True)")
    parser.add_argument("--search-shoulders", action="store_true",
                        help="Also search shoulder/elbow joints (narrow range)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previous study from the database")
    parser.add_argument("--show-best", action="store_true",
                        help="Show best result and exit")
    parser.add_argument("--study-name", default="posture_opt",
                        help="Optuna study name (default: posture_opt)")
    parser.add_argument("--db", type=str, default=str(DB_PATH),
                        help=f"SQLite database path (default: {DB_PATH})")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet down optuna's own logging unless verbose
    if not args.verbose:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    fix_shoulders = args.fix_shoulders and not args.search_shoulders

    # Fixed seed implies single run (repeating the same seed is pointless)
    runs_per_trial = 1 if args.seed is not None else args.runs_per_trial

    storage = f"sqlite:///{args.db}"

    if args.show_best:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        print_best(study)
        print_summary(study)
        return

    # Create or load study
    if args.resume:
        study = optuna.load_study(study_name=args.study_name, storage=storage)
        logger.info(f"Resuming study '{args.study_name}' with {len(study.trials)} existing trials")
    else:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=storage,
            direction="minimize",
            load_if_exists=True,  # don't error if DB already exists
        )
        logger.info(f"Study '{args.study_name}' — {len(study.trials)} existing trials")

    # Enqueue the default config as the first trial so we have a baseline
    if len(study.trials) == 0:
        default_params = {
            "wrist_1": DEFAULT_WRIST_1,
            "wrist_2": DEFAULT_WRIST_2,
            "wrist_3": DEFAULT_WRIST_3,
        }
        if not fix_shoulders:
            default_params.update({
                "shoulder_pan": DEFAULT_SHOULDER_PAN,
                "shoulder_lift": DEFAULT_SHOULDER_LIFT,
                "elbow": DEFAULT_ELBOW,
            })
        study.enqueue_trial(default_params)
        logger.info("Enqueued default posture_config as first trial (baseline)")

    objective = make_objective(
        target_count=args.target_count,
        seed=args.seed,
        runs_per_trial=runs_per_trial,
        timeout_seconds=args.timeout,
        fix_shoulders=fix_shoulders,
    )

    logger.info(f"Starting optimization: {args.num_trials} trials, "
                f"{runs_per_trial} run(s)/trial, "
                f"{args.target_count} targets/run, "
                f"timeout={args.timeout}s/run, "
                f"fix_shoulders={fix_shoulders}")
    logger.info(f"Database: {args.db}")
    logger.info(f"Logs: {LOG_DIR}/")

    try:
        study.optimize(objective, n_trials=args.num_trials)
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")

    print_best(study)
    print_summary(study)


if __name__ == "__main__":
    main()
