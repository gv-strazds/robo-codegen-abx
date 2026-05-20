"""Harvest random seeds from a logs directory produced by run_all_simulation_tasks.sh
or run_all_teleport_tasks.sh, and emit a JSON seed map suitable for SEEDS_FILE.

Per-run logs are named "<task_idx>-<run_j>-<mode>.out" (mode is "sim" or "teleport").
Each contains a line like:
    [Warning] [__main__] Running Task TableTask1v2 with seed:62462.
We extract (task_name, run_j, seed) tuples and group seeds by task_name, ordered
by run_j so the resulting map is consumable by _run_all_tasks_lib.sh.

Usage:
    python utility_scripts/harvest_seeds.py                      # scan ./logs
    python utility_scripts/harvest_seeds.py _results/logs-001    # scan a different dir
    python utility_scripts/harvest_seeds.py --mode teleport
    python utility_scripts/harvest_seeds.py -o seeds.json        # write to file
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

LOG_NAME_RE = re.compile(r"^(?P<idx>\d+)-(?P<run>\d+)-(?P<mode>sim|teleport)\.out$")
SEED_RE = re.compile(r"Running Task\s+(?P<task>\S+)\s+with\s+seed:(?P<seed>\d+)")


def _extract_from_file(path: Path) -> tuple[str, int] | None:
    """Return (task_name, seed) from the first matching line in *path*, or None."""
    try:
        with path.open("r", errors="replace") as f:
            for line in f:
                m = SEED_RE.search(line)
                if m:
                    return m.group("task"), int(m.group("seed"))
    except OSError as e:
        print(f"WARNING: could not read {path}: {e}", file=sys.stderr)
    return None


def harvest(logs_dir: Path, mode: str) -> dict[str, list[int]]:
    """Scan *logs_dir* for per-run logs and return {task_name: [seed_for_run1, ...]}.

    *mode* is one of "sim", "teleport", or "auto". In "auto" mode we prefer the
    mode with more matching files; ties go to "sim".
    """
    by_mode: dict[str, list[tuple[int, int, str, int]]] = defaultdict(list)
    # entries are (idx, run_j, task_name, seed) tuples per mode
    for entry in sorted(logs_dir.iterdir()):
        if not entry.is_file():
            continue
        m = LOG_NAME_RE.match(entry.name)
        if not m:
            continue
        idx = int(m.group("idx"))
        run_j = int(m.group("run"))
        file_mode = m.group("mode")
        result = _extract_from_file(entry)
        if result is None:
            continue
        task_name, seed = result
        by_mode[file_mode].append((idx, run_j, task_name, seed))

    if mode == "auto":
        if not by_mode:
            return {}
        if len(by_mode) == 1:
            mode = next(iter(by_mode))
        else:
            sim_n = len(by_mode.get("sim", []))
            tel_n = len(by_mode.get("teleport", []))
            mode = "sim" if sim_n >= tel_n else "teleport"
            print(
                f"NOTE: auto-selected mode={mode} (sim files: {sim_n}, teleport files: {tel_n}). "
                f"Use --mode to force.",
                file=sys.stderr,
            )

    rows = by_mode.get(mode, [])
    if not rows:
        return {}

    # Group by task_name, sorted by (idx, run_j) so seed order matches run order.
    grouped: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for idx, run_j, task_name, seed in rows:
        grouped[task_name].append((idx, run_j, seed))

    out: dict[str, list[int]] = {}
    for task_name, items in grouped.items():
        items.sort(key=lambda r: (r[0], r[1]))
        out[task_name] = [seed for _, _, seed in items]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs_dir", nargs="?", default="logs",
                    help="Directory containing per-run logs (default: logs)")
    ap.add_argument("--mode", choices=("sim", "teleport", "auto"), default="auto",
                    help="Which run mode's logs to harvest (default: auto)")
    ap.add_argument("-o", "--output", default=None,
                    help="Write JSON to this file instead of stdout")
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.is_dir():
        print(f"ERROR: '{logs_dir}' is not a directory", file=sys.stderr)
        return 2

    seed_map = harvest(logs_dir, args.mode)
    payload = json.dumps(seed_map, indent=2, sort_keys=True)

    if args.output:
        Path(args.output).write_text(payload + "\n")
        total = sum(len(v) for v in seed_map.values())
        print(f"Wrote {total} seeds across {len(seed_map)} tasks to {args.output}",
              file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
