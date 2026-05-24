"""Shared CLI utilities for run_task.py.

Consolidates task discovery, argument parsing helpers, task selection
resolution, seed setup, and count resolution.
"""
import argparse
import ast
import logging
import random
from dataclasses import dataclass, field
from importlib import import_module
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


_TASK_BASE_CLASSES = {"UR10MultiPickPlaceTask"}


def _base_name(node: ast.expr) -> str | None:
    """Extract the simple name from an AST base-class node."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _has_task_base(cls_node: ast.ClassDef, extra_base_names: set[str] | None = None) -> bool:
    """Return True if *cls_node* directly extends a known task base class.

    *extra_base_names* lets callers extend recognition to subclasses of
    already-discovered tasks (e.g., ``tasks2/`` cortex-tree variants that
    inherit from a v1 task in ``tasks/``).
    """
    accepted = _TASK_BASE_CLASSES if not extra_base_names else _TASK_BASE_CLASSES | extra_base_names
    return any(_base_name(b) in accepted for b in cls_node.bases)


_FLAWED_TASKS_PACKAGE = "flawed_tasks"
_DEFAULT_TASKS_PACKAGE = "tasks"
_SECONDARY_TASKS_PACKAGE = "tasks2"
_TASKS2_SEPARATOR_LABEL = "Tasks2"


def resolve_tasks_package(flawed: bool = False) -> str:
    """Return the primary tasks package name based on whether --flawed mode is active."""
    return _FLAWED_TASKS_PACKAGE if flawed else _DEFAULT_TASKS_PACKAGE


def discover_task_modules(
    tasks_package: str = _DEFAULT_TASKS_PACKAGE,
    extra_base_names: set[str] | None = None,
) -> dict[str, str]:
    """Return a mapping of task class names to their module paths discovered in *tasks_package*.

    *extra_base_names* extends recognition to subclasses of those names
    (in addition to ``UR10MultiPickPlaceTask``).  Used for ``tasks2/``
    where v2 classes inherit from their v1 counterpart in ``tasks/``.
    Within a single package we also do a fixed-point pass so a class
    that inherits from another class in the same package is discovered
    regardless of file-iteration order.
    """
    tasks_path = Path(__file__).parent / tasks_package
    available_tasks: dict[str, str] = {}
    if not tasks_path.exists():
        return available_tasks

    # Parse every file once; reuse ASTs across the fixed-point passes.
    parsed: list[tuple[Path, str, ast.Module]] = []
    for task_file in tasks_path.glob("*.py"):
        if task_file.name == "__init__.py":
            continue
        module_name = f"{tasks_package}.{task_file.stem}"
        try:
            source = task_file.read_text(encoding="utf-8")
        except OSError:
            continue
        try:
            module_ast = ast.parse(source, filename=str(task_file))
        except SyntaxError:
            continue
        parsed.append((task_file, module_name, module_ast))

    accepted_extra: set[str] = set(extra_base_names or ())
    while True:
        added = False
        for _task_file, module_name, module_ast in parsed:
            for node in module_ast.body:
                if not isinstance(node, ast.ClassDef):
                    continue
                if node.name in available_tasks:
                    continue
                if _has_task_base(node, accepted_extra):
                    available_tasks[node.name] = module_name
                    accepted_extra.add(node.name)
                    added = True
        if not added:
            break

    return available_tasks


@dataclass
class TaskCatalog:
    """Two-tier catalog of discovered tasks: primary (tasks/) + secondary (tasks2/).

    Listing/indexing semantics:
      - Default: primary names (sorted) followed by secondary names (sorted),
        with continuous 1-based numbering.
      - tasks2_only=True at construction time restricts the catalog to the
        secondary tier (numbering then starts from 1 within tasks2/).
    """
    primary: dict[str, str] = field(default_factory=dict)
    secondary: dict[str, str] = field(default_factory=dict)
    tasks2_only: bool = False

    @property
    def primary_names(self) -> list[str]:
        return [] if self.tasks2_only else sorted(self.primary)

    @property
    def secondary_names(self) -> list[str]:
        # When duplicates exist across tiers, the secondary entry is hidden
        # in default mode (primary wins). With --tasks2 the secondary tier
        # is shown standalone, so duplicates are not filtered.
        if self.tasks2_only:
            return sorted(self.secondary)
        return sorted(n for n in self.secondary if n not in self.primary)

    @property
    def ordered_names(self) -> list[str]:
        return self.primary_names + self.secondary_names

    @property
    def modules(self) -> dict[str, str]:
        """Combined name -> module path used for resolving a task to import."""
        if self.tasks2_only:
            return dict(self.secondary)
        merged = dict(self.secondary)
        merged.update(self.primary)  # primary wins on duplicates
        return merged

    def __contains__(self, name: str) -> bool:
        return name in self.modules

    def __bool__(self) -> bool:
        return bool(self.modules)


def discover_task_catalog(flawed: bool = False, tasks2_only: bool = False,
                          tasks1_only: bool = False) -> TaskCatalog:
    """Discover tasks across the primary package and the secondary tasks2/ package.

    With --flawed or --tasks1, tasks2/ is not consulted (single-tier).
    Emits a warning for any name present in both tiers.
    """
    if tasks1_only and tasks2_only:
        raise SystemExit("--tasks1 and --tasks2 are mutually exclusive.")
    primary_pkg = resolve_tasks_package(flawed)
    primary = discover_task_modules(primary_pkg)
    if flawed or tasks1_only:
        return TaskCatalog(primary=primary, secondary={}, tasks2_only=False)

    secondary = discover_task_modules(_SECONDARY_TASKS_PACKAGE, extra_base_names=set(primary))
    duplicates = sorted(set(primary) & set(secondary))
    for name in duplicates:
        logger.warning(
            "Task '%s' found in both `%s/` and `%s/`; using `%s/` version.",
            name, primary_pkg, _SECONDARY_TASKS_PACKAGE, primary_pkg,
        )
    return TaskCatalog(primary=primary, secondary=secondary, tasks2_only=tasks2_only)


def resolve_task_class(task_name: str, task_modules) -> type:
    """Import and return the task class for *task_name*.

    Accepts either a TaskCatalog or a plain dict (legacy callers).
    """
    if isinstance(task_modules, TaskCatalog):
        modules = task_modules.modules
    else:
        modules = task_modules
    if task_name not in modules:
        available = ", ".join(sorted(modules))
        raise ValueError(
            f"Task '{task_name}' not found. Available: {available}"
            if available else "No task classes found."
        )
    module = import_module(modules[task_name])
    return getattr(module, task_name)


def add_common_task_arguments(parser, available_task_names: str) -> None:
    """Add the CLI arguments shared across task runner entry points."""
    parser.add_argument(
        "--flawed", action="store_true",
        help=(
            f"Load tasks from the `{_FLAWED_TASKS_PACKAGE}/` package instead of `{_DEFAULT_TASKS_PACKAGE}/`."
        ),
    )
    parser.add_argument(
        "--tasks2", action="store_true",
        help=(
            f"Restrict listing/indexing to the `{_SECONDARY_TASKS_PACKAGE}/` package "
            "(numbering then starts from 1 within tasks2)."
        ),
    )
    # Hidden inverse of --tasks2: hide the tasks2/ tier entirely.
    parser.add_argument(
        "--tasks1", action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help=(
            "With --list: show task_description for each task."
        ),
    )
    parser.add_argument(
        "task_positional",
        nargs="?",
        help="Task name, index (>=1), or random (<0). --task takes precedence if both provided.",
    )
    parser.add_argument(
        "--task", default=None,
        help=(
            "Specify the task class to run. "
            f"Available options: {available_task_names}."
        ),
    )
    parser.add_argument(
        "--list", "-l", action="store_true",
        help="List available tasks and exit",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--no-randomize", action="store_true", help="Disable randomization in item generation.")
    parser.add_argument("--teleport", "-t", action="store_true", help="Teleport mode: skip robot movement and teleport objects to targets.")
    parser.add_argument("--no-incremental-checks", action="store_true", help="Disable per-item incremental verification after each pick-place cycle.")
    parser.add_argument("--pause", action="store_true", help="Pause after each pick-place cycle for inspection.")

    # Pick count arguments
    parser.add_argument("--pick-count", type=int, default=None, help="Fixed number of pick objects.")
    parser.add_argument("--pick-count-min", type=int, default=None, help="Minimum number of pick objects.")
    parser.add_argument("--pick-count-max", type=int, default=None, help="Maximum number of pick objects.")

    # Target count arguments
    parser.add_argument("--target-count", type=int, default=None, help="Fixed number of target objects.")
    parser.add_argument("--target-count-min", type=int, default=None, help="Minimum number of target objects.")
    parser.add_argument("--target-count-max", type=int, default=None, help="Maximum number of target objects.")

    # Dynamic (incremental) spawn interval overrides (seconds).  Apply only
    # when the task's TaskSpec configures a pick/target incremental scheduler
    # — ignored silently otherwise.
    parser.add_argument(
        "--dynamic-pick-interval", type=float, default=None,
        help="Override the batch interval (seconds) for incremental pick spawning.",
    )
    parser.add_argument(
        "--dynamic-target-interval", type=float, default=None,
        help="Override the batch interval (seconds) for incremental target spawning.",
    )

    # Cycle-time gate: enforce a minimum elapsed time between pick-place
    # cycles so mock / teleport runs (which would otherwise complete at near-
    # zero simulated time) take long enough for conveyor-driven scenarios
    # (item drift, fall-off-the-edge) to be observable. 0 disables the gate.
    # Mock mode evaluates against the 10 Hz mock_time; real-sim against
    # CortexWorld simulation time. Inserted before SelectNextPick in both BTs.
    parser.add_argument(
        "--min-cycle-time", type=float, default=0.0,
        help=(
            "Minimum elapsed seconds between pick-place cycles (default 0 = "
            "off). Useful for mock/teleport modes to keep the BT idle long "
            "enough to see conveyor-driven dynamics."
        ),
    )


def _extract_task_description(module_path: str) -> str | None:
    """Extract the task_description string literal from a task module file via AST."""
    parts = module_path.split(".")
    file_path = Path(__file__).parent / Path(*parts[:-1]) / f"{parts[-1]}.py"
    if not file_path.exists():
        return None
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))
    except (OSError, SyntaxError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            func_name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else None
            )
            # Match TaskSpec(...) for primary definitions, plus replace(...)
            # for wrapper variants that customize a parent's spec via
            # dataclasses.replace(spec, task_description=..., ...).
            if func_name in ("TaskSpec", "replace"):
                for kw in node.keywords:
                    if kw.arg == "task_description" and isinstance(kw.value, ast.Constant):
                        return str(kw.value.value)
    return None


def list_tasks(catalog: TaskCatalog, verbose: bool = False,
               only_index: int | None = None, only_name: str | None = None) -> None:
    """Print available tasks with 1-based indices.

    Listing order:
      - Default: primary tier (tasks/) first, then a 'Tasks2' separator,
        then the secondary tier (tasks2/), with continuous numbering.
      - tasks2_only catalog: only the secondary tier, numbered from 1.

    If *only_index* is a positive integer, print only the entry at that index.
    If *only_name* is a task name string, print only that entry.
    """
    primary_names = catalog.primary_names
    secondary_names = catalog.secondary_names
    ordered_names = primary_names + secondary_names
    modules = catalog.modules

    def _print_entry(idx: int, name: str) -> None:
        print(f"  {idx:2d}.{name}")
        if verbose:
            desc = _extract_task_description(modules[name])
            if desc:
                print(f"\t{desc}")

    if only_name is not None:
        if only_name not in modules:
            raise SystemExit(f"Task '{only_name}' not found. Available: {', '.join(ordered_names)}")
        if only_name in primary_names:
            idx = primary_names.index(only_name) + 1
        else:
            idx = len(primary_names) + secondary_names.index(only_name) + 1
        _print_entry(idx, only_name)
        return

    if only_index is not None:
        if not (1 <= only_index <= len(ordered_names)):
            raise SystemExit(f"Task index {only_index} out of range (1-{len(ordered_names)})")
        _print_entry(only_index, ordered_names[only_index - 1])
        return

    print("Available tasks:")
    for i, name in enumerate(primary_names, 1):
        _print_entry(i, name)
    if secondary_names and not catalog.tasks2_only:
        print(f"--- {_TASKS2_SEPARATOR_LABEL} ---")
    start = len(primary_names) + 1
    for offset, name in enumerate(secondary_names):
        _print_entry(start + offset, name)


def handle_list_request(args, catalog: TaskCatalog) -> bool:
    """If --list was requested, print matching tasks and return True (caller should return)."""
    if not args.list:
        return False
    only_index = None
    only_name = None
    filter_arg = args.task if args.task is not None else args.task_positional
    if filter_arg is not None:
        try:
            idx = int(filter_arg)
            if idx > 0:
                only_index = idx
        except ValueError:
            only_name = filter_arg
    list_tasks(catalog, verbose=args.verbose, only_index=only_index, only_name=only_name)
    return True


def resolve_task_selection(args, catalog: TaskCatalog, default_task: str) -> None:
    """Resolve the task name from positional arg / --task / default into args.task."""
    available_task_names_list = catalog.ordered_names

    positional_task = None
    if args.task_positional is None and args.task is None and default_task is None:
        args.task_positional = -1 # randomly select one of the available tasks
    if args.task_positional is not None:
        try:
            idx = int(args.task_positional)
            if idx >= 1:
                if 1 <= idx <= len(available_task_names_list):
                    positional_task = available_task_names_list[idx - 1]
                    logger.info(f"Positional index {idx} resolved to task: {positional_task}")
                else:
                    raise SystemExit(f"Task index {idx} out of range (1-{len(available_task_names_list)})")
            elif idx < 0:
                positional_task = random.choice(available_task_names_list)
                logger.info(f"Positional random index {idx} resolved to task: {positional_task}")
        except ValueError:
            positional_task = args.task_positional

    if args.task is not None:
        if positional_task is not None and args.task != positional_task:
            logger.warning(
                f"Both positional task and --task specified. --task '{args.task}' "
                f"takes precedence over positional '{positional_task}'."
            )
    elif positional_task is not None:
        args.task = positional_task
    else:
        args.task = default_task
    logger.info(f"Selected task: {args.task}")


def setup_seed(args) -> None:
    """Initialize RNG seeds from args.seed (generating one if None)."""
    if args.seed is None:
        args.seed = random.randint(1, 99999)
        logger.info(f"No seed specified. Using random seed: {args.seed}")
    else:
        logger.info(f"Using specified seed: {args.seed}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
    except ImportError:
        pass


def resolve_counts(args):
    """Resolve pick_count and target_count from CLI args.

    Returns (pick_count, target_count) where each is an int, tuple, or None.
    """
    pick_count = args.pick_count
    if pick_count is None and (args.pick_count_min is not None or args.pick_count_max is not None):
        pick_count = (args.pick_count_min or 1, args.pick_count_max)

    target_count = args.target_count
    if target_count is None and (args.target_count_min is not None or args.target_count_max is not None):
        target_count = (args.target_count_min or 1, args.target_count_max)

    return pick_count, target_count


def resolve_dynamic_intervals(args):
    """Resolve CLI overrides for incremental pick/target batch intervals.

    Returns ``(dynamic_pick_interval, dynamic_target_interval)`` where each
    element is a float (seconds) or ``None`` when the user did not override.
    """
    return (
        getattr(args, "dynamic_pick_interval", None),
        getattr(args, "dynamic_target_interval", None),
    )
