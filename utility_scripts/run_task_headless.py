# Copied and modified from run_task.py for headless execution with timeout
import argparse
import ast
from importlib import import_module
from pathlib import Path
import time
import sys
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path to allow imports from parent directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def _discover_task_modules() -> dict[str, str]:
    """Return a mapping of task class names to their module paths discovered in ``tasks``."""
    tasks_package = "tasks"
    # tasks directory is in project root
    tasks_path = Path(project_root) / tasks_package
    available_tasks: dict[str, str] = {}
    if not tasks_path.exists():
        return available_tasks

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
        for node in module_ast.body:
            if isinstance(node, ast.ClassDef):
                available_tasks[node.name] = module_name
    return available_tasks


def _resolve_task_class(task_name: str, task_modules: dict[str, str]) -> type:
    if task_name not in task_modules:
        raise ValueError(f"Task '{task_name}' not found.")
    module = import_module(task_modules[task_name])
    return getattr(module, task_name)


def main() -> None:
    # Use TableTaskMixedPacking by default for this test
    # Force headless=True
    # Force auto-exit logic
    
    available_task_modules = _discover_task_modules()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="TableTaskMixedPacking")
    parser.add_argument("--auto-exit", action="store_true", default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--pick-count", type=int)
    args = parser.parse_args()

    if args.seed is not None:
        import random
        random.seed(args.seed)
        np.random.seed(args.seed)

    if args.task not in available_task_modules:
        print(f"Error: {args.task} not found")
        sys.exit(1)

    print("Initializing SimulationApp in HEADLESS mode...")
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})

    from isaacsim.core.api import World
    from isaacsim.core.utils.stage import get_stage_units

    my_world = World(stage_units_in_meters=1.0)
    
    task_cls = _resolve_task_class(args.task, available_task_modules)
    
    # Instantiate task (using pick-count if provided to limit duration if needed)
    my_task = task_cls(seed=args.seed, pick_count=args.pick_count)

    my_world.add_task(my_task)
    my_world.reset()

    reset_needed = False
    success_checked = False
    
    start_time = time.time()
    MAX_DURATION = 180 # 3 minutes

    print("Starting simulation loop...")
    while simulation_app.is_running():
        # Watchdog
        if time.time() - start_time > MAX_DURATION:
            print("TIMEOUT: Simulation ran longer than 3 minutes. Forcing exit.")
            break
            
        my_world.step(render=False) 
        
        if my_world.is_stopped() and not reset_needed:
            reset_needed = True
        
        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                reset_needed = False
            
            current_tasks = my_world.get_current_tasks()
            for task_name in current_tasks:
                task = current_tasks[task_name]
                if hasattr(task,"task_step"):
                    task.task_step()
                
                if hasattr(task,"can_exit") and task.can_exit():
                    if not success_checked:
                        print(f"Task {task_name} signaled completion. Checking success...")
                        task.check_groundtruth_task_success()
                        success_checked = True
                    elif args.auto_exit:
                        print(f"Task {task_name} signaled exit. Shutting down.")
                        my_world.clear()
                        simulation_app.close()
                        sys.exit(0)

    simulation_app.close()

if __name__ == "__main__":
    main()
