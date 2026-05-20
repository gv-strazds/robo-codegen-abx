# Gemini Guidelines

This document provides guidelines for the Gemini AI assistant to follow when working in this repository.

## Project Overview

Isaac Sim 5.1 simulation of a UR10 robotic arm performing multi-object pick-and-place tasks. Objects are picked from a bin or conveyor and placed onto targets (markers, containers, or specific locations on a cart). Tasks are parameterizable with different object types, colors, shapes, and arrangement patterns.

Active development is on the `pt-cortex` and `curobo` branches. `pt-cortex` integrated IsaacSim's Cortex framework (CortexWorld, CortexUr10, MotionCommand) alongside the existing `py_trees` BT architecture, and there are now two parallel task sets in `tasks/` (classic) and `tasks2/` (cortex-style). The `pytrees-integration` branch is legacy.

## Environment Setup
This repository requires a specific Conda (Mamba) environment to run correctly.

**Environment Name:** `env_isaacsim51`

Ensure this environment is activated before running any scripts:
```bash
mamba activate env_isaacsim51
```

## Tech Stack

*   **Primary Language:** Python
*   **Simulation Environment:** Isaac Sim 5.1
*   **Behavior Framework:** `py_trees`

## Getting Started

### Running Simulation Tasks

The main entry point for the simulation is `run_task.py`.

```bash
python run_task.py --task <TaskName>
```

The `--task` argument corresponds to class names in the `tasks/` directory (e.g., `TableTask3`, `TableTaskColors1`, `TableTaskSoupCans1`).

### Running Mock Tasks (No Isaac Sim required)

For testing behavior trees and logic without launching the full simulation:

```bash
python run_mock_task.py --task TableTask3
python run_mock_task.py --task TableTaskColors1 --seed 42 --show-status
python run_mock_task.py --list   # list available tasks
```

`run_mock_task.py` runs any task configuration through the mock py_trees BT, with task verification, configurable pick/target counts, and optional `--show-status` to display the py_trees tree status and DEBUG logging after each tick. See `--help` for all options.

### Running Tests

The testing suite requires the `env_isaacsim51` environment and proper `PYTHONPATH` setup to include mocks.

**Run all tests:**
```bash
mamba run -n env_isaacsim51 env PYTHONPATH=$(pwd)/extsMock:$(pwd) python -m pytest tests/
```

**Run a single test:**
```bash
mamba run -n env_isaacsim51 env PYTHONPATH=$(pwd)/extsMock:$(pwd) python -m pytest tests/test_mock_controllers.py
```

Note: `pytest` is normally installed as a bash command in `env_isaacsim51`; if it isn't on a given machine, fall back to `python -m pytest`.

### Generating PDF Documentation

To regenerate PDF files from Markdown in the `docs/` directory, use `npx md-to-pdf`:

```bash
npx md-to-pdf docs/mock-system-and-testing-design.md
npx md-to-pdf docs/task-environment-setup-design.md
```

## Architecture

### Task System

*   `run_task.py`: Main entry point; dynamically imports task classes.
*   `multi_pickplace_task.py`: `UR10MultiPickPlaceTask` base class. Thin bridge to IsaacSim's `BaseTask`. Manages robot, scene objects, and task lifecycle.
*   `task_spec.py`: `TaskSpec` (scene/description) plus a nested `TaskImplementationSpec` (policy: pairing strategy factory, BT tree factory, postures, hover heights, watchdog timeouts, cuRobo flags). Task classes build a `TaskSpec` with `implementation=TaskImplementationSpec(...)` and pass it to `super().__init__(task_spec=spec, ...)` instead of overriding `setup_workspace()` / `_create_strategy()`.
*   `simulation_configurator.py`: `SimulationConfigurator` manages scene objects, geometry cache, and verification. Extracted from `multi_pickplace_task.py`.
*   `task_controller.py`: `TaskController` policy layer connecting strategy, `TaskContext`, and the BT controller. Extracted from `multi_pickplace_task.py`.
*   `multi_pick_strategy.py`: Strategy subclasses (`ColorMatchStrategy`, `TypeBasedStrategy`, etc.) own pick-to-target pairing, iteration, EE orientation, and verification hooks.
*   `table_setup.py`: Isaac Sim scene setup helpers (tables, bins, objects). Re-exports workspace constants from `env_config_values.py` for back-compat.
*   `env_config_values.py`: Isaac-Sim-free workspace geometry constants (`BIN_X_COORD`, `ITEM_SPAWN_REFERENCE_Z`, `CART_SURFACE_CENTER`, `Region2D`, `compute_region_2d`, etc.) — importable from `TaskSpec`, tests, and mock paths without requiring Isaac Sim.

### Controller Hierarchy (py_trees)

The system uses a `py_trees` BehaviourTree for task orchestration:
- `ContextMonitorBehaviour` (in `robot_controllers/pt_context_monitor.py`) queries the `TaskContext` and updates the blackboard.
- The tree handles picking, placing, and error recovery through a sequence of phases.
- Task-level behaviours query `TaskContext` directly to avoid stale data.

### Context and Mocking

*   `task_context_base.py`: Shared base class for interacting with robot hardware/simulation.
*   `task_context.py`: Implementation for real Isaac Sim environments.
*   `task_context_mock.py`: `MockTaskContext` for testing without Isaac Sim.
*   `extsMock/`: Mock implementations of Isaac Sim and USD (pxr) interfaces.

### Generation System

*   `item_generation.py`: `ItemGenerator` creates objects using strategy patterns (Fixed, Random, Sequential).
*   `asset_utils.py`: Isaac-Sim-dependent asset scene operations (prim creation, live-prim AABB computation, semantic-label helpers). Re-exports pure-data symbols from `asset_data_utils`.
*   `asset_data_utils.py`: Isaac-Sim-free asset metadata and geometry (`AssetMetaData`, `ITEMS_MAP`, `PRIM_TYPES`, `PrimGeometry`, `scale_aabb`, `lookup_prim_geometry`) — importable from mock tasks and unit tests.

## AI Assistant Guidelines

### Key Constraints

*   **Do not modify files in the `exts/` directory.** These are external Isaac Sim libraries.
*   **Do not modify files in `extsMock/`** unless extending mock coverage for new Isaac Sim APIs.
*   **PYTHONPATH Management:** When running tests or mock scripts, `PYTHONPATH` must include `extsMock/` FIRST to shadow real Isaac Sim packages. When running real simulation tasks via `run_task.py`, DO NOT include `extsMock/` in `PYTHONPATH`.
*   **Documentation:** When editing Markdown files in the `docs/` directory, always regenerate the corresponding PDF files using `npx md-to-pdf <file>.md` to keep them in sync.

### Code Style

*   **Formatting:** PEP 8, 4-space indentation, 100-120 char line length.
*   **Naming:** `snake_case` for files/functions/variables, `PascalCase` for classes.
*   **Commits:** Use imperative mood (e.g., "Add feature", "Fix bug"). Prefixes like `feat:`, `fix:`, `refactor:` are preferred.
