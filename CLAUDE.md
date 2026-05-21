# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Isaac Sim 5.1 simulation of a UR10 robotic arm performing multi-object pick-and-place tasks. Objects are picked from a bin/conveyor and placed onto targets (markers, containers, or specific locations on a cart). Tasks are parameterizable with different object types, colors, shapes, and arrangement patterns.

Active development is currently on the `pt-cortex` and `curobo` branches. The `pt-cortex` branch integrated IsaacSim's Cortex framework for robot control (CortexWorld, CortexUr10, MotionCommand) alongside the existing py_trees behavior tree architecture. There are now two parallel sets of task implementations, in the `tasks/` and `tasks2/` subdirectories. The former are the "classic" py_trees versions; the `tasks2/` versions differ primarily in that they use the newer "cortex-style" behavior tree for controlling the robot (a few have more significant differences from their original version). The `pytrees-integration` branch is now a legacy branch that has the pre-Cortex py_trees work.

## Environment & Commands

**Required conda environment:** `env_isaacsim51`

```bash
mamba activate env_isaacsim51
```

**Run a simulation task:**

```bash
python run_task.py --task TableTask3
```

Task names correspond to classes in `tasks/` (e.g., `TableTaskColors1`, `TableTaskBottles1`, `TableTaskMixedPacking`).

**Run project tests:**

```bash
mamba run -n env_isaacsim51 env PYTHONPATH=$(pwd)/extsMock:$(pwd) python -m pytest tests/
```

**Run a single test:**

```bash
mamba run -n env_isaacsim51 env PYTHONPATH=$(pwd)/extsMock:$(pwd) python -m pytest tests/test_mock_controllers.py
```

**Run mock task (no Isaac Sim required):**

```bash
python run_mock_task.py --task TableTask3
python run_mock_task.py --task TableTaskColors1 --seed 42 --show-status
python run_mock_task.py --list   # list available tasks
```

`run_mock_task.py` runs any task configuration through the mock py_trees BT, with task verification, configurable pick/target counts, and optional `--show-status` to display the py_trees tree status and DEBUG logging after each tick. See `--help` for all options.

Common diagnostic flags for `run_task.py`:
- `--auto-exit` / `--headless` — exit when the task completes; `--headless` implies `--auto-exit`.
- `--video` / `--snapshots` / `--snapshot-errors` — capture to `_results/snapshots/<task>_<ts>/` (PNG + sidecar JSON) and `_results/videos/<task>_<ts>.mp4`. `--snapshot-errors` only fires on failure events (plus a task-final frame). These artifacts are agent-readable — see "Visual Debugging with Snapshots and Video" in `docs/mock-system-and-testing-design.md` for the recommended `--snapshot-errors` → `--snapshots` → `--video` escalation workflow.
- `--physics-dt`, `--rendering-dt`, `--psteps-per-render` — physics/render rates and substep ratio.
- `--max-sim-time SECONDS` — hard cap on simulated time (`World.current_time`), not wall-clock.
- `--telemetry-csv PATH` — log per-step telemetry.
- `--tasks1` / `--tasks2` — restrict task discovery to one of the two parallel implementation sets.

**Capturing output without hanging (agent / automation note):** when invoking `run_task.py` via `mamba run` non-interactively, redirect stdout+stderr to a file (`> /tmp/run.log 2>&1`) instead of piping to `tail` / `head` / `grep`. Piping has been observed to leave the `mamba run` wrapper hanging indefinitely after Isaac Sim itself has exited cleanly (the simulation produces its snapshot artifacts and shuts down, but the wrapper never returns). Redirect-then-`tail -n N` the log afterward is the safe pattern; if a hang does occur, `kill -9` the lingering `mamba run …` pid.

## Architecture

### Task System

- `run_task.py` — Main entry point. Dynamically imports task classes from `tasks/` via `--task` argument.
- `multi_pickplace_task.py` — `UR10MultiPickPlaceTask` base class. Thin bridge to IsaacSim's `BaseTask`. Accepts a `TaskSpec` and delegates to `SimulationConfigurator` (scene/geometry/verification) and `TaskController` (strategy/context/BT). Still manages robot and task lifecycle.
- `task_spec.py` — Two dataclasses split by concern:
  - `TaskSpec` (scene/description side): generators, workspace, conveyor, verification semantics, scene metadata.
  - `TaskImplementationSpec` (policy side, nested under `TaskSpec.implementation`): pairing strategy factory, BT tree factory, virtual-target generation, postures, hover heights, watchdog timeouts, reachability gates, cuRobo flags.
  - Tasks build a `TaskSpec` with `implementation=TaskImplementationSpec(...)` and pass it to `super().__init__(task_spec=spec, ...)`. The split lets `SimulationConfigurator` be built from `TaskSpec` alone, with `TaskController`/`TaskContext` constructed from both halves later.
  - Helpers: `TaskSpec.with_impl(**kw)` for partial impl overrides (used by v2 `_customize_spec`); `TaskSpec.impl` returns a default-constructed `TaskImplementationSpec` when `implementation` is `None`.
- `simulation_configurator.py` — `SimulationConfigurator` manages scene objects, geometry cache, and verification (extracted from `UR10MultiPickPlaceTask`).
- `task_controller.py` — `TaskController` policy layer wrapping `MultiPickStrategy`, owns strategy creation, `TaskContext`, and BT controller.
- `multi_pick_strategy.py` — `MultiPickStrategy` and subclasses (`ColorMatchStrategy`, `TypeBasedStrategy`, `BottlePickStrategy`). Owns pick-to-target pairing computation, pick iteration, EE orientation decisions (pick and drop), completion tracking, and verification hooks. Geometry computations live in `TaskContextBase`, not here.
- `table_setup.py` — Isaac Sim scene setup helpers (tables, robot mount, picking bin, objects). Re-exports pure-Python constants/helpers from `env_config_values.py` for back-compat.
- `env_config_values.py` — Isaac-Sim-free workspace geometry constants (`BIN_X_COORD`, `ITEM_SPAWN_REFERENCE_Z`, `CART_SURFACE_CENTER`, `Region2D`, `compute_region_2d`, etc.). Safe to import from `TaskSpec`, tests, and mock code paths without dragging in Isaac Sim.

### Controller Hierarchy (py_trees)

Two behaviour tree implementations are available, selectable via `tree_factory` on `TaskSpec.implementation`:

**Default tree** (`make_task_controller_tree` in `pt_task_tree.py`) — 9-phase time-interpolated pick-place:
```
UR10MultiPickPlaceController (robot_controllers/) — wraps a py_trees BehaviourTree
  └── Parallel("TaskRoot", SuccessOnOne)
        ├── ContextMonitorBehaviour (robot_controllers/pt_context_monitor.py)
        │     Queries TaskContext every tick, writes to /pickplace/ and /task/ blackboard
        └── Selector("task_orchestration")
              ├── CheckAllDone — SUCCESS if task finished
              └── Sequence("finish_or_fail")
                    ├── Repeat → do_one_pick_place
                    │     ├── SelectNextPick
                    │     ├── CheckTargetAvailable
                    │     ├── ResetPickPlaceTree
                    │     ├── pick_then_place (9-phase PickPlaceBehaviour sequence)
                    │     └── MarkPickComplete
                    └── SetTaskFinished
```

**Cortex-style tree** (`make_cortex_task_controller_tree` in `pt_cortex_tree.py`) — MotionCommand-based, threshold-checked completion:
```
  └── Parallel("TaskRoot", SuccessOnOne)
        ├── ContextMonitorBehaviour
        └── Selector("task_orchestration")
              ├── CheckAllDone
              └── Sequence("finish_or_fail")
                    ├── Repeat → do_one_pick_place
                    │     ├── SelectNextPick
                    │     ├── CheckTargetAvailable
                    │     ├── pick_item: MoveToPick → CloseGripper → Timer → MoveRelative(lift)
                    │     ├── place_item: MoveToPlace → DownToInsert → OpenGripper → Timer → MoveRelative(lift)
                    │     └── MarkPickComplete
                    └── SetTaskFinished
```

- `task_context_base.py` — `TaskContextBase` shared base class. Holds robot refs, `arm_commander`/`gripper_commander` (IArmCommander/IGripperCommander), owns the `_prim_geometry` cache and geometry-based computations (EE offsets, placing positions), and delegates pairing/iteration/EE-orientation calls to the `MultiPickStrategy`.
- `task_context.py` — `TaskContext(TaskContextBase)` for real Isaac Sim scenes. Auto-detects CortexUr10 and creates CortexArmAdapter/CortexGripperAdapter.
- `task_context_mock.py` — `MockTaskContext(TaskContextBase)` for testing without Isaac Sim. Creates MockArmCommander/MockGripperCommander.
- `robot_controllers/pt_pick_place_behaviours.py` — 9 time-interpolated pick-place phase behaviours used by the default tree (`create_pick_place_sequence()` factory).
- `robot_controllers/pt_cortex_behaviours.py` — Cortex-style behaviours (`CortexMoveToPick`, `CortexMoveToPlace`, `CortexDownToInsert`, `CortexMoveRelative`, `CortexCloseGripper`, `CortexOpenGripper`). Send `MotionCommand` objects and check `robot_at_target()` for completion.
- `robot_controllers/pt_context_monitor.py` — `ContextMonitorBehaviour` refreshes blackboard from `TaskContext`.
- `robot_controllers/pt_task_behaviours.py` — Task-level behaviours for multi-pick orchestration (shared by both tree variants).
- `robot_controllers/pt_task_tree.py` — `make_task_controller_tree()` factory builds the default (9-phase) tree.
- `robot_controllers/pt_cortex_tree.py` — `make_cortex_task_controller_tree()` factory builds the cortex-style tree. Selected via `tree_factory` on `TaskSpec.implementation`.
- `robot_controllers/pt_curobo_behaviours.py` + `pt_curobo_tree.py` — cuRobo plan-and-stream variant. Activate by setting both `implementation.tree_factory=make_curobo_task_controller_tree` and `implementation.use_curobo=True` (the latter swaps `TaskContext`'s arm commander for `CuroboMotionGenDriver`). Synchronous warmup (~5–15 s) at `set_up_scene`/`post_reset`. See `tasks2/table_task_3_curobo.py` for a reference variant.
- `robot_controllers/curobo_driver.py` — `IArmMotionDriver` protocol + `CuroboMotionGenDriver` (plan-and-stream) + `CuroboMpcDriver` (stub). All cuRobo / Isaac Sim imports are lazy.
- `robot_controllers/curobo_world_config.py` — builds the cuRobo `WorldConfig` (picking table, cart top, conveyor surface). Per-task obstacles via `TaskImplementationSpec.curobo_obstacles_fn`.
- `robot_controllers/robot_interfaces.py` — Protocol definitions including `IArmCommander` (`send_motion_command`, `get_fk_p`, `get_fk_pq`), `IGripperCommander` (Cortex-aligned), and legacy `IGripper`, `IEndEffectorController`.
- `robot_controllers/cortex_adapters.py` — `CortexArmAdapter`, `CortexGripperAdapter`, `LegacyArmAdapter`, `LegacyGripperAdapter`.

Key behaviors:

- `pick_position` is latched from `picking_position` at grasp time (`CloseGripper.initialise()`); `placing_position` is sampled continuously.
- Task-level behaviours query `TaskContext` directly (not via blackboard) to avoid stale-data timing issues.
- Both BT variants send commands via `arm_commander.send_motion_command(cmd)` / `gripper_commander` (no blackboard `current_action`). The controller's `forward()` just ticks the tree.

### PyTrees Integration

- Source code and documentation for the py_trees framework can be found in the additional source directory `../py_trees/` relative to the robo-codegen-exp base project. Note that py_trees is already included in the mamba env `env_isaacsim51`; the `py_trees/` directory is for reference only. Online documentation for py_trees is at <https://py-trees.readthedocs.io/en/devel/>.

### Design docs

Design notes and reference docs live in `docs/`. Stable references include `mock-system-and-testing-design.md`, `task-environment-setup-design.md`, and `PickPlaceAPIs.md`. Many other files in `docs/` are status notes and plans for in-progress work — read with that caveat (they may be stale or describe abandoned approaches).

### Generation System

- `item_generation.py` — `ItemGenerator` creates objects using strategy patterns.
- `asset_utils.py` — Isaac-Sim-dependent asset scene operations: primitive/USD prim creation (`add_asset`, `add_prim_asset`, `add_usd_asset`), live-prim AABB computation (`compute_prim_geometry`, `get_or_compute_prim_geometry`), and semantic-label helpers (`has_label`, `has_color`, `is_of_type`, etc.). Re-exports pure-data symbols from `asset_data_utils` for back-compat.
- `asset_data_utils.py` — Isaac-Sim-free asset metadata and geometry math: `AssetMetaData`, `ITEMS_MAP`, `PRIM_TYPES`, `PrimGeometry`, `scale_aabb`, `lookup_prim_geometry`, `simready_assets`. Safe to import from mock tasks and unit tests without dragging in Isaac Sim.
- Strategy classes: `FixedValue`, `RandomChoice`, `SequentialChoice` for colors, scales, positions.

### Mock System (for testing without Isaac Sim)

- `extsMock/` — Mock implementations of Isaac Sim interfaces, including Cortex types (`MotionCommand`, `PosePq`, `math_util`).
- `tasks_mock/mock_task_utils.py` — Core infrastructure for mock task execution: config extraction, strategy creation, BT tick loop, and task verification.
- `task_context_mock.py` — `MockTaskContext` for testing the py_trees task tree. Creates `MockArmCommander` and `MockGripperCommander`.
- `robot_controllers/mock_robot.py` — Mock robot, gripper, controllers, and Cortex-aligned commanders (`MockArmCommander`, `MockGripperCommander`, `MockCortexRobot`) for testing.


## Key Constraints

- **Never modify files under `exts/`** — this is a read-only Isaac Sim extension snapshot. Use adapters/wrappers in top-level scripts instead.
- **Never modify files under `extsMock/`** unless extending mock coverage for new Isaac Sim APIs.
- When running scripts that are not intended to require the IsaacSim SimulationApp (like run_mock_task.py or the tests in the tests/ directory), the `PYTHONPATH` must include both `extsMock/` (first, to shadow the real `isaacsim` package) and the project root for imports to resolve (handled by the `mamba run` command above). However, when running the Tasks in the tasks/ directory (as launched via the main loop in run_task.py), the IsaacSim SimulationApp is in fact required, and this modification of PYTHONPATH to include extsMock should NOT be done.

## Code Style

- Python 3.x, PEP 8, 4-space indentation, ~100-120 char line length.
- Files/modules: `snake_case.py`. Classes: `PascalCase`. Functions/vars: `snake_case`.
- Commit messages: imperative mood, optionally prefixed (`feat:`, `fix:`, `refactor:`).
