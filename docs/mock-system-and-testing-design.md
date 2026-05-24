# Mock Task System & Testing Infrastructure Design Document

> **Note (curobo branch):** The mock system uses `MockArmCommander` /
> `MockGripperCommander` (from `mock_robot.py`); behaviours send commands via
> `IArmCommander` / `IGripperCommander` in both real and mock modes.
> `MockTaskContext` creates the mock commanders automatically and, when a task
> selects the cuRobo tree (`TaskImplementationSpec.use_curobo=True`), also wires
> a `NullArmMotionDriver` so the cuRobo behaviours' `setup()` finds an
> `arm_motion_driver` on the context.
>
> Tasks now express execution policy on a nested `TaskImplementationSpec`
> (`TaskSpec.implementation`). The behaviour tree variant — default 9-phase,
> cortex-style, or cuRobo — is selected via `implementation.tree_factory`; the
> mock runner picks it up from the same field.

## 1. Purpose and Motivation

The project simulates a UR10 robotic arm performing multi-object pick-and-place tasks in NVIDIA Isaac Sim 5.1. Isaac Sim is a heavyweight dependency: it requires a GPU, takes significant time to initialize, and runs a full physics simulation for every tick. This makes rapid iteration on task logic — designing new tasks, debugging pairing strategies, verifying placement correctness — impractical if every change requires launching the full simulator.

The **mock task system** was created to solve this problem. It provides:

1. **Fast feedback** — Tasks complete in under a second (vs. minutes in simulation).
2. **No GPU required** — Tests run on any machine, including CI.
3. **Full behavior tree execution** — The same py_trees behavior tree that runs in Isaac Sim is ticked against mock hardware, exercising all task orchestration, pick selection, target pairing, completion tracking, and verification logic.
4. **Spatial verification** — Synthesized axis-aligned bounding boxes (AABBs) allow the real `PlacementChecker` to check placement correctness without a physics engine.

Two additional features — **`--teleport`** and **`--pause`** — were added for the real Isaac Sim execution path. They enable fast visual evaluation and manual inspection of placements inside the simulator without waiting for realistic robot motion.


## 2. Architecture Overview

### 2.1 Inheritance Hierarchy

```
TaskContextBase                       (task_context_base.py)
  ├── TaskContext                     (task_context.py)       — real Isaac Sim
  └── MockTaskContext                 (task_context_mock.py)  — testing
        └── MockTaskContextWithPlaceUpdate                    — dynamic subclass
                                      (tasks_mock/mock_task_utils.py)
```

`TaskContextBase` is the shared abstract base that holds:
- A reference to the robot (real or mock)
- `arm_commander` (`IArmCommander`) and `gripper_commander` (`IGripperCommander`) for Cortex-aligned control
- The `_prim_geometry` cache and geometry-based computations (EE offsets, placing positions)
- Delegation of pairing/iteration/EE-orientation calls to a `MultiPickStrategy`
- The `teleport_mode` flag — when enabled, `NullArmCommander`/`NullGripperCommander` are used so no commands reach the robot

Both real and mock contexts use the same `MultiPickStrategy` subclass (created by the task's `create_strategy` factory on `TaskSpec.implementation`), so pairing and placement logic is identical in both paths.

### 2.2 Entry Points

| Entry Point | Environment | Purpose |
|---|---|---|
| `run_task.py --task <T>` | Isaac Sim (GPU) | Full physics simulation with robot kinematics |
| `run_task.py --task <T> --teleport` | Isaac Sim (GPU) | Visual evaluation with instant object placement |
| `run_task.py --task <T> --pause` | Isaac Sim (GPU) | Step-by-step inspection after each cycle |
| `run_mock_task.py --task <T>` | CPU only | Fast BT execution and verification without Isaac Sim |
| `python -m pytest tests/` | CPU only | Unit tests for all mock components |
| `bash run_all_mock_tasks.sh` | CPU only | Batch run every (or a subset of) mock task(s) |
| `bash run_all_teleport_tasks.sh` | Isaac Sim (GPU) | Batch run every task in teleport mode |
| `bash run_all_simulation_tasks.sh` | Isaac Sim (GPU) | Batch run every task with full physics |

See section 9 for details on the batch runner scripts.


## 3. The Mock Task System

### 3.1 Module Shadowing Strategy

Isaac Sim exposes its API through deeply nested Python packages (e.g., `isaacsim.core.api.tasks`, `isaacsim.robot.manipulators.examples.universal_robots`). These packages are unavailable outside the simulator runtime. The mock system uses a three-layer approach to make task classes importable:

**Layer 1: `extsMock/` directory** — Contains real Python implementations of core Isaac Sim types that the project depends on:
- `isaacsim/core/utils/types.py` — `ArticulationAction`, `JointsState`, `DataFrame`, etc.
- `isaacsim/core/utils/semantics.py` — In-memory semantic label store (dict-of-dicts instead of USD properties)
- `isaacsim/core/utils/math.py`, `numpy/`, `torch/` — Rotation and transformation utilities
- `pxr/__init__.py` — Minimal USD type stubs (`Gf.Vec3d`, `Gf.Rotation`, `Gf.Quatd`, `UsdGeom`, `Usd`)

The `extsMock/` directory is inserted at the front of `sys.path` so its modules shadow the real Isaac Sim packages.

**Layer 2: `setup_mock_modules()`** (in `tasks_mock/mock_task_utils.py`) — Pre-installs ~50 module stubs into `sys.modules` for Isaac Sim APIs that don't need real implementations:
- `MockBaseTask` — Minimal task interface (`set_up_scene()`, `pre_step()`, `post_reset()`)
- `MockBaseController` — Controller stub with `forward()` and `reset()`
- `MockRMPFlowController` — RMP flow controller stub
- `_Dummy` — Placeholder for object creation classes (`DynamicCuboid`, `FixedCuboid`, etc.)
- All necessary parent packages (`isaacsim.robot.manipulators`, etc.) with `__path__ = []`

**Layer 3: Targeted patches** — Specific return values and function stubs:
- `get_stage_units()` → `1.0`
- `get_assets_root_path_or_die()` → `/tmp/assets`
- `is_prim_path_valid()` → `False`
- `find_unique_string_name()` → identity function

**Critical ordering requirement:** `extsMock/` must be on `sys.path` and `setup_mock_modules()` must execute *before* any task module is imported. `run_mock_task.py` enforces this at the module level (lines 17-23).

### 3.2 LightweightObj

`LightweightObj` (defined in `task_context_base.py`) is a minimal object representation that provides the same pose interface as Isaac Sim prim objects:

```python
class LightweightObj:
    def __init__(self, name, position=None, orientation=None)
    def get_local_pose(self) -> (position, orientation)
    def get_world_pose(self) -> (position, orientation)
    def set_position(self, position)
    def set_orientation(self, orientation)
    def set_world_pose(self, position=None, orientation=None)
    def get_local_scale(self) -> [1, 1, 1]
    _semantic_labels: dict     # in-memory labels (checked by has_label() before USD)
    _local_half_extents: array # precomputed AABB half-extents (avoids USD queries)
```

`LightweightObj` is used in two contexts:
1. **Mock execution** — representing both pick and target objects when testing without Isaac Sim.
2. **Virtual targets** — representing hidden/utility markers generated at pairing time by `TaskController._generate_virtual_targets()`. These exist even in real Isaac Sim sessions as lightweight in-memory objects without USD prims.

The `_semantic_labels` dict stores labels (type, color, name) directly on the object. `asset_utils._get_semantic_labels()` (Isaac-Sim-dependent) checks this attribute first before falling back to USD API queries, so label-based matching (e.g., `has_label()`, `has_color()`) works on both real prims and `LightweightObj` instances.

The `_local_half_extents` array enables AABB computation in `get_corrected_aabb()` without USD scene queries — essential for virtual targets that have no USD prim.

The factory function `create_lightweight_objs_from_items(items, prefix, prim_geometry_out)` in `task_context_base.py` converts a list of `ItemSpec` to `LightweightObj` instances, populating semantic labels and looking up geometry via `lookup_prim_geometry()` (imported from the Isaac-Sim-free `asset_data_utils`).

Mock execution uses `extract_task_config()` which calls `task.get_task_spec()` to obtain a `TaskSpec`, then delegates to `prepare_mock_from_spec()`. This function creates `LightweightObj` instances from the spec's generation strategies, applies semantic labels (to the in-memory label store), caches `PrimGeometry`, generates any virtual targets (via `TaskImplementationSpec.virtual_target_generation_strategy`), and invokes the implementation spec's `create_strategy` factory (`task_spec.implementation.create_strategy`, or falls back to default `MultiPickStrategy`). When the spec carries an incremental / spatial-trigger scheduler config, `prepare_mock_from_spec()` builds an `ItemSpawner` with a `MockPrimFactory` for materialising replenishment batches as `LightweightObj` instances and aligns mock pick / target name sequences with the real-sim factory. No centralized dispatch registry: each task class provides its configuration through `TaskSpec` (with execution policy on `TaskSpec.implementation`).

### 3.3 Mock Hardware

`robot_controllers/mock_robot.py` provides mock implementations of the robot
protocols. The classes most exercised by the mock BT path are the Cortex-aligned
commanders:

| Class | Purpose | Key Behavior |
|---|---|---|
| `MockGripper` | Parallel-gripper-shaped object | Tracks open/close state via `open()` / `close()` |
| `MockArmCommander` | `IArmCommander` | Records `send_ee_target()` / `send_motion_command()` calls; simulates motion via a tick countdown (`ticks_per_move`) so `get_fk_p()` / `get_fk_pq()` "arrive" at the commanded pose after a few ticks. `tick()` is called once per BT tick by the mock runner. |
| `MockGripperCommander` | `IGripperCommander` | Records `open()` / `close()` counts and exposes a `grasp_state_override` hook so cortex-tree `VerifyGrasp` failure can be injected from tests. |
| `MockCortexRobot` | Complete robot assembly | Provides `.arm` (a `MockArmCommander`) and `.suction_gripper` (a `MockGripperCommander`) so cortex-shaped code finds the expected attributes. |

Legacy classes also live in `mock_robot.py` for tests that pre-date the
commander split: `MockEndEffectorController` and `MockArticulationController`
record `ArticulationAction` history; `MockRobotArticulation` composes a
`MockGripper` plus an articulation controller. The BT itself only touches the
commander objects.

The arm/gripper commanders are created automatically inside
`MockTaskContext.__init__`; the mock runner does not need to construct them
directly. After each `tree.tick()`, `run_mock_task` calls
`context.arm_commander.tick()` so cortex/curobo behaviours that gate on
`robot_at_target()` see the commanded pose materialise after a deterministic
number of ticks.

These are intentionally minimal. The mock system tests control flow and task
logic, not kinematics.

### 3.4 MockTaskContextWithPlaceUpdate

The standard `MockTaskContext` doesn't move objects after placement — it only tracks completion in the strategy. `MockTaskContextWithPlaceUpdate` (created via dynamic subclassing in `mock_task_utils.py`) overrides `mark_pick_complete()` to actually move the pick object to its target position:

```python
def mark_pick_complete(self, pick_name):
    target_name = self.get_placing_target_name(pick_name)
    drop_orient = self.get_end_effector_orientation_for_drop(pick_name, target_name)
    target_name, place_pos, place_orient = self.get_placing_info(pick_name, drop_orient)
    MockTaskContext.mark_pick_complete(self, pick_name)  # strategy bookkeeping
    if place_pos is not None:
        pick_obj.set_position(place_pos)
        pick_obj.set_orientation(place_orient)
```

This is essential for:
- Verification (objects must be at their target positions for AABB checks)
- Stacking tasks (subsequent placements depend on where earlier items were placed)

### 3.5 AABB Infrastructure for Verification

Task verification uses axis-aligned bounding boxes (AABBs) to check spatial relationships (e.g., "is pick object on top of target?"). In Isaac Sim, AABBs come from the physics engine. The mock system synthesizes them from object positions and `PrimGeometry`:

```python
def compute_mock_aabb_from_geometry(position, geom):
    hx, hy, hz = geom.local_half_extents
    return [pos[0]-hx, pos[1]-hy, pos[2]-hz, pos[0]+hx, pos[1]+hy, pos[2]+hz]
```

A module-level `_mock_aabb_registry` (keyed by prim path) is updated before verification. The real `PlacementChecker` class (in `task_verification.py`) is used with monkeypatched `compute_aabb()` and `create_bbox_cache()` functions pointing to the mock registry.

When `obj_asset_info` is available, AABBs are recomputed using the object's *current* orientation (post-placement), so rotated objects get correct bounding boxes. The recomputation also overwrites the `LightweightObj._local_half_extents` cache so `get_corrected_aabb()` (which short-circuits via that attribute) reflects the placed orientation instead of the spawn orientation.

### 3.6 End-to-End Mock Execution Pipeline

`run_mock_task()` in `tasks_mock/mock_task_utils.py` orchestrates the complete pipeline:

```
1. setup_mock_modules()              — Shadow Isaac Sim imports
2. clear_all_labels()                — Reset semantic label store
3. extract_task_config(task_class)    — Instantiate task, get TaskSpec, call prepare_mock_from_spec()
   └── task.get_task_spec()          — Build/return TaskSpec (+ nested TaskImplementationSpec)
   └── prepare_mock_from_spec()      — Generate LightweightObj items, apply labels,
                                       cache PrimGeometry, generate virtual targets,
                                       build the strategy from impl_spec.create_strategy,
                                       and (optionally) wire an ItemSpawner+MockPrimFactory
                                       for incremental / spatial-trigger schedulers
4. create_mock_context(config)        — Build MockTaskContextWithPlaceUpdate (auto-creates
                                       MockArmCommander/MockGripperCommander; wires a
                                       NullArmMotionDriver when impl_spec.use_curobo=True)
5. tree_factory = config["tree_factory"]   — Selected from impl_spec.tree_factory, else
                                             pt_task_tree.make_task_controller_tree
   root = tree_factory(fake_fast=True)
   tree = py_trees.trees.BehaviourTree(root)
6. tree.setup(context, arm_commander, gripper_commander)  — Wire commanders to behaviours
7. Tick loop (up to max_ticks):
   ├── mock_time += MOCK_TICK_DT_S   — Sim clock published as context.simulation_time
   ├── spawner.tick(...)             — Release replenishment items + check BT-start gate
   ├── tree.tick()                   — Execute one BT cycle
   ├── context.arm_commander.tick()  — Advance mock arm so cortex/curobo behaviours can
                                       detect arrival via robot_at_target()
   ├── Incremental verification      — Check newly completed picks (PlacementChecker)
   ├── conveyor.advance(...)         — Drift on-belt items in -Y when belt is moving
   ├── Optional status display       — py_trees.display.unicode_tree(root, show_status)
   └── Break when not RUNNING
8. Final verification                 — Full PlacementChecker.verify() on all placements
9. Report results                    — Positions, pairings, success/failure
```

The runner samples mock time at a coarse `MOCK_TICK_HZ` (10 Hz, see
`MOCK_TICK_DT_S` constant in `mock_task_utils.py`) and publishes it to
`context.simulation_time` so `WaitForCycleTime` and any other time-aware
behaviours read the same clock as the schedulers. When the task uses
incremental or spatial-trigger generation, a `bt_started` gate suppresses BT
ticks until `ItemSpawner.bt_should_start()` returns True (so the tree does not
spin while waiting for the first batch to spawn).

**Exit codes:** 0 = success, 1 = task didn't finish, 2 = finished but verification failed.

### 3.7 Verification Modes

The mock system supports two verification approaches, both implemented by the
shared `PlacementChecker` class in `task_verification.py`:

**Standard (1:1 marker):** Each pick is checked against its paired target using a spatial check function (default: `is_on_top`). Strategies can provide a custom check via `strategy.get_spatial_check_fn()` (e.g., `is_vertical` for bottles); the `TaskSpec.spatial_check_fn` field is the per-task fallback.

**Box containment (multi-occupancy):** For tasks like `TableTaskColorShapes` and `TableTaskSoupCanPacking` where multiple picks go into shared containers. The mock path calls `build_box_verification_hooks(box_specs, pick_objs, is_pick_expected=...)` from `task_verification.py` to synthesise virtual box targets, the spatial check function (`is_within_box_geometry`), and the valid-targets-for-pick function. Box specs support optional `match_labels` for restricting which picks can go into which box (e.g., `{"color": "red"}`).

**Centralized box containment via TaskSpec:** Tasks with `box_verification_info` set on their `TaskSpec` get the box containment path automatically — both the real `UR10MultiPickPlaceTask` and the mock runner read the same field, so tasks no longer need to override `check_groundtruth_task_success()`.

**Containment mode via TaskSpec:** Tasks with `containment_check=True` on their `TaskSpec` pass `containment_mode=True` to `PlacementChecker`, which implies multi-occupancy and uses containment-specific failure messages (distinguishing "not inside container" from "not on top of target").

**Incremental verification:** After each pick-place cycle completes, only the newly placed item is verified. The mock path calls `verify_mock_task_incremental(config, context, pick_names)`, which converts the pick names to indices internally and runs `PlacementChecker.verify(pick_indices=...)`. This provides early failure detection and per-item logging without re-checking all previous placements.

### 3.8 Conveyor, Schedulers, and Incremental Spawning

Tasks that involve a moving conveyor belt, or incremental / spatial-trigger
generation of picks or targets, work transparently in mock mode through three
cooperating pieces in `tasks_mock/mock_task_utils.py`:

**`MockConveyor`** — Drifts on-belt items in -Y each tick, mirroring what
`PhysX SurfaceVelocityAPI` does in real Isaac Sim. Items resting on the belt
(Z near `CONVEYOR_SURFACE_TOP_Z`, X/Y within the belt surface) drift by
`speed * dt` each tick; items that cross below `CONVEYOR_END_Y` are dropped in
Z so reachability filters treat them as fallen-off. Placed picks resting on a
drifting target ride along with their carrier (`ride_with={pick_name:
target_obj}`) so the per-tick spatial check still sees them aligned, matching
the friction-driven behaviour in real sim. Constructed once when
`TaskSpec.conveyor_speed` is non-zero.

**`MockPrimFactory`** — Implements the `PrimFactory` protocol that
`ItemSpawner` uses to materialise picks/targets. Mirrors `IsaacPrimFactory`:
owns its own pick/target name counters (seeded from the initial-batch sizes in
`prepare_mock_from_spec`) so virtual targets do NOT advance the scene-target
counter and mock names line up with the real-sim names.

**`ItemSpawner` (shared with real sim)** — When `TaskSpec.pick_incremental_config`,
`pick_spatial_trigger_config`, `target_incremental_config`, or
`target_spatial_trigger_config` is set, `prepare_mock_from_spec` builds an
`IncrementalItemScheduler` or `SpatialTriggeredItemScheduler` plus an
`ItemSpawner` and stores them on the context. Each mock tick calls
`spawner.tick(mock_time, live_picks, live_targets)`; newly-released items are
appended to the strategy via `add_incremental_picks` / `add_incremental_targets`
and the `more_items_expected` / `more_targets_expected` flags are cleared once
the scheduler has released its full count. A `bt_started` gate suppresses BT
ticks until `spawner.bt_should_start()` returns True so the tree does not spin
while waiting for the first batch.

Spatial-trigger schedulers depend on conveyor motion to drive replenishment;
if the belt is stationary, `prepare_mock_from_spec` suppresses
`more_*_expected` so the BT can complete on the initial batch instead of
waiting forever.

### 3.9 `--show-status` Flag

When `run_mock_task.py` is invoked with `--show-status`, the runner:
- Enables py_trees DEBUG-level logging
- Prints the full behavior tree with status indicators after each tick using `py_trees.display.unicode_tree(root, show_status=True)`
- Defaults to 0.05s inter-tick delay (20 ticks/sec) for readability

This is invaluable for debugging tree execution flow — seeing which behaviours are RUNNING, SUCCESS, or FAILURE at each step.

`--verbose` (`-v`) implies `--show-status`; `--quiet` suppresses the per-task
banner and final summary.


## 4. The `--teleport` Option

### 4.1 Purpose

The `--teleport` flag enables fast task evaluation inside the real Isaac Sim environment. Instead of computing inverse kinematics, commanding joint motions, and waiting for the robot to physically traverse each phase, objects are instantly teleported to their target positions. This allows:

- **Rapid scene validation** — Verify that objects spawn correctly, targets are reachable, and the scene layout is sensible.
- **Quick placement preview** — See where objects will end up without waiting for robot motion.
- **Debugging workflow** — The commit log records a recommended pattern: "Run with `--teleport` first to verify scene setup, then run without for real physics."

### 4.2 Implementation

Teleport mode is activated by `--teleport` on the command line (`run_task.py`) and flows through the system as follows:

**CLI → Task → Context → Controller:**
```python
# run_task.py
parser.add_argument("--teleport", action="store_true", ...)

# UR10MultiPickPlaceTask.__init__()
self._teleport_mode = teleport_mode

# post_reset() — passed to both context and controller
self._task_context = TaskContext(..., teleport_mode=self._teleport_mode)
self._task_controller = UR10MultiPickPlaceController(..., fake_fast=self._teleport_mode)
```

**Two effects:**

1. **Objects teleport on completion** (`task_context_base.py`):
   ```python
   def mark_pick_complete(self, pick_name):
       if self._teleport_mode:
           target_name = self.get_placing_target_name(pick_name)
           drop_orient = self.get_end_effector_orientation_for_drop(pick_name, target_name)
           target_name, place_pos, place_orient = self.get_placing_info(pick_name, drop_orient)
       self._strategy.mark_pick_complete(pick_name)
       if self._teleport_mode and place_pos is not None:
           pick_obj.set_world_pose(position=place_pos, orientation=place_orient)
   ```

2. **Phases complete quickly** — `fake_fast=True` sets all phase durations to 0.5s (or 1.0s for gripper actions), so each phase completes in ~2 ticks instead of hundreds.

### 4.3 Behaviour Tree Variants

The task's `TaskImplementationSpec.tree_factory` chooses which behaviour tree
is built; teleport mode does not change the structure, only the commanders
behind the leaves.

| Variant | Factory | When to use |
|---|---|---|
| Default (time-interpolated) | `pt_task_tree.make_task_controller_tree` | Default; cheap to tick, well-suited to mock mode and the 9-phase visual evaluation pattern used by `--teleport`. |
| Cortex-style | `pt_cortex_tree.make_cortex_task_controller_tree` | Real Isaac Sim runs that use Cortex's `MotionCommand` API. Threshold-checked completion (`robot_at_target`) gives smoother motion than the time-stepped default. Active in most `tasks2/` v2 variants. |
| cuRobo | `pt_curobo_tree.make_curobo_task_controller_tree` | Plan-and-stream variant on top of the cuRobo motion generator. Requires `implementation.use_curobo=True` so the context wires a `CuroboMotionGenDriver` (real sim) or a `NullArmMotionDriver` (mock / teleport). See `tasks2/table_task_3_curobo.py`. |

All three variants share the same task-level skeleton:

```
Parallel("TaskRoot", SuccessOnOne)
├── ContextMonitorBehaviour
└── Selector("task_orchestration", memory=False)
    ├── CheckAllDone
    └── Sequence("finish_or_fail")
        ├── FailureIsSuccess(Repeat("repeat_picks", num_success=-1))
        │     └── Sequence("do_one_pick_place")
        │           └── < variant-specific pick + place children >
        └── SetTaskFinished
```

The variants differ only in the children of `do_one_pick_place`. The
behaviours `CheckAllDone`, `SelectNextPick`, `CheckTargetAvailable`,
`MarkPickComplete`, `SetTaskFinished`, `WaitForCycleTime`, and `ContextMonitor`
are reused across all three.

#### 4.3.1 Default tree — 9-phase time-interpolated pick-place

Each pick-place cycle consists of 9 phases, implemented as individual
`PickPlaceBehaviour` subclasses in `pt_pick_place_behaviours.py`. Each phase
uses time-based progression: an internal timer `t` advances by `dt` each tick
(scaled by sim time when a live World is available — see `BT_TICK_REFERENCE_HZ`
in `pt_pick_place_behaviours.py`), and the phase completes when `t >= 1.0`.

| Phase | Class | Normal `dt` | `fake_fast` `dt` | Action |
|---|---|---|---|---|
| 0 | `MoveToPickXYBehaviour` | 0.01 | 0.5 | Move EE above pick at move height |
| 1 | `LowerToPickBehaviour` | 0.005 | 0.5 | Sinusoidal descent to grasp position |
| 2 | `WaitSettlingBehaviour` | 0.1 | 0.5 | Hold position for inertia to settle |
| 3 | `CloseGripperBehaviour` | 1.0 | 1.0 | Close gripper to grasp item |
| 4 | `LiftPickedBehaviour` | 0.008 | 0.5 | Ascent to move height |
| 5 | `MoveToPlaceXYBehaviour` | 0.005 | 0.5 | Interpolate XY from pick to place |
| 6 | `LowerToPlaceBehaviour` | 0.005 | 0.5 | Sinusoidal descent to placement height |
| 7 | `OpenGripperBehaviour` | 1.0 | 1.0 | Open gripper to release item |
| 8 | `LiftAfterPlaceBehaviour` | 0.08 | 0.5 | Ascent after release |

The factory wires the 9 phases inside a `ResetPickPlaceTree` decorator (which
re-initialises the subtree at the start of each cycle) and a `Sequence`:

```
do_one_pick_place (Sequence, memory=True)
├── WaitForCycleTime          [no-op when min_cycle_time_s=0]
├── SelectNextPick
├── CheckTargetAvailable
├── ResetPickPlaceTree        [resets the 9-phase subtree]
├── pick_then_place           [9 PickPlaceBehaviour phases]
└── MarkPickComplete          [TELEPORTS object in teleport mode]
```

With `fake_fast`, each phase completes in 2 ticks (one to compute the final
action, one to return SUCCESS). A full 9-phase cycle completes in ~18 ticks
instead of hundreds.

The optional `track_picked_item_during_lift` factory kwarg switches between
holding the latched pick XY during the post-grasp lift (default) and following
the live picked item position (legacy behaviour, useful for tasks where a
conveyor surface drags the held item).

#### 4.3.2 Cortex-style tree — `MotionCommand` with threshold-based completion

`pt_cortex_tree.make_cortex_task_controller_tree` builds a richer tree that
matches the Cortex framework's `MotionCommand` API. Each movement leaf sends a
`MotionCommand` (optionally with `ApproachParams`) and returns `RUNNING` until
`context.robot_at_target(command, p_thresh, R_thresh)` is satisfied.

Three additional concerns appear in this variant:

- **Per-cycle perception cache.** `PrepareGrasp` / `PreparePlacement` populate
  cached `GraspPose` / `PlacePose` on the context; all downstream cortex
  behaviours read from this cache so pre-grasp, approach, and descent agree
  on a single resolved pose. `reset_cycle_cache()` clears it between attempts.
- **Retry with deferral.** A `pick_attempt` Sequence is wrapped in
  `Retry(num_failures=PICK_RETRY_BUDGET=5)`. A guarded `Sequence` around the
  Retry re-evaluates `IsPickReachableGuard` every tick so a pick that goes
  permanently unreachable mid-Retry (e.g. drops below the Z-floor) aborts
  immediately. On exhaustion, `DeferPickAndRelease` runs through a
  `FailureIsSuccess` and the cycle falls through to `release_and_skip`.
- **Sim-time watchdogs.** Each `CortexMove*` leaf is wrapped by
  `sim_timeout_to_success` (from `pt_sim_time_decorators.py`). The duration
  resolver reads from the live `TaskContext` at `initialise()` time
  (`get_move_timeout_s()` / `get_approach_timeout_s()` / `get_insert_timeout_s()`)
  so per-task `TaskSpec.{move,approach,insert}_timeout_s` overrides take
  effect without any factory-time wiring. On expiry the watchdog surfaces a
  rich diagnostic (target_p, fk_p, distance) via the wrapped behaviour's
  `_timeout_diagnostic()` callback.

The cycle structure:

```
do_one_pick_place (Sequence, memory=True)
├── CheckCycleProgress             [no-progress safety net — aborts after 50 idle cycles]
├── WaitForCycleTime
├── SelectNextPick
├── CheckTargetAvailable
├── Selector("pick_or_defer", memory=True)
│   ├── Sequence("guard_then_retry", memory=False)
│   │   ├── IsPickReachableGuard
│   │   └── Retry(num_failures=5)
│   │       └── Sequence("pick_attempt", memory=True)
│   │           ├── CheckPickReachable
│   │           ├── PrepareGrasp
│   │           ├── CheckGraspPoseReachable
│   │           ├── _watchdog(CortexMoveToPreGrasp)
│   │           ├── _watchdog(CortexExecuteApproach)
│   │           ├── CortexCloseGripper
│   │           ├── SimTimer(wait_for_grip ≈ 0.5s)
│   │           ├── LatchCurrentPick
│   │           ├── _watchdog(CortexMoveRelative(lift_after_pick, +Z 0.13m))
│   │           └── VerifyGrasp
│   └── FailureIsSuccess(DeferPickAndRelease)
├── Selector("place_or_recover", memory=True)
│   ├── Sequence("place_item", memory=True)
│   │   ├── HaveItemInGripper
│   │   ├── PreparePlacement
│   │   ├── LatchPlacementTarget
│   │   ├── _watchdog(CortexMoveToPlace)
│   │   ├── _watchdog(CortexDownToInsert)
│   │   ├── CortexOpenGripper
│   │   ├── SimTimer(wait_for_release ≈ 0.3s)
│   │   └── _watchdog(CortexMoveRelative(lift_after_place, +Z 0.13m))
│   └── Sequence("release_and_skip", memory=True)
│       ├── CortexOpenGripper
│       ├── SimTimer(recovery_wait)
│       └── _watchdog(CortexMoveRelative(recovery_lift, +Z 0.13m))
└── MarkPickComplete                [skips deferred picks]
```

The transport lifts (`lift_after_pick`, `lift_after_place`) use
`cap_to_ee_height_for_move=True` so the absolute Z floor of the lift is at
least `context.get_ee_height_for_move()` — tasks setting a higher
`ee_height_for_move` (for obstacle clearance) get the lift *extended* without
forcing a per-task tweak of the lift constant.

In `fake_fast` mode, the cortex movement behaviours return SUCCESS immediately
(skipping `robot_at_target`) and the `SimTimer` durations collapse to ~1 ms;
the cycle still exercises the full Retry / defer / recover logic.

#### 4.3.3 cuRobo tree — plan-and-stream variant

`pt_curobo_tree.make_curobo_task_controller_tree` is structurally identical to
the cortex tree but swaps every `CortexMove*` leaf for the corresponding
`CuroboMove*` behaviour, and reads/writes through `context.arm_motion_driver`
(a `CuroboMotionGenDriver` in real sim, `NullArmMotionDriver` in mock /
teleport) instead of `context.arm_commander`. The Cortex `ApproachParams`
funnels are dropped in this first cut; the BT's pre-grasp/approach phase split
is the substitute. Activated by setting both
`TaskImplementationSpec.tree_factory=make_curobo_task_controller_tree` and
`TaskImplementationSpec.use_curobo=True`. cuRobo synchronously warms up once
during `set_up_scene`/`post_reset` (5–15 s).

### 4.4 Interaction with the Behavior Tree

In teleport mode all phases still execute (maintaining tree consistency), but
`NullArmCommander` / `NullGripperCommander` silently discard motion and gripper
commands. The visible effect is that objects appear at their target positions
when `MarkPickComplete` fires at the end of each cycle, regardless of which
tree variant is in use. When `use_curobo=True`, a `NullArmMotionDriver` is
additionally wired so the `CuroboMove*` behaviours see an immediately-arrived
arm in teleport mode.

The behaviour tree structure is identical in normal and teleport modes for all
three variants — only the action application and object placement mechanics
change.


## 5. The `--pause` Option

### 5.1 Purpose

The `--pause` flag pauses the task after each pick-place cycle completes, allowing the user to:
- Visually inspect each placement in the Isaac Sim GUI
- Manually reposition objects if needed (e.g., to test verification with adjusted positions)
- Step through the task one cycle at a time for debugging

This feature was added alongside incremental verification (commit `6bd39eb`), enabling a "place, inspect, verify, continue" workflow.

### 5.2 Implementation

**State tracking** (`multi_pickplace_task.py`, near `UR10MultiPickPlaceTask.__init__`):
```python
self._paused_for_inspection = False
self._pause_cycle_count = 0
self._prev_completed_picks = set()
self._pending_incremental_pick_names: list[str] = []
```

**Cycle detection and pause trigger** (`multi_pickplace_task.py`, inside `task_step()`):

After each `task_step()` tick, newly completed picks are detected by comparing the current completed set against the previous one. If `--pause` is active and new picks exist, the system sets `_paused_for_inspection = True` and logs a prompt.

**Non-blocking ENTER check** (`_check_stdin_enter` in `multi_pickplace_task.py`):
```python
def _check_stdin_enter():
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if ready:
        sys.stdin.readline()
        return True
    return False
```

This uses `select.select()` with a zero timeout for non-blocking stdin polling. The Isaac Sim GUI continues rendering while waiting — only the behavior tree is paused.

**Deferred incremental verification** (`multi_pickplace_task.task_step` →
`_check_incremental`):

When a pause resumes, any pending incremental verification pick names are processed. This deferral serves two purposes:
1. It gives the physics engine at least one step to process pose changes (e.g., from teleport via `set_world_pose()`), ensuring AABB queries return up-to-date bounding boxes.
2. It allows the user to manually reposition objects during the pause, with verification reflecting the adjusted positions.

**Headless suppression** (`run_task.py`):
```python
pause_after_cycle=args.pause and not args.headless
```
If `--headless` is set (no GUI), `--pause` is automatically disabled since there's nothing to inspect.

### 5.3 Teleport + Pause Combination

When both flags are used together:
1. Objects are teleported instantly to targets (no robot motion)
2. The tree pauses after each teleported placement
3. User inspects the teleported position in the GUI
4. On ENTER, incremental verification runs
5. Tree resumes for the next pick

This is the fastest way to visually validate a task configuration end-to-end: each cycle takes a fraction of a second of simulation time, and the user controls the pacing.


## 6. Incremental Verification

### 6.1 Motivation

Full task verification runs only after all picks are complete. For tasks with many objects, a failure on the first pick isn't discovered until the entire task finishes. Incremental verification (commit `c4619ba`) checks each item immediately after placement.

### 6.2 Implementation

**In the real system** (`multi_pickplace_task.py`, `task_step` →
`_check_incremental`):
- New completions are detected each `task_step()`.
- Pick names are queued in `_pending_incremental_pick_names`.
- On the *next* step (or after a pause resumes), `_check_incremental()` creates a `PlacementChecker` and calls `verify(pick_indices=...)` for only the queued items (the names are translated to indices internally).
- Results are logged: `DEBUG` for successes, `WARNING` for failures.

**In the mock system** (`tasks_mock/mock_task_utils.py`,
`verify_mock_task_incremental`):
- `run_mock_task()` tracks `prev_completed` across ticks.
- `verify_mock_task_incremental()` creates a `PlacementChecker` with monkeypatched AABBs and checks only the new picks.
- Failures are collected in `incremental_failures` and reported at the end.

### 6.3 Deferral Strategy

Verification is deferred by one step (real system) because:
- `set_world_pose()` (used by teleport mode) may not immediately update physics-engine AABBs.
- Deferring ensures the physics engine has processed the pose change before AABB queries run.

In the mock system, this isn't strictly necessary (AABBs are computed directly from `LightweightObj` positions), but the same pattern is used for consistency.


## 7. Test Infrastructure

### 7.1 Test Environment Setup

`tests/conftest.py` ensures `extsMock/` is first on `sys.path` and adds the repo root. Tests are run with:
```bash
mamba run -n env_isaacsim51 env PYTHONPATH=$(pwd)/extsMock:$(pwd) python -m pytest tests/
```

### 7.2 Test Files

The `tests/` directory has grown well beyond the initial mock-coverage scope.
The current set, grouped by area:

**Task context, strategies, and verification**
| File | Focus |
|---|---|
| `test_task_context.py` | `MockTaskContext` initialization, queries, mutations, reset; bottle / layered / stacking strategies |
| `test_task_verification.py` | `PlacementChecker` spatial checks with stub AABBs |
| `test_dynamic_top_pick_strategy.py` | Dynamic top-of-stack pick selection |
| `test_strategy_name_based.py` | Name-based pairing strategies |
| `test_target_reachability.py` | Reachability filters and Z-floor gating |
| `test_pick_deferral.py` | Deferred-pick bookkeeping in the strategy layer |

**Behaviour trees and behaviours**
| File | Focus |
|---|---|
| `test_task_tree.py` | Task-level behaviours: `CheckAllDone`, `SelectNextPick`, `MarkPickComplete`, `ContextMonitor`, etc. |
| `test_pick_place_behaviours.py` | The 9 phase behaviours from `pt_pick_place_behaviours.py` |
| `test_cortex_perception_behaviours.py` | `PrepareGrasp`, `PreparePlacement`, `VerifyGrasp`, `HaveItemInGripper`, `Defer*`, … |
| `test_cortex_tree_defer.py` | End-to-end cortex tree deferral + recovery paths |
| `test_cycle_progress_safety.py` | `CheckCycleProgress` no-progress safety-net behaviour |
| `test_sim_time_decorators.py` | `sim_timeout_to_success` / `SimTimer` semantics |
| `test_motion_methods.py` | Context-side motion command builders |
| `test_perception_utils.py` | `compute_grasp_pose` / `compute_place_pose` / approach funnel math |

**Generation, schedulers, conveyor**
| File | Focus |
|---|---|
| `test_item_gen_randomization.py` | Item generation randomization strategies |
| `test_incremental_generation.py` | `IncrementalItemScheduler` mechanics |
| `test_incremental_targets.py` | Incremental target replenishment |
| `test_spatial_trigger_scheduler.py` | `SpatialTriggeredItemScheduler` mechanics |
| `test_item_spawner.py` | `ItemSpawner` + `PrimFactory` protocol |
| `test_mock_conveyor.py` | `MockConveyor` drift / fall-off behaviour |
| `test_conveyor_falloff.py` | Reachability filtering when items cross the belt edge |
| `test_conveyor_proximity_strategy.py` | `ConveyorProximityStrategy` JIT target binding |

**Mock infrastructure and CLI**
| File | Focus |
|---|---|
| `test_mock_controllers.py` | `MockGripper`, `MockEndEffectorController`, `MockArmCommander`, articulation recording |
| `test_mock_semantics.py` | In-memory semantic label store |
| `test_asset_utils.py` | Asset metadata and geometry math |
| `test_prim_geometry.py` | `PrimGeometry` lookup + caching |
| `test_virtual_targets.py` | Virtual target generation, semantic labels, `build_box_verification_hooks` |
| `test_task_spec.py` | `TaskSpec` / `TaskImplementationSpec` serialization round-trip |
| `test_args_propagation.py` | CLI → task-class argument flow |
| `test_ur10_controller.py` | `UR10MultiPickPlaceController` wiring |


## 8. Key Design Decisions

### 8.1 Separation of Strategy and Geometry Concerns

Pairing, iteration, EE-orientation decisions, and completion logic live in `MultiPickStrategy` and its subclasses. Geometry-based computations (EE offsets, placing positions, drop offsets) live in `TaskContextBase` using the `_prim_geometry` cache. Both `TaskContext` and `MockTaskContext` use the same strategy instance and the same geometry computation code. This means the mock system tests the actual task logic, not a simplified approximation.

### 8.2 TaskSpec-Driven Mock Execution

Each task class provides a `get_task_spec()` method that returns a `TaskSpec` capturing all configuration declaratively. The `TaskSpec` includes both executable config (generation strategies, strategy factory, workspace setup, verification hooks) and human-readable metadata fields (`scenario`, `pick_description`, `target_description`, `strategy_description`, `verification_description`, `rationale`). The mock pipeline calls `extract_task_config()` which obtains the `TaskSpec` and delegates to `prepare_mock_from_spec()` to generate `LightweightObj` items, apply semantic labels, cache geometry, and create the strategy.

### 8.3 Dynamic Subclassing for Position Updates

`MockTaskContextWithPlaceUpdate` uses `type()` to dynamically create a subclass of `MockTaskContext` at runtime. This avoids circular imports (since `MockTaskContext` is defined in a different module) while providing the position-update behavior needed for verification and stacking.

### 8.4 Monkeypatching for Verification

The mock verifier temporarily replaces `compute_aabb` and `create_bbox_cache` in the `task_verification` module, then restores them via a cleanup callback. This allows using the real `PlacementChecker` class without modification.

### 8.5 Teleport Mode Preserves Tree Structure

Teleport mode doesn't skip or shortcircuit behavior tree phases. All 9 phases still execute; only action application is suppressed and objects are teleported at completion. This ensures the tree's state management (blackboard writes, status transitions, memory sequences) is exercised identically in both modes.


## 9. Batch Runner Scripts

Three top-level bash wrappers run every task (or a specified subset of tasks)
through one of the entry points and collate per-task pass/fail lines into a
single `*_results.out` file. All three share a single implementation in
`_run_all_tasks_lib.sh`; each wrapper only fills in mode-specific defaults
(label, default skip list, default `--tasks*` filter, extra task args, runner
script) and sources the library.

### 9.1 The wrappers

| Wrapper | Mode | Runner | Default `MODE_TASK_ARGS` | Default `NUM_RUNS` |
|---|---|---|---|---|
| `run_all_mock_tasks.sh` | mock | `run_mock_task.py` | (none) | 1 |
| `run_all_teleport_tasks.sh` | teleport | `run_task.py` | `--teleport` | 3 |
| `run_all_simulation_tasks.sh` | sim | `run_task.py` | (none) | 1 |

Every wrapper appends `--show-status --auto-exit` to each invocation; the mock
runner accepts `--auto-exit` as a no-op for compatibility.

### 9.2 Usage

```bash
# Run all tasks (skipping any in DEFAULT_SKIP_TASKS) once each.
bash run_all_mock_tasks.sh

# Run only specific task indices (skip list is ignored when indices are passed).
bash run_all_mock_tasks.sh 1 3 5

# Run every task 10× with random seeds.
NUM_RUNS=10 bash run_all_mock_tasks.sh

# Run across both task tiers (tasks/ + tasks2/) instead of the default --tasks2.
TASK_SET=ALL bash run_all_mock_tasks.sh

# Pin seeds per task per run via JSON or text file.
SEEDS_FILE=seeds.json bash run_all_teleport_tasks.sh 1 3 5

# Direct outputs somewhere other than ./logs.
SIM_LOGS_DIR=/tmp/run42 bash run_all_simulation_tasks.sh
```

### 9.3 Environment variables

All three wrappers honour the same env vars (implemented in
`_run_all_tasks_lib.sh::run_all_tasks`):

| Variable | Purpose | Notes |
|---|---|---|
| `NUM_RUNS` | Runs per task | Overrides each wrapper's `DEFAULT_NUM_RUNS` |
| `TASK_SET` | Which tier to enumerate | `--tasks1`, `--tasks2`, or `ALL` (no filter). Default `--tasks2`. Empty string is treated as `ALL`. |
| `SKIP_TASKS` | Space-separated names to skip | Setting `SKIP_TASKS=""` skips nothing. Ignored when positional indices are supplied. |
| `SEEDS_FILE` | JSON or text seed map | See section 9.4. When unset, runs use random seeds and tag the log line with `[seed=…]` only when one was pinned. |
| `EXTRA_ARGS` | Extra args appended verbatim | Lets one wrapper pass extra flags to `run_task.py`. |
| `SIM_LOGS_DIR` | Output directory | Default `./logs`. Created with `mkdir -p` if missing. Holds `i-j-<mode>.out` and `i-j-<mode>.stderr` per run, plus `<mode>_results.out`. |

### 9.4 Seeds-file format

`SEEDS_FILE` may point to either a JSON or whitespace-text seed map:

**JSON:**
```json
{
  "TableTask3": [42, 1234, 5678],
  "TableTaskColors1": [100]
}
```

**Text (one task per line, `#` comments allowed):**
```text
# TaskName seed1 seed2 seed3 ...
TableTask3       42  1234  5678
TableTaskColors1 100
```

For each task `TASK_NAME`, the `j`-th run uses `SEEDS_MAP[TASK_NAME][j-1]`
when present; otherwise no `--seed` is passed and the runner picks a random
seed. Detection is by file extension (`.json`) with a fallback to
JSON-then-text.

### 9.5 What the output looks like

`$SIM_LOGS_DIR/<mode>_results.out` collects the human-readable header and a
single status line per run, interleaved with skip notices:

```text
----FAILURE MESSAGES-----
  1. TableTask3                ...
--- Task 1.1 (1.1 of N.M) [seed=42] ---
Task TableTask3 Completed successfully (seed: 42).
----------------------
  2. TableTaskBottles1         ...
  ** task TableTaskBottles1 skipped **
----------------------
  3. TableTaskColors1          ...
--- Task 3.1 (2.1 of N.M) [seed=100] ---
Task TableTaskColors1 Verification checks reported UNSUCCESSFUL completion (seed: 100).
----------------------
```

The full stdout / stderr for each run lives in
`$SIM_LOGS_DIR/<i>-<j>-<mode>.out` and `<i>-<j>-<mode>.stderr` so failures can
be re-run and inspected without re-running the whole batch.

### 9.6 When to use which wrapper

- **`run_all_mock_tasks.sh`** — fastest regression check. Every task runs the
  full BT against the mock harness in well under a minute. The default
  task set is `--tasks2`; use `TASK_SET=ALL` to include the legacy `tasks/`
  tier as well. No GPU required.
- **`run_all_teleport_tasks.sh`** — fast visual / scene validation in Isaac
  Sim. Defaults to 3 runs per task so randomisation issues surface quickly.
  Useful before commits that touch scene generation, virtual-target wiring,
  or verification hooks.
- **`run_all_simulation_tasks.sh`** — full physics regression. Default skip
  list excludes the long-running cuRobo / dense-scenario tasks
  (`TableTask3Curobo`, `TableTaskBottlesToConveyor2x`, `TableTaskSortAndStack`,
  `TableTaskSortAndStack2`) — override `SKIP_TASKS=""` to include them.


## 10. Typical Workflows

### Developing a New Task

```bash
# 1. Write the task class in tasks/
# 2. Quick mock validation
python run_mock_task.py --task MyNewTask --show-status

# 3. Run with multiple seeds to catch randomization issues
for seed in 1 2 3 42 99; do
    python run_mock_task.py --task MyNewTask --seed $seed
done

# 4. Visual check in Isaac Sim with teleport
python run_task.py --task MyNewTask --teleport

# 5. Step-by-step inspection (interactive)
python run_task.py --task MyNewTask --teleport --pause

# 6. Full physics simulation
python run_task.py --task MyNewTask
```

### Debugging a Failed Placement

```bash
# Mock with verbose output
python run_mock_task.py --task MyTask --show-status --seed 42

# Check specific pick/target counts
python run_mock_task.py --task MyTask --pick-count 3 --target-count 3

# Isaac Sim teleport to see positions visually
python run_task.py --task MyTask --teleport --pause
# → Press ENTER after each cycle, inspect in GUI
```

### Running the Test Suite

```bash
# All tests
mamba run -n env_isaacsim51 env PYTHONPATH=$(pwd)/extsMock:$(pwd) python -m pytest tests/

# Single test file
mamba run -n env_isaacsim51 env PYTHONPATH=$(pwd)/extsMock:$(pwd) python -m pytest tests/test_task_context.py -v

# Single test
mamba run -n env_isaacsim51 env PYTHONPATH=$(pwd)/extsMock:$(pwd) python -m pytest tests/test_mock_controllers.py::TestMockGripper -v
```

### Batch Regression Sweep

```bash
# Fast: every mock task once, results to ./logs/mock_results.out
bash run_all_mock_tasks.sh

# Run only TableTask3-style indices a few times each with pinned seeds
SEEDS_FILE=seeds.json NUM_RUNS=3 bash run_all_teleport_tasks.sh 1 3 5

# Full physics regression, redirect logs
SIM_LOGS_DIR=/tmp/sim-$(date +%s) bash run_all_simulation_tasks.sh
```
