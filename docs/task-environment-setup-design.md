# Task and Environment Setup System: Design Document

This document describes the architecture and key design choices of the task definition, environment setup, item generation, pick-to-target pairing, and automated success verification systems.

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Task Discovery and Registration](#2-task-discovery-and-registration)
3. [Task Class Hierarchy](#3-task-class-hierarchy)
4. [Environment and Workspace Setup](#4-environment-and-workspace-setup)
5. [Item Generation Pipeline](#5-item-generation-pipeline)
6. [Position Generators](#6-position-generators)
7. [Attribute Strategies and Randomization](#7-attribute-strategies-and-randomization)
8. [Asset System and Geometry](#8-asset-system-and-geometry)
9. [Pick-to-Target Pairing Strategies](#9-pick-to-target-pairing-strategies)
10. [Placement Position Computation](#10-placement-position-computation)
11. [Stacking Constraints and Pick Ordering](#11-stacking-constraints-and-pick-ordering)
12. [Automated Success Verification](#12-automated-success-verification)
13. [Mock Execution Path](#13-mock-execution-path)
14. [Example Task Walkthrough](#14-example-task-walkthrough)

---

## 1. System Overview

The task system enables rapid creation of multi-object pick-and-place scenarios for a UR10 robotic arm in Isaac Sim. Each task is a self-contained definition that specifies:

- **What objects to create** (pick items and target markers) via composable generation strategies
- **Where to place them** in the scene via position generators
- **How to pair picks with targets** via pairing strategies (sequential, color-matching, type-based, etc.)
- **How to verify success** via spatial checks against expected placements

The design follows a **strategy pattern** throughout: tasks compose generators and strategies rather than implementing placement logic directly, making it straightforward to create new tasks by combining existing building blocks.

### Key Files

| File | Role |
|------|------|
| `run_task.py` | Main entry point; task discovery, CLI, simulation loop |
| `multi_pickplace_task.py` | `UR10MultiPickPlaceTask` base class |
| `item_generation.py` | `ItemGenerator`, position generators, attribute strategies |
| `asset_utils.py` | Isaac-Sim-dependent asset scene ops: prim creation, live-prim AABB computation, semantic-label helpers. Re-exports pure-data symbols from `asset_data_utils`. |
| `asset_data_utils.py` | Isaac-Sim-free asset metadata and geometry (`AssetMetaData`, `ITEMS_MAP`, `PRIM_TYPES`, `PrimGeometry`, `scale_aabb`, `lookup_prim_geometry`) |
| `multi_pick_strategy.py` | `MultiPickStrategy` and subclasses (pairing logic) |
| `task_spec.py` | `TaskSpec` dataclass capturing task configuration declaratively (strategies, metadata, rationale) |
| `simulation_configurator.py` | `SimulationConfigurator` — scene objects, geometry cache, verification |
| `task_controller.py` | `TaskController` — policy layer wrapping strategy, context, BT controller; generates virtual targets |
| `table_setup.py` | Isaac-Sim-dependent workspace setup (`setup_two_tables`, `spawn_open_box`, etc.). Re-exports constants from `env_config_values.py`. |
| `env_config_values.py` | Isaac-Sim-free workspace geometry constants (`BIN_X_COORD`, `ITEM_SPAWN_REFERENCE_Z`, `CART_SURFACE_CENTER`, `Region2D`, `compute_region_2d`, etc.) |
| `task_verification.py` | Spatial verification primitives and `TaskVerifier` |
| `tasks/*.py` | Individual task definitions (20 classes) |
| `tasks_mock/mock_task_utils.py` | Mock execution and verification infrastructure |

---

## 2. Task Discovery and Registration

Tasks are discovered at runtime via AST parsing, requiring no manual registration.

**Discovery process** (`run_task.py:_discover_task_modules()`):

1. Scans the `tasks/` directory for `.py` files (excluding `__init__.py`)
2. Parses each file's AST to extract `ClassDef` nodes
3. Maps each class name to its module path (e.g., `"TableTaskColors1"` -> `"tasks.table_task_colors_1"`)
4. When a task is requested (via `--task` CLI arg), dynamically imports and instantiates the class

This approach avoids circular imports and allows new tasks to be added by simply dropping a new file into `tasks/` with a class that extends `UR10MultiPickPlaceTask`.

**Currently available tasks** (20 classes):

| Category | Tasks |
|----------|-------|
| Basic geometry | `TableTask2`, `TableTask3`, `TableTask4`, `TableTask5` |
| Color-based | `TableTaskColors1`, `TableTaskColorShapes`, `TableTaskColorCircle`, `TableTaskColorBinSort` |
| Object-specific | `TableTaskBottles1`, `TableTaskSoupCans1`, `TableTaskCrackerBoxes1` |
| Complex packing | `TableTaskMixedPacking`, `TableTaskSoupCanPacking` |
| Layered/stacking | `TableTaskLayeredCubes`, `TableTaskLayeredCircle` |
| Conveyor-based | `TableTaskConveyorColorStacks`, `TableTaskCartToConveyor` |
| Mixed types | `TableTaskMixedCircle` |

---

## 3. Task Class Hierarchy

All tasks extend `UR10MultiPickPlaceTask`, which extends Isaac Sim's `BaseTask`.

```
isaacsim.core.api.tasks.BaseTask
  └── UR10MultiPickPlaceTask (multi_pickplace_task.py)
        ├── TableTask2 (tasks/table_task_2.py)
        ├── TableTaskColors1 (tasks/table_task_colors_1.py)
        ├── TableTaskLayeredCubes (tasks/table_task_layered_cubes.py)
        └── ... (20 task classes)
```

### Task `__init__` Pattern

All tasks use `TaskSpec` to declaratively capture all configuration, passing it to `super().__init__(task_spec=spec, ...)`. The base class constructor no longer accepts `pick_generation_strategy` or `target_generation_strategy` as direct arguments — these must be provided via `TaskSpec`. The `task_description` parameter has also been removed from task constructors; it is set directly on `TaskSpec`. **Execution-policy fields** (pairing strategy factory, BT tree, virtual targets, postures, timeouts, reachability gates, cuRobo) live on a nested `TaskImplementationSpec` (assigned via `implementation=`).

**Standard pattern (TaskSpec + TaskImplementationSpec):**

```python
class TableTaskExample(UR10MultiPickPlaceTask):
    def __init__(self, task_name="TableTaskExample", **kwargs):
        from item_generation import ItemGenerator, GridPositionGenerator, FixedValue
        from table_setup import setup_two_tables
        from task_spec import TaskImplementationSpec, TaskSpec

        stage_units = get_stage_units()
        expected_scale = np.array([0.0515, 0.0515, 0.0515]) / stage_units

        pick_gen = ItemGenerator(
            position_generator=GridPositionGenerator(...),
            asset_type_strategy=FixedValue("cube"),
            color_strategy=RandomChoice(["red", "green", "blue"]),
            scale_strategy=FixedValue(expected_scale),
        )
        target_gen = ItemGenerator(
            position_generator=GridPositionGenerator(...),
            asset_type_strategy=FixedValue("marker"),
            hidden_strategy=FixedValue(True),
        )

        spec = TaskSpec(
            task_name=task_name,
            task_description="Example pick-place task",
            pick_generation_strategy=pick_gen,
            target_generation_strategy=target_gen,
            setup_workspace=setup_two_tables,
            containment_check=False,
            # Scene-side metadata
            scenario={"source": "bin", "destination": "dropzone_grid", "workspace": "two_tables"},
            pick_description={"asset_types": ["cube"], "count": 12, "colors": "RandomChoice(...)"},
            target_description={"type": "visible_markers", "arrangement": "grid on dropzone"},
            rationale={"create_strategy": "Cubes must be placed on same-color targets"},
            # Execution policy (nested)
            implementation=TaskImplementationSpec(
                create_strategy=lambda picks, targets: ColorMatchStrategy(
                    picks, targets, color_palette=["red", "green", "blue"],
                ),
                strategy_description={"class": "ColorMatchStrategy", "pairing": "color_match"},
            ),
        )
        super().__init__(task_spec=spec, **kwargs)
```

The old pattern (passing generation strategies directly to `super().__init__()` and overriding `_create_strategy()`) is no longer supported.

### Key Extension Points

Tasks customize behavior through `TaskSpec` fields. Scene-side and impl-side are split:

**Scene-side (outer `TaskSpec`):**
- **`setup_workspace`** — callable `(scene, assets_root_path) -> None` for workspace setup.
- **`containment_check`** — enables multi-occupancy containment verification automatically.
- **`box_verification_info`** — dict with `"box_specs"` list for centralized box containment verification.
- **`spatial_check_fn`** / **`placement_constraints_fn`** — custom verification hooks.
- **`stacking_enabled`** — scene fact: pick objects are pre-stacked.

**Impl-side (nested `TaskImplementationSpec`):**
- **`create_strategy`** — callable `(pick_objs, target_objs) -> MultiPickStrategy` factory.
- **`virtual_target_generation_strategy`** — generator for hidden/utility markers created at pairing time, not during scene setup. Used for box-packing tasks where targets don't need scene prims. Uses outer `target_count` for the generation count.
- **`tree_factory`** — selects the BT variant (default 9-phase / cortex / cuRobo).
- **`target_reachable_fn`**, **`pick_min_reachable_z`**, **`pick_max_reachable_radius_xy`** — runtime reachability gates.
- **`pick_approach_p_thresh`**, **`pick_approach_std_dev`** — approach funnel tuning.
- **`move_/approach_/insert_timeout_s`** — sim-time watchdog deadlines.
- **`ee_height_for_move`**, **`place_hover_above_z`**, **`place_approach_distance`** — transport/hover geometry.
- **`pick_posture_config`**, **`place_posture_config`** — null-space posture biases.
- **`use_curobo`**, **`curobo_robot_yaml`**, **`curobo_obstacles_fn`** — cuRobo planner config.
- **`startup_delay_seconds`** — sim-time delay before BT begins ticking.

Helpers: `TaskSpec.with_impl(**kw)` copies the spec with given impl fields overridden (used in v2 `_customize_spec`); `TaskSpec.impl` property returns a default `TaskImplementationSpec()` when `implementation` is None.

#### Human-Readable Metadata Fields

`TaskSpec` and `TaskImplementationSpec` each carry their own optional metadata fields. Scene-related metadata lives on the outer spec; implementation-related metadata lives on the inner one:

- **`TaskSpec.scenario`** — dict describing source, destination, and workspace (e.g., `{"source": "bin", "destination": "dropzone_grid", "workspace": "two_tables"}`).
- **`TaskSpec.pick_description`** — dict describing pick items (asset types, count, arrangement, colors, orientation).
- **`TaskSpec.target_description`** — dict describing target objects (type, arrangement, count, containers, virtual).
- **`TaskSpec.verification_description`** — dict describing verification config (spatial check, placement constraints).
- **`TaskSpec.rationale`** — scene-side justification strings (generators, constraints, verification).
- **`TaskImplementationSpec.strategy_description`** — dict describing the strategy (class, pairing logic, details).
- **`TaskImplementationSpec.rationale`** — impl-side justification strings (strategy choice, tree variant, motion tuning, postures).

### Refactored Internal Architecture

`UR10MultiPickPlaceTask` now delegates to two helper objects:

- **`SimulationConfigurator`** (`simulation_configurator.py`) — owns pick/target object lists, `_prim_geometry` cache, bounding-box cache, object generation (`add_source_objects`, `add_target_objects`), and verification (`check_groundtruth`, `check_incremental`).
- **`TaskController`** (`task_controller.py`) — policy layer that owns strategy creation, `TaskContext` construction, BT controller creation, and observation augmentation. Takes a `strategy_factory` callable and `prim_geometry` dict. Also handles **virtual target generation**: if a `virtual_target_generation_strategy` is configured, `TaskController._generate_virtual_targets()` creates `LightweightObj` instances (with cached geometry and semantic labels) and appends them to the target list before strategy creation. Scene targets keep their original indices.

The task class creates a `SimulationConfigurator` in `__init__` and a `TaskController` in `post_reset()`, keeping simulation setup separate from execution policy.

---

## 4. Environment and Workspace Setup

### Scene Layout

The workspace is defined by constants in `env_config_values.py` (re-exported from `table_setup.py` for back-compat):

```
                          ┌─────────────────┐
                          │  Conveyor Belt   │
                          │  (pick source)   │
                          │  X≈0.13, Y≈0.15 │
                          └─────────────────┘
    ┌──────────────────────────────┐
    │          Table               │
    │   Center: [-0.81, 0.33, 0.1]│
    │   Size: 1.2m × 0.7m         │
    │                              │
    │   ┌─────────┐               │
    │   │Pick Bin │               │
    │   │[-0.62,  │   Dropzone    │
    │   │ 0.60]   │   [0.25, 0.69]│
    │   └─────────┘               │
    └──────────────────────────────┘

              UR10 Robot
              [0, 0, 0]

    ┌──────────────────┐
    │   Cart + Surface │
    │ [-0.78, 0.27]    │
    └──────────────────┘
```

**Key coordinate constants:**

| Constant | Value | Description |
|----------|-------|-------------|
| `UR_COORDS` | `[0, 0, 0]` | Robot base position |
| `ITEM_SPAWN_REFERENCE_Z` | `0.1` | Table surface height |
| `TABLETOP_CENTER_POINT` | `[-0.81, 0.33, 0.1]` | Center of the main table |
| `BIN_COORDS` | `[-0.62, 0.60, 0.2]` | Center of the pick bin |
| `DROPZONE_CENTER_POINT` | `[0.04, 0.69, 0]` | Center of the drop zone |
| `GROUND_PLANE_Z_OFFSET` | `-0.5` | Ground plane Z position |

### Setup Functions

**`setup_two_tables(scene, assets_root_path, standard_objs=True, add_bin=True)`** — the primary workspace setup:
- Adds a conveyor belt (visual prop)
- Adds the robot mount
- Adds a cart with surface below the table
- Optionally adds default YCB objects (cracker box, sugar box, soup can, mustard bottle)
- Optionally adds a KLT picking bin

---

## 5. Item Generation Pipeline

The item generation system uses a **strategy composition** pattern where an `ItemGenerator` orchestrates position computation and attribute assignment to produce a list of `ItemSpec` objects.

### Data Flow

```
Task.__init__()
    │
    ├── Creates ItemGenerator(pick) with position + attribute strategies
    ├── Creates ItemGenerator(target) with position + attribute strategies  [scene targets]
    └── Optionally creates virtual_target_generation_strategy  [hidden/utility markers]
            │
            ▼
UR10MultiPickPlaceTask.set_up_scene()
    │
    ├── add_source_objects()
    │     ├── pick_generator.generate(count_range, seed) → List[ItemSpec]
    │     └── For each ItemSpec:
    │           ├── add_asset(scene, ...) → creates Isaac Sim prim
    │           └── get_or_compute_prim_geometry() → caches PrimGeometry
    │
    └── add_target_objects()  [scene targets only]
          ├── target_generator.generate(count_range, seed) → List[ItemSpec]
          └── For each ItemSpec:
                ├── add_asset(scene, ...) → creates Isaac Sim prim
                └── get_or_compute_prim_geometry() → caches PrimGeometry
            │
            ▼
UR10MultiPickPlaceTask.post_reset()
    │
    └── TaskController.create_strategy()
          ├── _generate_virtual_targets()  [if configured]
          │     ├── strategy.generate(count_range, seed) → List[ItemSpec]
          │     └── create_lightweight_objs_from_items() → List[LightweightObj]
          │           ├── LightweightObj with _semantic_labels (no USD API)
          │           └── lookup_prim_geometry() → caches in _prim_geometry
          ├── Merges: scene_targets + virtual_targets
          └── strategy_factory(pick_objs, combined_targets) → MultiPickStrategy
```

### Virtual Target Generation

For tasks placing items into boxes or box-like containers, target markers (hidden `"marker"` objects) can be generated **at pairing time** rather than during scene setup. This avoids cluttering the USD scene with invisible utility objects and allows the target layout to adapt to the pick objects.

**How it works:**

1. The task sets `virtual_target_generation_strategy` on `TaskSpec.implementation` (a `TaskImplementationSpec`) — an `ItemGenerator` or custom generator that produces `ItemSpec` items for hidden markers.
2. During `post_reset()`, `TaskController._generate_virtual_targets()` calls the generator and converts items to `LightweightObj` instances via `create_lightweight_objs_from_items()` (`task_context_base.py`).
3. Each `LightweightObj` gets in-memory semantic labels (type, color, name) stored in `_semantic_labels` (checked by `has_label()` before falling back to USD queries), and precomputed geometry via `_local_half_extents`.
4. Virtual targets are appended to the scene target list. The combined list is passed to the strategy factory.
5. The task's `_target_objs` is updated to include virtual targets after strategy creation.

**This is the recommended default** for new tasks that place items into box-like containers. Tasks with visible scene-spawned targets (markers, rects, discs, pads) continue to use `target_generation_strategy` as before.

**Example (from `TableTaskSoupCanPacking`):**

```python
spec = TaskSpec(
    ...
    pick_generation_strategy=pick_strategy,
    # No target_generation_strategy — all targets are virtual
    box_verification_info={"box_specs": box_specs},
    containment_check=True,
    implementation=TaskImplementationSpec(
        virtual_target_generation_strategy=target_strategy,  # hidden markers
    ),
)
```

### ItemSpec

Each generated item is captured as an `ItemSpec` dataclass (`item_generation.py:7`):

```python
@dataclass
class ItemSpec:
    asset_type: str                              # e.g., "cube", "soup_can"
    position: np.ndarray                         # [x, y, z] world coordinates
    orientation: Optional[np.ndarray] = None     # quaternion [w, x, y, z]
    scale: Optional[np.ndarray] = None           # [sx, sy, sz]
    color: Optional[Union[str, np.ndarray]] = None  # named color or RGB
    name: Optional[str] = None                   # auto-generated if None
    hidden: bool = False                         # True for invisible markers
```

### ItemGenerator

The `ItemGenerator` class (`item_generation.py:281`) composes a position generator with attribute strategies:

```python
class ItemGenerator:
    def __init__(self,
        position_generator: PositionGenerator,
        asset_type_strategy: AttributeStrategy = FixedValue("cube"),
        orientation_strategy: AttributeStrategy = FixedValue(None),
        scale_strategy: AttributeStrategy = FixedValue(None),
        color_strategy: AttributeStrategy = FixedValue(None),
        hidden_strategy: AttributeStrategy = FixedValue(False),
    ): ...
```

**Count resolution** in `generate(count_range, seed)`:

| `count_range` value | Behavior |
|---------------------|----------|
| `None` | Uses position generator's capacity (or 1 if unlimited) |
| `int` | Exact count |
| `(min, max)` | Random integer in range; max defaults to generator capacity if `None` |

The `--pick-count`, `--target-count`, `--pick-count-min/max`, and `--target-count-min/max` CLI arguments flow through to `count_range`, allowing runtime override of how many objects a task generates.

---

## 6. Position Generators

All position generators implement the `PositionGenerator` abstract base class:

```python
class PositionGenerator(ABC):
    def get_positions(self, count: int, seed: Optional[int] = None) -> List[np.ndarray]: ...
    def capacity(self) -> Optional[int]: ...  # Max positions, or None if unlimited
```

### GridPositionGenerator

Creates a rectangular grid centered at a given point (`item_generation.py:100`).

**Parameters:** `center`, `rows`, `cols`, `spacing_x`, `spacing_y`, `z_offset`, `randomize`

**Behavior:**
- Computes grid origin: `start = center - (size - 1) * spacing / 2`
- Generates all `rows × cols` slot positions as `(row, col)` indices
- If `randomize=True`: shuffles slots using the seed, selects first `count` slots
- If `randomize=False`: selects slots sequentially (row-major order)
- **Capacity:** `rows × cols`

**Negative spacing** is supported and commonly used for targets (e.g., `spacing_x=-0.15` flips the X direction of the grid).

### CircularPositionGenerator

Arranges items on a circle (`item_generation.py:144`).

**Parameters:** `center`, `radius`, `z_offset`, `count` (slot count), `randomize`

**Behavior:**
- With `randomize=False` (default): dynamically spaces items evenly based on the *requested* count: `angle = 2π × i / count`. Requesting 3 items from an 8-slot circle gives 120-degree spacing.
- With `randomize=True`: pre-divides the circle into `count` fixed slots, shuffles, and selects `count` of them. This produces randomized angular positions while maintaining consistent slot geometry.
- **Capacity:** `count` (the slot count parameter)

### ConveyorPositionGenerator

Linear arrangement along the Y axis simulating a conveyor belt (`item_generation.py:214`).

**Parameters:** `center_x`, `center_y`, `z`, `spacing`, `jitter_x`, `jitter_y`

**Behavior:**
- Centers items around `center_y`: `start_y = center_y - spacing × (count - 1) / 2`
- For each item, applies uniform random jitter in X and Y within `[-jitter, +jitter]`
- Jitter is seeded for reproducibility
- **Capacity:** unlimited (no fixed slot count)

### LayeredPositionGenerator

Replicates a base generator's positions across multiple Z layers for stacking scenarios (`item_generation.py:234`).

**Parameters:** `base_generator`, `num_layers`, `layer_height`

**Behavior:**
- Gets one full layer of positions from the base generator
- Replicates positions for each layer, adding `layer_idx × layer_height` to Z
- Fills bottom-up: layer 0 first, then layer 1, etc.
- Supports partial top layers when `count` is not a full multiple
- **Capacity:** `base_capacity × num_layers`

---

## 7. Attribute Strategies and Randomization

Attribute strategies implement the `AttributeStrategy` interface, returning a value per item index.

### Strategy Classes

| Strategy | Behavior | Typical Use |
|----------|----------|-------------|
| **`FixedValue(value)`** | Returns `value` for every item | Fixed asset type, scale, visibility |
| **`RandomChoice(options)`** | Random selection from list | Random colors per item |
| **`SequentialChoice(options, loop=True)`** | Cycles through list by index | Layer-based colors, repeating patterns |
| **`MixedScaleStrategy(types, scale)`** | Primitives get `scale`; USD assets get identity | Tasks mixing cubes and USD models |
| **`MixedOrientationStrategy(types)`** | Applies -90-degree X rotation to specific USD assets | Upright boxes and bottles |

### Seeded Randomization

All randomization is seed-controlled for reproducibility:

- **`RandomChoice`** uses `seed + index` per item, ensuring the same item index always gets the same value across runs with the same seed, while different indices get different values.
- **Position generators** create `random.Random(seed)` instances for shuffling and jitter.
- **`ItemGenerator.generate()`** passes the seed through to both position generators and attribute strategies.

The seed flows from CLI (`--seed`) through the task constructor to `ItemGenerator.generate()`.

### Randomization Override

The `--randomize` CLI flag overrides the `randomize` attribute on position generators at the task level, allowing grid/circle slot selection to be toggled between randomized and sequential without modifying task code.

---

## 8. Asset System and Geometry

### Asset Registry

Assets are registered in `ITEMS_MAP` (`asset_data_utils.py`) as `AssetMetaData` entries:

```python
@dataclass
class AssetMetaData:
    asset_type: str
    is_primitive: bool            # auto-detected from PRIM_TYPES
    usd_path: Optional[str]       # path to USD file (for non-primitives)
    color: str = "blue"
    grasp_height: Optional[float] # override for geometry
    rest_height: Optional[float]
    top_surface_height: Optional[float]
```

**Primitive types** (names in `PRIM_TYPES`; created via Isaac Sim API through `PRIMS_MAP` in `asset_utils`):
`cube`, `disc`, `ball`, `cylinder`, `capsule`, `cone`, `rect` (FixedCuboid), `marker` (VisualCuboid)

**USD-backed assets** (loaded from USD files):
YCB objects (`cracker_box`, `sugar_box`, `soup_can`, `mustard_bottle`), bottles (`madara_bottle`), bins (`KLT_Bin`, `sorting_bin_blue`), mugs, factory parts (gears, bolts), etc.

### PrimGeometry

`PrimGeometry` (`asset_data_utils.py`) captures the intrinsic geometry of each spawned object:

```python
@dataclass
class PrimGeometry:
    grasp_height: float          # Z from origin to EE grasp point
    rest_height: float           # Z from origin to bottom surface
    top_surface_height: float    # Z from origin to top surface
    local_half_extents: np.ndarray  # [half_x, half_y, half_z] bounding box
    needs_aabb_scale_correction: bool
```

These values are used for:
- **Picking:** computing the EE offset to approach the object at the correct grasp height
- **Placing:** computing the Z coordinate where the pick object should land on the target
- **Verification:** synthesizing mock AABBs for testing without Isaac Sim

**Geometry computation** happens in two ways:
1. **Live computation** (`compute_prim_geometry()` in `asset_utils.py`) — from the loaded prim's axis-aligned bounding box (AABB) in Isaac Sim
2. **Precomputed lookup** (`lookup_prim_geometry()` in `asset_data_utils.py`) — from `asset_prim_geometry.json`, with scale and orientation transforms applied. Used by the mock system (does not require Isaac Sim).

### AABB Scale Correction

Isaac Sim has a known bug where primitive cuboid types (cube, rect, marker) report AABBs at double their actual scale. The system corrects for this:
- `needs_aabb_scale_correction` flag is set on cuboid-type prims
- `get_corrected_aabb()` applies an inverse-scale correction before any spatial check
- `scale_aabb()` performs the correction: scale the AABB extents relative to center

---

## 9. Pick-to-Target Pairing Strategies

The `MultiPickStrategy` class (`multi_pick_strategy.py`) is the central abstraction for deciding which pick object gets placed on which target. It manages:

- **Pairing computation** — which pick maps to which target
- **Pick iteration** — the order in which picks are attempted
- **EE orientation decisions** — how the gripper should orient for pick and drop
- **Completion tracking** — recording finished pick-place cycles

Note: Placement position computation (drop Z from geometry) and EE offset computation are handled by `TaskContextBase` using the `_prim_geometry` cache.

### Pairing Initialization

`initialize_pairings()` performs these steps:

1. Calls `pair_picks_with_targets()` (overridable) to produce `(pick_index, Optional[target_index])` pairs
2. Builds lookup dicts: `_pairings_by_pick_name`, `_pick_name_to_index`
3. Applies stacking order constraints via `_apply_stacking_order()` (reorders top-down)
4. Redistributes targets via `_reassign_targets_by_picking_order()` (ensures early picks get targets)

### Strategy Subclasses

| Strategy | Pairing Logic | Key Behaviors |
|----------|--------------|---------------|
| **`MultiPickStrategy`** | Sequential: `pick[i]` -> `target[i]` | Default; supports stacking via `stacking_map` |
| **`ColorMatchStrategy`** | Match by semantic color label | Builds `color_to_targets` map; restricts valid targets by color |
| **`TypeBasedStrategy`** | Route by asset type to type-specific zones | Separates picks by type (e.g., cubes vs. balls), routes each to dedicated targets |
| **`BottlePickStrategy`** | Sequential pairing | Overrides drop orientation (90-degree X rotation for horizontal placement); drop offset computed by TaskContextBase from `_prim_geometry` |
| **`LayeredStackStrategy`** | Multi-layer stacking by property value | Bottom layer uses marker targets; upper layers dynamically add completed items as targets |
| **`SingleStackStrategy`** | All picks -> one growing stack | Dynamically adds each placed item as the target for the next; computes recommended EE height |

### ColorMatchStrategy Detail

Used by `TableTaskColors1`, `TableTaskColorBinSort`, `TableTaskColorShapes`.

1. Builds a map of `color_name -> [list of target indices]` from semantic labels on targets
2. For each pick object, looks up its semantic color
3. Finds the first unused target of that matching color
4. Yields `(pick_index, matching_target_index)` or `(pick_index, None)` if no match exists
5. `valid_targets_for_pick()` restricts verification to color-matched targets only

### LayeredStackStrategy Detail

Used by `TableTaskConveyorColorStacks`.

- **Layer 0:** Objects are placed on marker targets (fixed scene objects)
- **Layer k (k > 0):** Objects are placed on top of completed items from layer k-1, which are dynamically added to the target list via `_extend_target_objs()`
- Tracks `_layer_target_starts` to know where each layer's targets begin in the target array
- Classifies picks by a property (e.g., color) and computes `num_complete_stacks`

---

## 10. Placement Position Computation

When the behavior tree is ready to place an object, it queries `get_placing_info()` on `TaskContextBase`:

```python
def get_placing_info(self, pick_name, end_effector_orientation_for_drop=None) -> (target_name, position, orientation):
```

The strategy provides `get_placing_target_name(pick_name)` for pure pairing lookup, while `TaskContextBase` computes the actual placement position using `_prim_geometry`.

### Drop Height Calculation

The critical Z-coordinate computation uses geometry from both the pick and target objects:

```
drop_z = target_pos[2] + target_geom.top_surface_height + pick_geom.rest_height
```

Where:
- `target_pos[2]` — the target object's world Z position (its origin)
- `target_geom.top_surface_height` — distance from target's origin to its top surface
- `pick_geom.rest_height` — distance from pick's origin to its bottom surface

This ensures the pick object's bottom surface makes contact with the target's top surface without penetration.

### Drop Orientation

- Default: preserves the pick object's current orientation
- `BottlePickStrategy` overrides to rotate bottles 90 degrees for horizontal placement
- Strategies return `None` from `get_end_effector_orientation_for_drop()` to use the default

### End-Effector Offsets

For picking, the EE offset is computed on `TaskContextBase` from `PrimGeometry.grasp_height`:

```python
def get_end_effector_offset(self, pick_name) -> np.ndarray:
    geom = self._prim_geometry.get(pick_name)
    if geom is not None:
        return np.array([0.0, 0.0, geom.grasp_height])
    return self._EE_OFFSET_FALLBACK.copy()
```

For dropping, `get_end_effector_offset_for_drop(pick_name, end_effector_orientation_for_drop)` returns the world-frame EE-to-item-center vector after the EE rotates from its pick orientation to the supplied drop orientation. The vector from the held item's center to the EE flange is fixed in the EE's *local* frame (set at grasp time as `[0, 0, grasp_height]`), so the world-frame offset at drop is the pick offset rotated by `R_drop · R_pick⁻¹`. When the drop orientation matches the pick orientation the relative rotation is identity and the result is identical to the pick offset; for the bottle case (90° EE rotation between pick and drop) the result is a horizontal world-frame offset of `grasp_height` magnitude.

The default EE orientation for picking is gripper-down: π/2 rotation around the Y axis.

### Transport Height

Between pick and place, the robot arm moves at `_ee_height_for_move` (default 0.3m). Tasks with tall obstacles (e.g., upright bottles) override this in their `__init__`:

```python
self._ee_height_for_move = 0.45 / stage_units
```

---

## 11. Stacking Constraints and Pick Ordering

When pick objects are physically stacked (items on top of other items), the system enforces top-down picking order.

### Stacking Map Computation

`compute_stacking_map(pick_objs, xy_tolerance=0.01)` (`multi_pick_strategy.py:25`):

1. Groups pick objects into columns by XY proximity (within `xy_tolerance`)
2. Within each column, sorts by Z ascending
3. Maps each lower item to the list of items directly above it:
   ```python
   {"cube_0": ["cube_6"], "cube_6": ["cube_12"]}
   # cube_12 is on top of cube_6, which is on top of cube_0
   ```

### Stacking Order Enforcement

`_apply_stacking_order()` reorders the picking list so topmost items come first:
- Computes depth for each item (items with nothing above = depth 0)
- Stable-sorts by depth ascending (shallowest/topmost first)

`_is_pick_available(name)` checks whether all items above `name` have been completed before allowing it to be picked.

### Wrap-Around Scanning

`_scan_for_available_pick()` handles the case where the current pick is blocked:
1. Scans forward from the current index for an unblocked, uncompleted pick with a target
2. If forward scan exhausts the list, wraps around from index 0
3. This allows previously-blocked items (now freed by completed picks above them) to be picked

---

## 12. Automated Success Verification

### Verification Architecture

The verification system is built on spatial primitives that check whether pick objects ended up at their expected target locations.

```
Task Completion
    │
    ▼
TaskVerifier
    ├── check_occupancy()      ← builds occupancy map from spatial checks
    └── verify(pick_indices)   ← produces VerificationResult
         ├── PlacementCheck[]  ← per-pick pass/fail + detail
         └── success: bool
```

### Spatial Check Functions

All checks use axis-aligned bounding boxes (AABBs) with tolerances:

| Function | Check | Key Tolerances |
|----------|-------|----------------|
| `is_on_top(pick, target)` | Pick rests on target: XY overlap + Z proximity | `z_tol=0.02m` (2cm vertical gap) |
| `is_within(pick, container)` | Pick is inside container: XY overlap + Z within | `z_tol=0.01m` (1cm check), `0.02m` (resting) |
| `is_within_box_geometry(obj, box_params)` | Object in defined box boundaries (no box prim needed) | `xy_tol=0.01m`, `z_tol=0.02m` (floor), `z_tol_top=0.01m` (wall top) |
| `is_vertical(obj)` | Longest axis aligned with Z | `tol=0.1` (10% relative) |
| `is_horizontal(obj)` | Longest axis in XY plane | `tol=0.1` (10% relative) |

### TaskVerifier

`TaskVerifier` (`task_verification.py:351`) is a composable verifier that accepts task-specific hooks as callable parameters:

```python
class TaskVerifier:
    def __init__(self,
        pick_objs, target_objs,
        spatial_check_fn=is_on_top,           # (pick, target, bb_cache, scale) -> bool
        valid_targets_fn=all_targets,          # (pick_index) -> [target_indices]
        placement_constraints_fn=always_true,  # (pick_index, target_index) -> bool
        allow_multi_occupancy=False,           # multiple picks per target?
    ): ...
```

### Verification Flow

1. **Build occupancy map** (`check_occupancy()`): for each pick, test against all targets using `spatial_check_fn`. Records which target each pick is on.

2. **Per-pick validation** (`verify()`): for each pick:
   - Find which valid target (if any) it occupies
   - If placed on a valid target: **PASS**
   - If NOT placed but valid targets remain available: **FAIL** (the pick should have been placed)
   - If NOT placed and all valid targets are occupied by other picks: **PASS** (acceptable overflow — more picks than targets)

3. **Result** — `VerificationResult` containing:
   - `success: bool` — True if no failures
   - `checks: list[PlacementCheck]` — per-pick results with detail strings
   - `failures: list[str]` — backward-compatible failure messages
   - `summary()` — multi-line diagnostic for logging

### Strategy-Specific Verification Hooks

Strategies provide custom verification via:

- **`valid_targets_for_pick(pick_index)`** — restricts which targets are valid for a given pick (e.g., `ColorMatchStrategy` only allows same-color targets)
- **`placement_constraints_satisfied(pick_index, target_index)`** — additional constraints beyond spatial checks
- **`get_spatial_check_fn()`** — overrides the default `is_on_top` with alternatives like `is_within_box_geometry`

### Centralized Box Containment Verification

For tasks placing items into boxes/bins/containers, box containment verification is handled centrally by the base class using `TaskSpec.box_verification_info` and `build_box_verification_hooks()` (`task_verification.py`). Tasks no longer need to override `check_groundtruth_task_success()` for box containment — they just set the appropriate `TaskSpec` fields.

**Configuration via TaskSpec:**

```python
spec = TaskSpec(
    ...
    box_verification_info={"box_specs": box_specs},
    containment_check=True,  # enables multi-occupancy mode
)
```

**Box spec dict format** (each entry in `box_specs`):

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Box identifier (e.g., `"red_collection_box"`) |
| `center_xy` | np.ndarray | `[x, y]` center of box interior |
| `floor_z` | float | Z coordinate of box floor surface |
| `inner_size` | np.ndarray | `[width, depth]` inner dimensions |
| `height` | float | Wall height |
| `match_labels` | dict (optional) | Label matching criteria, e.g. `{"color": "red"}` |
| `z_tol` | float (optional) | Lower-bound Z tolerance: object bottom vs box floor (default 0.02) |
| `z_tol_top` | float (optional) | Upper-bound Z tolerance: object bottom vs box wall top (default 0.01) |

**`build_box_verification_hooks(box_specs, pick_objs, strategy)`** is a factory that creates three components for `TaskVerifier`:

1. **`box_targets`** — lightweight namedtuple targets (one per box, only `.name` is needed)
2. **`spatial_check_fn`** — closure using `is_within_box_geometry()` with per-box geometry from the spec
3. **`valid_targets_fn`** — closure that checks `match_labels` via `has_label()` for each pick, and excludes unpaired picks (overflow picks with no assigned target)

The base class `_check_incremental()` and `check_groundtruth_task_success()` automatically use this mechanism when `box_verification_info` is present, creating a `TaskVerifier` with `containment_mode=True`.

**When to use:**
- **Default for all box-packing tasks** — any task where items go into boxes, bins, crates, or similar containers.
- Combines naturally with `virtual_target_generation_strategy` — virtual markers provide pairing targets while `box_verification_info` provides containment verification.
- Tasks needing additional per-pick constraints (e.g., upright orientation) add a `placement_constraints_fn` to the `TaskSpec`.

### Incremental Verification

Verification runs both incrementally (after each pick-place cycle) and at task completion:

**Incremental** — after each pick is placed:
- Detects newly completed picks by comparing `strategy.completed_picks` against previous state
- Runs `TaskVerifier.verify(pick_indices=[...])` on just the new picks
- Logs per-item results: DEBUG for success, WARNING for failures
- In real sim, deferred by one physics step so AABBs are up-to-date

**Final** — after task completion:
- Runs `TaskVerifier.verify()` on all picks
- Reports overall success/failure with diagnostic summary

### Exit Codes (Mock Execution)

| Code | Meaning |
|------|---------|
| 0 | Task finished AND verification passed |
| 1 | Task did not finish (incomplete) |
| 2 | Task finished BUT verification failed |

---

## 13. Mock Execution Path

The mock system enables testing task logic without Isaac Sim by replacing physics objects with lightweight data objects.

### Mock Object Generation

Mock execution uses `extract_task_config()` which instantiates the task, calls `task.get_task_spec()` to obtain the `TaskSpec`, then delegates to `prepare_mock_from_spec()` (`tasks_mock/mock_task_utils.py`):

1. Calls `ItemGenerator.generate()` for picks and targets from the spec's generation strategies
2. Creates `LightweightObj` instances instead of Isaac Sim prims — minimal objects with position, orientation, and get/set methods
3. Applies semantic labels (color, type) for strategy matching
4. Looks up precomputed geometry from `asset_prim_geometry.json` via `lookup_prim_geometry()`
5. Calls the spec's `create_strategy` factory (or falls back to default `MultiPickStrategy`) and `strategy.initialize_pairings()`
6. Returns a metadata dict for the mock test harness

### Mock Verification

The mock system monkeypatches the AABB computation functions in `task_verification.py`:

- Maintains a `_mock_aabb_registry` mapping prim paths to synthetic AABBs
- Computes AABBs from `position + PrimGeometry.local_half_extents`
- Updates the registry after each placement (when objects are moved to their targets)
- The same `TaskVerifier` code runs for both mock and real sim

---

## 14. Example Task Walkthrough

### TableTaskColors1 — Color-Matching Pick-and-Place

**Objective:** Pick colored cubes from a bin and place them onto markers of matching colors.

**1. Item Generation**

*Pick items:* 12 cubes in a 4×3 grid inside the pick bin
- Position: `GridPositionGenerator(center=[BIN_X, BIN_Y, z], rows=3, cols=4, spacing=0.08)`
- Color: `RandomChoice(["red", "green", "blue"])` — each cube gets a random color
- Scale: `FixedValue([0.0515, 0.0515, 0.0515])` — uniform cube size

*Target items:* 12 markers in a 3×4 grid on the dropzone
- Position: `GridPositionGenerator(center=[dropzone_center], rows=4, cols=3, spacing_x=-0.15, spacing_y=0.15)`
- Color: `SequentialChoice(["red", "cyan", "yellow", "green", "blue", "magenta"], loop=True)`
- Scale: `FixedValue(target_scale)` — flat marker rectangles

**2. Strategy Creation**

The `create_strategy` factory on `TaskSpec` returns a `ColorMatchStrategy` with `color_palette=["red", "green", "blue", "yellow"]`.

**3. Pairing**

`pair_picks_with_targets()` builds a color-to-targets map:
- `"red" -> [indices of red targets]`
- `"green" -> [indices of green targets]`
- etc.

For each pick cube, it finds the first unused target of the same color. Picks with colors not present in any target (e.g., a blue pick when all blue targets are taken) get `target_index=None` and are skipped.

**4. Execution**

The behavior tree iterates through picks:
- `SelectNextPick` gets the next pick name from the strategy
- `CheckTargetAvailable` verifies the paired target exists
- The 9-phase pick-place sequence executes (approach, grip, lift, move, descend, release, retreat)
- `MarkPickComplete` records the placement

**5. Verification**

After all picks are placed (or targets exhausted):
- `check_groundtruth_task_success()` creates a `TaskVerifier` with `valid_targets_fn` from the strategy (only same-color targets are valid)
- For each red pick, checks `is_on_top(pick, red_target)`
- **Pass:** each pick is on a same-color target, OR all same-color targets are occupied by other picks
- **Fail:** a pick is not on any matching target while matching targets remain available

### TableTaskLayeredCubes — Stacked Pick with Ordering Constraints

**Objective:** Pick 18 cubes from a 3-layer stack and place them on flat markers.

**1. Item Generation**

*Pick items:* 18 cubes in a 2×3 grid, 3 layers high
- Position: `LayeredPositionGenerator(base=GridPositionGenerator(rows=3, cols=2, randomize=False), num_layers=3, layer_height=cube_size)`
- Color: `SequentialChoice(["red"]*6 + ["green"]*6 + ["blue"]*6, loop=False)` — 6 red on bottom, 6 green in middle, 6 blue on top

*Target items:* 18 hidden markers in a flat 6×3 grid on the dropzone.

**2. Strategy with Stacking**

The `create_strategy` factory on `TaskSpec` computes `stacking_map` from pick object positions and returns a `MultiPickStrategy` with stacking constraints:
```python
stacking_map = compute_stacking_map(self._pick_objs)
# Groups objects by XY proximity, maps bottom -> [items above]
```

**3. Pick Ordering**

`_apply_stacking_order()` reorders picks so blue cubes (top layer) are picked first, then green, then red. If a green cube can't be picked because a blue cube is still above it, `_scan_for_available_pick()` wraps around to find another available pick.

---

## Design Principles

1. **Composition over inheritance** — Tasks compose generators and strategies rather than implementing logic directly. A new task is typically 50-130 lines of configuration.

2. **Strategy pattern throughout** — Position generators, attribute strategies, and pairing strategies are all interchangeable via the strategy pattern.

3. **Seed-controlled reproducibility** — All randomization is deterministic given the same seed, enabling reproducible task configurations for debugging and benchmarking.

4. **Dual execution paths** — The same task definition runs both in real Isaac Sim (with physics) and in mock mode (CPU-only, instant feedback), sharing the same verification code.

5. **Extensible verification** — The `TaskVerifier` accepts hook callables rather than requiring subclassing, so strategy-specific verification logic (color matching, box containment) plugs in cleanly.

6. **Lazy imports** — Task `__init__` methods import Isaac Sim utilities lazily to avoid circular dependencies and allow the task discovery system (AST parsing) to work without importing Isaac Sim.
