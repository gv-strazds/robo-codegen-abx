# Task Structure Reference

A task in this project is a Python class that inherits from `UR10MultiPickPlaceTask`. It defines the objects to be picked, where they are placed, and the logic for pairing them.

## Key Components

### 1. Class Definition
Inherit from `UR10MultiPickPlaceTask`.

```python
from multi_pickplace_task import UR10MultiPickPlaceTask

class MyNewTask(UR10MultiPickPlaceTask):
    """Description of the task."""
```

### 2. `__init__` Method
The constructor sets up the generation strategies for both pick objects and target objects, bundles them into a `TaskSpec`, and passes it to `super().__init__()`.

#### Lazy Imports
Always import Isaac Sim and project-specific utilities inside `__init__` to avoid initialization order issues.

```python
def __init__(self, task_name="my_task", task_description=None, offset=None, **kwargs):
    from isaacsim.core.utils.stage import get_stage_units
    from item_generation import ItemGenerator, GridPositionGenerator, FixedValue
    from table_setup import setup_two_tables  # requires Isaac Sim
    from env_config_values import BIN_X_COORD, BIN_Y_COORD, CART_SURFACE_CENTER, DROPZONE_X, DROPZONE_Y, DROPZONE_Z, ITEM_SPAWN_REFERENCE_Z  # also re-exported from table_setup
    from task_spec import TaskImplementationSpec, TaskSpec
```

#### Scale Calculation
Calculate object scales relative to stage units (usually 1.0 or 0.01).

```python
stage_units = get_stage_units()
expected_scale = np.array([0.05, 0.05, 0.05]) / stage_units
```

#### Pick Generation Strategy
Defines how objects appear in the source bin.

```python
pick_pos_gen = GridPositionGenerator(
    center=np.array([BIN_X_COORD, BIN_Y_COORD, pick_z]),
    rows=3, cols=2, spacing_x=0.08, spacing_y=0.08
)
pick_strategy = ItemGenerator(
    position_generator=pick_pos_gen,
    asset_type_strategy=FixedValue("cube"),
    scale_strategy=FixedValue(expected_scale),
    color_strategy=None # Random
)
```

#### Target Generation Strategy
Defines where the objects should be placed.

```python
target_pos_gen = GridPositionGenerator(
    center=np.array([DROPZONE_X, DROPZONE_Y, center_grid_z]),
    rows=4, cols=3, spacing_x=-0.15, spacing_y=0.15
)
target_strategy = ItemGenerator(
    position_generator=target_pos_gen,
    asset_type_strategy=FixedValue("disc"),
    color_strategy=FixedValue("red"),
    scale_strategy=FixedValue(expected_scale)
)
```

After defining pick and target strategies, build a `TaskSpec` and pass it to `super().__init__()`. Scene-side fields go on the outer `TaskSpec`; execution-policy fields (strategy factory, BT tree, virtual targets, postures, timeouts, etc.) go inside a nested `TaskImplementationSpec`, assigned via `implementation=`:

```python
    # Simple marker-based tasks (items onto visible markers/rects/discs):
    spec = TaskSpec(
        task_name=task_name,
        task_description=task_description,
        pick_generation_strategy=pick_strategy,
        target_generation_strategy=target_strategy,
        setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
        # Default sequential pairing needs no implementation block.
        # For non-default strategy, wrap inside TaskImplementationSpec:
        # implementation=TaskImplementationSpec(
        #     create_strategy=lambda picks, targets: ColorMatchStrategy(picks, targets, ...),
        # ),
    )
    super().__init__(task_spec=spec, offset=offset, **kwargs)
```

For box-packing tasks (items into box containers), use virtual targets (on the implementation spec — they're policy helpers) and centralized box containment verification (outer TaskSpec — verification semantics):

```python
    spec = TaskSpec(
        task_name=task_name,
        task_description=task_description,
        pick_generation_strategy=pick_strategy,
        setup_workspace=_workspace_setup,
        box_verification_info={"box_specs": box_specs},  # centralized verification (scene-side)
        containment_check=True,
        # Optional: placement_constraints_fn=_check_verticality,
        implementation=TaskImplementationSpec(
            virtual_target_generation_strategy=target_strategy,  # hidden markers (LightweightObj)
        ),
    )
    super().__init__(task_spec=spec, offset=offset, **kwargs)
```

Each box spec dict needs: `name`, `center_xy`, `floor_z`, `inner_size`, `height`, and optionally `match_labels` (e.g., `{"color": "red"}`). The base class automatically handles box containment verification via `build_box_verification_hooks()`. See `TableTaskSoupCanPacking` for a complete working example.

### 3. Workspace Setup
Workspace setup is provided as a lambda in the `TaskSpec` `setup_workspace` field. Most tasks use `setup_two_tables`.

### 4. Pairing Strategy (Optional)
For non-sequential pairing, set `create_strategy` inside `TaskImplementationSpec`. See `strategies.md` for options. v2 / cortex / cuRobo subclasses override `_customize_spec(spec)` and use `spec.with_impl(tree_factory=..., ...)` to swap policy fields without touching the description side.
