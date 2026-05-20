# Generation Patterns

## ItemSpec

Every generated item is an `ItemSpec` dataclass:

```python
@dataclass
class ItemSpec:
    asset_type: str
    position: np.ndarray           # [x, y, z]
    orientation: Optional[np.ndarray] = None  # quaternion [w, x, y, z]
    scale: Optional[np.ndarray] = None
    color: Optional[Any] = None    # str name or np.array RGB
    name: Optional[str] = None     # auto-generated if None
    hidden: bool = False           # True for invisible markers
```

## Position Generators

### GridPositionGenerator
Most common. Creates a rectangular grid of positions.

```python
from item_generation import GridPositionGenerator

pos_gen = GridPositionGenerator(
    center=np.array([x, y, z]),   # center of the grid
    rows=4,                        # number of rows (along Y)
    cols=3,                        # number of columns (along X)
    spacing_x=0.08,               # distance between columns
    spacing_y=0.08,               # distance between rows
    randomize=False,               # shuffle slot order if True
)
# Capacity: rows * cols = 12
```

### CircularPositionGenerator
Arranges items in a circle.

```python
from item_generation import CircularPositionGenerator

pos_gen = CircularPositionGenerator(
    center=np.array([x, y, z]) + np.array([0, 0, height]),
    radius=0.15,
    count=8,                       # number of positions
    randomize=False,
)
```

### ConveyorPositionGenerator
Linear spacing along the Y axis (simulates a conveyor belt). Items are centered around `center_y` and spaced symmetrically.

```python
from item_generation import ConveyorPositionGenerator

pos_gen = ConveyorPositionGenerator(
    center_x=0.04,                 # X position for all items
    center_y=0.69,                 # Y center of the line
    z=0.035,                       # Z position for all items
    spacing=0.12,                  # distance between items along Y
    jitter_x=0.01,                 # random X offset (0.0 for none)
    jitter_y=0.01,                 # random Y offset (0.0 for none)
)
```

Note: `ConveyorPositionGenerator` uses `get_positions(count, seed)` (not `generate()`), so it's typically used inside a custom generator class rather than with `ItemGenerator`. See `TableTaskColorShapes` for an example.

### LayeredPositionGenerator
Wraps a base `PositionGenerator` and replicates its positions across multiple Z layers. Fills bottom-up: layer 0 first, then layer 1, etc. Supports partial top layers when count < capacity.

```python
from item_generation import GridPositionGenerator, LayeredPositionGenerator

base_gen = GridPositionGenerator(
    center=np.array([x, y, z]),
    rows=3, cols=2,
    spacing_x=0.08, spacing_y=0.08,
)
pos_gen = LayeredPositionGenerator(
    base_generator=base_gen,       # any PositionGenerator
    num_layers=3,                  # number of Z layers
    layer_height=0.0515,           # Z offset between layers
)
# Capacity: base_capacity * num_layers = 6 * 3 = 18
```

**Stacking constraint**: For stacked items in a real physics sim, picks must be done top-down (can't pull from under a stack). Use `compute_stacking_map()` with the `MultiPickStrategy` to enforce top-down pick ordering:

```python
from multi_pick_strategy import MultiPickStrategy, compute_stacking_map

def _strategy_factory(picks, targets):
    stacking_map = compute_stacking_map(picks)
    return MultiPickStrategy(
        pick_objs=picks, target_objs=targets, stacking_map=stacking_map,
    )

spec = TaskSpec(
    ...
    stacking_enabled=True,   # scene fact: items are pre-stacked
    implementation=TaskImplementationSpec(
        create_strategy=_strategy_factory,
    ),
)
```

`compute_stacking_map(pick_objs, xy_tolerance=0.01)` groups objects by XY proximity, sorts by Z within each column, and maps each item to the items directly above it. The strategy then ensures items with objects above them are deferred until the upper items are picked.

See `TableTaskLayeredCubes` for a simple working example.

### Stacking with Skip-Color Cubes and Permanently Blocked Detection

When stacked source items include "distractor" objects (skip-colors like yellow) that will never be picked, cubes below them can never be accessed in real physics. The `ColorSortStackStrategy` in `flawed_tasks/table_task_sort_and_stack.py` demonstrates how to handle this:

1. **Permanently blocked detection**: Fixed-point analysis marks skip-color cubes as "never completed", then propagates transitively — any cube with a never-completed cube above it is also permanently blocked. These are excluded from pairings.

2. **Dual stacking constraints**: When the picking order (destination-layer order) conflicts with source stacking constraints, override `_is_pick_available()` to check BOTH source readiness (all cubes above completed) AND destination readiness (dynamic stacking target cube completed).

3. **Wrap-around scanning**: The base class `advance_pick_index()` uses `_scan_for_available_pick()` with wrap-around when stacking is active, finding previously-blocked items that are now available behind the cursor.

See the deprecated `TableTaskSortAndStack` in `flawed_tasks/` for the complete pattern.

### Stacking with Distractor Relocation (No Permanently Blocked Cubes)

An alternative to skipping distractor cubes is to **relocate** them to separate stacks elsewhere on the dropzone. This eliminates permanently blocked cubes — all source cubes become reachable. The `ColorSortRelocateStackStrategy` in `tasks/table_task_sort_and_stack.py` demonstrates this:

1. **Per-color stack counts**: Instead of uniform `stacks_per_box`, use a `stacks_per_color` dict allowing different stack counts per color (e.g., 4 for box colors, 6 for relocated colors).

2. **No skip colors**: All colors are sort colors. Former distractors get their own destination stacks on the dropzone floor.

3. **Virtual verification region**: Relocated stacks on the dropzone (no physical box) use a virtual `box_spec` region for containment verification with `match_labels` for color filtering.

4. **Non-overlapping placement**: Position relocated stacks so their X and Y ranges don't overlap with the source grid. From the robot's perspective: right = +X, closer to robot = -Y. See [assets-and-workspace.md](assets-and-workspace.md) § "Robot-Relative Directions".

See `TableTaskSortAndStack` for the complete pattern.

## Attribute Strategies

```python
from item_generation import FixedValue, RandomChoice, SequentialChoice

# Same value for every item:
FixedValue("cube")
FixedValue(np.array([0.05, 0.05, 0.05]))
FixedValue(None)  # use default

# Random selection per item:
RandomChoice(["red", "green", "blue"])

# Cycle through a list:
SequentialChoice(["red", "green", "blue"], loop=True)
```

### Mixed-Type Strategies
For tasks with multiple object types that need different scales/orientations:

```python
from item_generation import MixedScaleStrategy, MixedOrientationStrategy

# Pre-sample types to keep scale/orientation consistent:
sampled_types = ["cube", "soup_can", "madara_bottle", "ball"]

# Primitives get expected_scale; USD assets get identity scale:
MixedScaleStrategy(sampled_types, default_scale=expected_scale)

# USD assets get -90 deg X rotation; primitives get identity:
MixedOrientationStrategy(sampled_types)
```

## ItemGenerator

Combines a position generator with attribute strategies:

```python
from item_generation import ItemGenerator

strategy = ItemGenerator(
    position_generator=pos_gen,
    asset_type_strategy=FixedValue("cube"),
    scale_strategy=FixedValue(expected_scale),
    color_strategy=RandomChoice(["red", "green", "blue"]),
    orientation_strategy=FixedValue(None),      # optional
    hidden_strategy=FixedValue(False),           # optional, True for markers
)

# Generate items:
items = strategy.generate(count_range=None, seed=42)
# count_range: None = use capacity, int = exact count, (min, max) = random range
```

## Custom Generator Pattern

For complex layouts that don't fit ItemGenerator, write a class with a `generate()` method. This is common for box-packing tasks, multi-type tasks, and interleaved pick orders:

```python
from item_generation import ItemSpec, resolve_count

class MyCustomGenerator:
    def __init__(self, ...):
        # Store configuration

    def generate(self, count_range=None, seed=None):
        items = []
        # Build ItemSpec list with custom logic
        items.append(ItemSpec(
            asset_type="cracker_box",
            position=np.array([x, y, z]),
            orientation=some_quaternion,
            name=f"cracker_box_{i}",
        ))
        # Use resolve_count for CLI --pick-count override support
        count = resolve_count(count_range, capacity=len(items), seed=seed)
        if count is not None and count < len(items):
            items = items[:count]
        return items
```

This is used in `TableTaskMixedPacking`, `TableTaskSoupCanPacking`, and `TableTaskConveyorSort` for complex multi-type layouts.

### resolve_count Utility

`resolve_count(count_range, capacity=None, seed=None)` in `item_generation.py` standardizes count resolution:

| `count_range` | Behavior |
|---------------|----------|
| `None` | Returns `capacity` (or `None` if no capacity) |
| `int` | Returns as-is |
| `(min, max)` | Random integer in range (using seed) |
| `(min, None)` | Uses `capacity` as max |

Custom generators should use `resolve_count()` to support CLI `--pick-count` / `--target-count` overrides consistently.

## Virtual Target Generation (for box-packing tasks)

For tasks placing items into boxes or box-like containers, use `virtual_target_generation_strategy` on `TaskImplementationSpec` (NOT the outer `target_generation_strategy`) for the hidden marker objects inside the boxes. Virtual targets are `LightweightObj` instances generated at pairing time by `TaskController` — they are **not spawned as USD prims** in the scene. They live on the implementation spec because they're policy helpers, not real scene objects.

```python
class BoxMarkerGenerator:
    """Generate hidden markers inside each box."""
    def __init__(self, box_specs, z_floor):
        self.box_specs = box_specs
        self.z_floor = z_floor

    def generate(self, count_range=None, seed=None):
        targets = []
        for spec in self.box_specs:
            center = spec["center"]
            # Create a grid of markers inside each box
            for row in range(3):
                for col in range(2):
                    ox = (col - 0.5) * 0.07
                    oy = (row - 1.0) * 0.07
                    targets.append(ItemSpec(
                        asset_type="marker",
                        position=np.array([center[0]+ox, center[1]+oy, self.z_floor]),
                        scale=np.array([0.05, 0.05, 0.001]),
                        hidden=True,
                    ))
        return targets

spec = TaskSpec(
    ...
    pick_generation_strategy=pick_strategy,
    # No target_generation_strategy needed when all targets are virtual
    box_verification_info={"box_specs": box_specs},
    containment_check=True,
    implementation=TaskImplementationSpec(
        virtual_target_generation_strategy=BoxMarkerGenerator(box_specs, box_floor_z),
    ),
)
```

**Key points:**
- Virtual targets get in-memory semantic labels and precomputed geometry automatically
- Use `target_count` in `TaskSpec` to override the number of targets generated (applies to both regular and virtual targets)
- The generator can be an `ItemGenerator`, a custom class with `generate()`, or a plain callable `(pick_objs, scene_target_objs) -> List[ItemSpec]`
- See `TableTaskSoupCanPacking` for a complete working example
