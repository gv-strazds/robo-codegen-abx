# Pairing Strategies

All strategies live in `multi_pick_strategy.py`. Pass `create_strategy` as a lambda inside `TaskImplementationSpec` (assigned via `implementation=` on the outer `TaskSpec`) to use a non-default strategy.

## MultiPickStrategy (Default)

Sequential 1:1 pairing: pick[0] -> target[0], pick[1] -> target[1], etc.

No `create_strategy` needed — omit the `implementation=` block entirely (or set only `strategy_description` for metadata). Used by most simple tasks (TableTask3, TableTaskCrackerBoxes1, TableTaskSoupCans1).

## ColorMatchStrategy

Pairs pick objects to targets that share the same semantic color label.

```python
spec = TaskSpec(
    ...
    implementation=TaskImplementationSpec(
        create_strategy=lambda picks, targets: ColorMatchStrategy(
            picks, targets, color_palette=["red", "green", "blue"],
        ),
    ),
)
```

**Requirements**:
- Pick items must have color labels (via `color_strategy` or `_apply_semantic_labels`)
- Target items must have matching color labels
- `color_palette` lists the colors to match on

**Used by**: TableTaskColors1, TableTaskColorShapes, TableTaskColorBinSort

## TypeBasedStrategy

Routes picks to per-type target groups. Works for arbitrary asset types (not just cube/ball) — supply a `target_indices_by_type` dict mapping each type key to the target indices reserved for it.

```python
# Dynamic source_types from a custom generator that stores them:
def _create_strategy(picks, targets):
    source_types = pick_strategy.source_types[:len(picks)]
    n_cubes = sum(1 for t in source_types if t == "cube")
    n_balls = sum(1 for t in source_types if t == "ball")
    cube_markers = target_strategy.markers_per_box[0]
    ball_markers = target_strategy.markers_per_box[1]
    cube_indices = list(range(0, min(n_cubes, cube_markers)))
    ball_start = cube_markers
    ball_indices = list(range(ball_start, ball_start + min(n_balls, ball_markers)))
    return TypeBasedStrategy(
        picks, targets,
        target_indices_by_type={"cube": cube_indices, "ball": ball_indices},
        source_types=source_types,
    )

spec = TaskSpec(
    ...
    implementation=TaskImplementationSpec(
        create_strategy=_create_strategy,
    ),
)
```

**Constructor**:

```python
TypeBasedStrategy(
    pick_objs, target_objs,
    target_indices_by_type: Dict[str, List[int]],   # required
    source_types: Optional[List[str]] = None,       # explicit per-pick type
    type_detect_fn: Optional[Callable] = None,      # (pick_obj) -> Optional[str]
)
```

Per-pick type resolution order: `source_types` → `type_detect_fn` → default name-prefix match against `target_indices_by_type` keys (longest-first, so `"cracker_box"` wins over `"box"`).

**Requirements**:
- `target_indices_by_type` maps each type string to the list of target indices reserved for it. Keys are arbitrary — use semantic type names like `"cube"`, `"ball"`, `"cracker_box"`, `"sugar_box"`.
- If picks are named `"<type>_<i>"` (e.g., `"cracker_box_3"`) the default name-prefix detection works and `source_types` can be omitted.
- When target counts are variable, compute the per-type index lists dynamically from the target generator's output.

**Used by**: TableTaskConveyorSort, TableTaskShapeSortBoxes, TableTaskConveyorTypeSort

## BottlePickStrategy

Specialized for bottles. Changes the end-effector orientation between pick and place so the robot picks bottles on their side and places them upright into pads.

```python
spec = TaskSpec(
    ...
    implementation=TaskImplementationSpec(
        create_strategy=lambda picks, targets: BottlePickStrategy(picks, targets),
    ),
)
```

**Key behaviors**:
- `get_end_effector_orientation_for_drop()` returns a different orientation than pick
- In teleport/mock mode, placed bottles get the target's orientation (upright)
- Typically paired with `is_within` + `is_vertical` verification

**Used by**: TableTaskBottles1

## SingleStackStrategy

Places all picks into a single growing stack at one target location. Respects source-side stacking constraints (from `stacking_map`) to determine pick order — topmost items are picked first. Items are placed in pick order: first picked = bottom of destination stack.

```python
# With bin_geometry (convenience shorthand for bin targets):
spec = TaskSpec(
    ...
    stacking_enabled=True,
    implementation=TaskImplementationSpec(
        create_strategy=lambda picks, targets: SingleStackStrategy(
            pick_objs=picks, target_objs=targets,
            stacking_map=compute_stacking_map(picks),
            bin_geometry=bin_geometry,
        ),
    ),
)

# With custom base_check_fn (for non-bin targets, e.g. table surface):
spec = TaskSpec(
    ...
    stacking_enabled=True,
    implementation=TaskImplementationSpec(
        create_strategy=lambda picks, targets: SingleStackStrategy(
            pick_objs=picks, target_objs=targets,
            stacking_map=compute_stacking_map(picks),
            base_check_fn=my_custom_check,  # (pick_obj, target_obj, bb_cache, obj_scale) -> bool
        ),
    ),
)
```

**Requirements**:
- `target_objs` should contain a single marker at the base of the stack
- `stacking_map` from `compute_stacking_map(picks)` if picks are stacked at source
- One of:
  - `bin_geometry` dict with `center_xy`, `inner_size`, `floor_z`, `height`, and optionally `z_tol` — convenience shorthand for bin containment. For KLT bin targets, `floor_z` must be the settled value (~0.063), not the spawn value (see note below).
  - `base_check_fn` — custom callable for bottom-layer spatial verification. Takes priority over `bin_geometry` when both provided.

**Key behaviors**:
- Automatically extends target list: each stacked item targets the previously placed item
- `valid_targets_for_pick()` returns only the assigned target per pick (prevents false matches in occupancy check)
- `get_spatial_check_fn()` returns: `base_check_fn` for base marker target, position-based XY proximity + Z ordering for stacked items
- `get_recommended_ee_height()` dynamically computes transport height to clear the current stack top

**Used by**: TableTaskLayeredCircle

## LayeredStackStrategy

Builds multiple stacks with property-based layer ordering. Objects are classified by a property (e.g., color) and arranged into stacks where each layer corresponds to a property value.

```python
# With bin_geometry (convenience shorthand for bin targets):
spec = TaskSpec(
    ...
    stacking_enabled=True,
    implementation=TaskImplementationSpec(
        create_strategy=lambda picks, targets: LayeredStackStrategy(
            pick_objs=picks, target_objs=targets,
            layer_order=["blue", "green", "red"],
            max_stacks=3,
            classify_fn=my_classifier,  # obj -> Optional[str]
            bin_geometry=bin_geometry,
        ),
    ),
)

# With custom base_check_fn (for non-bin targets):
spec = TaskSpec(
    ...
    stacking_enabled=True,
    implementation=TaskImplementationSpec(
        create_strategy=lambda picks, targets: LayeredStackStrategy(
            pick_objs=picks, target_objs=targets,
            layer_order=["blue", "green", "red"],
            max_stacks=3,
            classify_fn=my_classifier,
            base_check_fn=my_custom_check,  # (pick_obj, target_obj, bb_cache, obj_scale) -> bool
        ),
    ),
)
```

**Requirements**:
- `target_objs` should contain bottom-layer markers (one per stack position)
- `layer_order` lists property values from bottom to top
- `classify_fn` maps objects to property values (defaults to color-based)
- `max_stacks` caps the number of parallel stacks
- One of:
  - `bin_geometry` for bottom-layer containment checks (same as SingleStackStrategy)
  - `base_check_fn` — custom callable for bottom-layer spatial verification. Takes priority over `bin_geometry` when both provided.

**Key behaviors**:
- Automatically computes how many complete stacks can be formed from available objects
- Upper layers target objects from the layer below (dynamically added)
- Excess/unclassifiable objects are skipped (paired with None)
- `valid_targets_for_pick()` returns only targets matching the pick's layer
- `get_spatial_check_fn()` returns: `base_check_fn` for bottom layer, `is_on_top` for upper layers

**Used by**: TableTaskConveyorColorStacks (via ColorStackStrategy subclass)

## ColorSortStackBase (Shared base)

Defined in `multi_pick_strategy.py`. Provides the shared logic for any color-sorted dynamic-stacking strategy: pick classification by color (`_classify_pick`), layer-completeness gating (`_is_pick_available`), dynamic-target readiness checks, and the combined spatial check that uses a base-marker check on bottom-layer placements and XY-proximity-plus-Z-ordering on stacked cubes (`get_spatial_check_fn`). It does not implement `pair_picks_with_targets`, `initialize_pairings`, or `get_recommended_ee_height` — subclasses provide those, since the per-stack-count layout differs.

When writing a new color-sort-stack variant, subclass `ColorSortStackBase` directly — do not subclass either of the concrete strategies below.

## ColorSortStackStrategy (Custom, deprecated)

Combines color-based routing with dynamic stacking into multiple boxes. Sorts picks by color into separate boxes, building round-robin stacks within each box. Supports source stacking constraints (pick from stacks), destination stacking constraints (place on previously placed cubes), and permanently blocked cube detection (cubes under skip-color distractors).

Defined in `flawed_tasks/table_task_sort_and_stack.py`. Inherits from `ColorSortStackBase` and adds uniform `stacks_per_box` plus `_find_permanently_blocked`.

```python
from multi_pick_strategy import compute_stacking_map

def _strategy_factory(picks, targets):
    stacking_map = compute_stacking_map(picks)
    return ColorSortStackStrategy(
        pick_objs=picks,
        target_objs=targets,
        sort_colors=["red", "green", "blue"],
        stacks_per_box=4,
        skip_colors=["yellow"],
        base_check_fn=combined_box_check,  # spatial check for base markers
        stacking_map=stacking_map,
    )

spec = TaskSpec(
    ...
    stacking_enabled=True,
    implementation=TaskImplementationSpec(
        create_strategy=_strategy_factory,
    ),
)
```

**Requirements**:
- `target_objs` should contain bottom-layer markers (N per box, ordered by sort_color then stack position)
- `sort_colors` lists colors to route to matching boxes
- `stacks_per_box` is the number of stack positions per box (e.g., 4 for 2x2)
- `skip_colors` lists colors to ignore (distractors)
- `base_check_fn` for bottom-layer spatial verification (typically `build_bin_geometry_check` per box)
- `stacking_map` from `compute_stacking_map(picks)` for source stacking constraints

**Key behaviors**:
- `pair_picks_with_targets()`: classifies picks by color, assigns round-robin to stack positions, uses `_extend_target_objs` for upper layers (each placed cube becomes the next target)
- `_find_permanently_blocked()`: fixed-point analysis identifies sort-color cubes transitively blocked by skip-color cubes above — these are excluded from pairings entirely
- `_is_pick_available()`: checks both source constraints (cubes above must be completed) AND destination constraints (dynamic stacking target must be completed before placing on it)
- `_reassign_targets_by_picking_order()`: no-op override to preserve color-based target assignments
- `get_recommended_ee_height()`: dynamically computes transport height to clear growing stacks
- `get_spatial_check_fn()`: base check for bottom markers, position-based XY proximity + Z ordering for stacked cubes
- Task stops when no more picks satisfy both source and destination stacking constraints

**Used by**: deprecated `TableTaskSortAndStack` (in `flawed_tasks/`)

## ColorSortRelocateStackStrategy (Custom)

Sibling of `ColorSortStackStrategy` — both inherit from `ColorSortStackBase`. Adds per-color stack counts (instead of a uniform `stacks_per_box`) and forces `skip_colors=[]` so every color is sorted. Used when all colors (including former "distractors") should be routed to their own destination stacks, potentially with different numbers of stacks per color.

Defined in `tasks/table_task_sort_and_stack.py`.

```python
from multi_pick_strategy import compute_stacking_map
from tasks.table_task_sort_and_stack import ColorSortRelocateStackStrategy

def _strategy_factory(picks, targets):
    stacking_map = compute_stacking_map(picks)
    return ColorSortRelocateStackStrategy(
        pick_objs=picks,
        target_objs=targets,
        sort_colors=["red", "green", "blue", "yellow"],
        stacks_per_color={"red": 4, "green": 4, "blue": 4, "yellow": 6},
        base_check_fn=combined_region_check,
        stacking_map=stacking_map,
    )

spec = TaskSpec(
    ...
    stacking_enabled=True,
    implementation=TaskImplementationSpec(
        create_strategy=_strategy_factory,
    ),
)
```

**Key differences from ColorSortStackStrategy**:
- `stacks_per_color` (dict) replaces uniform `stacks_per_box` — each color can have a different number of stacks
- `skip_colors` is always empty — all colors are sortable, no cubes are permanently blocked
- Target indices use cumulative offsets per color (e.g., red=0..3, green=4..7, blue=8..11, yellow=12..17)
- Base markers for different colors can be in different locations (e.g., R/G/B in boxes on cart, yellow on dropzone floor)

**Virtual verification regions**: When some color groups are placed outside physical boxes (e.g., yellow stacks on the dropzone floor), create a virtual box_spec region for verification:
```python
yellow_region_spec = {
    "name": "yellow_stacks_region",
    "center_xy": np.array([center_x, center_y]),
    "floor_z": DROPZONE_Z + 0.001,
    "inner_size": np.array([width, depth]),  # must cover all marker positions + margins
    "height": 0.50,  # generous for growing stacks
    "match_labels": {"color": "yellow"},
    "z_tol": 0.03,
}
all_verification_specs = box_specs + [yellow_region_spec]
```

**Used by**: TableTaskSortAndStack

### Stacking Strategy Base-Layer Verification

Both stacking strategies support two mechanisms for bottom-layer spatial checks:

1. **`base_check_fn`** (general) — any callable with signature `(pick_obj, target_obj=None, bb_cache=None, obj_scale=None) -> bool`. Use this for non-bin targets (table surfaces, custom containers, etc.). Example:
   ```python
   from task_verification import is_on_top
   base_check_fn=lambda pick, tgt, **kw: is_on_top(pick, tgt, **kw)
   ```

2. **`bin_geometry`** (convenience shorthand) — a dict converted internally to a `base_check_fn` via `build_bin_geometry_check()`. Uses `is_within_box_geometry()` with no runtime settling adjustment, so `floor_z` must already reflect the settled position. For the KLT bin on the cart:
   ```python
   bin_floor_z = 0.0573 + 0.005  # cart surface + small lift (approximate settled value)
   bin_geometry = {
       "center_xy": np.array([BIN_X_COORD, BIN_Y_COORD]),
       "inner_size": np.array(BIN_SIZE[:2]),
       "floor_z": bin_floor_z,
       "height": 0.15,   # generous wall height for containment check
       "z_tol": 0.03,    # generous Z tolerance for physics settling
   }
   ```

When both are provided, `base_check_fn` takes priority. The `build_bin_geometry_check()` helper is also available as a public function for tasks that want to build a bin check fn and compose it with other checks.

This differs from the `box_verification_info` path (used by non-stacking tasks like TableTaskMixedCircle) where `prim_path` + `spawn_position` enable automatic runtime adjustment.

## Custom Strategy Hooks

Any strategy subclass can override these methods:

- `get_end_effector_orientation(pick_name)` — EE orientation for picking (default: gripper down)
- `get_end_effector_orientation_for_drop(pick_name, target_name)` — EE orientation for dropping (default: None = same as pick). Return non-None to change the item's orientation at placement.
- `valid_targets_for_pick(pick_index)` — Which targets are valid for a given pick (default: all)
- `placement_constraints_satisfied(pick_index, target_index)` — Extra constraints beyond spatial check

Note: `get_end_effector_offset_for_drop()` and `get_placing_info()` are now implemented on `TaskContextBase` using `_prim_geometry`, not on the strategy. The strategy provides `get_placing_target_name(pick_name)` for pure pairing lookup.
