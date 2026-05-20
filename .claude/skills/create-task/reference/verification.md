# Verification Patterns

Task success is checked via verification hooks configured in the `TaskSpec`. The preferred approach is to use `TaskSpec` fields (`spatial_check_fn`, `placement_constraints_fn`, `box_verification_info`) rather than overriding methods on the task class.

## Default: is_on_top

The base class uses `is_on_top` — checks that the pick object's AABB is resting on the target's AABB with XY overlap. No configuration needed for this default.

Suitable for: items placed on flat markers, rects, or discs.

## Custom Spatial Check: spatial_check_fn (Preferred)

Use `spatial_check_fn` in the `TaskSpec` to replace the default `is_on_top` check:

```python
from task_verification import is_on_top, is_vertical, is_within

# Items must be on top AND upright:
def _spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None):
    return (
        is_on_top(pick_obj, target_obj, bb_cache=bb_cache, obj_scale=obj_scale)
        and is_vertical(pick_obj, bb_cache=bb_cache, obj_scale=obj_scale)
    )

spec = TaskSpec(
    ...
    spatial_check_fn=_spatial_check,
)
```

```python
# Items must be inside a container:
def _spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None):
    return is_within(pick_obj, target_obj, bb_cache, obj_scale)

spec = TaskSpec(
    ...
    spatial_check_fn=_spatial_check,
)
```

### Per-Type Checks (Mixed Tasks)

For tasks with mixed item types where only some need orientation checks:

```python
from task_verification import is_on_top, is_vertical

_VERTICAL_ASSET_TYPES = {"madara_bottle", "cracker_box", "soup_can", "mustard_bottle"}

def _spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None):
    if not is_on_top(pick_obj, target_obj, bb_cache=bb_cache, obj_scale=obj_scale):
        return False
    if any(pick_obj.name.startswith(t) for t in _VERTICAL_ASSET_TYPES):
        return is_vertical(pick_obj, bb_cache=bb_cache, obj_scale=obj_scale)
    return True

spec = TaskSpec(
    ...
    spatial_check_fn=_spatial_check,
)
```

Used by: TableTaskMixedPacking.

Note: For containment tasks (items into bins/boxes), prefer the `box_verification_info` pattern below with a separate `placement_constraints_fn` for orientation checks, rather than embedding `is_vertical` into `spatial_check_fn`. See TableTaskMixedCircle for an example.

## Orientation Checks: is_vertical

For items that must be placed upright (their tallest axis aligned with world Z). Use via `spatial_check_fn` (see above) or `placement_constraints_fn` (see below for box containment). Note: if items to be picked are originally not vertical, 
by default they will also not be vertical when placed somewhere else. In such cases, check is_vertical only if the user mentions vertical placement or if the task implementation explicitly reorients the item while moving it.

## Which Items Might Need is_vertical

Elongated USD assets that have a clear "upright" orientation:
- `madara_bottle` — tall bottle
- `soup_can` — cylinder taller than wide
- `cracker_box` — box taller than wide
- `mustard_bottle` — bottle shape
- `sugar_box` — box taller than wide

Primitives and symmetric objects generally do NOT need vertical checks, unless asymmetrically scaled:
- `cube`, `ball` — symmetric, no meaningful orientation (asymmetric scaling can elongate an axis)
- `cone` `cylinder`, `disc` — depends on task intent: These shapes have one axis that is conceptually their vertical (z) axis

## Available Verification Functions

All in `task_verification.py`:

| Function | Checks |
|----------|--------|
| `is_on_top(a, b, ...)` | A rests on top surface of B (XY overlap + Z proximity) |
| `is_within(a, b, ...)` | A is inside container B (XY overlap + Z inside bounds) |
| `is_vertical(obj, ...)` | Object's Z extent >= max horizontal extent (within tolerance) |
| `is_horizontal(obj, ...)` | Object's max horizontal extent >= Z extent (within tolerance) |

## Box Containment Verification (Centralized)

When multiple items are placed into the same box (or bin), the default 1:1 marker-based verification doesn't work well — physics can shift items slightly off their assigned marker within the box, causing false failures. The system provides **centralized box containment verification** via `TaskSpec` fields.

**This is the default approach when the task places items into boxes or box-like containers (bins, crates, etc.).**

### Pattern: TaskSpec Configuration (Recommended)

Set `box_verification_info` and `containment_check` in the TaskSpec. **No need to override `check_groundtruth_task_success()`** — the base class handles everything automatically via `build_box_verification_hooks()`.

```python
box_specs = [
    {
        "name": "red_collection_box",
        "center_xy": np.array([x, y]),
        "floor_z": box_floor_z,
        "inner_size": box_inner_size,   # np.array([width, depth])
        "height": box_height,
        "match_labels": {"color": "red"},  # optional: restrict which picks can go here
    },
    # ... more boxes
]

spec = TaskSpec(
    ...
    box_verification_info={"box_specs": box_specs},
    containment_check=True,
    implementation=TaskImplementationSpec(
        virtual_target_generation_strategy=target_strategy,  # hidden markers in boxes
    ),
)
```

### Box Spec Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Box identifier (e.g., `"red_collection_box"`) |
| `center_xy` | np.ndarray | `[x, y]` center of box interior |
| `floor_z` | float | Z coordinate of box floor surface |
| `inner_size` | np.ndarray | `[width, depth]` inner dimensions |
| `height` | float | Wall height |
| `match_labels` | dict (optional) | Label matching criteria. Can match any semantic label key: `{"color": "red"}` for color routing, `{"type": "cube"}` for type routing. |
| `z_tol` | float (optional) | Lower-bound Z tolerance: object bottom vs box floor (default 0.02). Use 0.03 for round objects or approximate floor estimates. |
| `z_tol_top` | float (optional) | Upper-bound Z tolerance: object bottom vs box wall top (default 0.01). Allows objects piled slightly above the walls. |
| `prim_path` | str (optional) | USD prim path of the container (e.g., `"/KLT_Bin"`). Enables physics-settling adjustment. |
| `spawn_position` | np.ndarray (optional) | `[x, y, z]` original spawn position of the container prim. Used with `prim_path` to compute settling delta. |

### Physics Settling Adjustment (KLT Bin / pick_bin)

Containers that are spawned slightly above a surface (e.g., the KLT bin spawned above the cart) settle downward during physics simulation. The static `floor_z` and `center_xy` in the box spec then become stale, causing false containment failures.

To handle this, include `prim_path` and `spawn_position` in the box spec. At verification time, the base class queries the prim's current world position, computes the delta from `spawn_position`, and adjusts `floor_z` and `center_xy` automatically.

**When to use:** Always include `prim_path` + `spawn_position` when items are placed into the **KLT pick bin** (or any other container that is spawned as a physics-enabled prim that may settle). Custom boxes created via `spawn_open_box()` use `FixedCuboid` parts that don't move, so they do NOT need this.

**KLT Bin example** (items placed into the pick bin on the cart):

```python
from table_setup import BIN_X_COORD, BIN_Y_COORD, BIN_SIZE, ITEM_SPAWN_REFERENCE_Z

bin_spawn_pos = np.array([BIN_X_COORD, BIN_Y_COORD, ITEM_SPAWN_REFERENCE_Z + 0.05])
bin_floor_z = 0.0573 + 0.005  # cart surface + small lift (approximate)
box_specs = [
    {
        "name": "pick_bin",
        "center_xy": np.array([BIN_X_COORD, BIN_Y_COORD]),
        "inner_size": np.array(BIN_SIZE[:2]),
        "floor_z": bin_floor_z,
        "height": 0.15,
        "z_tol": 0.03,
        "prim_path": "/KLT_Bin",
        "spawn_position": bin_spawn_pos,
    },
]
```

See `TableTaskMixedCircle` for a complete reference.

### Strategy-Level Base-Layer Verification (Stacking Tasks)

Stacking strategies (`SingleStackStrategy`, `LayeredStackStrategy`) accept either `base_check_fn` (general mechanism) or `bin_geometry` (convenience shorthand) passed directly to the strategy constructor — NOT via `box_verification_info`. When `bin_geometry` is provided, it is internally converted to a check fn via `build_bin_geometry_check()` with **no runtime settling adjustment**, so `floor_z` must be pre-computed to the settled value. `base_check_fn` takes priority when both are provided.

Use `box_verification_info` for non-stacking containment tasks. Use strategy-level `base_check_fn` or `bin_geometry` for stacking tasks. See [strategies.md](strategies.md) § "Stacking Strategy Base-Layer Verification" for details.

### How it works

`build_box_verification_hooks(box_specs, pick_objs, strategy)` in `task_verification.py` creates:

1. **`box_targets`** — lightweight namedtuple targets (one per box, only `.name` is needed for diagnostics)
2. **`spatial_check_fn`** — closure using `is_within_box_geometry()` with per-box geometry
3. **`valid_targets_fn`** — closure checking `match_labels` via `has_label()`, excluding unpaired overflow picks

The base class `_check_incremental()` and `check_groundtruth_task_success()` automatically invoke this when `box_verification_info` is present.

### Key elements

- **`containment_check=True`**: Implies `allow_multi_occupancy=True` and uses containment-specific failure messages.
- **`match_labels`**: Optional per-box label filter. Without it, all paired picks can go in any box. With `{"color": "red"}`, only picks with a `"red"` color label are valid for that box.
- **Unpaired pick handling**: `build_box_verification_hooks` checks `strategy._pairings` and returns `[]` (no valid targets) for overflow picks that were never assigned a target, so they aren't counted as failures.
- **Virtual targets**: Use `virtual_target_generation_strategy` on `TaskImplementationSpec` for the hidden markers inside boxes (recommended over `target_generation_strategy` for box-packing tasks — they're policy helpers, not scene objects).

### When to add `is_vertical` to box containment

For items that must be upright inside the box, add a `placement_constraints_fn` to the `TaskSpec`:

```python
def _check_verticality(pick_index, target_index):
    pick_obj = task_ref._pick_objs[pick_index]
    if not is_vertical(pick_obj, bb_cache=task_ref._get_bb_cache()):
        return (False, "item is not vertical (upright orientation required)")
    return (True, "")

spec = TaskSpec(
    ...
    placement_constraints_fn=_check_verticality,
    box_verification_info={"box_specs": box_specs},
    containment_check=True,
)
```

Used by: TableTaskSoupCanPacking (soup cans must be upright in boxes).

### Reference tasks

- **`TableTaskMixedCircle`** — single bin container + selective verticality constraints (USD assets only)
- **`TableTaskSoupCanPacking`** — virtual targets + multi-box containment + verticality constraints
- **`TableTaskColorShapes`** — scene targets + box containment with `match_labels` color matching
- **`TableTaskColorBinSort`** — box containment with color-matched sorting bins
- **`TableTaskConveyorSort`** — virtual targets + type-based `match_labels` routing (cubes/balls into separate boxes)

## TaskVerifier

The verification framework is automatic — you only need to configure `TaskSpec` fields. The base class `check_groundtruth_task_success()` creates a `TaskVerifier` that:
1. Checks each pick against its valid targets via the spatial check function
2. Applies placement constraints (from strategy or TaskSpec)
3. Reports structured pass/fail results

### Multi-occupancy / containment mode

When `containment_check=True` or `allow_multi_occupancy=True`:
- Spatial check is called directly per pick per valid target (no occupancy map)
- Targets are never "full" — multiple picks can match the same target
- `is_within_box_geometry()` logs DEBUG on failure automatically — no `log_failure` parameter needed
- Spatial check functions should use the standard signature: `(pick_obj, target_obj, bb_cache=None, obj_scale=None)`
