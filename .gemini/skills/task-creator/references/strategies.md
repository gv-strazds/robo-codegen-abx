# Multi-Pick Strategy Reference

The `MultiPickStrategy` class (found in `multi_pick_strategy.py`) defines how objects in the "pick" group are paired with "target" locations. Pass `create_strategy` as a lambda inside `TaskImplementationSpec` (assigned via `implementation=` on the outer `TaskSpec`) to use a non-default strategy.

## Available Strategies

### 1. `MultiPickStrategy` (Default)
Provides sequential 1-to-1 pairing (Pick 0 -> Target 0, Pick 1 -> Target 1, etc.).

**Use Case:** Simple grid-to-grid stacking where order doesn't matter beyond sequence. No `create_strategy` needed — omit the `implementation=` block entirely (or set only `strategy_description` for metadata).

### 2. `ColorMatchStrategy`
Pairs pick objects with target objects that have the same semantic color label.

**Use Case:** "Sort red cubes into red bins."

**Implementation:**
```python
spec = TaskSpec(
    ...
    implementation=TaskImplementationSpec(
        create_strategy=lambda picks, targets: ColorMatchStrategy(
            picks, targets, color_palette=["red", "green", "blue", "yellow"],
        ),
    ),
)
```

### 3. `TypeBasedStrategy`
Pairs objects based on their asset type. Works for arbitrary type keys — not limited to cube/ball.

**Use Case:** Sorting multiple object types into type-specific zones.

**Implementation:**
```python
spec = TaskSpec(
    ...
    implementation=TaskImplementationSpec(
        create_strategy=lambda picks, targets: TypeBasedStrategy(
            picks, targets,
            target_indices_by_type={"cube": [0, 2], "ball": [1]},
            source_types=["cube", "ball", "cube"],
        ),
    ),
)
```

Per-pick type resolution: `source_types` → optional `type_detect_fn(pick_obj)` → default name-prefix match against `target_indices_by_type` keys (longest-first).

### 4. `BottlePickStrategy`
A specialized strategy for placing bottles on their side.

**Use Case:** Handling cylindrical objects that must be laid flat.

**Implementation:**
```python
spec = TaskSpec(
    ...
    implementation=TaskImplementationSpec(
        create_strategy=lambda picks, targets: BottlePickStrategy(picks, targets),
    ),
)
```

## Custom Strategy hooks
Subclasses can override these methods for unique behavior:
- `get_end_effector_orientation(self, pick_name: str)`: Change how the gripper faces for a specific item.
- `get_end_effector_orientation_for_drop(self, pick_name: str, target_name: Optional[str] = None)`: Customize orientation at the drop zone.

Note: `get_end_effector_offset_for_drop()` and `get_placing_info()` are now implemented on `TaskContextBase` using `_prim_geometry`, not on the strategy. The strategy provides `get_placing_target_name(pick_name)` for pure pairing lookup.
