# Implementation Guide

Detailed patterns for implementing a task class. Read the relevant sections based on your task requirements.

## Table of Contents

1. [Position Generators](#position-generators)
2. [Attribute Strategies](#attribute-strategies)
3. [Pairing Strategies](#pairing-strategies)
4. [Virtual Targets for Box-Packing Tasks](#virtual-targets-for-box-packing-tasks)
5. [USD Asset Orientation](#usd-asset-orientation)
6. [Workspace Setup](#workspace-setup)
7. [Verification (spatial_check_fn)](#verification)
8. [TaskSpec Metadata and Rationale](#taskspec-metadata-and-rationale)

---

## Position Generators

Reference: [generation-patterns.md](generation-patterns.md) for full API details including `GridPositionGenerator`, `CircularPositionGenerator`, `ConveyorPositionGenerator`, `LayeredPositionGenerator`, and custom generators.

## Attribute Strategies

- `FixedValue(value)` — same for all items
- `RandomChoice(options)` — random from list
- `SequentialChoice(options, loop=True)` — cycles through list
- `MixedScaleStrategy(types, default_scale)` — different scale per asset type
- `MixedOrientationStrategy(types)` — applies -90 deg X rotation to USD assets (boxes, bottles, cans)

## Pairing Strategies

Reference: [strategies.md](strategies.md) for full details on each strategy class.

Pass `create_strategy` as a lambda inside `TaskImplementationSpec` (only when a non-default strategy is needed):
- **Sequential** (default): pick[i] -> target[i]. Omit `create_strategy` (or omit `implementation=` entirely).
- **ColorMatchStrategy**: pairs by semantic color label.
- **TypeBasedStrategy**: routes different object types to different target groups.
- **BottlePickStrategy**: specialized for bottles placed upright into pads.
- **SingleStackStrategy**: all picks into one growing stack. Set `stacking_enabled=True` (scene-side, on the outer TaskSpec — it's a scene fact, not a policy choice).
- **LayeredStackStrategy**: multiple stacks with property-based layer ordering. Set `stacking_enabled=True`.
- **ColorSortStackBase** (in `multi_pick_strategy.py`): shared base for color-sorted dynamic stacking. Provides classification, layer-completeness gating, dynamic-target readiness, and combined spatial verification. Subclass it for new color-sort-stack variants — do NOT subclass either of the concrete strategies below.
- **ColorSortStackStrategy** (custom, in `flawed_tasks/table_task_sort_and_stack.py`, deprecated): adds uniform `stacks_per_box` and `_find_permanently_blocked` for skip-color distractor handling. Used only by the retired flawed task. Set `stacking_enabled=True`.
- **ColorSortRelocateStackStrategy** (custom, in `tasks/table_task_sort_and_stack.py`): adds per-color stack counts and no skip colors — all colors are sorted to their own stacks. Set `stacking_enabled=True`.

Example:
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

## Virtual Targets for Box-Packing Tasks

For tasks placing items into boxes or box-like containers, use `virtual_target_generation_strategy` on the `TaskImplementationSpec` (NOT the outer `target_generation_strategy`) for the hidden marker objects inside the boxes. Virtual targets are `LightweightObj` instances generated at pairing time by `TaskController` (not during scene setup) — they're policy helpers, not real scene objects. They get in-memory semantic labels and precomputed geometry automatically. This is the **recommended default** for all box-packing tasks. See [generation-patterns.md](generation-patterns.md) and `TableTaskSoupCanPacking` for the pattern.

## USD Asset Orientation

USD assets (soup_can, cracker_box, mustard_bottle, madara_bottle, sugar_box) have their tall axis along the local Y axis, NOT Z. They need a **-90 deg X rotation** as their spawn orientation so they appear upright in the world:
```python
from isaacsim.core.utils import rotations
from pxr import Gf
default_orientation = rotations.gf_rotation_to_np_array(
    Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)
)
```

## Workspace Setup

Most tasks use the standard two-table workspace via the `TaskSpec`:
```python
spec = TaskSpec(
    ...
    setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
)
```
Pass `standard_objs=False` and/or `add_bin=False` to `setup_two_tables` if spawning custom workspace objects.

For tasks with custom boxes, see [assets-and-workspace.md](assets-and-workspace.md) for `spawn_open_box` details.

## Verification

Reference: [verification.md](verification.md) for full verification patterns.

**For tasks placing items into boxes/bins/containers** (the common case for container tasks):
Set `box_verification_info={"box_specs": box_specs}` and `containment_check=True` in the `TaskSpec`. The base class automatically uses `build_box_verification_hooks()` — no need to override `check_groundtruth_task_success()`.

**Important**: When the destination is the **KLT pick bin** (or any physics-enabled container that may settle after spawning), include `prim_path` and `spawn_position` in each box spec to enable automatic physics-settling adjustment at verification time. See [verification.md](verification.md) § "Physics Settling Adjustment" for details and the KLT Bin example.

**For simpler tasks** (items onto markers/rects/pads) with custom spatial checks:
Use `spatial_check_fn` in the `TaskSpec` to replace the default `is_on_top` check:
```python
from task_verification import is_on_top, is_vertical

def _custom_spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None):
    return (
        is_on_top(pick_obj, target_obj, bb_cache=bb_cache, obj_scale=obj_scale)
        and is_vertical(pick_obj, bb_cache=bb_cache, obj_scale=obj_scale)
    )

spec = TaskSpec(
    ...
    spatial_check_fn=_custom_spatial_check,
)
```

## TaskSpec Metadata and Rationale

The `TaskSpec` includes human-readable metadata fields that document the task design. Scene-side metadata (`scenario`, `pick_description`, `target_description`, `verification_description`, scene-side `rationale`) lives on the outer `TaskSpec`. Implementation-side metadata (`strategy_description` and impl-side `rationale`) lives on the nested `TaskImplementationSpec`:

```python
from task_spec import TaskImplementationSpec, TaskSpec

spec = TaskSpec(
    task_name=task_name,
    task_description="Pick cubes from the bin and place onto markers.",
    pick_generation_strategy=pick_strategy,
    target_generation_strategy=target_strategy,
    setup_workspace=lambda scene, assets_root: setup_two_tables(scene, assets_root),
    # Scene-side metadata
    scenario={
        "source": "bin",               # "bin" | "conveyor" | "cart" | "dropzone"
        "destination": "dropzone_grid", # "dropzone_grid" | "dropzone_circle" | "boxes_on_cart" | etc.
        "workspace": "two_tables",      # "two_tables" | "two_tables_custom_boxes"
    },
    pick_description={
        "asset_types": ["cube"],
        "count": 6,                     # int, "random(2,16)", or "3 cubes + 3 balls"
        "arrangement": "3x2 grid in pick bin",
        "colors": "RandomChoice(['red', 'green', 'blue'])",  # or "USD default", "green (fixed)"
    },
    target_description={
        "type": "visible_markers",      # "visible_markers" | "hidden_markers" | "carrier_pads"
        "arrangement": "3x4 grid on dropzone",
        "count": 12,
        # Include containers dict for box-packing tasks:
        # "containers": {"count": 4, "layout": "2x2 grid on cart", "capacity_per_box": 6},
    },
    # Only include verification_description for non-default verification:
    # verification_description={
    #     "spatial_check": "is_within + is_vertical",
    #     "placement_constraints": "is_vertical (for USD asset types)",
    #     "containment_check": True,
    # },
    rationale={
        # Scene-side rationale (generators, constraints, verification).
        # Examples:
        # "pick_generation_strategy"             — why this generator layout
        # "spatial_check_fn"                     — why custom spatial check is needed
        # "placement_constraints_fn"             — why placement constraints are needed
        # "containment_check"                    — why containment check is enabled
        # "stacking_enabled"                     — why stacking constraints are needed
        # "pick_count" / "target_count"          — why these counts (if non-obvious)
    },
    implementation=TaskImplementationSpec(
        # Default sequential pairing needs no create_strategy — omit it.
        strategy_description={
            "class": "MultiPickStrategy",   # strategy class name
            "pairing": "sequential",        # "sequential" | "color_match" | "type_based" | "stacking"
            # "details": "strategy-specific notes",
        },
        rationale={
            # Implementation-side rationale (strategy choice, tree, motion tuning).
            "create_strategy": "Default sequential pairing — simple placement without matching",
            # Other impl-side keys:
            # "tree_factory"                         — why this BT variant
            # "use_curobo"                           — why cuRobo over the cortex tree
            # "virtual_target_generation_strategy"  — why targets are virtual (never spawned)
            # "ee_height_for_move"                   — why transport height is overridden
            # "place_hover_above_z"/"_approach_distance" — why custom place geometry
            # "pick_approach_p_thresh"               — why tighter/looser approach close
            # "move_/approach_/insert_timeout_s"     — why custom watchdog deadlines
            # "pick_posture_config"/"place_posture_config" — why posture override
        },
    ),
)
super().__init__(task_spec=spec, offset=offset, **kwargs)
```

**Rationale guidelines:**
- The `create_strategy` rationale is **always required** when a non-default strategy is configured — explain why this strategy class was chosen. (Default `MultiPickStrategy` may omit it.)
- Other rationale entries are only needed for non-default choices (custom verification, stacking, ee_height override, virtual targets, etc.)
- Each spec owns rationale for its own fields: scene-related keys on outer `TaskSpec.rationale`, implementation-related keys on `TaskImplementationSpec.rationale`.
- Keep rationale strings concise (one sentence) but specific about the *why*
- Only include non-None/non-default fields in description sub-dicts
