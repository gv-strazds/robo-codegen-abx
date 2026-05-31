# TableTaskConveyorColorRows

## User Request

create a new task: A random number (3 to 5 of each color) of colored cubes (red, green, or blue, randomly interleaved), are spawned on the slowly moving conveyor. Initially, 5 of the cubes are spawned in a row with x-position jitter; New cubes are spawned incrementally, one at a time, replenishing the initially spawned sequence as they move along the conveyor. The robot picks the cubes approaching the end of the conveyor and places them onto matching colored small rectangular markers in three rows of 5 (along Y) on the initially empty cart, filling each color's row from +Y to -Y.

## Task Overview

Colored primitive cubes (`asset_type="cube"`) arrive on the slowly-moving conveyor and must be sorted onto color-coded rows of `rect` (`FixedCuboid`) markers on the cart. Per-color counts are sampled independently in [3, 5] using the task's `seed`, so the total varies in [9, 15] across runs. The pre-shuffled color list is split between two `FixedSpecsGenerator` wrappers — the primary generator returns the 5 initial cubes (laid out in a row along Y with X/Y jitter inside the UR10 working radius), and the `replenishment_generation_strategy` returns the remainder at a fixed +Y feed point.

Targets are 15 visible colored rectangular markers laid out in 3 color-coded rows of 5 markers each: rows offset along X toward the +X (robot) edge of the cart, markers within a row spaced along Y. Markers are pre-ordered +Y → -Y within each color group so `ColorMatchStrategy` (which assigns each pick to the first unused matching target) naturally fills each row +Y → -Y as picks arrive.

Spawning uses `SpatialTriggerConfig(initial_count=5, items_per_batch=1, invert=True, trigger_delay=2.0)` over a small region around the +Y feed point: a new cube is released ~2 s after the previous spawn drifts out of the region. The belt runs at `DEFAULT_CONVEYOR_SPEED` (-0.015 m/s); falloff verification auto-enables. The cart is set up empty (`standard_objs=False, add_bin=False`).

## Concise Task Description

Pick colored cubes (red, green, blue) arriving on the slowly-moving conveyor and place each onto a matching-color rectangular marker on the cart, filling three color rows from +Y to -Y.

## Pick Items

- **Type**: `cube` (primitive `DynamicCuboid`), edge ≈ 5.15 cm
- **Arrangement**: Initial 5 cubes spawn in a single row along Y on the conveyor inside the robot's working radius (Y ∈ {0.45, 0.55, 0.65, 0.75, 0.85}), with ±1 cm X-position jitter and ±1 cm Y-position jitter. Subsequent cubes spawn one at a time at the +Y feed point (Y ≈ 0.85) with the same X/Y jitter, triggered when the spawn region is empty (`invert=True`) after a 2.0 s delay.
- **Count**: 9–15 total (independent random 3–5 per color × 3 colors), pre-computed at `__init__` time from the seed and pinned to `TaskSpec.pick_count` so the `SpatialTriggeredItemScheduler` queues are sized exactly.
- **Color/Appearance**: Red, green, blue (the pre-shuffled color list is split between the primary and replenishment generators so colors are randomly interleaved across the spawn sequence).

## Target Objects

- **Type**: `rect` (static `FixedCuboid`) markers, ~5 cm × 4 cm × 2 mm.
- **Arrangement**: 3 color-coded rows of 5 markers each = 15 markers total, on the initially empty cart surface. The 3 rows are aligned along X (perpendicular to the fill axis), shifted toward the +X edge of the cart (robot side) per learnings.md Issue 18. Within each row, the 5 markers are spaced 0.14 m along Y, centered slightly toward the robot's "easy" Y range to keep the +Y-most marker inside the comfortable reach band.
- **Markers**: Visible (not hidden), one per placement slot. Pre-ordered in the target list as: all reds (+Y→-Y), then all greens (+Y→-Y), then all blues (+Y→-Y). `ColorMatchStrategy` walks the unused targets in this order per color so each row fills +Y→-Y.
- **Color/Appearance**: Row 0 (closest to robot, +X-most) is red; row 1 (middle) is green; row 2 (deepest, -X-most) is blue.

## PickPlace Pairing and Sequencing

- **Pairing**: `ColorMatchStrategy` with `color_palette=["red", "green", "blue"]`. Each pick is paired to the first unused same-color target in target-list order. Because the target list is pre-sorted +Y→-Y within each color, the assignment naturally fills each row from +Y to -Y.
- **Sequencing**: Picks are processed in spawn order. For the initial 5 cubes laid out along Y, this means the cube closest to the belt edge (smallest Y) is the oldest and is picked first — matching the user's "approaching the end of the conveyor" description. Replenishment cubes append at the tail; new arrivals enter the order as they spawn.
- **Surplus safety**: Per-color counts are bounded above by 5 (= row length), so every pick is guaranteed to find an unused matching target. No surplus / "no target for" stall is possible by construction.

## Success Condition

Every spawned cube is placed on top of an unused same-color marker in its row, with each row filled from +Y to -Y.

## Success Checks

- Each pick rests on a same-color target (default `is_on_top` via `ColorMatchStrategy.is_pick_successfully_placed`, which enforces color match).
- The unused targets in each color row are at the -Y end (because picks consume the +Y-most unused matching target first).
- Conveyor falloff is auto-enabled (`conveyor_speed` is truthy), so any unpicked cube that drifts off the belt is flagged.
- `pick_min_reachable_z = CONVEYOR_SURFACE_TOP_Z - 0.10 m` defers below-belt cubes from BT consideration.
