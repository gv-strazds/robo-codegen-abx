# TableTaskConveyorTypeSort2

## User Request

> Three boxes are on the cart, sized large enough to hold a row of 5 cracker boxes each. Every 1.5 seconds a new item appears on the conveyor (which is moving at the default conveyor speed). The items should be spawned at the far end (high y coord) of the drop zone, positioned at a random offset from -2 to +2 centimeters in the x dimension from the centerline of the drop zone. The robot should pick up these items and sort them into a corresponding boxes based on the type of the item. The sequence of items should be randomized as follows: total number of items from 5 to 15, type of each item randomly chosen from [cracker box, sugar box, mustard bottle]

### Clarifications provided by the user

- **Overflow**: constrain random sampling so no type exceeds 5 (matches each box's row-of-5 capacity).
- **Spacing**: increase the conveyor speed to `1.5 * DEFAULT_CONVEYOR_SPEED = -0.0225 m/s` and spawn one item every **4.0 seconds** so successive items are ~8 cm apart along the belt.
- **Completion**: succeed when the count threshold is met. The default `target_count` equals the number of items actually spawned (unless overridden via CLI).
- **Verification**: use box-containment checking **and** require each placed item to be vertical.

## Task Overview

Randomly generate a batch of `5..15` upright YCB items drawn uniformly from `{cracker_box, sugar_box, mustard_bottle}` (capped at 5 per type) and spawn them one-by-one at the far end of the drop zone onto a conveyor that runs at twice the default surface speed. The UR10 picks each arriving item and places it in one of three open-top collection boxes on the cart, chosen by the item's type (cracker → box #1, sugar → box #2, mustard → box #3).

- **Pick assets**: USD assets `cracker_box`, `sugar_box`, `mustard_bottle` (all upright, −90° X rotation).
- **Target objects**: three `spawn_open_box` containers on the cart, each inner 0.40 × 0.18 m, wall height 0.05 m. Three hidden stand-in targets populate `target_objs` so the BT can pair picks to a "target"; verification uses `box_verification_info` + `containment_check=True`, so the stand-ins are not spatially meaningful.
- **Pick generation**: custom `ConveyorTypeSpawnGenerator` using `IncrementalGenerationConfig(items_per_batch=1, batch_interval=4.0, bt_start_threshold=1)`.
- **Strategy**: `TypeBasedStrategy` (from `multi_pick_strategy.py`) routes each pick to its type-matched marker slot; pick type is inferred from the item's name prefix (default detector in `TypeBasedStrategy`) and looked up in `target_indices_by_type`.
- **Workspace**: `setup_two_tables(... conveyor_speed = 1.5 * DEFAULT_CONVEYOR_SPEED, add_bin=False, standard_objs=False)` plus three `spawn_open_box` calls.

## Concise Task Description

Pick items arriving one at a time on a moving conveyor and sort them into the cart-top box that corresponds to each item's type.

## Pick Items

- **Type**: one of `cracker_box`, `sugar_box`, `mustard_bottle` per item (uniform random, capped at 5 per type).
- **Arrangement**: sequential conveyor spawning at the far end of the drop zone (`x = DROPZONE_CENTER_POINT[0] ± 0.02 m`, `y = far-end of dropzone`, `z = DROPZONE_Z + asset rest height`); belt physics transports items toward the robot.
- **Count**: random integer in `[5, 15]` total (CLI-overridable via `--pick-count`).
- **Color/Appearance**: USD asset defaults (no color override).
- **Orientation**: upright (`Gf.Rotation(Gf.Vec3d(1,0,0), −90°)`).

## Target Objects

- **Type**: three open-top collection boxes spawned via `spawn_open_box`, one per asset type. Inner size 0.40 × 0.18 m, wall height 0.05 m, wall thickness 0.01 m, base 0.01 m.
- **Arrangement**: three boxes in a row along the cart's long (Y) axis at `cy + [−0.28, 0.0, +0.28]`, shared `cx = CART_SURFACE_CENTER[0]`.
- **Markers**: three hidden stand-in prims (`box_target_<type>`), one per box. Not spatially meaningful — verification is purely box-geometry based (`is_within_box_geometry` + verticality extra check).
- **Color/Appearance**: cardboard-ish tones, one per type (cracker = tan, sugar = pale buff, mustard = pale yellow).

## PickPlace Pairing and Sequencing

- **Pairing**: `TypeBasedStrategy.pair_picks_with_targets()` reads `target_indices_by_type` (built from `marker_<type>_<j>` target names) and pairs each pick to the next unused marker in its type's index list.
- **Valid targets per pick**: restricted to the marker indices registered under the pick's type, so containment verification only runs against the correct box (`match_labels={"type": <asset_type>}` on each `box_spec`).
- **Sequencing**: follows spawn order (the `IncrementalItemScheduler` releases items one by one as time progresses, and each release is appended via `strategy.add_incremental_picks`).

## Success Condition

All spawned items end up inside the correct-type box in an upright (≤ 15° tilt) pose.

## Success Checks

- Each pick object is inside the box matched by its `type` label (centralized `box_verification_info` + `containment_check=True`).
- Each placed item is vertical (`is_vertical` with `max_tilt_deg=15`, upright half-extents looked up per-type); wired in via the new `box_verification_info["extra_pick_check"]` extension that AND-s with the containment result.
- Default task completion triggers when the count threshold is met; default `target_count` equals the spawned pick count.
