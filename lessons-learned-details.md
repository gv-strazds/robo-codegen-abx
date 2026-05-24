# Lessons Learned: Debugging Pick-and-Place Tasks

## Case Study: TableTaskCartToConveyor (Feb 2025)

Task: Pick 4 types of USD assets (cracker_box, soup_can, mustard_bottle, sugar_box) from the cart and place them into boxes on the conveyor. Items placed vertically (upright).

### Issue 1: Items overlapping on the cart cause collisions during picking

**Symptom**: Robot picks an item and collides with adjacent items on the cart, knocking them over or displacing them. Subsequent picks fail because items are no longer at expected positions.

**Root cause**: Cart column spacing (0.08m between X offsets) was smaller than the actual item widths. The cracker_box has a half-width of 0.082m in X, so it physically overlapped the adjacent soup_can column 0.08m away.

**How we found it**: Computed actual footprints from `asset_prim_geometry.json` half-extents after applying the -90° X rotation for upright placement, then checked whether adjacent items in the cart layout overlap.

**Fix**: Space cart columns based on actual item widths plus a 2cm margin:
```
cracker_box half-width: 0.082m    →  type_x_offset: -0.18
soup_can half-width:    0.034m    →  type_x_offset: -0.04
mustard_bottle half-w:  0.048m    →  type_x_offset:  0.06
sugar_box half-width:   0.046m    →  type_x_offset:  0.18
```

**General rule**: Always compute actual world-frame footprints (accounting for spawn orientation) before setting layout spacing. For USD assets spawned with -90° X rotation, local Y becomes world Z (height) and local Z becomes world Y (depth), while local X stays as world X.

### Issue 2: Boxes too narrow for items

**Symptom**: Items placed at marker positions inside boxes extend past the box walls. In simulation, items bounce off walls or land outside the box.

**Root cause**: Box inner width (0.30m) was too small. With a 2×2 marker grid, the cracker_box at offset X=-0.075 extended to X=-0.157, past the box wall at X=-0.15.

**How we found it**: Computed item positions (marker offset + item half-width) and compared against box wall positions (inner_size/2).

**Fix**: Increased box inner X from 0.30m to 0.38m. Verified each item type fits at its marker position with clearance from box walls.

**General rule**: When designing boxes/containers for a 2×2 or NxM grid of items, verify that the largest item at the most extreme grid position still fits within the box walls. Formula: `box_inner_half > abs(marker_offset) + item_half_width + margin`.

### Issue 3: Transport height too low — carried items collide with other objects

**Symptom**: Robot successfully picks an item but collides with other tall items on the cart while moving horizontally toward the target. Items get knocked over, causing cascading failures.

**Root cause**: The default `ee_height_for_move = 0.3m` was barely above the tops of upright cracker_boxes on the cart (~0.296m). The bottom of a carried cracker_box at move height is at `0.3 - rest_height = 0.3 - 0.107 = 0.193m`, well below the 0.296m tops.

**How we found it**: Traced the move height through the code:
- `ee_height_for_move` (0.3m) is the EE target Z during horizontal moves
- EE offset (grasp_height) shifts the EE above the item center
- The carried item's bottom extends `rest_height` below the EE target
- Compared carried item bottom against tops of obstacles in the workspace

**Fix**:
1. Added per-task `_ee_height_for_move` attribute support (in `task_context.py` and `multi_pickplace_task.py`)
2. Set `self._ee_height_for_move = 0.45 / stage_units` in the task

**Clearance calculation**:
```
obstacle_top = cart_z + item_clearance_above_surface + item_full_height
             = 0.0573 + 0.025 + 0.213 = 0.296m  (for upright cracker_box)

carried_item_bottom = ee_height_for_move - rest_height
                    = ee_height - 0.107

Need: carried_item_bottom > obstacle_top + margin
      ee_height > 0.296 + 0.107 + 0.03 = 0.433m  →  used 0.45m
```

**General rule**: When items are picked from a surface with other tall items nearby, verify that:
```
ee_height_for_move > tallest_obstacle_top + carried_item_rest_height + safety_margin
```
The default 0.3m move height works for most tasks where picks are from a bin (low items) to a dropzone, but fails when picks are from surfaces with tall upright items.

### Issue 4: Box wall height vs. item height

**Symptom**: Increasing box wall height (0.15m → 0.23m) to contain tall items caused items to collide with box walls during transport just prior to the lowering phase of placement.

**Root cause**: The robot lowers items from move height into the box. If walls are tall, the item must pass between narrow walls while descending. With the default approach trajectory and gripper width, tall walls create a tight channel that items can't enter cleanly.

**Fix**: Kept box walls at 0.15m (shorter than the tallest item at 0.213m). Items stick out above the walls but are contained laterally. The walls provide lateral containment during placement without obstructing the descent path.

**General rule**: Box wall height is a tradeoff — tall enough for lateral containment but short enough that items can be lowered in without collision. For the current gripper and approach trajectory, walls should generally be shorter than the item height, not taller.

## Case Study: TableTaskShapeSortBoxes (Mar 2025)

Task: Sort randomly colored cubes (0.0515m) and balls (initially 0.0515m) from the conveyor into two boxes (0.16m × 0.16m inner, 0.06m walls) on the cart.

### Issue 5: Round objects don't fit in box despite grid fitting

**Symptom**: Even with taller walls, 4 balls at 0.0515m diameter placed in a 0.16m × 0.16m box with 2×2 grid at 0.06m spacing was too tight. Balls bounced off each other during physics settling and some escaped or piled up.

**Root cause**: The grid spacing (0.06m) and box inner size (0.16m) were geometrically sufficient for point positions, but balls are physical objects that bounce and interact. Four 0.05m balls in a 0.16m box leaves only ~0.06m total clearance in each axis — not enough for physics settling of round objects that push each other around.

**How we found it**: User observed that balls were too large for the box in the simulation viewport. The geometric calculation didn't account for dynamic physics interactions between round objects.

**Fix**: Reduced ball scale from 0.0515m to 0.035m. This gives each ball ~0.035m diameter, so 4 balls in a 0.16m box have ample room (~0.09m clearance per axis). Cubes kept at 0.0515m since they settle stably.

**General rule**: When placing multiple round/bouncy objects into a container, the geometric grid fit is necessary but not sufficient. Account for physics bouncing and object-to-object interactions. Either (a) make the container larger, (b) reduce object count, or (c) reduce object size. As a rule of thumb, total object diameter across any row should be under 60% of box inner dimension for round objects, vs 80% for stable flat-bottomed objects.

## Case Study: TableTask3b (Mar 2025)

Task: Pick 3-6 balls from the bin and place them into gaps between discs in a tight 3×4 grid on the dropzone. Balls nestle stably in pockets formed by 4 adjacent disc rims rather than balancing on a single flat disc surface.

### Issue 6: Default `is_on_top` verification fails for pocket/nestled placements

**Symptom**: All balls visually placed correctly in pockets between discs. Final verification reports 4 of 6 balls as FAIL. Pattern: earlier-placed balls fail, later-placed ones pass. The incremental checks (run one step after each placement) also fail for the first balls.

**Root cause**: The `is_on_top` check compares `ball_bottom` vs `marker_top` with z_tol=0.02m. In pocket geometry, the ball sinks partially below the disc top surface (by design — the ball contacts disc rims, not a flat surface). Physics interactions between dynamic discs (DynamicCylinder) and multiple balls cause small cumulative position shifts. The thin hidden markers (scale [0.04, 0.04, 0.001]) have an AABB of only ±0.0005m in Z, so even tiny shifts push `|ball_bottom - marker_top|` beyond the 0.02m tolerance.

**How we found it**: The mock runner passed (exact positioning, no physics). The real sim showed verification failures despite visually correct placement. The pattern of early balls failing but late ones passing pointed to cumulative physics settling rather than a constant offset.

**Fix**: Replaced the default `is_on_top` check with a custom `spatial_check_fn` using position proximity:
```python
def _spatial_check(pick_obj, target_obj, bb_cache=None, obj_scale=None):
    pick_pos, _ = pick_obj.get_world_pose()
    target_pos, _ = target_obj.get_world_pose()
    xy_dist = np.linalg.norm(pick_pos[:2] - target_pos[:2])
    z_diff = abs(pick_pos[2] - expected_pocket_z)
    return xy_dist < xy_tol and z_diff < z_tol
```
- XY tolerance: 40% of disc spacing (~42mm) — generous but won't match wrong gap
- Z tolerance: one ball radius (51.5mm) — accommodates physics settling

**General rule**: When objects rest in non-flat geometries (pockets, gaps, nestled positions), the AABB-based `is_on_top` check is inappropriate because:
1. The object's resting Z depends on contact geometry, not surface-on-surface stacking
2. Physics settling causes small shifts that exceed tight AABB-based tolerances
3. Dynamic support objects (like DynamicCylinder discs) can shift under load

Use a custom `spatial_check_fn` with position-proximity checks for such tasks. The tolerances should be:
- XY: tight enough to distinguish adjacent target positions (< half the target spacing)
- Z: generous enough for physics settling (one object radius is a good default)

## Debugging Workflow

1. **Run with `--teleport` first** to verify scene setup and marker positions are correct (no physics, instant placement)
2. **Run without `--teleport`** to test real robot motion and physics
3. **Check verification output** — `[FAIL]` lines identify which specific items failed and why
4. **Compute actual dimensions** from `asset_prim_geometry.json`, applying the spawn orientation rotation to get world-frame extents
5. **Trace the motion heights** through the code: `ee_height_for_move`, `grasp_height`, `rest_height`, `top_surface_height`

## Key Reference: USD Asset Dimensions (upright, after -90° X rotation)

| Asset | World X (width) | World Y (depth) | World Z (height) | Grasp Height | Rest Height |
|-------|----------------|-----------------|-------------------|--------------|-------------|
| cracker_box | 0.164m | 0.072m | 0.213m | 0.107m | 0.107m |
| soup_can | 0.068m | 0.068m | 0.102m | 0.051m | 0.051m |
| mustard_bottle | 0.096m | 0.058m | 0.191m | 0.096m | 0.096m |
| sugar_box | 0.093m | 0.045m | 0.176m | 0.088m | 0.088m |

Source: `asset_prim_geometry.json` half-extents × 2, with Y↔Z swap from -90° X rotation.

## Case Study: TableTaskConveyorTypeSort (Apr 2026)

### Issue 7 Details: Incremental-spawn strategy captured per-pick metadata at init time

**Setup**: Task uses `IncrementalGenerationConfig(items_per_batch=1, batch_interval=4.0, bt_start_threshold=1)` to spawn items one at a time over a simulated 60+ seconds. The pick generator creates `ItemSpec` objects for the full 15-item sequence up-front and records each item's type in `self.source_types`. A custom `MultiTypeSortStrategy` was initially constructed with `source_types = pick_strategy.source_types[:len(picks)]` inside `create_strategy`.

**Failure trace**:
1. Task initialization: only the first batch (`items_per_batch=1`) is spawned → `len(picks) == 1`.
2. `create_strategy(picks, targets)` is called → strategy stores `source_types=[first_item_type]` (length 1).
3. First item picked and placed successfully.
4. `pre_step` releases batch 2; `strategy.add_incremental_picks(new_objs)` appends to `_pick_objs`.
5. Base class recomputes pairings by calling `pair_picks_with_targets()`, which loops `range(len(self._pick_objs))`.
6. For `idx >= 1`, `self._source_types[idx]` is out of range → type lookup returns `None` → no target → `CheckTargetAvailable` logs `no target for '<pick>'` and the BT calls `SetTaskFinished` prematurely.

**Root cause**: `create_strategy` runs once, at strategy-construction time, when only the initial batch exists. Capturing a sliced copy of a pick-indexed list there guarantees it will go stale as incremental picks arrive. Base `MultiPickStrategy` already supports growth of `_pick_objs` via `add_incremental_picks` — strategies must respect that and derive per-pick data from the pick objects themselves each call.

**Fix**: Removed the `source_types` parameter from the strategy. Replaced with a `_type_for_pick(pick_obj)` helper that matches the pick's name prefix against the set of known types. Both `pair_picks_with_targets()` and `valid_targets_for_pick()` now call this helper, so they work for any `pick_obj` currently in `_pick_objs` including those added incrementally.

**Reusable check**: When writing a custom strategy for an incremental-spawn task, audit any constructor argument that has one element per initial pick. If it does, it will be silently wrong once `add_incremental_picks` runs. Either replace it with a per-pick lookup function, or query the full generator object (not a sliced copy) at strategy-creation time.

### Issue 8 Details: Same-position stacking when many picks share one target

**Setup**: First attempt at box-containment verification used three hidden stand-in target prims (one per type). The `MultiTypeSortStrategy` paired every pick to the stand-in whose index matched the pick's type. `containment_check=True` with `box_verification_info={"box_specs": ..., "extra_pick_check": _vertical_check}` handled verification.

**Failure trace**: Verification passed (every item was inside its correct box), but in both the mock dump and Isaac Sim the placed items of one type were sitting on top of each other at the box-center XY, not spread along the box length. Mock runner output confirmed it: all sugar_box items at `pos=[-0.503, 0.8034, 0.157]`, etc.

**Root cause**: The BT's `get_placing_info(pick_name, ...)` returns the world pose of the paired target prim for the robot to drop the item at. When multiple picks pair to the same target prim, they all drop at the same (x, y, z). `containment_check=True` lets the *verifier* accept multi-occupancy of a single box target, but it does nothing about the BT's choice of placement position.

**Fix**: Reverted to a per-slot marker scheme — a fixed `MAX_PER_TYPE=5` row of hidden markers inside each box, evenly spaced along the box's long Y axis (step = inner_y / 5), named `marker_<type>_<j>`. Strategy's `pair_picks_with_targets()` keeps per-type iterators over the markers of that type and yields the next unused one per pick. Kept `containment_check=True` so verification is still box-geometry + verticality, independent of exact marker hit precision.

**Design note**: The slot count is deliberately fixed (5 per box, sized for the widest item `cracker_box` whose narrow face is 0.072 m → 0.08 m slot pitch). It does not consult actual spawned counts, so the layout does not leak foreknowledge of the random sequence into the scene setup.

**Reusable check**: `containment_check=True` is about verification. For *side-by-side* packing/sorting inside a container (picks land at distinct slots along a row or grid), you need one target marker per placement slot — a single per-container target would pile every pick at the same XY. A single per-container target is perfectly fine when picks are meant to *stack vertically* at one location (the stacking strategy itself adjusts the drop Z as picks accumulate); the guidance here is specific to the sort/pack case, not the stack case.

### Issue 9 Details: YCB mustard_bottle occasionally stalled on the moving conveyor

**Setup**: Items spawned incrementally at the far end of the conveyor drop zone, belt at `2 × DEFAULT_CONVEYOR_SPEED = −0.030 m/s` via `PhysxSurfaceVelocityAPI` applied to `/World/conveyor_surface` — a 1 mm-thin DynamicCuboid (scale `[0.7, 1.6, 0.001]`, kinematic, gravity disabled). Items spawned upright (−90° X rotation) at Z = `DROPZONE_Z + rest_height + 0.005`.

**Failure trace**: `cracker_box` and `sugar_box` items (flat-bottomed collision primitives) always moved with the belt toward the robot. `mustard_bottle` items sometimes sat stationary at their spawn location while the belt slid under them. If a later-spawned item collided with a stuck bottle and knocked it onto its side, the bottle immediately began moving.

**Root cause**: `PhysxSurfaceVelocityAPI` transfers surface velocity only via friction where a dynamic rigid body is in actual contact with the kinematic surface. The YCB `mustard_bottle` USD collision mesh follows the scanned real-world bottle shape, including its concave base, so only a thin outer rim contacts a flat surface when upright. After a ~8 mm fall from `DROPZONE_Z + rest_height + 0.005` onto a 1 mm-thin slab, the rim can land with marginal contact depth / near-zero normal force on part of the rim. Friction becomes negligible → no momentum transfer → the bottle stays put. Once a collision topples the bottle, its long cylindrical side provides a large contact patch and friction recovers.

The other types are immune because cracker and sugar boxes are box-shaped; their collision primitive has a flat bottom that always beds solidly onto the slab.

**Fix (two layers, both kept)**:

1. *Spawn without hover*: change the per-item spawn Z from `DROPZONE_Z + rest_height + 0.005` to `DROPZONE_Z + rest_height`. Items start already pressed against the slab rather than dropping onto it, so there's no fall-and-settle transient that can leave the bottle balanced on its rim.

2. *Seed initial linear velocity*: override `pre_step` in the task so that every newly-spawned pick gets `SingleRigidPrim(prim_path=prim.prim_path).set_linear_velocity([0, conveyor_speed, 0])` the first time the task sees it. The velocity is idempotent (re-applied only to names not yet in `self._belt_velocity_initialized`). `set_linear_velocity` can fail for items spawned during `set_up_scene` because the rigid-prim view may not be ready yet; swallow that exception and let the next tick retry.

Why keep both: (1) alone reduced but didn't eliminate the stalls (observed directly in the user's Isaac Sim run). (2) alone would still leave items dropping onto the belt; seeding velocity helps even marginal-contact cases but belt-and-suspenders is more robust.

**Reusable check**: any task that spawns items on a surface that uses `PhysxSurfaceVelocityAPI` should (a) avoid mid-air spawn hovers for curved- or concave-based assets, and (b) seed the spawn linear velocity to the surface velocity when the asset's expected contact patch is small. The general principle: `PhysxSurfaceVelocityAPI` needs *real* friction contact; any geometry/settling configuration that can produce near-zero normal force at contact will stall that object on the belt.

### Issue 10 Details: Static-primitive targets do not move with the belt

**Setup**: `TableTaskSoupCans2` spawns thin red rectangle targets at the far-Y end of the drop zone on a conveyor running at `DEFAULT_CONVEYOR_SPEED = -0.015 m/s`. First implementation used `asset_type="rect"` (scale `[0.1, 0.1, 0.002]`, color red), matching the static dropzone markers in `TableTaskSoupCans1`.

**Failure trace**: In the live Isaac Sim run the rectangles appeared at the correct spawn position and the soup cans were visible being picked, but the rectangles sat motionless on the belt. `cracker_box` / `sugar_box` pick items on the same belt in other tasks move correctly, so the conveyor itself was working.

**Root cause**: `PRIMS_MAP["rect"] = FixedCuboid` — a visual/collider primitive with no RigidBodyAPI. `PhysxSurfaceVelocityAPI` couples through friction only to rigid bodies; static colliders (FixedCuboid, VisualCuboid) never receive surface-velocity drag, regardless of mass/material settings. So the rectangle was a correctly-placed static collider, and the belt slid frictionlessly under it.

**Fix**: Switch the target primitive type from `"rect"` → `"cube"` (→ `DynamicCuboid`, which does apply RigidBodyAPI + default MassAPI + default physics material). Two shape tweaks made it robust:
1. *Non-paper-thin*: bump target thickness from 2 mm → 1 cm. A 2 mm-thin dynamic cuboid tends to jitter or penetrate the kinematic belt surface at contact; 1 cm is thin enough to still read visually as a "marker" while giving stable rigid-body contact.
2. *Velocity kickstart*: add a `pre_step` override that mirrors the Issue 9 pattern — on the first tick after each new target prim exists, call `SingleRigidPrim(prim_path=...).set_linear_velocity([0, conveyor_speed, 0])`. Same exception-swallow-and-retry behavior for rigid-view initialization races.

**Reusable check**: any dynamic behavior on a conveyor — carrying, tumbling, colliding — requires a *dynamic* primitive. When the default asset type is known-static (rect, marker) and the task uses a moving belt, either switch the primitive class (cube/disc/ball) or extend `PRIMS_MAP` / `asset_utils` with a dynamic-variant entry. Thickness ≥ 1 cm and an initial belt-velocity seed are the two cheap robustness knobs.

### Issues 11–12 Details: Tuning burst interval and burst size to the robot's service rate

**Setup**: `TableTaskSoupCans2` releases target rectangles in bursts of 1-3 items at a fixed interval along `DEFAULT_CONVEYOR_SPEED = -0.015 m/s`. First attempt: `TARGET_BATCH_INTERVAL = 4.0 s`, `MAX_BURST = 3`, positions spread over a 0.28 m half-range along X within each burst.

**Failure traces**:
- *Interval too tight (Issue 11)*: newly-spawned bursts overlapped the trailing edge of the previous burst in Y. 4 s × 1.5 cm/s = 6 cm of belt travel, less than the 10 cm Y length of the rectangle, so the freshly-spawned burst was born partly on top of the previous one.
- *Burst size too large for throughput (Issue 12)*: even at a non-overlapping 6 s interval, max-burst = 3 meant up to 3 new targets appeared every 6 s. The robot's per-pick cycle (move-to-pick, grasp, transport, place, return) is ~4-5 s; with 3 targets/burst, the queue grew monotonically and later targets passed the robot's reach before a pick could commit.

**Fixes**:
1. *Pick interval to clear the previous footprint* (Issue 11): require `batch_interval > item_length_along_Y / |conveyor_speed|`. For 10 cm items at 1.5 cm/s that's > 6.67 s in the strict-non-overlap sense, but 6 s (≈ 9 cm travel) was accepted as the right compromise between visual tightness and clearance — the user preferred a small amount of brief overlap to an over-sparse scene.
2. *Cap burst size to match service rate* (Issue 12): drop `MAX_BURST` from 3 to 2. The spawn pattern still uses a fixed 3-slot row across conveyor X (preserves the "multi-lane conveyor" visual), but each burst randomly occupies only 1 or 2 of those slots via `rng.sample(slot_offsets, size)`. This keeps the per-spawn throughput ≤ 2 items per 6 s ≈ 1 item per 3 s, within the robot's cycle time.

**Reusable check**: for time-based spawning on a conveyor, compute two independent rate constraints and take the larger interval:
- *Geometry*: `batch_interval ≥ item_length_along_conveyor / conveyor_speed` (prevents spatial overlap at spawn).
- *Throughput*: `batch_interval ≥ max_burst × per_item_service_time` (prevents queue growth beyond the robot's service rate).

If only the geometry rule is enforced but the robot is slow, the belt carries away targets faster than they can be serviced. If only the throughput rule is enforced but the belt is slow, new spawns interpenetrate old ones. Both rules must hold simultaneously.

### Issue 13 Details: `disc` (DynamicCylinder) scale_xy acts as radius, not diameter

**Setup**: `TableTaskCrackerCircleMarkers` targets are 4 colored `disc` markers arranged in a circle on the dropzone. First implementation used `target_scale = np.array([DISC_DIAMETER, DISC_DIAMETER, DISC_THICKNESS]) / stage_units` with `DISC_DIAMETER = 0.15`, expecting a 0.15 m wide disc.

**Failure trace**: The headless self-check passed verification, but the snapshots showed the 4 discs visibly overlapping each other. The chord between adjacent disc positions on a `radius=0.18 m`, `count=4` circle is `0.18 * sqrt(2) ≈ 0.255 m`, but the rendered discs were ≈ 0.30 m wide, so they intersected each other by about 0.05 m at the rims.

**Root cause**: In `asset_utils.PRIMS_MAP`, both `"disc"` and `"cylinder"` map to `isaacsim.core.api.objects.cylinder.DynamicCylinder`. That class defaults to `radius=1.0` and `height=1.0` when neither is passed in (and `add_prim_asset` never passes them — only `scale`). So the underlying USD prim's base radius is 1.0 m, and `scale_xy` multiplies that radius. The intuitive reading — "scale_xy is the diameter" — is double the actual size. Passing `scale=[0.15, 0.15, ...]` produces a cylinder of radius 0.15 m (diameter 0.30 m), not a 0.15 m wide disc. The same factor-of-2 applies to `"cylinder"`. (Other primitives behave differently — `DynamicCuboid` is created at `size=1.0` so `scale_xy` *is* the side length.)

**Fix**: Compute the scale explicitly as half the intended diameter: `target_scale = np.array([DISC_DIAMETER / 2, DISC_DIAMETER / 2, DISC_THICKNESS]) / stage_units`. Added a comment in the task explaining the DynamicCylinder default-radius convention so future edits don't fall into the same trap.

**Reusable check**: when sizing `disc` or `cylinder` assets, remember `scale_xy = radius`, not diameter. The two convenient sanity checks at the call site:
- Adjacent-marker chord on a `count=n`, `radius=r` circle is `2 * r * sin(π/n)`; the disc diameter must stay under it (with a comfortable gap) to avoid overlap.
- For DynamicCuboid (`"cube"` / `"rect"`) `scale_xy` is the side length, but for DynamicCylinder it's the radius — don't reuse a `scale` value across the two primitive families without halving / doubling.
