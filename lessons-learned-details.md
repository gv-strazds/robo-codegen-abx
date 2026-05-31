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

**Failure trace**: Verification passed (every item was inside its correct box), but in both the mock dump and Isaac Sim the placed items of one type were sitting on top of each other at the box-center XY, not spread along the box length. `run_mock_task.py` output confirmed it: all sugar_box items at `pos=[-0.503, 0.8034, 0.157]`, etc.

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

### Issue 14 Details: Default cart decoration props collide with custom cart-spawned picks

**Setup**: `TableTaskGreenCubesRowToYellowGrid` spawns 6 green cubes in a single row along Y at `(CART_SURFACE_CENTER[0] − 0.18, CART_SURFACE_CENTER[1] + j·0.08, …)` for j ∈ {−2.5 .. 2.5}, i.e. row X ≈ −0.72 and Y ∈ [0.53, 0.93]. The workspace lambda initially used the default `setup_two_tables(scene, assets_root)`, which is equivalent to passing `standard_objs=True, add_bin=True`.

**Failure trace**: The Phase 5 self-check passed verification (`task_successful: true`, no failure-event frames), but visual inspection of the task-start snapshot showed one of the green cubes interpenetrating the default `cracker_box` on the cart. `setup_two_tables` places that cracker_box at `TABLETOP_CENTER_POINT + [−0.19, −0.08, 0.15]` ≈ `[−0.75, 0.625, 0.25]`, which falls inside the cube row's footprint in both axes (X ≈ −0.72 is within the cracker_box X half-extent; Y = 0.625 is inside the cube row's Y range).

**Root cause**: `setup_two_tables` has two flags that default to `True`:
- `standard_objs=True` — adds `cracker_box`, `sugar_box`, `soup_can`, `mustard_bottle` decorative props on top of the cart at fixed offsets from `TABLETOP_CENTER_POINT`. Intended as set dressing for tasks that pick from the bin (so the cart top has visual context for the work area).
- `add_bin=True` — places the KLT_Bin at `(BIN_X_COORD, BIN_Y_COORD, …)`, the +X half of the cart top.

For tasks where picks come from the bin those defaults are visually correct. But for any task that uses the cart surface as the *pick source*, the four props and/or the bin will sit in the same XY region as the custom layout and intersect it physically.

**Fix**: pass both flags as `False` in the workspace lambda:
```python
setup_workspace=lambda scene, assets_root: setup_two_tables(
    scene, assets_root, standard_objs=False, add_bin=False,
),
```
This leaves the cart, its invisible collision surface, the conveyor + its dropzone surface, and the camera setup in place — only the decoration props and the bin are suppressed. After the change, the cart starts empty except for the six green cubes.

**Reusable check**: any task whose pick generator emits positions inside `CART_SURFACE_REGION` should audit `setup_two_tables` keyword arguments before relying on the defaults. As a quick sanity test, draw the four prop positions (`TABLETOP_CENTER_POINT + [(−0.19,−0.08,0.15), (−0.06,−0.08,0.01), (0.11,−0.08,0.10), (−0.055, 0.235, 0.12)]`) and the bin footprint (`BIN_INNER_REGION`) on top of the planned pick layout — any overlap means `standard_objs=False` and/or `add_bin=False` should be passed. Conversely, if the bin or any prop is part of the intended scene, keep that flag on and just steer the cart layout around it.

### Issue 13 Details: `disc` (DynamicCylinder) scale_xy acts as radius, not diameter

**Setup**: `TableTaskCrackerCircleMarkers` targets are 4 colored `disc` markers arranged in a circle on the dropzone. First implementation used `target_scale = np.array([DISC_DIAMETER, DISC_DIAMETER, DISC_THICKNESS]) / stage_units` with `DISC_DIAMETER = 0.15`, expecting a 0.15 m wide disc.

**Failure trace**: The headless self-check passed verification, but the snapshots showed the 4 discs visibly overlapping each other. The chord between adjacent disc positions on a `radius=0.18 m`, `count=4` circle is `0.18 * sqrt(2) ≈ 0.255 m`, but the rendered discs were ≈ 0.30 m wide, so they intersected each other by about 0.05 m at the rims.

**Root cause**: In `asset_utils.PRIMS_MAP`, both `"disc"` and `"cylinder"` map to `isaacsim.core.api.objects.cylinder.DynamicCylinder`. That class defaults to `radius=1.0` and `height=1.0` when neither is passed in (and `add_prim_asset` never passes them — only `scale`). So the underlying USD prim's base radius is 1.0 m, and `scale_xy` multiplies that radius. The intuitive reading — "scale_xy is the diameter" — is double the actual size. Passing `scale=[0.15, 0.15, ...]` produces a cylinder of radius 0.15 m (diameter 0.30 m), not a 0.15 m wide disc. The same factor-of-2 applies to `"cylinder"`. (Other primitives behave differently — `DynamicCuboid` is created at `size=1.0` so `scale_xy` *is* the side length.)

**Fix**: Compute the scale explicitly as half the intended diameter: `target_scale = np.array([DISC_DIAMETER / 2, DISC_DIAMETER / 2, DISC_THICKNESS]) / stage_units`. Added a comment in the task explaining the DynamicCylinder default-radius convention so future edits don't fall into the same trap.

**Reusable check**: when sizing `disc` or `cylinder` assets, remember `scale_xy = radius`, not diameter. The two convenient sanity checks at the call site:
- Adjacent-marker chord on a `count=n`, `radius=r` circle is `2 * r * sin(π/n)`; the disc diameter must stay under it (with a comfortable gap) to avoid overlap.
- For DynamicCuboid (`"cube"` / `"rect"`) `scale_xy` is the side length, but for DynamicCylinder it's the radius — don't reuse a `scale` value across the two primitive families without halving / doubling.

### Issue 15 Details: Default to hidden virtual targets unless the user specifies visible target geometry

**Setup**: `TableTaskSugarBoxesRowToCircle` was created from the user request *"Starting with a row of 6 vertical sugar boxes in the pick bin, arrange them in a circle on the drop zone."* The user described the picks concretely (6 sugar boxes, vertical, row, bin) and the destination *arrangement* (a circle on the dropzone), but said nothing about what should occupy each drop position. The initial implementation followed the closest existing template — `TableTaskCrackerBoxes1`, which is also a row of upright YCB boxes into dropzone markers — and inherited its visible-marker target style: 6 white `"rect"` markers (scale `[0.06, 0.06, 0.002]`) via `target_generation_strategy`.

**Failure trace**: Phase 5 self-check passed (`task_successful: true`, no failure-event frames). After viewing the snapshot the user's very next message was: *"make the destination targets not visible"*. The fix was mechanical — switch `asset_type` from `"rect"` to `"marker"`, add `hidden_strategy=FixedValue(True)`, move the generator from `TaskSpec.target_generation_strategy` to `TaskImplementationSpec.virtual_target_generation_strategy`, re-run Phase 4 mock and Phase 5 sim. About one full verification round trip that could have been saved.

**Root cause**: Two different target-spawn patterns coexist in the codebase, and the implementation defaulted to the "wrong" one given the user's actual intent:
- *Visible targets* (`TaskSpec.target_generation_strategy`, no hidden flag) — used when the marker geometry is part of the *task description*: e.g. `TableTaskCrackerBoxes1` (*"onto thin green rectangles"*), `TableTaskGreenCubesRowToYellowGrid` (*"onto yellow rectangles"*), `TableTaskCrackerCircleMarkers` (*"onto colored disc markers"*). The user's words name the marker's color or shape.
- *Hidden virtual targets* (`TaskImplementationSpec.virtual_target_generation_strategy` + `hidden_strategy=FixedValue(True)`) — used when the user describes *where* items should land but not what should be there: e.g. `TableTaskColorCircle` (*"place them in a circle on the drop zone"*). The dropzone surface stays bare; targets exist only as logical pairing/placement anchors, never spawned as USD prims.

The cracker-boxes-1 layout was the closest *pick-side* match (row of upright YCB boxes in the bin), so the implementation reused that task's *target-side* style too — without re-checking the user's words for visible-marker intent. They weren't there.

**Fix**: Treat hidden virtual targets as the default for new tasks whose user description gives a destination layout but no target geometry. Concrete pattern:
```python
target_strategy = ItemGenerator(
    position_generator=...,
    asset_type_strategy=FixedValue("marker"),
    color_strategy=FixedValue("white"),
    scale_strategy=FixedValue(marker_scale),
    hidden_strategy=FixedValue(True),
)

spec = TaskSpec(
    ...
    pick_generation_strategy=pick_strategy,
    # NO target_generation_strategy here.
    implementation=TaskImplementationSpec(
        virtual_target_generation_strategy=target_strategy,
        ...
    ),
)
```
This matches `TableTaskColorCircle` and produces clean dropzone snapshots without visible scaffolding.

**Reusable check**: when interpreting a new task request, ask:
- *Did the user name the targets themselves?* (color, asset type, "marker"/"rectangle"/"disc", explicit size or thickness) → use **visible** `target_generation_strategy`.
- *Did the user only describe the layout?* ("in a circle", "in a 2×3 grid", "evenly spaced", "on the dropzone") → use **hidden** `virtual_target_generation_strategy` + `hidden_strategy=FixedValue(True)`.

The bar for visible markers is that the user *names them*. If the description is layout-only, the markers are scaffolding and should stay out of the scene.

### Issue 16 Details: Stationary dropzone markers should default to Fixed* primitives, not Dynamic*

**Setup**: `TableTaskSoupCansDiscs1` targets are 6 colored disc markers arranged in a 2x3 grid on the dropzone. First implementation used `asset_type="disc"` (→ `DynamicCylinder`), copying the convention used by `TableTaskCrackerCircleMarkers`. Discs are spawned at `DROPZONE_Z + 0.001 + thickness/2`; soup cans are placed on top of them by the BT and the discs themselves are never expected to move.

**Failure trace**: Phase 5 headless self-check passed (`task_successful: true`, no failure-event frames). But in the live `--teleport` GUI run the user reported soup cans visibly jittering / vibrating continuously after each placement. The motion was small but persistent and did not damp out within several seconds. Bumping thickness from 2 mm → 1 cm (matching Issue 10's fix) reduced amplitude but did not eliminate the jitter.

**Root cause**: `"disc"` resolves to `DynamicCylinder` — a full rigid body with mass, friction, and physics-material assignments. After a can lands on the disc, the contact stack is `(kinematic dropzone table) ↔ (dynamic disc) ↔ (dynamic soup can)`. The solver has to maintain stable normal forces through all three layers, and small mismatches in physics material parameters plus uneven contact between the disc bottom and the USD dropzone collision geometry produce tiny normal-force oscillations. The disc wobbles; the can on top transmits and amplifies the motion visually. Tuning the disc's mass / thickness / material parameters can attenuate the wobble but does not remove the underlying instability — the disc still has a rigid body.

The clean fix is to remove the disc's rigid body entirely. `PRIMS_MAP["fixed_disc"] = FixedCylinder` (asset_utils.py:85) provides a static collider with no RigidBodyAPI / MassAPI — the solver treats it as part of the static world. The disc cannot oscillate at all because it has no mass / velocity state. Visually identical, physically inert.

**Fix**: change `asset_type_strategy=FixedValue("disc")` → `FixedValue("fixed_disc")`. No other changes (scale, color, thickness, position) required. After the swap, the user confirmed the jitter was gone in the live GUI run.

**Reusable check**: when designing a task with decorative target markers on a static surface (dropzone, cart top, table top), default to the **Fixed*** primitive variant. Quick lookup:

| Intended target shape | Static (default for stationary markers) | Dynamic (only when target must move) |
|-----------------------|------------------------------------------|---------------------------------------|
| Disc / Cylinder       | `"fixed_disc"` → `FixedCylinder`         | `"disc"` / `"cylinder"` → `DynamicCylinder` |
| Box / Rectangle       | `"rect"` → `FixedCuboid`                 | `"cube"` → `DynamicCuboid`            |
| Sphere / Ball         | *(no Fixed variant in `PRIMS_MAP`)*      | `"ball"` → `DynamicSphere`            |

### Issue 17 Details: Stacked YCB cracker boxes lean, inflating their AABB along Z and breaking `is_on_top`

**Setup**: `TableTaskCrackerStacksToMarkers` unstacks 18 horizontal cracker boxes from a 3-layer 2×3 pre-stacked source on the dropzone and restacks them as three 6-high horizontal stacks on three green markers on the cart. Verification used a composite `spatial_check_fn = is_on_top(z_tol=0.04) AND is_horizontal(max_tilt_deg=20°)` so each box must rest on its paired target (the marker for layer 0, the previous box for layers 1–5) and lie roughly flat.

**Failure trace**: Phase 5 headless self-check (`--headless --teleport --snapshot-errors`) reported `Verification checks reported UNSUCCESSFUL completion` with 3 failures: `cracker_box_3`, `cracker_box_4`, `cracker_box_5` — all the top-of-stack picks. The snapshot showed three intact, visually correct stacks. Log:

```
is_on_top FAIL: 'cracker_box_3' on 'cracker_box_0':
  cracker_box_3 aabb=[..., 0.3848, ..., 0.5049]   (Z span 0.1201)
  cracker_box_0 aabb=[..., 0.3205, ..., 0.4315]   (Z span 0.1110)
  xy_overlap=True   z_diff=-0.0467 (tol=0.0400, z_ok=False)
```

**Root cause**: A flat cracker box has full Z extent 0.0718 m (per `asset_prim_geometry.json`: half_extent 0.0359 m). The observed Z span of 0.12 m is far above that. Working backward through `aabb_z = thickness × cos(θ) + long_dim × sin(θ)` with `thickness = 0.072 m`, `long_dim = 0.213 m`, solving for the observed `aabb_z = 0.12 m` gives `θ ≈ 14°`. Each box in the stack leans by ~13–14° because the YCB `003_cracker_box.usd` asset's contact surfaces aren't perfectly flat (the cardboard panels are modelled with slight curvature / wobble). The leans don't accumulate dangerously (each box still sits stably on the box below), but they inflate the per-box AABB symmetrically around the true rest position:

- Upper box's AABB-bottom is `~0.5 × (aabb_z - thickness) ≈ 0.024 m` *below* its true bottom face.
- Lower box's AABB-top is `~0.024 m` *above* its true top face.
- Total AABB overlap: `~0.048 m` — exactly the observed `|z_diff|`.

So `is_on_top`'s `z_tol=0.04` (a value tuned for flat, settled placements) gives a false negative. The boxes really are on top of each other; the AABBs say they interpenetrate.

`is_horizontal(max_tilt_deg=20°)` would have passed at 14° tilt (the up_axis would still be within 20° of the horizontal plane), but it never runs in this test — the composite check short-circuits on the `is_on_top` failure.

**Fix**: Relax both tolerances on the verification function:

```python
def _cracker_horizontal_on_top(pick_obj, target_obj, ...):
    on_top = is_on_top(pick_obj, target_obj, ..., z_tol=0.08)   # was 0.04
    if not on_top:
        return False
    return is_horizontal(pick_obj, ..., max_tilt_deg=30.0)       # was 20.0
```

A `z_tol` of 0.08 m absorbs lean up to ~22° per box; `max_tilt_deg=30°` keeps the orientation check meaningful (a box truly tipped on its long edge would tilt 90° and still fail) while tolerating the YCB asset's inherent skew.

**Reusable check**: when verifying horizontal-stacked YCB assets (cracker_box, sugar_box, etc.):
- For stacks of ≥ 3 boxes, default `z_tol ≥ 0.07 m` in `is_on_top`.
- For tilt orientation, default `max_tilt_deg ≥ 25°` in `is_horizontal`.
- If even those are insufficient (very tall stacks or asymmetric assets), switch to a position-based verifier: compute the expected stack-tip XYZ at pairing time and check `np.linalg.norm(pick_pos - expected_pos) < tol`. AABB-based checks fundamentally lose precision once items lean.

### Issue 18 Details: Centered cart-marker stacks fall outside the UR10's comfortable reach; build furthest stack first

**Setup**: `TableTaskCrackerStacksToMarkers` placed three target stacks in a row along Y on the cart, centered at `CART_SURFACE_CENTER[0]` with `spacing_y=0.30 m`. Headless `--teleport` runs passed all 18 verifications; the user then ran the full-sim variant (with motion planning) and reported that the robot "couldn't reach" the furthest (largest +Y) stack — motion planning either failed outright or churned for very long before producing a path.

**Root cause**: Two interacting issues:

1. **Reachability**: `CART_SURFACE_CENTER` is at `X ≈ -0.79` (relative to UR10 base at origin). The cart half-width in X is 0.35 m, so the cart spans `X ∈ [-1.14, -0.44]`. The far-X side is at the outer edge of the UR10's working radius (`UR10_WORKING_RADIUS ≈ 1.7 m` measured straight-line, but motion planning has to avoid self-collision and the cart itself, which constrains the practical reach much tighter). Stacks centered on the cart sit closer to the far-X edge than necessary; shifting them toward `+X` (toward the robot) buys margin without leaving the cart surface. With marker scale `[0.22, 0.17, 0.005]`, a +0.22 m offset gives marker-center at `X = -0.57`, marker edge at `X = -0.46` — still ~5 cm from the cart's near edge at `X = -0.44`, comfortable.

2. **Build order vs. obstruction**: `LayeredStackStrategy` fills stacks in `target_objs` order. With the default `GridPositionGenerator(rows=3, cols=1, spacing_y=+0.30)`, `target_0` is at smallest +Y, `target_2` at largest +Y. So the nearest-to-robot stack got built first, and by the time the robot tried to build the furthest stack, the two completed nearer stacks (0.43 m tall each) obstructed the arm's natural arc to the back of the cart. With the order reversed (`spacing_y=-0.30`), the furthest stack is built first when the path to it is clear, and the robot finishes by building the nearest stack with no in-front obstacles.

**Fix**: Two coupled adjustments in the target generator:

```python
# Before
center=np.array([cart_x, cart_y, marker_z]),
spacing_y=0.30,

# After
marker_x_offset = 0.22          # toward +X (closer to robot) edge of cart
marker_spacing_y = -0.30        # negative → target_0 at largest +Y (furthest stack first)
center=np.array([cart_x + marker_x_offset, cart_y, marker_z]),
spacing_y=marker_spacing_y,
```

After the change the user confirmed the full-sim run worked (all stacks built and reached without motion-plan stalls).

**Reusable check**: when a task places targets on the cart and uses motion planning (not just `--teleport`):
- Default target X to `CART_SURFACE_CENTER[0] + 0.20 m` (or further toward +X if cart geometry / marker size allow). The cart center is convenient for layout but bad for reachability.
- For a row of stacks built sequentially, build the **furthest** stack first so completed stacks don't sit between the robot and its next target. `LayeredStackStrategy` and any `MultiPickStrategy` that fills in `target_objs` order can be reordered cheaply via negative `spacing_y` (or by flipping the markers list when building it manually).
- `--teleport` mode masks both problems because it skips motion planning entirely; always verify reachability in a full-sim run before considering a cart-stacking task complete.

Reach for the dynamic variant only when the target must physically respond to forces — e.g. ride a moving conveyor (Issue 10), tumble after collision, or get nudged by another rigid body. For stationary visual markers, dynamic primitives are over-specified and invite contact-stack-instability jitter. Compare with Issue 10, which is the exact opposite situation (a `"rect"` target on a conveyor failed to move *because* it was static); pick the variant that matches whether the target must move, not by copying a sibling task's choice.

### Issue 19 Details: `TaskSpec.conveyor_speed` is logic-side only; the physics-side velocity is applied by `setup_two_tables`

**Setup**: `TableTaskConveyorColorRows` set `TaskSpec.conveyor_speed = DEFAULT_CONVEYOR_SPEED`, expecting the belt to drag cubes. Mock and `--teleport` runs passed; full-sim showed cubes stuck exactly at their spawn positions.

**Root cause**: `TaskSpec.conveyor_speed` is read in three places, none of which apply the physics velocity:

1. `multi_pickplace_task._update_more_expected_flags` clears `more_items_expected` for spatial-trigger schedulers when the belt is stationary (so the BT can complete on what's already in flight).
2. `SpatialTriggeredItemScheduler.tick()` early-exits when `conveyor_speed` is 0 / None — preventing replenishment from firing on a paused belt.
3. `TaskSpec.falloff_is_enabled()` auto-enables conveyor falloff verification when `conveyor_speed` is truthy.

The actual `PhysxSurfaceVelocityAPI` is applied by `table_setup.setup_two_tables(..., conveyor_speed=value)` — line 182:

```python
if conveyor_speed != 0.0:
    PhysxSchema.PhysxSurfaceVelocityAPI.Apply(usd_prim)
    surface_velocity = PhysxSchema.PhysxSurfaceVelocityAPI(usd_prim)
    velocity = Gf.Vec3f(0.0, conveyor_speed, 0.0)
    surface_velocity.CreateSurfaceVelocityAttr(velocity)
```

If the `setup_workspace` lambda doesn't forward `conveyor_speed`, the surface API is never created → the kinematic surface has zero velocity → cubes don't move.

**Why mock and `--teleport` hid the bug**:
- Mock has no physics at all; the belt is irrelevant.
- `--teleport` mode (in `task_context_base.teleport_to_target`) snaps the held item directly to its target prim's pose, skipping the entire grasp+lift+carry+place motion. Items are removed from the belt within ~5 sim seconds before they could have drifted noticeably.

**Fix**: Forward `conveyor_speed` through both call sites in the task class:

```python
conveyor_speed = DEFAULT_CONVEYOR_SPEED * 0.5  # or whatever the task needs

spec = TaskSpec(
    ...,
    conveyor_speed=conveyor_speed,                   # logic side
    setup_workspace=lambda scene, assets_root: setup_two_tables(
        scene, assets_root,
        standard_objs=False, add_bin=False,
        conveyor_speed=conveyor_speed,                # physics side
    ),
)
```

**Reusable check**: every conveyor task should grep both `TaskSpec(...)` and the `setup_workspace=` lambda body for the same `conveyor_speed` value. A working precedent is `tasks/table_task_conveyor_type_sort.py` (the only task in the repo that does this end-to-end correctly under full sim). Tasks like `table_task_conveyor_color_stacks.py` set `conveyor_speed` on TaskSpec but not on `setup_two_tables` — they happen to work because they use `--teleport` for verification and never exercised full physics.

### Issue 20 Details: Pick reachability gates only exist in the cortex BT; the default tree is open-loop on pick selection

**Setup**: `TableTaskConveyorColorRows` initially used the default 9-phase tree (`make_task_controller_tree` via `pt_task_tree.py`). The user reported that fallen cubes kept being targeted by the robot.

**Root cause**: The default 9-phase tree's pick sequence is:

```
MoveToPickXY → LowerToPick → WaitSettling → CloseGripper → LiftPicked
```

None of these check whether the pick is reachable — they just send the EE to wherever the strategy's `get_current_pick_name()` reports. The strategy doesn't know the cube is on the floor; it just returns the smallest-Y candidate, which after fall-off is still the fallen cube.

The cortex tree (`make_cortex_task_controller_tree` via `pt_cortex_tree.py`) inserts two gates:

```
CheckPickReachable → PrepareGrasp → CheckGraspPoseReachable → ...
```

`CheckPickReachable` polls the live pick prim each tick:
- If `pos[axis] < ctx.get_pick_min_reachable_z()` (the value set on `TaskImplementationSpec.pick_min_reachable_z`), it calls `strategy.mark_pick_permanently_unreachable(pick_name)` and returns FAILURE.
- If the pick is outside the XY working radius, it returns RUNNING and starts an `UNREACHABLE_GRACE_S` ~10 s timer; if the grace window expires it also marks the pick permanently unreachable.

`IsPickReachableGuard` is then placed outside the Retry decorator so a pick that goes permanent mid-retry aborts immediately rather than burning the full retry budget.

Once a pick is in `_permanently_unreachable_picks`, every standard `MultiPickStrategy` (`ColorMatchStrategy`, `ConveyorProximityStrategy`, custom JIT strategies) is expected to filter it from selection — both the base class's `_scan_for_available_pick` and `advance_pick_index` exclude it, and any custom strategy should query the set in its own candidate filter.

**Fix**: One line on `TaskImplementationSpec`:

```python
from robot_controllers.pt_cortex_tree import make_cortex_task_controller_tree

implementation=TaskImplementationSpec(
    tree_factory=make_cortex_task_controller_tree,
    pick_min_reachable_z=CONVEYOR_SURFACE_TOP_Z - 0.10,
    ...
)
```

The `pick_min_reachable_z=CONVEYOR_SURFACE_TOP_Z - 0.10` value matches the precedent in `tasks/table_task_conveyor_type_sort.py` — a 10 cm margin below the belt top catches cubes that have fallen but not yet hit the floor, while staying generous enough that a cube riding on the belt with mild physics settling jitter doesn't get falsely flagged.

**Reusable check**: any task that uses `pick_spatial_trigger_config` or `pick_incremental_config` with `conveyor_speed != 0` must use the cortex tree — otherwise unpicked items that fall off the belt edge become an infinite-retry hazard. The `pick_min_reachable_z` field on `TaskImplementationSpec` is the standard knob; setting it without `tree_factory=make_cortex_task_controller_tree` is silently a no-op.

### Issue 21 Details: JIT pick-selection breaks `all_picks_done` semantics when the cursor outruns the picking-order list

**Setup**: `ColorMatchConveyorProximityStrategy` in `tasks/table_task_conveyor_color_rows.py` overrode `get_current_pick_name` and `advance_pick_index` to select picks dynamically by world-Y proximity rather than spawn order. With `SpatialTriggerConfig` replenishing 13 cubes total, the headless `--teleport` self-check completed 12 picks then immediately fired "Task finished" and verification reported the 13th cube (cube_12) "not on a valid target while 1 valid target(s) remain available".

**Trace** (excerpt):
```
DEBUG MarkPickComplete: completed 'cube_11'      # 12th completed
DEBUG SelectNextPick: waiting for more items...   # ~120 idle ticks
DEBUG SpatialTriggeredItemScheduler: released 1 items (13/13) at t=19.433
INFO  Incremental generation complete: all 13 pick objects spawned
WARN  Task 'table_task_conveyor_color_rows' has finished.    # <-- BT ended here
FAIL  Pick 'cube_12': not placed on any valid target
```

**Root cause**: `UR10MultiPickPlaceController.is_done()` returns True when `task_context.all_picks_done` is True AND `more_items_expected` is False. The base `MultiPickStrategy.all_picks_done` definition is `_current_pick_index >= len(_picking_order_item_names)`.

The 9-phase tree's `SelectNextPick` calls `advance_pick_index()` on every tick after the first, regardless of whether a new pick was completed. During the ~120 idle ticks waiting for cube_12, the override's `self._current_pick_index += 1` ran once per tick, driving the cursor far past the list length (12 at that time).

When cube_12 was finally spawned via `add_incremental_picks`, the override appended it to `_picking_order_item_names` (now length 13) — but the cursor was already at ~130. `all_picks_done` immediately returned True, the spawner's `all_picks_released = True` set `more_items_expected = False`, and `is_done()` fired in the *same* simulation step that cube_12 was added — before the BT could call `SelectNextPick` again.

**Fix**: Two coupled overrides (final code in `tasks/table_task_conveyor_color_rows.py:130-180`):

```python
def advance_pick_index(self) -> Optional[str]:
    self._active_pick_name = None
    next_name = self.get_current_pick_name()
    if next_name is not None:
        self._current_pick_index += 1
    return next_name

@property
def all_picks_done(self) -> bool:
    names = self._picking_order_item_names
    if not names:
        return False
    for name in names:
        if name in self._completed_picks:
            continue
        if name in self._permanently_unreachable_picks:
            continue
        return False
    return True
```

Now the cursor only increments when there is genuinely something to advance to, and the completion check is semantic (set membership) rather than positional. Both are necessary: just fixing the cursor without fixing `all_picks_done` would still leave a stale cursor from earlier "advance, then no candidate" cycles; just fixing `all_picks_done` without fixing the cursor would still leak the cursor past the list size and complicate any other code that reads `_current_pick_index`.

**Reusable check**: any strategy that overrides `get_current_pick_name` / `advance_pick_index` for JIT pick selection should also override `all_picks_done` semantically. The base class's cursor-based check assumes the cursor moves through the list exactly once — JIT strategies don't preserve that invariant. The same applies to any strategy where `_picking_order_item_names` grows after initialization (incremental spawn + non-sequential pick order).

### Issue 22 Details: ColorMatchStrategy + conveyor proximity requires JIT on both sides; subclass mirroring `ConveyorProximityStrategy`

**Setup**: User specified "robot picks the cubes approaching the end of the conveyor" (proximity-order pick selection) AND "places onto matching colored markers, filling each row from +Y to -Y" (color-matched target assignment with deterministic fill order). Default `ColorMatchStrategy` iterates picks in spawn order: a freshly-spawned cube at the +Y feed point could be selected ahead of an older cube near the belt edge. `ConveyorProximityStrategy` does proximity-order target selection but assumes default sequential picking.

**Root cause**: The two existing strategies are mirror images of each other on different axes:

| Strategy                         | Pick selection          | Target assignment              |
|----------------------------------|-------------------------|--------------------------------|
| `ColorMatchStrategy`             | spawn-order iteration   | color-match, pre-paired at init |
| `ConveyorProximityStrategy`      | spawn-order iteration   | JIT proximity, latched per place |
| (what this task needs)           | **JIT proximity**       | **JIT color-match, latched**     |

So neither off-the-shelf strategy fits. The new combined strategy subclasses `ColorMatchStrategy` (inheriting `_has_color`, `_color_palette`, base pairings bookkeeping) and adds two JIT layers, one on each side.

**Fix**: `ColorMatchConveyorProximityStrategy` in `tasks/table_task_conveyor_color_rows.py`, ~110 lines. Key overrides:

```python
# Pick side
def _proximity_key(self, pick_obj):
    pos, _ = pick_obj.get_world_pose()
    return -self.SIGN * float(pos[self.AXIS_IDX])  # smaller = closer to edge

def get_current_pick_name(self):
    if self._targets_exhausted: return None
    if self._active_pick_name and self._pick_is_candidate(self._active_pick_name):
        return self._active_pick_name
    self._active_pick_name = self._select_next_pick()  # smallest-Y candidate
    return self._active_pick_name

def latch_current_pick(self, pick_name):
    self._active_pick_name = pick_name  # cortex-tree post-grasp commit

# Target side
def _first_unused_matching_target(self, pick_name):
    color = self._pick_color(pick_name)
    for tgt in self._target_objs:  # target list pre-sorted +Y -> -Y per color
        if self._has_color(tgt, color) and tgt.name not in occupied + latched_by_others:
            if self.is_target_reachable(tgt.name):
                return tgt.name

def get_placing_target_name(self, pick_name):
    if pick_name in self._completed_picks:
        return self._pairings_by_pick_name.get(pick_name)
    latched = self._latched_target_by_pick.get(pick_name)
    if latched and still_valid(latched):
        return latched
    new_tgt = self._first_unused_matching_target(pick_name)
    self._pairings_by_pick_name[pick_name] = new_tgt
    return new_tgt

def latch_current_target(self, pick_name):
    tgt = self._pairings_by_pick_name.get(pick_name) or self._first_unused_matching_target(pick_name)
    self._latched_target_by_pick[pick_name] = tgt

# Incremental
def add_incremental_picks(self, new_objs):
    self._extend_pick_objs(new_objs)
    for obj in new_objs:
        if obj.name not in self._picking_order_item_names:
            self._picking_order_item_names.append(obj.name)
    self._targets_exhausted = False  # new picks may unblock
```

Combined with target list pre-sorted +Y → -Y within each color group, this gives:
1. Each tick the proximity scan finds the smallest-Y cube → latched as `_active_pick_name`.
2. When the cortex tree calls `LatchPlacementTarget`, the JIT assignment picks the first unused matching-color target from the pre-sorted list — which is always the +Y-most empty slot in that color's row.
3. After mark_pick_complete, both latches are cleared; next cycle picks the new smallest-Y candidate.

**Verification** (full-sim non-teleport, 25 s cap):
- Pick 0: `cube_4` (spawned at Y=0.45, smallest of initial 5) → `target_10` (+Y-most blue slot — cube_4 was blue).
- Pick 1: `cube_3` (Y=0.55, next-smallest) → `target_5` (+Y-most green slot — cube_3 was green).

**Reusable check**: when a task needs JIT pick selection AND color-matched target assignment, the existing `ConveyorProximityStrategy` is a structural template — the same latch lifecycle, same `add_incremental_targets` (or `add_incremental_picks` for the pick side) pattern, just with the proximity scan applied to the opposite side from the original. Don't pre-pair in `initialize_pairings`; let `get_placing_target_name` JIT-assign from `_target_objs` order so the target-list ordering (which the task class controls) drives the fill semantics.
