# Lessons Learned: Debugging Pick-and-Place Tasks

## Case Study: TableTaskCartToConveyor (Feb 2025)

Task: Pick 4 types of USD assets (cracker_box, soup_can, mustard_bottle, sugar_box) from the cart and place them into boxes on the conveyor. Items placed vertically (upright).

### Issue 1: Items overlapping on the cart cause collisions during picking

**Symptom**: Robot picks an item and collides with adjacent items on the cart, knocking them over or displacing them. Subsequent picks fail because items are no longer at expected positions.

**General rule**: Always compute actual world-frame footprints (accounting for spawn orientation) before setting layout spacing. For USD assets spawned with -90° X rotation, local Y becomes world Z (height) and local Z becomes world Y (depth), while local X stays as world X.

### Issue 2: Boxes too narrow for items

**Symptom**: Items placed at marker positions inside boxes extend past the box walls. In simulation, items bounce off walls or land outside the box.

**General rule**: When designing boxes/containers for a 2×2 or NxM grid of items, verify that the largest item at the most extreme grid position still fits within the box walls. Formula: `box_inner_half > abs(marker_offset) + item_half_width + margin`.

### Issue 3: Transport height too low — carried items collide with other objects

**Symptom**: Robot successfully picks an item but collides with other tall items on the cart while moving horizontally toward the target. Items get knocked over, causing cascading failures.

**General rule**: When items are picked from a surface with other tall items nearby, verify that:
```
ee_height_for_move > tallest_obstacle_top + carried_item_rest_height + safety_margin
```
The default 0.3m move height works for most tasks where picks are from a bin (low items) to a dropzone, but fails when picks are from surfaces with tall upright items.

### Issue 4: Box wall height vs. item height

**Symptom**: Increasing box wall height to contain tall items caused items to collide with box walls during transport just prior to the lowering phase of placement.

**General rule**: Box wall height is a tradeoff — tall enough for lateral containment but short enough that items can be lowered in without collision. For the current gripper and approach trajectory, walls should generally be shorter than the item height, not taller.

## Case Study: TableTaskShapeSortBoxes (Mar 2025)

Task: Sort randomly colored cubes and balls from the conveyor into two boxes on the cart (one per shape type).

### Issue 5: Round objects don't fit in box despite grid fitting

**Symptom**: 4 balls at 0.0515m scale placed in a 0.16m × 0.16m box bounced off each other and escaped. The 2×2 grid markers fit geometrically but balls couldn't physically coexist.

**General rule**: When placing round/bouncy objects into containers, account for physics settling — balls bounce and push each other. Either make the box larger, reduce object count, or reduce object size. As a rule of thumb, for N round objects in a box, ensure total object volume is well under 50% of box floor area, and use generous `z_tol` (0.03+) in verification for settling tolerance.

## Case Study: TableTask3b (Mar 2025)

Task: Pick balls from the bin and place them into gaps between discs arranged in a tight 3×4 grid. Balls nestle in pockets formed by 4 adjacent disc rims.

### Issue 6: Default `is_on_top` verification fails for pocket/nestled placements

**Symptom**: Balls visually placed correctly in pockets between discs, but `is_on_top` verification reports failures. The first balls placed fail while the last ones pass.

**General rule**: When objects rest in non-flat geometries (pockets, gaps between supports, nestled positions), the default `is_on_top` check (ball AABB bottom vs marker AABB top, z_tol=0.02m) is too strict. Physics settling shifts object Z relative to the thin hidden marker. Use a custom `spatial_check_fn` based on position proximity (XY distance to target + Z within expected range) instead of AABB-based `is_on_top`.

## Case Study: TableTaskConveyorTypeSort (Apr 2026)

Task: Sort items arriving one at a time on a moving conveyor into 3 type-specific collection boxes on the cart. Uses `IncrementalGenerationConfig` to spawn picks over time.

### Issue 7: Incremental-spawn strategy captured per-pick metadata at init time

**Symptom**: The first item placed correctly, then the task stalled. `CheckTargetAvailable: no target for '<pick>'` was logged for every subsequently-spawned pick. Every pick after the initial batch had no paired target even though the strategy's `pair_picks_with_targets()` covered all picks conceptually.

**General rule**: With `pick_incremental_config`, `create_strategy(picks, targets)` is called when only the initial batch exists. Any per-pick list (e.g. `source_types`) captured and sliced at that moment will be shorter than the pick list as new picks are added via `add_incremental_picks`. Infer per-pick attributes from the pick object itself (name prefix, semantic labels) inside `pair_picks_with_targets()` / `valid_targets_for_pick()`, so the strategy stays correct as `_pick_objs` grows.

### Issue 9: YCB mustard_bottle occasionally stalled on the moving conveyor

**Symptom**: Spawned `mustard_bottle` items sometimes stayed exactly at their spawn location while `cracker_box` and `sugar_box` items always moved with the belt. When a later-spawned item later collided with a stalled bottle and knocked it onto its side, the bottle then moved normally.

**General rule**: `PhysxSurfaceVelocityAPI` couples through friction only where the object makes real contact with the kinematic surface prim. YCB mustard bottles have a concave base, so upright they contact a thin kinematic belt surface only on their outer rim. A small spawn hover + a narrow rim means landings are sometimes marginal (near-zero contact depth/normal force) and friction is too low to transfer belt velocity. The bottle sits still on a belt that's sliding under it. On its side, the large cylindrical contact patch restores friction and the belt drags the bottle.

**Fix**: (a) Spawn items at `surface_z + rest_height` (no hover) so they start in contact rather than dropping onto the belt. (b) Belt-and-suspenders: seed each newly-spawned pick with initial linear velocity `(0, conveyor_speed, 0)` via `SingleRigidPrim(...).set_linear_velocity(...)` in the task's `pre_step`, so marginal-contact items still move immediately after spawn. Retrying each tick handles rigid-body-view initialization timing for items spawned in `set_up_scene`.

### Issue 8: Multi-occupancy containment with a single per-type target placed all items at the same point

**Symptom**: With `containment_check=True` and a single stand-in target per type, every item of the same type landed at the identical XY position (on top of each other). The BT reads the placement position from the paired target prim, so all picks paired to the same target drop at the same spot.

**General rule**: `containment_check=True` enables multi-pick-to-same-target pairing for *verification* (the verifier tests box containment via `box_verification_info`), but the BT still uses the paired target's prim position for placement. To spread placements across distinct slots, emit one marker per placement slot (e.g. a fixed row of N markers per box) and pair each pick to the next unused same-type marker. Keep `containment_check=True` for verification — markers only drive placement; box geometry verifies success.

## Case Study: TableTaskSoupCans2 (Apr 2026)

Task: Soup cans picked from the bin and placed onto thin red rectangles that arrive in randomized 1-2 item bursts on a moving conveyor.

### Issue 10: Static-primitive targets do not move with the belt

**Symptom**: Target rectangles spawned with `asset_type="rect"` (→ `FixedCuboid`) appear at the correct spawn location but remain stationary on the moving conveyor. Picks can still be placed onto them in teleport mode, but the intended moving-target behavior is absent.

**General rule**: `PhysxSurfaceVelocityAPI` only drags dynamic rigid bodies. `FixedCuboid` / `VisualCuboid` primitives have no RigidBody / MassAPI, so the belt's surface velocity cannot act on them. For any task where a target, marker, or container is expected to be carried by a moving conveyor, spawn it as a **dynamic** primitive (`asset_type="cube"` / `"disc"` / `"ball"` → `DynamicCuboid` / `DynamicCylinder` / `DynamicSphere`) and give it a non-paper-thin thickness (≥ ~1 cm) for stable rigid-body contact. The usual conveyor-spawn hygiene from Issue 9 still applies: seed `set_linear_velocity([0, conveyor_speed, 0])` in `pre_step` on the first tick after spawn so the belt couples immediately through friction.

### Issue 11: Burst-spawn spacing too tight for the slow default belt

**Symptom**: At `DEFAULT_CONVEYOR_SPEED = -0.015 m/s`, 4 s between bursts of 10 cm-long target rectangles made successive bursts overlap in Y because the belt only travels ~6 cm per interval.

**General rule**: For burst-spawned items on a moving conveyor, pick `batch_interval > item_length_along_Y / conveyor_speed`. A margin of ~50% above that floor (e.g. 6 s for 10 cm items at 1.5 cm/s = ~9 cm travel, just under the item length) keeps successive bursts visually tight without interpenetration. Going much higher makes the scene feel empty; much lower and items pile up at the spawn point.

### Issue 12: Burst size exceeded what the robot could service at the chosen interval

**Symptom**: With bursts of up to 3 targets arriving every 6 s and a pick-and-place cycle longer than 6/3 = 2 s/item, the robot fell progressively further behind and targets queued up on the conveyor before the robot reached them.

**General rule**: Burst size must be compatible with the *per-item* service rate. Rule of thumb: `max_burst × per_item_service_time ≤ spawn_interval`. If the robot's cycle time is about 4-5 s/item, a 6 s spawn interval can sustain a max burst of 1 comfortably (1 item × 5 s ≤ 6 s) or 2 with queueing (2 × 5 s = 10 s > 6 s but recoverable over a few idle bursts); 3 is too many. For 3-slot row layouts, randomly filling 1-2 of the 3 slots keeps the visual "multi-lane" character without starving service capacity.

## Case Study: TableTaskCrackerCircleMarkers (May 2026)

Task: Pick upright cracker boxes from the bin and place them onto colored disc markers arranged in a circle on the dropzone.

### Issue 13: `disc` (DynamicCylinder) scale_xy acts as radius, not diameter

**Symptom**: Discs sized by an "intended diameter" appeared roughly twice as large as expected and overlapped each other on the dropzone (e.g. `scale=[0.15, 0.15, ...]` produced a ~0.30 m wide disc, larger than the 0.255 m chord between adjacent positions on a `radius=0.18 m`, `count=4` circle).

**General rule**: `DynamicCylinder` (asset types `"disc"` and `"cylinder"`) is created with default `radius=1.0` and `height=1.0`, so the `scale` argument multiplies those — `scale_xy` ends up acting as the **radius**, not the diameter, and `scale_z` is the full height. To get a target diameter `D`, pass `scale_xy = D / 2`. The same logic applies to height: `scale_z = thickness`. When laying out discs on a circle of radius `r` with `count=n`, the chord between adjacent positions is `2 * r * sin(π/n)`; keep `D < chord` for no overlap (e.g. `r=0.18, n=4` → chord ≈ 0.255 m).

## Case Study: TableTaskGreenCubesRowToYellowGrid (May 2026)

Task: Pick 6 green cubes pre-arranged in a single row on the cart and place them onto yellow rectangle markers in a 2x3 grid on the dropzone.

### Issue 14: Default cart decoration props collide with custom cart-spawned picks

**Symptom**: A green cube spawned at the configured row position on the cart intersected the default `cracker_box` placed by `setup_two_tables` (one of the four `standard_objs`), visible as overlap in the task-start snapshot.

**General rule**: When a task spawns pick items on the cart surface (rather than inside the picking bin), the default `setup_two_tables(...)` call adds four decorative YCB props (`cracker_box`, `sugar_box`, `soup_can`, `mustard_bottle`) and a `KLT_Bin` to the cart top, which will occupy the same workspace and likely collide with the custom layout. Pass `standard_objs=False` and `add_bin=False` to the `setup_workspace` lambda so the cart starts empty — keep them only if the bin/props are genuinely part of the task scenario.

## Case Study: TableTaskSugarBoxesRowToCircle (May 2026)

Task: Pick 6 upright sugar boxes from a row in the bin and place them onto a circle of 6 markers on the dropzone.

### Issue 15: Default to hidden virtual targets unless the user specifies visible target geometry

**Symptom**: User asked for items to be arranged in a circle on the dropzone. First implementation spawned 6 visible white rectangle markers (modelled after `TableTaskCrackerBoxes1`), then the user immediately asked to make them invisible. A round trip of re-implementation + re-verification could have been avoided by defaulting to hidden virtual targets from the start.

**General rule**: When the user describes *where* items should go (an arrangement: "in a circle", "in a 2×3 grid", "on the dropzone") without describing what the *targets themselves* should look like (no color, no thickness, no asset type, no "use red discs", etc.), default to hidden virtual targets via `TaskImplementationSpec.virtual_target_generation_strategy` with `hidden_strategy=FixedValue(True)` (and omit `target_generation_strategy` from the `TaskSpec`). The placement positions are still well-defined and verifiable; the dropzone surface just stays uncluttered. Reach for visible `target_generation_strategy` only when the user explicitly names target appearance (color, size, asset/shape — "yellow rectangles", "colored disc markers") or otherwise asks for a visible cue at each drop spot.

## Case Study: TableTaskSoupCansDiscs1 (May 2026)

Task: Pick soup cans from the bin and place them onto colored disc markers in a 2x3 grid on the dropzone.

### Issue 16: Dynamic-primitive markers jitter under items resting on them

**Symptom**: Soup cans placed onto `"disc"` (`DynamicCylinder`) target markers on the static dropzone vibrated continuously after placement. Verification still passed, but the live GUI showed visible jitter that thickness tweaks (2 mm → 1 cm) reduced without eliminating.

**General rule**: For target markers that should stay still throughout the task (decorative discs, rectangles, pads on a static dropzone or cart top), default to the **static** primitive variant — `"fixed_disc"` → `FixedCylinder`, `"rect"` → `FixedCuboid`. Reserve the dynamic variants (`"disc"`, `"cylinder"`, `"cube"`, `"ball"`) only for cases where the target itself must respond to physics, e.g. ride a moving conveyor (see Issue 10) or be jostled by other objects. A dynamic primitive squeezed between two rigid bodies (a placed item above, kinematic table below) gets caught in a contact-resolution loop driven by uneven surface contact and physics-material mismatch; the disc is the wobble source and the placed item amplifies it visually. No thickness or material tweak fully eliminates this — switch to the Fixed variant instead.

## Case Study: TableTaskCrackerStacksToMarkers (May 2026)

Task: Unstack 18 horizontal cracker boxes from a 3-layer 2×3 footprint on the dropzone and restack them as three 6-high horizontal stacks on three green markers in a row on the cart.

### Issue 17: Tall horizontal stacks of YCB cracker boxes lean and break AABB-based `is_on_top`

**Symptom**: Headless self-check verification failed for the top box of each stack with `is_on_top FAIL ... z_diff=-0.0467 (tol=0.0400, z_ok=False)`. Each box's AABB Z-extent measured ~0.12 m instead of the expected 0.072 m (the box's local Z half-extent × 2), so the upper box's AABB-bottom dipped below the lower box's AABB-top even though the stack was physically intact in the snapshot.

**General rule**: When stacking YCB asset boxes (cracker_box, sugar_box) on their largest face, every layer adds a small (~10–15°) lean because the asset's contact surfaces aren't perfectly flat. The lean inflates each box's world-frame AABB along Z by ~`0.5 × long_dim × sin(tilt)`, and a strict `is_on_top` check sees that inflation as overlap. For stacks of 3+ horizontal YCB boxes, use a generous `z_tol` (≥ 0.07 m for cracker boxes) and a relaxed `is_horizontal` `max_tilt_deg` (≥ 25–30°), or write a position-based verifier that ignores AABBs entirely.

### Issue 18: Centered cart-marker layout placed the far stack outside the UR10's comfortable reach

**Symptom**: All verification checks passed in `--teleport` mode (which skips motion planning), but the user reported in full-sim testing that the robot struggled to reach the third (furthest +Y) stack — motion planning would fail or take excessively long, even though the marker was nominally within the kinematic envelope.

**General rule**: When placing target stacks on the cart, default to the **+X edge** of the cart (the side closest to the UR10 base), not the cart center. The cart's far-X half is at the edge of the UR10's reachable workspace and motion planning becomes brittle there. Concretely: shift marker X by `~+0.20 m` from `CART_SURFACE_CENTER[0]` (cart half-width is 0.35 m, so a 0.22 m offset still leaves ~5 cm clearance to the cart edge for a 0.22 m-wide marker). Also consider **stack-build order**: for a row of stacks, build the furthest-from-robot stack first while the dropzone is densest and the robot's path to it is least obstructed by already-built stacks. Reversing the marker order in `GridPositionGenerator` via a negative `spacing_y` is the simplest way to achieve this with `LayeredStackStrategy` (which fills stacks in `target_objs` order).

## Case Study: TableTaskConveyorColorRows (May 2026)

Task: Pick colored cubes (3–5 of each of red/green/blue, randomly interleaved) arriving on a slowly-moving conveyor — 5 spawned initially in a row, the rest replenished one at a time at the +Y feed point — and place each onto a matching-color rectangular marker arranged in 3 color rows of 5 on the cart, filling each row from +Y to -Y.

### Issue 19: `TaskSpec.conveyor_speed` alone does not move the belt

**Symptom**: Full-sim interactive run showed the conveyor surface velocity API never engaging — cubes stayed exactly where they spawned even though `TaskSpec.conveyor_speed = DEFAULT_CONVEYOR_SPEED` was set. Mock and headless `--teleport` runs both passed without revealing the bug (mock has no physics, `--teleport` skips motion planning and teleports items off the belt before they would have drifted).

**General rule**: `TaskSpec.conveyor_speed` is consumed by the *spawner / scheduler* gate ("is the belt moving so spatial-trigger replenishment can fire?") and by *falloff auto-enable*. It does NOT itself apply `PhysxSurfaceVelocityAPI` to the conveyor surface prim — that happens only in `setup_two_tables(scene, assets_root_path, ..., conveyor_speed=...)`. Tasks that want a physically moving belt must pass the same value through both: TaskSpec.conveyor_speed (logic side) AND the `setup_workspace` lambda's `setup_two_tables(conveyor_speed=...)` kwarg (physics side). Forgetting the second one is silent under mock and `--teleport`; only full-sim catches it.

### Issue 20: Default 9-phase BT has no pick-reachability gate; cubes falling off the belt keep getting selected

**Symptom**: In a full-sim conveyor task, after a cube drifted past `CONVEYOR_END_Y` and fell to the floor, the robot kept trying to plan motions toward it (or toward whatever Y value its strategy still saw as "closest"), churning through retries instead of moving on.

**General rule**: Setting `TaskImplementationSpec.pick_min_reachable_z` has no effect under the default 9-phase tree (`make_task_controller_tree`) — only the cortex-style tree (`make_cortex_task_controller_tree`) wires up `CheckPickReachable` + `IsPickReachableGuard`, which mark items below the Z floor as `permanently_unreachable` and short-circuit the retry loop. For any conveyor task where unpicked items can drop off the edge, set `tree_factory=make_cortex_task_controller_tree` on the `TaskImplementationSpec`. The standard `MultiPickStrategy._permanently_unreachable_picks` set is honored by every existing strategy's pick-iteration logic, so the strategy automatically stops returning fallen items.

### Issue 21: Conveyor + color-matching wants pick-side proximity JIT and target-side color-match JIT — neither built-in strategy does both

**Symptom**: When the user specified "robot picks the cubes approaching the end of the conveyor" + "places them onto matching-color markers, filling each row from +Y to -Y", default `ColorMatchStrategy` honored the colors but iterated picks in spawn order — so a freshly-replenished cube at the +Y feed point could be selected ahead of an older cube about to fall off. The mirror-image `ConveyorProximityStrategy` does proximity-based selection but on the *target* side and assumes a single sequential pairing.

**General rule**: For a conveyor pick + cart-color-match task, subclass `ColorMatchStrategy` with the following overrides, modelled on `ConveyorProximityStrategy`'s latch pattern but reversed:

- `get_current_pick_name` → scan all uncompleted picks for the smallest world Y (closest to `-Y` belt edge), latch the choice in `_active_pick_name` until completion or advance.
- `advance_pick_index` → clear the latch, re-scan, and only consume a cursor slot when a candidate exists (see Issue 21).
- `get_placing_target_name` → JIT-assign the first unused matching-color target in `_target_objs` list order; latch in `_latched_target_by_pick` for mid-place stability.
- `latch_current_pick` / `clear_pick_latch` / `latch_current_target` / `clear_target_latch` → wire up the cortex-tree post-grasp / pre-place latch hooks (no-ops in the default 9-phase tree but needed when switching to the cortex tree per Issue 20).
- `add_incremental_picks` → extend `_pick_objs` and append new names to `_picking_order_item_names` directly; do NOT call `recompute_pairings` (which would clobber the JIT pairing state, same as `ConveyorProximityStrategy.add_incremental_targets`).
- `_has_target` → query `_first_unused_matching_target` rather than `_pairings_by_pick_name`, so picks aren't falsely skipped while their JIT target is still being assigned.

Combined with target list pre-sorted +Y → -Y within each color group, this gives "closest-to-edge cube next, dropped onto the +Y-most empty matching-color slot" in a few dozen lines.

## More details
See [lessons-learned-details.md](lessons-learned-details.md) for full analysis including root causes, fix details, and clearance calculations.
