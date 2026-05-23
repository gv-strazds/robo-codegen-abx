# TableTaskCrackerSoupSort

## User Request

> create a new task: Pick a mix of cracker boxes and soup cans - 4 of each type randomly interleaved in one line on the conveyor - and sort them by type into two boxes on the cart (cracker boxes in the left box, soup cans in the right box). Both item types must be placed vertically. The conveyor is stationary. Both boxes are the same size.

## Task Overview

Pick 4 `cracker_box` and 4 `soup_can` USD assets that are randomly interleaved in a single line on the stationary conveyor, and sort them by type into two same-size open-top boxes on the cart: cracker boxes go into the **left** box (negative X from cart center), soup cans into the **right** box (positive X from cart center). Both item types must remain upright (vertical) after placement. The pairing logic uses `TypeBasedStrategy` with default name-prefix detection (`cracker_box_*` → left-box markers `0–3`; `soup_can_*` → right-box markers `4–7`). Verification combines `containment_check=True` + per-box `match_labels` filtering + a `placement_constraints_fn` that calls `is_vertical(max_tilt_deg=15)` on every placed item.

## Concise Task Description

Pick 4 cracker boxes and 4 soup cans that are randomly interleaved in a line on the stationary conveyor and sort them by type into two same-size open-top boxes on the cart (cracker boxes go in the left box, soup cans in the right box), keeping every item upright.

## Pick Items

- **Type**: USD assets `cracker_box` (×4) and `soup_can` (×4)
- **Arrangement**: Single line along the conveyor's Y axis (`ConveyorPositionGenerator`) at `DROPZONE_CENTER_POINT` with spacing `0.10 m` and small XY jitter. Order is randomly shuffled per seed.
- **Count**: 8 total (4 cracker_box + 4 soup_can), fixed.
- **Color/Appearance**: USD asset default. Spawned upright via `-90° X` rotation (`Gf.Rotation(Gf.Vec3d(1, 0, 0), -90)`), so each asset's local up-axis aligns with world +Z.

## Target Objects

- **Type**: Two open-top open boxes on the cart, built from `FixedCuboid` walls + base via `spawn_open_box`. Both boxes have **identical** dimensions (per user requirement).
- **Arrangement**: Two boxes side-by-side along the cart's short (X) axis, centered at the cart surface Y center.
  - Left box (cracker_box target): `CART_SURFACE_CENTER + [-0.18, 0, …]`
  - Right box (soup_can target): `CART_SURFACE_CENTER + [+0.18, 0, …]`
  - `inner_size = [0.20, 0.40]`, `wall_height = 0.10`, `wall_thickness = 0.01`, `base_thickness = 0.01`
- **Markers**: 4 hidden virtual markers per box (8 total), laid out in a 1×4 line along the cart's Y axis at the box floor: marker Y offsets `[-0.135, -0.045, +0.045, +0.135]` from the box center, X at box center, Z = `cart_z + base_thickness + 0.001`. Markers are `hidden=True`, not visible in the scene (per Issue 15: hidden when user didn't specify visible markers).
- **Color/Appearance**: Left box brown `[0.50, 0.40, 0.30]`; right box tan `[0.65, 0.55, 0.40]`. Markers hidden.

## PickPlace Pairing and Sequencing

- **Pairing**: `TypeBasedStrategy` with `target_indices_by_type={"cracker_box": [0,1,2,3], "soup_can": [4,5,6,7]}`. No explicit `source_types` is passed — the strategy auto-detects each pick's type via name-prefix matching against the keys (longest first), so `cracker_box_0..3` and `soup_can_0..3` resolve correctly even after the pick list is shuffled.
- **Sequencing**: Picks are visited in the order they were spawned (the interleaved/shuffled order on the conveyor). For each pick, the strategy consumes the next unused target index in that type's bucket, so each box fills 0→3 regardless of which order the types come up.

## Success Condition

All 8 items are placed inside their type-matched box on the cart and remain upright (≤15° tilt from world +Z).

## Success Checks

- Each pick object is inside the box that has matching `match_labels` (`{"type": "cracker_box"}` left, `{"type": "soup_can"}` right) — `containment_check=True` + `box_verification_info={"box_specs": box_specs}`.
- Each placed item satisfies `is_vertical(pick_obj, max_tilt_deg=15)` — checked through `placement_constraints_fn`.
- Hidden virtual markers spread placements across distinct slots inside each box (per Issue 8 — one marker per placement slot, not per box).
