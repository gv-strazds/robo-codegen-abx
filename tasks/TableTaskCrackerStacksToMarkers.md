# TableTaskCrackerStacksToMarkers

## User Request

> Cracker boxes are pre-stacked (lying face up) in a 3-layer 2x3 footprint on the dropzone. Unstack them and stack them (still horizontal) in three stacks, one on each of three green-square markers in a row on the cart.

## Task Overview

Pick 18 cracker boxes (YCB `cracker_box` asset, identity orientation = lying face up) from a 2×3 grid stacked 3 layers high on the dropzone, and redistribute them as three horizontal 6-high stacks — one stack on each of three visible green square markers (`rect` primitives, color `green`) arranged in a row along the Y axis on the cart surface. Pairing uses `LayeredStackStrategy` with `max_stacks=3` and a custom spawn-index-based `classify_fn` that maps source top→destination bottom (so the first pick exposes the next, and the stack-by-stack ordering is consistent with the pre-stacked source).

## Concise Task Description

Unstack 18 cracker boxes from a 3-layer 2×3 footprint on the dropzone and restack them as three 6-high horizontal stacks on three green square markers in a row on the cart.

## Pick Items

- **Asset type**: `cracker_box` (YCB USD asset)
- **Count**: 18 (6 per layer × 3 layers)
- **Arrangement**: 2×3 grid (2 cols X × 3 rows Y) on the dropzone, stacked 3 layers high
  - `spacing_x = 0.20 m` (~3.6 cm gap between adjacent boxes)
  - `spacing_y = 0.218 m` (~0.5 cm gap; tight fit, slight overhang of dropzone friction region)
  - `layer_height = 0.074 m` (box Z thickness 0.072 m + 2 mm margin)
- **Orientation**: Identity quaternion (lying face up — smallest local extent along world Z)
- **Colors**: USD asset default (red)

## Target Objects

- **Type**: Visible `rect` markers (FixedCuboid — static, no jitter per Issue 16)
- **Count**: 3
- **Color**: `green`
- **Scale**: `[0.22, 0.17, 0.005] m` (slightly larger than the horizontal cracker box footprint of 0.164 × 0.213 m, thin)
- **Arrangement**: Row along Y on the cart surface, at `CART_SURFACE_CENTER` X, spaced 0.30 m apart in Y
  - Y positions: `cart_y - 0.30`, `cart_y`, `cart_y + 0.30`
  - Z just above the settled cart surface (~`cart_z + 0.003`)

## PickPlace Pairing and Sequencing

- **Strategy**: `LayeredStackStrategy(layer_order=["lvl_0"..."lvl_5"], max_stacks=3, classify_fn=...)`
- **classify_fn**: parses spawn index from obj name (`cracker_box_0`..`cracker_box_17`); returns `f"lvl_{(2 - idx // 6) * 2 + (idx % 6) // 3}"`
  - Source top layer (idx 12–17) → dest `lvl_0` and `lvl_1` (bottom of destination)
  - Source middle layer (idx 6–11) → dest `lvl_2` and `lvl_3`
  - Source bottom layer (idx 0–5) → dest `lvl_4` and `lvl_5` (top of destination)
- **stacking_enabled=True**: with the classify mapping above, the strategy's stack-by-stack pick order (per stack: lvl_0 first, lvl_5 last) is also top-of-source first, satisfying both source unstacking and destination bottom-up stacking simultaneously
- **Destination stacking**: `LayeredStackStrategy._extend_target_objs()` extends the target list per layer — each placed cracker box becomes the target for the next layer's pick on that stack

## Success Condition

All 18 cracker boxes are picked from the dropzone and placed into three horizontal 6-high stacks (still lying face up) on the three green square markers.

## Success Checks

1. Each box is on top of its paired target (`is_on_top` with `z_tol=0.04` — generous for accumulated stack settling).
2. Each placed box is lying flat (`is_horizontal` with `max_tilt_deg=20°`).
3. Both checks combined apply to base-layer picks (cracker box on green marker) and to upper-layer picks (cracker box on cracker box).

## Implementation Notes

- `ee_height_for_move = 0.65 m` (world Z) — must clear the tallest stack (~0.49 m world Z) + carried box height (~0.07 m) + margin (~0.07 m).
- `startup_delay_seconds = 1.0` — lets the pre-stacked source settle before the BT begins ticking.
- `setup_two_tables(scene, assets_root, standard_objs=False, add_bin=False)` — cart starts empty so the standard YCB props and KLT bin don't collide with the marker stacks (per Issue 14).
- No drop-orientation override: picks are already face-up at identity orientation; the default behavior preserves EE orientation through transport so boxes stay face-up. If visual review shows tilting, an identity-quaternion override on `get_end_effector_orientation_for_drop` can be added.
- Targets are visible (not hidden virtual markers) per Issue 15 because the user explicitly named appearance ("green-square markers").
