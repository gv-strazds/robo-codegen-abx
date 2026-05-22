# TableTaskRedBlueCubesToGreenGrid

## User Request

> Pick cubes from two sources — the pick bin (3x3 grid of red cubes) and the cart surface (row of 3 blue cubes) — and place them onto square green markers in a 3x4 grid on the dropzone. The bin cubes are placed first, then the cart cubes.

## Task Overview

Mixed-source pick-and-place: 9 red cubes pre-arranged as a 3×3 grid in the pick bin, plus 3 blue cubes pre-arranged as a single row on the cart surface. The robot must place all 12 cubes onto a 3×4 grid of green square rectangle markers on the dropzone, picking the 9 bin cubes first and the 3 cart cubes last.

- **Pick asset type**: `cube` (primitive) — red for bin items, blue for cart items
- **Target asset type**: `rect` (static `FixedCuboid`) — green, 6 cm × 6 cm × 2 mm
- **Pick generator**: custom `_CombinedPickGenerator` class wrapping two `GridPositionGenerator` instances (one for the bin 3×3, one for the cart row of 3). Returns bin cubes first, then cart cubes, in a single deterministic list — pick order is therefore guaranteed by construction.
- **Target generator**: standard `ItemGenerator` with `GridPositionGenerator(rows=3, cols=4, randomize=False)` producing 12 markers on the dropzone.
- **Pairing strategy**: default sequential — `pick[i] → target[i]`, no custom strategy needed.

## Concise Task Description

Pick 9 red cubes from a 3×3 grid in the bin, then 3 blue cubes from a row on the cart, and place them onto 12 green square markers arranged in a 3×4 grid on the dropzone.

## Pick Items

- **Type**: `cube` (primitive)
- **Arrangement**:
  - **Bin source** — 3×3 grid centered at `(BIN_X_COORD, BIN_Y_COORD)` inside the pick bin, spacing 0.08 m × 0.08 m
  - **Cart source** — single row of 3 cubes along Y at `CART_SURFACE_CENTER[0] − 0.18` (offset in −X to keep clear of the bin), spacing 0.08 m
- **Count**: 12 (9 bin + 3 cart, fixed)
- **Color/Appearance**: 9 red bin cubes + 3 blue cart cubes (deterministic, assigned per source by the custom generator)
- **Cube size**: 0.0515 m edge

## Target Objects

- **Type**: Visible `rect` markers (static `FixedCuboid`)
- **Arrangement**: 3×4 grid (3 rows along Y × 4 columns along X) on the dropzone, spacing 0.13 m × 0.12 m (total footprint 0.26 × 0.36 m, centered on the dropzone)
- **Markers**: N/A — targets ARE the markers (no containers)
- **Color/Appearance**: Green, 6 cm × 6 cm × 2 mm thin rectangles

## PickPlace Pairing and Sequencing

- **Pairing**: Default sequential — `pick[i] → target[i]`.
- **Sequencing**: The custom `_CombinedPickGenerator` returns 9 bin cubes (indices 0–8) first, then 3 cart cubes (indices 9–11). Default sequential pairing therefore places all bin cubes before any cart cubes, satisfying the user's "bin cubes placed first, then cart cubes" requirement without a custom strategy.
- **Which marker gets which colour**: not specified by the user; the row-major iteration of the 3×4 target grid yields a deterministic but uncoordinated colour distribution. Acceptable per the request.

## Success Condition

All 12 cubes rest on top of their paired green markers in the dropzone grid.

## Success Checks

- Each pick object rests on its paired target (`is_on_top`, the default `spatial_check_fn`).
- No custom orientation or containment constraints — cubes are isotropic primitives on flat markers.
