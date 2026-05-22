# TableTaskColoredBallsToDiscGrid

## User Request

Pick small colored balls (randomly colored red, green, or blue) from the bin and place each one onto a disc target whose color matches the ball. Markers are arranged in a 3x3 grid on the dropzone. The balls should be small enough so that 6 can comfortably fit in the pick bin.

## Task Overview

A color-matching pick-and-place task:

- Pick `asset_type="ball"` (DynamicSphere) items, 6 in the picking bin, randomly colored red/green/blue.
- Target `asset_type="fixed_disc"` (FixedCylinder, static) discs arranged in a 3×3 grid on the dropzone, colored to give exactly 3 red / 3 green / 3 blue (color cycled across the 9 positions).
- Pairing via `ColorMatchStrategy(color_palette=["red","green","blue"])` so each ball goes to a same-color disc; unmatched picks (if a random seed yields more than 3 balls of a single color) are filtered out of the picking order by `ColorMatchStrategy.initialize_pairings()` and remain in the bin.
- Default `is_on_top` spatial verification; balls go directly on top of disc markers (Issue 6's pocket-placement caveat does not apply).

## Concise Task Description

Pick 6 small balls (randomly red/green/blue) from the bin and place each one onto the matching-color disc in a 3×3 grid of red/green/blue disc markers on the dropzone.

## Pick Items

- **Type**: `ball` (DynamicSphere primitive); scale `[0.025, 0.025, 0.025]` → radius 2.5 cm, diameter 5 cm.
- **Arrangement**: 3 rows × 2 cols grid in the picking bin (spacing 0.09 m in X, 0.10 m in Y, centered on `[BIN_X_COORD, BIN_Y_COORD]`).
- **Count**: 6.
- **Color/Appearance**: `RandomChoice(["red", "green", "blue"])` — colors assigned independently per ball at task setup.

## Target Objects

- **Type**: `fixed_disc` (FixedCylinder static primitive — chosen over dynamic `disc` to avoid jitter under placed balls per Issue 16); scale `[0.045, 0.045, 0.03]` → radius 4.5 cm, diameter 9 cm, thickness 3 cm. The 2 cm margin around the ball footprint lets the ball settle visibly centered with `is_on_top` passing.
- **Arrangement**: 3×3 grid on the dropzone, `dx=-0.12`, `dy=0.13`, centered on the dropzone reference corner using `DROPZONE_X`/`DROPZONE_Y` grid-centering math.
- **Markers**: The discs themselves are the visible targets (no separate marker layer).
- **Color/Appearance**: `SequentialChoice(["red", "green", "blue"], loop=True)` cycled across the 9 grid positions → R G B / R G B / R G B (3 of each color, interleaved across rows and columns).

## PickPlace Pairing and Sequencing

`ColorMatchStrategy` pairs each ball to the next unused same-color disc. The interleaved (rather than row-blocked) disc layout forces the strategy to do real color-search rather than degenerate to index-based pairing.

When a random seed produces 4 or more balls of one color (~20–25% chance in this random palette), the surplus ball is paired to `None` and `ColorMatchStrategy.initialize_pairings()` removes it from `_picking_order_item_names` entirely, so the robot never attempts to pick it. The remaining (matched) balls execute in their natural order.

## Success Condition

Every ball that was assigned a same-color disc by the strategy ends up resting on top of that disc; surplus-color balls (filtered out by the strategy) stay in the bin.

## Success Checks

- Each paired ball rests on its assigned same-color disc (default `is_on_top` spatial check, `z_tol=0.02 m` — sufficient for a 2 mm flat disc and a settled sphere).
- The verifier walks only the strategy's matched-pick list, so unmatched balls do not count as failures.
