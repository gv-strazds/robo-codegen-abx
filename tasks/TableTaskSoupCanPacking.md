# TableTaskSoupCanPacking

## User Request

Pick soup cans from the conveyor and pack them into 4 boxes on the cart.

## Task Overview

This task requires the robot to pick 24 soup cans arranged in 4 rows of 6 on the conveyor (drop zone) and place 6 soup cans into each of 4 boxes situated on the cart. Each box holds a 2x3 grid of cans. The boxes are arranged in a 2x2 layout on the cart surface. Hidden markers inside each box guide the placement positions. Default sequential pairing fills boxes one at a time (first 6 cans into box 1, next 6 into box 2, etc.).

## Concise Task Description

Pick soup cans from the conveyor and place 6 into each of 4 boxes on the cart.

## Pick Items

- **Type**: soup_can (USD asset)
- **Arrangement**: 4 rows of 6 cans on the conveyor (drop zone), spaced along Y axis. Rows spaced along X axis. Spawn orientation: -90 deg X (upright in world).
- **Count**: 24
- **Color/Appearance**: USD asset default

## Target Objects

- **Type**: 4 collection boxes on the cart, each containing 6 hidden markers
- **Arrangement**: Boxes in a 2x2 grid on the cart surface. Inside each box, 6 markers in a 2x3 grid (2 columns along X, 3 rows along Y).
- **Markers**: Hidden marker objects (asset_type "marker") at box floor level, one per soup can.
- **Color/Appearance**: Cardboard-colored boxes; hidden markers.

## PickPlace Pairing and Sequencing

- Sequential pairing: pick[0] -> target[0], pick[1] -> target[1], etc.
- Targets are ordered box-by-box: targets 0-5 in box 1, targets 6-11 in box 2, targets 12-17 in box 3, targets 18-23 in box 4.
- Picks are ordered row-by-row from the conveyor: row 0 items 0-5, row 1 items 6-11, etc.

## Success Condition

All 24 soup cans are placed upright into the 4 boxes on the cart, 6 per box.

## Success Checks

- `is_within_box_geometry` (containment mode): each soup can must be inside its target box, verified using per-box dimensions (`inner_size`, `height`) stored in each box spec dict.
- `is_vertical` (placement constraint): each placed soup can must be upright.
