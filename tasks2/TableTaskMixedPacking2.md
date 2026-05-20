# TableTaskMixedPacking2

## User Request

Pick Cracker Boxes and Soup Cans from the conveyor area and place one Cracker Box and four Soup Cans into each of two boxes on the cart. The Soup Cans should be arranged in a 2x2 grid within the boxes.

## Task Overview

Cracker boxes (`cracker_box` USD asset) and soup cans (`soup_can` USD asset) are picked from interleaved columns on the conveyor/dropzone and placed into 2 boxes on the cart. Each box receives 1 cracker box and 4 soup cans (cans in a 2×2 grid). Pick items are arranged in columns: 1 column of cracker boxes and 3 columns of soup cans. The pick order is interleaved (1 box then 4 cans, repeating) so sequential pairing fills each box in order. Both item types spawn upright (-90° X rotation). Verification checks both marker placement and vertical orientation.

## Concise Task Description

Pick Cracker Boxes and Soup Cans from the conveyor and place one Cracker Box and four Soup Cans into each of two boxes on the cart.

## Pick Items

- **Types**: cracker_box, soup_can (USD assets)
- **Count**: 7 cracker boxes (1 column) + 27 soup cans (3 columns of 9), interleaved pick order: 1 box then 4 cans per set
- **Arrangement**: Interleaved columns on the conveyor/dropzone; box column separated by 0.20m gap from can columns; can columns spaced 0.08m apart
- **Orientation**: Upright (-90° X rotation for both types)
- **Color/Appearance**: USD asset default

## Target Objects

- **Type**: Hidden rectangular markers inside 2 boxes on the cart
- **Arrangement**: 2 boxes on the cart; each box has 5 markers (1 green for cracker box, 4 red for soup cans in a 2×2 grid)
- **Markers**: Thin rectangles (0.08m × 0.08m × 0.002m), hidden
- **Color/Appearance**: Green markers for cracker box slots, Red markers for soup can slots

## PickPlace Pairing and Sequencing

- **Strategy**: Sequential (default). Pick order is interleaved (box, can, can, can, can, box, can, ...) and markers are ordered per-box (box_marker, can_markers × 4), so sequential pairing naturally fills each box with 1 cracker box + 4 soup cans.
- Only 2 boxes' worth of items are paired (2 cracker boxes + 8 soup cans = 10 items); remaining items on the conveyor are skipped.

## Success Condition

Each of the two cart boxes contains one cracker box and four soup cans, all placed vertically.

## Success Checks

- `is_within_box_geometry` (containment mode): each picked item must be inside its target box, verified using per-box dimensions (`inner_size`, `height`) stored in each box spec dict.
- `is_vertical` (placement constraint): each placed item must be upright.
