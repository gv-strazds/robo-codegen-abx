# TableTaskCrackerBoxes1

## User Request

Pick cracker boxes from the bin and place them onto thin green rectangles arranged in a 3x4 grid in the dropzone.

## Task Overview

Cracker boxes (`cracker_box` USD asset) are picked from a 4×1 grid in the pick bin and placed onto thin green rectangle markers arranged in a 3×4 grid on the dropzone. The target grid is spaced widely enough along X (spacing_x=-0.18m) that three upright cracker boxes can sit comfortably side-by-side on adjacent column targets. Cracker boxes spawn upright (-90° X rotation). Pairing is sequential (default). Verification checks both marker placement (`is_on_top`) and vertical orientation (`is_vertical`).

## Concise Task Description

Pick cracker boxes from the bin and place them onto thin green rectangles arranged in a 3x4 grid in the dropzone.

## Pick Items

- **Type**: cracker_box (USD asset)
- **Arrangement**: 4×1 grid in the pick bin (4 rows, 1 column; spacing_x=0.08m, spacing_y=0.08m)
- **Count**: 4
- **Orientation**: Upright (-90° X rotation)
- **Color/Appearance**: USD asset default

## Target Objects

- **Type**: rect (thin rectangle, height=0.002m, scale=[0.1m, 0.1m, 0.002m])
- **Arrangement**: 3×4 grid on the dropzone (spacing_x=-0.18m, spacing_y=0.15m). The wider X spacing leaves ~1.6cm clearance between adjacent upright cracker boxes (footprint ~0.164m × 0.072m world XY) so three can sit side-by-side on a single row.
- **Markers**: N/A (targets are visible green rectangles)
- **Color/Appearance**: Green

## PickPlace Pairing and Sequencing

Sequential (default): pick[0] → target[0], pick[1] → target[1], etc. Only 4 of 12 targets will be used.

## Success Condition

All 4 cracker boxes are placed upright on their corresponding green rectangle targets in the dropzone.

## Success Checks

- Each cracker box rests on top of its paired target rectangle (`is_on_top`).
- Each placed cracker box is vertical (`is_vertical`, max tilt 15° from world +Z).
