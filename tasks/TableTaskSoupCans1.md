# TableTaskSoupCans1

## User Request

Pick soup cans from the bin and place them onto thin red rectangles arranged in a 3x4 grid in the dropzone.

## Task Overview

Soup cans (`soup_can` USD asset) are picked from a 3×3 grid in the pick bin and placed onto thin red rectangle markers in a 3×4 grid on the dropzone. Soup cans spawn upright (-90° X rotation). Pairing is sequential (default). Verification checks both marker placement (`is_on_top`) and vertical orientation (`is_vertical`).

## Concise Task Description

Pick soup cans from the bin and place them onto thin red rectangles arranged in a 3x4 grid in the dropzone.

## Pick Items

- **Type**: soup_can (USD asset)
- **Arrangement**: 3×3 grid in the pick bin (spacing_x=0.08m, spacing_y=0.08m)
- **Count**: 9
- **Orientation**: Upright (-90° X rotation)
- **Color/Appearance**: USD asset default

## Target Objects

- **Type**: rect (thin rectangle, height=0.002m, scale=[0.1m, 0.1m, 0.002m])
- **Arrangement**: 3×4 grid on the dropzone (spacing_x=-0.15m, spacing_y=0.15m)
- **Markers**: N/A (targets are visible red rectangles)
- **Color/Appearance**: Red

## PickPlace Pairing and Sequencing

Sequential (default): pick[0] → target[0], pick[1] → target[1], etc. Only 9 of 12 targets will be used.

## Success Condition

All 9 soup cans are placed upright on their corresponding red rectangle targets in the dropzone.

## Success Checks

- Each soup can rests on top of its paired target rectangle (`is_on_top`).
- Each placed soup can is vertical (`is_vertical`).
