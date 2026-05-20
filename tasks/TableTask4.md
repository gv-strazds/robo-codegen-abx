# TableTask4

## User Request

Pick cubes from the bin and place them onto yellow rectangles arranged in a circle on the dropzone table.

## Task Overview

Cubes (`cube` asset type) are picked from a 3×3 grid in the pick bin and placed onto thin yellow rectangle markers arranged in a circle (radius=0.2m, 7 positions) on the dropzone. Uses `CircularPositionGenerator` for target layout. Pairing is sequential (default).

## Concise Task Description

Pick cubes from the bin and place them onto yellow rectangles arranged in a circle on the dropzone table.

## Pick Items

- **Type**: cube
- **Arrangement**: 3×3 grid in the pick bin (spacing_x=0.08m, spacing_y=0.08m)
- **Count**: 9
- **Color/Appearance**: Random (default)

## Target Objects

- **Type**: rect (thin rectangle, height=0.002m)
- **Arrangement**: Circle on the dropzone (`CircularPositionGenerator`, radius=0.2m, 7 positions)
- **Markers**: N/A (targets are visible yellow rectangles)
- **Color/Appearance**: Yellow

## PickPlace Pairing and Sequencing

Sequential (default): pick[0] → target[0], pick[1] → target[1], etc. Only 7 of 9 cubes will be picked (fewer targets than picks).

## Success Condition

All paired cubes are placed on their corresponding yellow rectangle targets in the circular arrangement.

## Success Checks

- Each pick object rests on top of its paired target rectangle (`is_on_top` — default).
