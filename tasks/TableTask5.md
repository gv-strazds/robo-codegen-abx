# TableTask5

## User Request

Pick green cubes from the dropzone table and place them onto red rectangles arranged in a circle.

## Task Overview

Green cubes (`cube` asset type) are picked from a 3×2 grid on the dropzone surface and placed onto red rectangle markers arranged in a circle (radius=0.18m, 7 positions) on the dropzone. This is a reversal of the typical bin-to-dropzone flow — both picks and targets are on the dropzone. Uses `CircularPositionGenerator` for target layout. Pairing is sequential (default).

## Concise Task Description

Pick green cubes from the dropzone table and place them onto red rectangles arranged in a circle.

## Pick Items

- **Type**: cube
- **Arrangement**: 3×2 grid on the dropzone surface (spacing_x=-0.15m, spacing_y=0.15m)
- **Count**: 6
- **Color/Appearance**: Green (`FixedValue("green")`)

## Target Objects

- **Type**: rect (taller than the pick cubes)
- **Arrangement**: Circle on the dropzone (`CircularPositionGenerator`, radius=0.18m, 7 positions)
- **Markers**: N/A (targets are visible red rectangles)
- **Color/Appearance**: Red

## PickPlace Pairing and Sequencing

Sequential (default): pick[0] → target[0], pick[1] → target[1], etc. Only 6 of 7 targets will be used (more targets than picks).

## Success Condition

All 6 green cubes are placed on their corresponding red rectangle targets in the circular arrangement.

## Success Checks

- Each pick object rests on top of its paired target rectangle (`is_on_top` — default).
