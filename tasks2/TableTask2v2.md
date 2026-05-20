# TableTask2v2

## User Request

Pick multiple cubes and place them onto blue cubes arranged in a 3x4 grid in the dropzone.

## Task Overview

Colored cubes (`cube` asset type) are picked from a line of 7 cubes on the dropzone surface and placed onto blue target cubes arranged in a 3x4 grid on the dropzone. Pairing is sequential (default). Target cubes are blue with FixedValue color strategy.

## Concise Task Description

Pick multiple cubes and place them on blue cubes arranged in a 3x4 grid.

## Pick Items

- **Type**: cube
- **Arrangement**: Line of 7 cubes (1×7 grid) on the dropzone surface (center_x=0.4m)
- **Count**: 7
- **Color/Appearance**: Random (default)

## Target Objects

- **Type**: cube (blue)
- **Arrangement**: 3×4 grid on the dropzone (spacing_x=-0.15m, spacing_y=0.15m)
- **Markers**: N/A (targets are visible blue cubes)
- **Color/Appearance**: Blue (`FixedValue("blue")`)

## PickPlace Pairing and Sequencing

Sequential (default): pick[0] → target[0], pick[1] → target[1], etc.

## Success Condition

All 7 cubes are placed on top of blue target cubes in the dropzone grid.

## Success Checks

- Each pick object rests on top of its paired target cube (`is_on_top` — default).
