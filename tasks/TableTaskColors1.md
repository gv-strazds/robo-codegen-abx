# TableTaskColors1

## User Request

Pick colored cubes from the bin and place them onto matching colored markers in the dropzone.

## Task Overview

Colored cubes (`cube` asset type) with random red/green/blue colors are picked from a 4×3 grid in the pick bin and placed onto color-matched target cubes in a 3×4 grid on the dropzone. Target colors cycle through red, cyan, yellow, green, blue, magenta — only red, green, and blue targets can receive picks. Uses `ColorMatchStrategy` with an extended palette. Picks without a matching available target are skipped.

## Concise Task Description

Pick colored cubes from the bin and place them onto matching colored markers in the dropzone.

## Pick Items

- **Type**: cube
- **Arrangement**: 4×3 grid in the pick bin (spacing_x=0.08m, spacing_y=0.08m)
- **Count**: 12
- **Color/Appearance**: `RandomChoice(["red", "green", "blue"])`

## Target Objects

- **Type**: cube (visible target cubes)
- **Arrangement**: 3×4 grid on the dropzone (spacing_x=-0.15m, spacing_y=0.15m)
- **Markers**: N/A (targets are visible colored cubes)
- **Color/Appearance**: `SequentialChoice(["red", "cyan", "yellow", "green", "blue", "magenta"], loop=True)`

## PickPlace Pairing and Sequencing

- **Strategy**: `ColorMatchStrategy` with `color_palette=["red", "green", "blue", "yellow"]`
- Each pick cube is paired to an available target of the same color
- Target colors cyan and magenta do not match any pick colors and remain unused
- Picks without a matching available target get no target and are skipped

## Success Condition

All paired cubes are placed on their color-matched target cubes.

## Success Checks

- Each paired pick object rests on top of its color-matched target cube (`is_on_top` — default).
