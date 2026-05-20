# TableTask3v2

## User Request

Pick balls from the bin and place them onto disc targets arranged in a 3x4 grid on the dropzone table.

## Task Overview

Balls (`ball` asset type) are picked from a 3×3 grid in the pick bin and placed onto colored disc markers arranged in a 3×4 grid on the dropzone. Disc colors cycle through purple, cyan, black, yellow using `SequentialChoice`. Pairing is sequential (default).

## Concise Task Description

Pick balls from the bin and place them onto disc targets arranged in a 3x4 grid on the dropzone table.

## Pick Items

- **Type**: ball
- **Arrangement**: 3×3 grid in the pick bin (spacing_x=0.08m, spacing_y=0.08m)
- **Count**: 9
- **Color/Appearance**: Random (default)

## Target Objects

- **Type**: disc (marker)
- **Arrangement**: 3×4 grid on the dropzone (spacing_x=-0.15m, spacing_y=0.15m)
- **Markers**: N/A (targets are visible disc markers)
- **Color/Appearance**: `SequentialChoice(["purple", "cyan", "black", "yellow"], loop=True)`

## PickPlace Pairing and Sequencing

Sequential (default): pick[0] → target[0], pick[1] → target[1], etc.

## Success Condition

All 9 balls are placed on top of their corresponding disc targets in the dropzone.

## Success Checks

- Each pick object rests on top of its paired target disc (`is_on_top` — default).
