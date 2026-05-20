# TableTaskBottles1

## User Request

Pick bottles from the bin and place them into carrier pads in a 3x4 grid in the dropzone.

## Task Overview

Madara bottles (`madara_bottle` USD asset) are picked from a 4×2 grid in the pick bin and placed into madara carrier pads (`madara_pad` USD asset) arranged in an N×4 grid on the dropzone, where N (columns) is randomly chosen from 1–3 each run. Uses `BottlePickStrategy` for specialized bottle handling (drop orientation and EE offset computation). Bottles spawn upright (-90° X rotation). Verification checks both containment (`is_within`) and vertical orientation (`is_vertical`).

## Concise Task Description

Pick bottles from the bin and place them into carrier pads in a 3x4 grid in the dropzone.

## Pick Items

- **Type**: madara_bottle (USD asset)
- **Arrangement**: 4×2 grid in the pick bin (spacing_x=0.08m, spacing_y=0.15m)
- **Count**: 8
- **Orientation**: Upright (-90° X rotation)
- **Color/Appearance**: USD asset default

## Target Objects

- **Type**: madara_pad (USD asset)
- **Arrangement**: N×4 grid on the dropzone (N columns randomly chosen 1–3; spacing_x=-0.15m, spacing_y=0.15m)
- **Count**: N×4 (4–12)
- **Markers**: N/A (pads are physical carrier objects)
- **Color/Appearance**: USD asset default

## PickPlace Pairing and Sequencing

Sequential via `BottlePickStrategy`: pick[0] → target[0], pick[1] → target[1], etc. The strategy handles bottle-specific drop orientation (pi/2 X rotation) and EE offset computation from pick item geometry.

## Success Condition

All 8 bottles are placed upright into their corresponding carrier pads.

## Success Checks

- Each bottle is within its paired carrier pad (`is_within`).
- Each placed bottle is vertical (`is_vertical`).
