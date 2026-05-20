# TableTaskColorCircle2

## User Request

Pick colored cubes from the pick bin and place them in a circle on the drop zone. Cubes have randomized colors from red, green, blue, or magenta. Random number of cubes between 2 and 16. Circle markers should not be visible.

## Task Overview

Colored cubes (`cube` asset type) are picked from a 4×4 grid in the pick bin and placed onto hidden markers (`marker` asset type) arranged in a circle on the drop zone. The number of cubes is randomized between 2 and 16 each run. Colors are randomly chosen from red, green, blue, magenta. Pairing is sequential (default). Circle targets use `CircularPositionGenerator` with `randomize=False` for even spacing.

## Concise Task Description

Pick randomly colored cubes from the bin and place them in a circle on the drop zone.

## Pick Items

- **Type**: cube
- **Arrangement**: 4×4 grid in the pick bin (`GridPositionGenerator`, `randomize=True`)
- **Count**: Random 2–16 (pre-sampled in `__init__`)
- **Color/Appearance**: `RandomChoice(["red", "green", "blue", "magenta"])`

## Target Objects

- **Type**: marker (hidden)
- **Arrangement**: Circle on the drop zone (`CircularPositionGenerator`, radius=0.15, `randomize=False`)
- **Markers**: Hidden (`hidden_strategy=FixedValue(True)`)
- **Color/Appearance**: white (not visible)

## PickPlace Pairing and Sequencing

Sequential (default): pick[0] → target[0], pick[1] → target[1], etc.

## Success Condition

All cubes are placed on their corresponding hidden circle markers.

## Success Checks

- Each pick object rests on top of its paired target marker (`is_on_top` — default).
