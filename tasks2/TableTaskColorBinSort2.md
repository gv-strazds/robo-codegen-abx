# TableTaskColorBinSort2

## User Request

Pick between one and five cubes and balls (each tinted red, green, or blue) from the conveyor and drop them into the color-matching collection boxes.

## Task Overview

Mixed cubes and balls (`cube` and `ball` asset types) with random red/green/blue colors are picked from the conveyor and sorted into 3 color-matching collection boxes on the table. Uses `ConveyorPositionGenerator` for pick layout, `ColorMatchStrategy` for pairing, and custom workspace setup to spawn 3 collection boxes. Item counts are randomized (1–5 cubes, 1–5 balls).

## Concise Task Description

Pick between one and five cubes and balls from the conveyor and drop them into the color-matching collection boxes.

## Pick Items

- **Type**: Mixed — cube and ball
- **Arrangement**: Conveyor (`ConveyorPositionGenerator`, spacing=0.12m, with jitter), shuffled order
- **Count**: Random 1–5 cubes + 1–5 balls (total 2–10)
- **Color/Appearance**: `RandomChoice(["red", "green", "blue"])`

## Target Objects

- **Type**: Hidden rectangular markers inside 3 collection boxes (red, green, blue)
- **Arrangement**: 3 boxes on the table; 6 marker slots per box at hardcoded positions
- **Markers**: Thin rectangles (0.04m × 0.04m × 0.002m) at box floor level
- **Color/Appearance**: Marker color matches box color; boxes are red, green, blue

## PickPlace Pairing and Sequencing

- **Strategy**: `ColorMatchStrategy` with `color_palette=["red", "green", "blue"]`
- Each item is paired to a marker inside the box matching its color
- Items are picked in shuffled conveyor order

## Success Condition

All cubes and balls are sorted into the collection box matching their color.

## Success Checks

- `is_within_box_geometry` (containment mode): each pick object must be inside its target box, verified using per-box dimensions (`inner_size`, `height`) stored in each box spec dict.
- Color matching is enforced via `match_labels` in each box spec, ensuring items are sorted into the correct color box.
- `allow_multi_occupancy=True`: multiple items per box.
