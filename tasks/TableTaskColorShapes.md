# TableTaskColorShapes

## User Request

Pick cubes, cylinders, cones, and balls spawned on the conveyor (each tinted red, green, or blue) and place them into the matching colored boxes on the table.

## Task Overview

Mixed shapes (`cube`, `cylinder`, `cone`, `ball` asset types) with random red/green/blue colors are picked from the conveyor and sorted into 3 color-matching collection boxes on the table. Uses `ConveyorPositionGenerator` for pick layout, `ColorMatchStrategy` for pairing, and custom workspace setup with 3 collection boxes. Item counts are randomized (1–2 of each type). Verification uses base-class `is_within_box_geometry` for box containment (no custom override needed).

## Concise Task Description

Pick cubes, cylinders, cones, and balls from the conveyor and place them into the matching colored boxes on the table.

## Pick Items

- **Type**: Mixed — cube, cylinder, cone, ball
- **Arrangement**: Conveyor (`ConveyorPositionGenerator`, spacing=0.12m, with jitter), shuffled order
- **Count**: Random 1–2 of each type (total 4–8)
- **Color/Appearance**: `RandomChoice(["red", "green", "blue"])`

## Target Objects

- **Type**: Hidden rectangular markers inside 3 collection boxes (red, green, blue)
- **Arrangement**: 3 boxes on the table; 6 marker slots per box in an evenly spaced grid
- **Markers**: Thin rectangles (0.05m × 0.04m × 0.002m) at box floor level, hidden
- **Color/Appearance**: Marker color matches box color; boxes are red, green, blue

## PickPlace Pairing and Sequencing

- **Strategy**: `ColorMatchStrategy` with `color_palette=["red", "green", "blue"]`
- Each item is paired to a marker inside the box matching its color
- Items are picked in shuffled conveyor order

## Success Condition

All shapes are placed inside the collection box matching their color.

## Success Checks

- `is_within_box_geometry` (containment mode): each picked item must be inside its target box, verified using per-box dimensions (`inner_size`, `height`) stored in each box spec dict. Color matching is enforced via `match_labels` in each box spec.
- `allow_multi_occupancy=True`: multiple items per box.
- Uses base-class box containment handling (no custom `check_groundtruth_task_success()` override needed).
