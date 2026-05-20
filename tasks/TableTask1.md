# TableTask1

## Task Overview

Pick 16 colored cubes (4 each of red, green, blue, orange, randomly shuffled) from a 4×4 grid in the pick bin and arrange them in a tightly packed 4×4 grid on the dropzone with horizontal color stripes (one color per row).

## Concise Task Description

Sort 16 randomly shuffled colored cubes from the bin into a 4×4 color-striped grid on the dropzone.

## Pick Items

- **Asset type**: `cube` (primitive)
- **Count**: 16
- **Arrangement**: 4×4 grid in the pick bin
  - spacing: cube_size + 0.005m (tight packing to prevent edge displacement)
- **Colors**: 4 each of red, green, blue, orange — randomly shuffled each run
- **Orientation**: default (axis-aligned)

## Target Objects

- **Type**: Hidden markers
- **Count**: 16
- **Arrangement**: 4×4 tightly packed grid on the dropzone
  - spacing: cube_size + 0.0025m
  - Row 0: red, Row 1: green, Row 2: blue, Row 3: orange
- **Deferred generation**: markers created at pairing time

## PickPlace Pairing and Sequencing

- **Pairing**: ColorMatchStrategy — each cube is paired to a marker of the same color
- **Ordering**: Sequential within each color group
- **Strategy class**: ColorMatchStrategy with `color_palette=['red', 'green', 'blue', 'orange']`

## Success Condition

All 16 cubes are placed onto markers with matching colors, forming a 4×4 grid with horizontal color stripes on the dropzone.

## Success Checks

1. Each cube is at its assigned target marker position (containment / proximity check)
