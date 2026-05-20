# TableTaskShapeSortBoxes2

## User Request

Pick randomly colored cubes and balls, two to four of each, randomly interleaved in a row along the conveyor with some horizontal jitter, and sort them into two boxes on the cart.

## Task Overview

Pick 2-4 cubes and 2-4 balls (each randomly colored from a palette) arranged in a shuffled row on the conveyor (using `ConveyorPositionGenerator` with X jitter), and sort them by type into two open boxes on the cart: cubes into one box, balls into the other. Uses `TypeBasedStrategy` for type-based routing and virtual hidden markers inside each box. Box containment verification with `match_labels` for type enforcement.

## Concise Task Description

Pick randomly colored cubes and balls from the conveyor and sort them by shape into two boxes on the cart.

## Pick Items

- **Type**: `cube` and `ball` (primitives)
- **Arrangement**: Single shuffled row along the conveyor (Y axis), with X jitter
- **Count**: 2-4 cubes + 2-4 balls (4-8 total, random per run)
- **Color/Appearance**: RandomChoice from palette (e.g., red, green, blue, yellow)

## Target Objects

- **Type**: Two open boxes on the cart with virtual hidden markers inside
- **Arrangement**: Side-by-side on cart surface (one for cubes, one for balls)
- **Markers**: 2x2 grid of hidden markers per box (capacity 4 each, matching max per type)
- **Color/Appearance**: Distinct box colors (e.g., light brown for cube box, dark brown for ball box)

## PickPlace Pairing and Sequencing

- TypeBasedStrategy routes cubes to cube-box markers and balls to ball-box markers
- Pick order follows the shuffled conveyor arrangement (random interleaving of cubes and balls)

## Success Condition

All cubes are inside the cube box and all balls are inside the ball box.

## Success Checks

- Each pick object is inside its assigned box (containment check via `box_verification_info`)
- Cube box contains only cubes (`match_labels: {"type": "cube"}`)
- Ball box contains only balls (`match_labels: {"type": "ball"}`)
