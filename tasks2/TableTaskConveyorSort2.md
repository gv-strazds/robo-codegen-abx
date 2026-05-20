# TableTaskConveyorSort2

## User Request

Pick cubes and balls (default colors, randomly interleaved) lined up on the conveyor and place them into separate boxes on the cart.

## Task Overview

Mixed cubes and balls (primitive types) with default system-assigned colors (no explicit color, `color=None`) are spawned in randomly interleaved order along the conveyor. The robot picks each item and places it into the correct box on the cart: cubes go into the cube box, balls go into the ball box. Uses `TypeBasedStrategy` for type-based routing and `ConveyorPositionGenerator` for pick positions. Two `spawn_open_box` containers on the cart with virtual hidden markers.

## Concise Task Description

Pick cubes and balls from the conveyor and sort them into separate boxes on the cart.

## Pick Items

- **Type**: `cube` and `ball` (primitives)
- **Arrangement**: Randomly interleaved single line on the conveyor (ConveyorPositionGenerator)
- **Count**: 4 cubes + 4 balls = 8 total
- **Color/Appearance**: Default (color=None, system-assigned, no color labels)

## Target Objects

- **Type**: Two open boxes on the cart with virtual hidden markers inside
- **Arrangement**: Side by side on the cart surface (offset in X)
- **Markers**: 2x2 grid of hidden markers per box (4 markers each, 8 total)
- **Color/Appearance**: Cube box is warm brown, ball box is steel blue

## PickPlace Pairing and Sequencing

- **Pairing**: TypeBasedStrategy — cubes route to cube_box markers (indices 0-3), balls route to ball_box markers (indices 4-7)
- **Sequencing**: Items are picked in conveyor order (interleaved types); each pick is routed to the correct box based on its type

## Success Condition

All cubes are inside the cube box and all balls are inside the ball box.

## Success Checks

- Each pick object is inside the correct box (centralized box_verification_info + containment_check=True)
- Cube box accepts only items with type label "cube" (match_labels: {"type": "cube"})
- Ball box accepts only items with type label "ball" (match_labels: {"type": "ball"})
