# TableTaskConveyorCubesBalls

## User Request

Pick three cubes followed by three balls from the conveyor and place them into two boxes on the table.

## Task Overview

Purple cubes and yellow balls are picked from the conveyor (3 cubes then 3 balls, fixed order) and placed into two separate boxes on the table — cubes into a "cube box" and balls into a "ball box". Uses `ConveyorPositionGenerator` for pick layout, `TypeBasedStrategy` for routing by object type, and custom workspace setup with 2 boxes. Each box has 3 marker slots.

## Concise Task Description

Pick three cubes followed by three balls from the conveyor and place them into two boxes on the table.

## Pick Items

- **Type**: Mixed — cube (3) then ball (3), fixed order
- **Arrangement**: Conveyor (`ConveyorPositionGenerator`, spacing=0.08m), cubes first then balls
- **Count**: 6 (3 cubes + 3 balls)
- **Color/Appearance**: Cubes purple, Balls yellow

## Target Objects

- **Type**: Hidden rectangular markers inside 2 boxes on the table
- **Arrangement**: 2 boxes on the table (cube box and ball box); 3 marker slots per box
- **Markers**: Thin rectangles (0.04m × 0.04m × 0.002m) at box floor level
- **Color/Appearance**: Purple markers in cube box, Yellow markers in ball box

## PickPlace Pairing and Sequencing

- **Strategy**: `TypeBasedStrategy` — routes cubes to cube box markers and balls to ball box markers
- Cubes are picked first (indices 0–2), then balls (indices 3–5)
- Each type is routed to its designated box's marker slots

## Success Condition

All cubes are in the cube box and all balls are in the ball box.

## Success Checks

- Each pick object rests on its paired marker inside the correct box (`is_on_top` — default via type-based routing).
