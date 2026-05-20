# TableTaskSpecExample

## User Request
\(The user's request, verbatim. For example: \)
Generate a new task in the tasks/ subdirectory, in which the robot picks Cracker Boxes and Soup Cans from the conveyor area (aka the "drop zone") and places one cracker box and 4 soup cans into each box on the cart. The initial configuration should have two boxes on the cart, and one row of Cracker Boxes and three rows of Soup Cans, of length 7 (increasing y coordinates) on the conveyor. The pick items should be pick-placed sequentially from the start of each row, not in a random order.

## Task Overview
\(A summary of the task requirements and initial configuration. For example: \)
This TableTask requires the robot to sequentially pick items from the drop zone and place a specific assortment — one cracker_box and four soup_cans — into each of two boxes situated on a cart (resting on the 'cart_surface' constructed in setup_two_tables() from table_setup.py). The initial configuration of pick items is arranged in four rows extending along the y-coordinate (one row of Cracker Boxes and three rows of Soup Cans), with each row containing exactly seven items. The initial configuration of target objects is two boxes situated on the cart with hidden objects of asset_type 'marker' for each item to be placed.

## Concise Task Description
\(Example for the cracker box and soup cans task: \)
Put one cracker box and four soup cans into each box on the cart.


## Pick Items
\(Example for the cracker box and soup cans task: \)
- **Type**: cracker_box, soup_can
- **Arrangement**: Rows (increasing in y direction) in the drop zone area. One row of cracker boxes and three rows of soup cans.
- **Count**: 7 items in each row.
- **Color/Appearance**: Default (None).


## Target Objects
\(Example for the cracker box and soup cans task: \)
- **Type**: Collection Boxes
- **Arrangement**: 2 Boxes fixed on the cart.
- **Markers**: Inside the boxes (virtual hidden markers — LightweightObj generated at pairing time, not spawned in scene).
- **Color/Appearance**: Default (None) for pick items; default color for boxes; virtual hidden markers for each item to be placed.


## PickPlace Pairing and Sequencing
\(Example for the cracker box and soup cans task: \)
- Pair each pick item with a corresponding marker in the boxes. Arrange the markers in the boxes with one marker centered near one end of the box for the cracker box, and a grid of markers for the soup cans.
- Pick sequentially in increasing y coordinates from the rows of pick items. Place sequentially: first into one box, then the next.


## Success Condition
\(Example for the cracker box and soup cans task: \)
The task is complete when one cracker box and 4 soup cans have been placed into each box.


## Success Checks
\(Example for the cracker box and soup cans task: \)
- Verify each picked object is in a box (via centralized box containment: `box_verification_info` + `containment_check=True` in TaskSpec).
- Verify that the placed cracker boxes are vertical (via `placement_constraints_fn`).
- Verify that the placed soup cans are vertical (via `placement_constraints_fn`).
