# TableTaskCartToConveyor2

## User Request

> Generate a new task for picking items from the cart and placing them into boxes on the conveyor. There should be a random number of boxes (from 1 to seven) in a row along the conveyor. On the cart there should be 4 sets of items: from between one and eight of each type of item - cracker boxes, soup cans, mustard bottles and sugar boxes. One of each should be placed into each box on the conveyor, stopping whenever a box cannot be filled because no more of the required item types are available. All of these items should be vertically placed in the boxes.

## Task Overview

Pick four types of USD assets (cracker_box, soup_can, mustard_bottle, sugar_box) from the cart and place one of each into boxes arranged in a row on the conveyor/dropzone table. Uses a custom generator for cart item layout, sequential pairing strategy, and box containment verification with vertical orientation checks.

## Concise Task Description

Pick cracker boxes, soup cans, mustard bottles, and sugar boxes from the cart and place one of each vertically into boxes on the conveyor.

## Pick Items

- **Types**: cracker_box, soup_can, mustard_bottle, sugar_box (USD assets)
- **Count**: Random 1–8 of each type (total 4–32 items on cart)
- **Arrangement**: Grouped by type in 4 rows on cart surface, each row spaced along X with items within each row spaced along Y
- **Orientation**: Upright (-90° X rotation)
- **Order**: Interleaved by fill order — cracker_0, soup_0, mustard_0, sugar_0, cracker_1, soup_1, ... (fillable items first, then extras at end)

## Target Objects

- **Types**: Hidden markers inside boxes on the conveyor
- **Boxes**: Random 1–7 boxes in a row along Y on the dropzone table, each large enough for a 2×2 grid of items; row is centered on the dropzone but clamped so the first box y ≥ 0.50m
- **Box inner size**: 0.30 × 0.18m, height 0.15m (accommodates cracker_box footprint)
- **Markers**: 4 hidden markers per box in a 2×2 grid (one slot per item type)

## PickPlace Pairing and Sequencing

- **Strategy**: Sequential (default). Pick items are ordered cracker_0, soup_0, mustard_0, sugar_0, cracker_1, ... and markers are ordered box_0_slot_0, box_0_slot_1, ..., so sequential pairing naturally fills each box with one of each type.
- **num_fillable** = min(num_boxes, n_cracker, n_soup, n_mustard, n_sugar). Only this many boxes' worth of items are paired; extra items on the cart get None targets and remain unpicked (CheckTargetAvailable stops the loop).

## Success Condition

Each filled box contains exactly one of each item type, all placed vertically.

## Success Checks

- `is_within_box_geometry` (containment mode): each picked item must be inside its target box, verified using per-box dimensions (`inner_size`, `height`) stored in each box spec dict.
- `is_vertical` (placement constraint): each picked item must be upright.
- `allow_multi_occupancy=True`: multiple items per box.
- Uses base-class box containment handling (no custom `check_groundtruth_task_success()` override needed).
