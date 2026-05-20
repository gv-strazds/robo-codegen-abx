# TableTaskBottlesToConveyor

## User Request

Create a new robot task that takes bottles from the pick bin and places them into madara_pads on the conveyor. There should be 9 pads arranged in a row along the conveyor, with some jitter. There should be two layers of bottles in the pick bin.

## Task Overview

Pick madara_bottles from the pick bin (4×2 grid stacked 2 layers high = 16 bottles) and place them upright into 9 madara_pads arranged in a single row along the conveyor (Y axis) with positional jitter. Only 9 bottles are picked (one per pad); the rest remain in the bin.

## Concise Task Description

Pick bottles from the bin and place them upright into carrier pads arranged in a row on the conveyor.

## Pick Items

- **Asset type**: `madara_bottle` (USD asset, custom bottle V3)
- **Count**: 16 (8 per layer × 2 layers)
- **Arrangement**: 4×2 grid (cols=4, rows=2) in the pick bin, stacked 2 layers high
  - spacing_x=0.08m, spacing_y=0.15m (matches existing bottle task spacing)
  - Layer height: 0.135m (full bottle height: rest_height + top_surface_height)
- **Orientation**: Upright (-90° X rotation, standard for USD assets)
- **Colors**: USD asset default (no color override)

## Target Objects

- **Asset type**: `madara_pad` (USD carrier pad)
- **Count**: 9
- **Arrangement**: Single row along conveyor (Y axis) using ConveyorPositionGenerator
  - center_x: DROPZONE_CENTER_POINT[0] (0.04m)
  - center_y: DROPZONE_CENTER_POINT[1] (0.69m)
  - spacing: 0.10m
  - jitter_x: 0.02m, jitter_y: 0.005m
  - z: DROPZONE_Z + 0.002

## PickPlace Pairing and Sequencing

- **Pairing**: JIT proximity — each tick, the next unoccupied pad nearest the
  -Y conveyor fall-off edge (closest to the robot) is selected as the target.
  No fixed pick→target pairing is computed up front; pairings are produced on
  demand by `ConveyorProximityStrategy._jit_select` and filtered through a
  Z-reachability check so pads that have dropped below the belt surface are
  skipped.
- **Ordering**: Top-down — upper layer bottles picked before lower layer
  (stacking_map from `compute_stacking_map`, applied via
  `_apply_stacking_order` in the base strategy).
- **Strategy class**: `BottleConveyorPickStrategy` — subclass of
  `ConveyorProximityStrategy` that forwards `stacking_map` and overrides
  `get_end_effector_orientation_for_drop` to return `HORIZONTAL_DROP_QUAT`
  (bottle-on-side drop orientation, matching BottlePickStrategy).

## Success Condition

All 9 paired bottles are placed upright within their corresponding carrier pads on the conveyor.

## Success Checks

1. Each bottle is within its target pad (is_within)
2. Each placed bottle remains vertical/upright (is_vertical)
