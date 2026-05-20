# TableTaskBottlesToConveyor2

## User Request

Take the existing TableTaskBottlesToConveyor and create a variant that uses the cortex-style behaviour tree (MotionCommand-based with threshold-checked completion) instead of the 9-phase time-interpolated pick-place sequence. Also add incremental bottle spawning so that bottles appear one at a time on the line, and the robot starts working before all bottles have been spawned.

## Task Overview

A variant of TableTaskBottlesToConveyor that replaces the default 9-phase BT with the cortex-style BT (`make_cortex_task_controller_tree`). Picks madara_bottles from the pick bin (4x2 grid stacked 2 layers high = 16 bottles) and places them upright into 10 madara_pads arranged in a row along the conveyor. Only 10 bottles are picked (one per pad). Bottles are spawned incrementally (1 every 0.5s) and the BT starts after 3 bottles have appeared, so the robot begins working while bottles are still arriving.

## Concise Task Description

Pick bottles from 2 stacked layers in the bin and place them into carrier pads in a row on the conveyor (cortex-style BT).

## Pick Items

- **Type**: `madara_bottle` (USD asset)
- **Arrangement**: 4x2 grid in pick bin, stacked 2 layers high (layer_height=0.135m)
- **Count**: 16 (8 per layer x 2 layers)
- **Color/Appearance**: USD asset default (no color override)
- **Orientation**: Upright (-90 deg X rotation)
- **Incremental generation**: 1 bottle every 0.5s; BT starts after 3 bottles spawned

## Target Objects

- **Type**: `madara_pad` (USD carrier pad)
- **Arrangement**: Single row of 10 along conveyor (Y axis) using ConveyorPositionGenerator
- **Count**: 10
- **Spacing**: 0.10m with jitter_x=0.02m, jitter_y=0.005m
- **Markers**: N/A (pads are visible targets)
- **Color/Appearance**: USD asset default

## PickPlace Pairing and Sequencing

- **Pairing**: Sequential (BottlePickStrategy default) -- first 10 bottles paired to 10 pads
- **Ordering**: Top-down -- upper layer bottles picked before lower layer (stacking_map from compute_stacking_map, recomputed as bottles arrive incrementally)
- **Strategy class**: BottlePickStrategy (handles bottle-specific drop orientation and EE offset)

## Success Condition

All 10 paired bottles are placed upright within their corresponding carrier pads on the conveyor.

## Success Checks

1. Each bottle is within its target pad (`is_within`)
2. Each placed bottle remains vertical/upright (`is_vertical`, max_tilt_deg=15)

## Implementation Notes

- Uses `make_cortex_task_controller_tree` (cortex-style BT with MotionCommand-based behaviours and threshold-checked completion) instead of the default 9-phase time-interpolated tree.
- `IncrementalGenerationConfig(items_per_batch=1, batch_interval=0.5, bt_start_threshold=3)` enables time-based bottle spawning with early BT start.
- `stacking_enabled=True` with `compute_stacking_map` enforces top-down pick ordering from the 2-layer stack.
- `conveyor_speed` (Optional[float], passed through `TaskSpec.conveyor_speed` and the `setup_workspace` lambda) controls conveyor belt motion: `None`/`0.0` means stationary; a non-zero value (e.g. `-0.015`) enables belt motion and adjusts the pad Y offset accordingly (`conveyor_offset = 0.7 if conveyor_speed else 0.2`).
