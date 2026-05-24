# TableTaskSugarBoxGrid

## User Request

Pick sugar boxes from the bin and place them, vertically, in a 3x4 grid in the dropzone. The sugar boxes appear incrementally one at a time in the center of the bin. Initially, there is only one — a new one spawns 2.2 seconds after there is no sugar box in the bin. The task is finished when the destination grid is full.

## Task Overview

Sugar boxes spawn one at a time at the bin center via a spatial-trigger mechanism: after the robot picks the current box (bin becomes empty), a 2.2-second cooldown elapses before the next box appears. The robot places each box vertically onto invisible virtual markers arranged in a 3x4 grid on the dropzone. The task completes when all 12 grid positions are filled.

## Concise Task Description

Pick incrementally-spawned sugar boxes from the bin center and place them vertically onto a 3x4 grid of markers on the dropzone.

## Pick Items

- **Asset type**: `sugar_box` (YCB USD asset)
- **Count**: 12 (spawned incrementally)
- **Arrangement**: Single point at bin center `[BIN_X_COORD, BIN_Y_COORD]`
- **Orientation**: Upright (-90° X rotation)
- **Spawning**: `SpatialTriggerConfig` with `invert=True` on the bin region; `initial_count=1`, `items_per_batch=1`, `trigger_delay=2.2`

## Target Objects

- **Type**: Hidden (virtual) markers
- **Arrangement**: 3 columns × 4 rows grid on the dropzone, filled sequentially by grid position
- **Count**: 12
- **Spacing**: `dx=-0.12`, `dy=0.10`

## PickPlace Pairing and Sequencing

Default sequential pairing (`MultiPickStrategy`). Grid is filled in sequential order (row by row). All items are identical sugar boxes, no matching needed.

## Success Condition

All 12 sugar boxes are placed vertically on their corresponding grid markers.

## Success Checks

- Each sugar box rests on top of its target marker (`is_on_top`)
- Each placed sugar box remains upright (`is_vertical`, max tilt 15°)
