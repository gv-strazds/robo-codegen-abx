# TableTaskSoupCans2

## User Request

> make a task (reference the create-task skill) that is like TableTaskSoupCans1, but has a
> moving conveyor (default speed) and instead of a grid of target objects, it dynamically
> adds from one to three targets (randomized per target spawn interval) at a y distance
> similar to where pick items are added in TableTaskConveyorTypeSort. Call it
> TableTaskSoupCans2 (table_task_soup_cans_2).

## Task Overview

Like `TableTaskSoupCans1` (soup cans from a 3x3 bin grid placed upright onto thin red
rectangles), but the drop zone is a moving conveyor rather than a static surface, and the
target rectangles are not pre-placed in a grid. Instead, they are spawned incrementally at
the far-Y end of the conveyor (mirroring the pick-item spawn point in
`TableTaskConveyorTypeSort`: `DROPZONE_CENTER_POINT[1] + 2 * dropzone_half_depth`). At each
spawn interval, 1-3 targets are released in a single burst, with the count randomized per
interval.

Pick side is unchanged from SoupCans1: 9 soup cans in a 3x3 grid in the KLT pick bin.

Target generation uses a custom `ItemGenerator`-compatible class that pre-plans group sizes
(each 1-3, summing to the total target count) using a seeded RNG, and a custom
`IncrementalItemScheduler` subclass that reads the per-group size from the generator before
each batch release so `items_per_batch` tracks the group size.

Pairing is sequential (default `MultiPickStrategy`).

## Concise Task Description

"Pick soup cans from the bin and place them upright onto thin red rectangles that arrive
in random-sized bursts (1-3 at a time) on a moving conveyor."

## Pick Items

- **Type**: `soup_can` (YCB USD asset)
- **Arrangement**: 3x3 grid in the KLT pick bin (same as SoupCans1)
- **Count**: 9
- **Color/Appearance**: USD asset default
- **Orientation**: upright (-90 deg X rotation)

## Target Objects

- **Type**: Thin red rectangular slabs (`cube` primitive / `DynamicCuboid`, 0.1 m x 0.1 m,
  0.01 m thick). Must be dynamic rigid bodies rather than `FixedCuboid` so the conveyor's
  surface velocity can carry them via friction.
- **Arrangement**: Dynamically spawned in bursts of 1 or 2 at `far_y` on the conveyor
  surface. Each burst randomly fills 1 or 2 of three fixed X-row slots (-0.14, 0, +0.14 m
  relative to the drop-zone centerline) so items in the same burst do not interpenetrate.
  The moving conveyor carries each burst toward the robot at `DEFAULT_CONVEYOR_SPEED`
  (-0.015 m/s). Max burst size is capped at 2 so the robot can keep up with the incoming
  stream at the chosen spawn interval.
- **Markers**: N/A (rectangles themselves act as placement markers).
- **Color/Appearance**: Red.
- **Count**: 9 total (matches pick count), released across ~5-9 bursts of size 1 or 2 each.
- **Spawn interval**: 6 s between bursts. At `DEFAULT_CONVEYOR_SPEED` = -0.015 m/s the
  belt travels ~9 cm per interval — just under the 10 cm rectangle length so successive
  bursts arrive visually tight along Y without fully overlapping.

## PickPlace Pairing and Sequencing

- **Strategy**: Default `MultiPickStrategy` with sequential pairing (pick[i] -> target[i]).
- **Sequencing**: Robot starts after the first burst of targets arrives
  (`bt_start_threshold=1`); incoming targets continue to spawn while the robot places
  earlier picks. `CheckTargetAvailable` idles if the pick queue gets ahead of the target
  stream.

## Success Condition

All 9 soup cans are placed upright on the 9 red rectangle targets.

## Success Checks

- Each soup can rests on its paired rectangle (`is_on_top`).
- Each placed soup can is vertical (`is_vertical`, max 15 deg tilt).
