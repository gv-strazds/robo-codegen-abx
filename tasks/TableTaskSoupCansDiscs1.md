# TableTaskSoupCansDiscs1

## User Request

> Pick soup cans from the bin and place them onto colored disc markers arranged in a 2x3 grid on the dropzone table. Disc colors should cycle through red, yellow, and blue.

## Task Overview

The UR10 picks soup cans (`asset_type="soup_can"`) from a 3x3 grid spawned in the pick bin and places them onto a 2x3 grid (3 rows x 2 cols) of colored disc markers (`asset_type="disc"`, the `DynamicCylinder` primitive) on the dropzone table. The 6 disc markers cycle in color through `red, yellow, blue` using `SequentialChoice`. Pairing uses the default `MultiPickStrategy` (sequential `pick[i] -> target[i]`); the 3 extra cans without a target are overflow and are not required to be placed for the task to complete successfully.

## Concise Task Description

Pick soup cans from the bin and place them onto colored disc markers (red, yellow, blue) arranged in a 2x3 grid on the dropzone.

## Pick Items

- **Type**: `soup_can` (USD asset)
- **Arrangement**: 3x3 grid in the pick bin
- **Count**: 9 (3 overflow vs. 6 targets)
- **Color/Appearance**: USD asset default; spawn orientation `-90°` about world X so cans stand upright.

## Target Objects

- **Type**: `fixed_disc` (FixedCylinder primitive, thin — static so cans resting on it do not jitter from uneven dynamic contact)
- **Arrangement**: 2x3 grid (3 rows along Y, 2 cols along X) on the dropzone, spacing `0.15 m` in each axis.
- **Markers**: The discs themselves are the markers (not hidden virtual targets, not containers).
- **Color/Appearance**: `SequentialChoice(["red", "yellow", "blue"], loop=True)` so the 6 markers cycle red, yellow, blue, red, yellow, blue. Disc diameter `0.07 m`, thickness `0.002 m` (sized so discs are visibly separated at `0.15 m` center-to-center spacing).

## PickPlace Pairing and Sequencing

- **Pairing**: Default `MultiPickStrategy` sequential pairing — `pick[i] -> target[i]` for `i in 0..min(9, 6)-1 = 0..5`. Picks 6, 7, 8 are emitted with `target=None` and are not placed.
- **Sequencing**: Sequential from start. No interleaving, no per-color matching.

## Success Condition

All 6 paired soup cans are placed upright on their disc markers; remaining picks may be unfulfilled.

## Success Checks

- Each placed soup can rests on its paired disc marker (`is_on_top`).
- Each placed soup can is upright after placement (`is_vertical`, `max_tilt_deg=15`).
- These two checks are combined in `spatial_check_fn=_soup_can_spatial_check`.
