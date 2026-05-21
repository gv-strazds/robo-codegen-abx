# TableTaskSugarBoxesRowToCircle

## User Request

Create a new task: Starting with a row of 6 vertical sugar boxes in the pick bin, arrange them in a circle on the drop zone

## Task Overview

Pick the 6 `sugar_box` YCB props that begin in an upright (vertical) row inside the pick bin and place each one onto a circular ring of 6 thin white rectangular markers laid out on the dropzone. Picks are spawned in a single column (`GridPositionGenerator(rows=6, cols=1)`) inside the bin with the standard YCB "stand-upright" orientation (-90° about X), and targets are generated with `CircularPositionGenerator(count=6, radius=0.18, randomize=False)` centered on `DROPZONE_CENTER_POINT`. Pairing is default sequential (`MultiPickStrategy`) — all items are identical so no color/type matching is needed.

## Concise Task Description

Pick 6 upright sugar boxes from a row in the bin and place them onto a circle of 6 markers on the dropzone, keeping each box vertical.

## Pick Items

- **Type**: `sugar_box` (YCB 004 sugar box; default yellow USD)
- **Arrangement**: Single column of 6 (rows=6, cols=1) along the bin's Y axis, centered at `(BIN_X_COORD, BIN_Y_COORD)`; `spacing_y=0.060 m`.
- **Count**: 6
- **Color/Appearance**: USD asset default (yellow). Standing upright via `-90°` rotation about X.

## Target Objects

- **Type**: Hidden virtual markers (`asset_type="marker"`, `hidden_strategy=FixedValue(True)`) generated at pairing time via `TaskImplementationSpec.virtual_target_generation_strategy` — not visible in the scene.
- **Arrangement**: Circle of 6 evenly spaced positions (60° apart) at radius 0.18 m, centered on `DROPZONE_CENTER_POINT`.
- **Markers**: Virtual only — no visible target geometry on the dropzone.
- **Color/Appearance**: N/A (hidden).

## PickPlace Pairing and Sequencing

- **Pairing:** Default sequential — `pick[i]` is placed on `target[i]`. All picks are identical sugar boxes and all targets are identical white markers, so no color/type matching is needed.
- **Sequencing:** Sequential from start (the default `MultiPickStrategy` pops picks in spawn order; targets are consumed in generation order).

## Success Condition

All 6 sugar boxes are placed on the 6 circle markers and remain standing upright.

## Success Checks

- Each placed sugar box rests on top of its paired marker (`is_on_top`).
- Each placed sugar box is vertical within 15° tilt (`is_vertical(max_tilt_deg=15)`).
- Both checks are combined in a custom `spatial_check_fn` (same pattern as `TableTaskCrackerBoxes1`).
