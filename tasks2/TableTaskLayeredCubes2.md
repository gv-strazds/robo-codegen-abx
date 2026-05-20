# TableTaskLayeredCubes2

## Task Overview

Pick 18 colored cubes from a 2×3 grid stacked 3 layers high in the pick bin and place them onto a flat 6×3 grid of hidden markers on the dropzone. Colors are layered: red (bottom layer), green (middle), blue (top).

## Concise Task Description

Unstack 18 cubes from a 3-layer grid in the bin and place them onto a flat grid on the dropzone.

## Pick Items

- **Asset type**: `cube` (primitive)
- **Count**: 18 (2×3 grid × 3 layers)
- **Arrangement**: 2×3 grid in the pick bin, stacked 3 layers high
  - spacing_x=0.08m, spacing_y=0.08m
  - layer_height=0.0515m (cube size)
- **Colors**: Sequential by layer — 6 red (layer 0, bottom), 6 green (layer 1), 6 blue (layer 2, top)
- **Orientation**: default (axis-aligned)

## Target Objects

- **Type**: Hidden markers
- **Count**: 18
- **Arrangement**: 6×3 flat grid on the dropzone
  - spacing_x=−0.10m, spacing_y=0.10m
- **Colors**: white (hidden)
- **Deferred generation**: markers created at pairing time

## PickPlace Pairing and Sequencing

- **Pairing**: Sequential (MultiPickStrategy default) — picks paired to targets in order
- **Ordering**: Top-down — upper layer cubes (blue) picked before middle (green) then bottom (red), enforced via `stacking_map` from `compute_stacking_map`
- **Strategy class**: MultiPickStrategy with `stacking_map`

## Success Condition

All 18 cubes are picked from the layered source and placed onto the flat grid markers on the dropzone.

## Success Checks

1. Each cube rests on its corresponding target marker (is_on_top)

## Implementation Notes

- `stacking_enabled=True` — enforces top-down pick ordering from the 3-layer source
- `stacking_map` computed from source pick positions via `compute_stacking_map`
