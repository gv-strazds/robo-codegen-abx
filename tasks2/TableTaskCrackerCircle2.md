# TableTaskCrackerCircle2

## User Request

Create a new task almost the same as TableTaskLayeredCircle, but using Cracker Boxes instead of sugar boxes. The cracker boxes should be initially positioned horizontally (with their longest dimension along the x axis). They should also end up stacked in that same horizontal orientation.

## Task Overview

Pick horizontal cracker boxes from a circular arrangement stacked 2 layers high (5 positions × 2 layers = 10 total) and restack them into a single growing column inside the KLT pick bin on the cart. Cracker boxes are rotated 90° about Z so their longest dimension (0.213m) aligns with the world X axis.

## Concise Task Description

Pick cracker boxes lying flat (longest dimension along X) from a layered circle on the dropzone and stack them into a single column in the bin.

## Pick Items

- **Asset type**: `cracker_box` (USD asset)
- **Count**: 10 (5 per layer × 2 layers)
- **Arrangement**: Circle (radius=0.22m, 5 evenly-spaced positions) on the dropzone, stacked 2 layers high
  - Layer height: 0.072m (horizontal cracker box Z thickness in native orientation)
  - Radius increased from 0.18m (sugar box task) to 0.22m to accommodate larger cracker box footprint
- **Orientation**: Horizontal, rotated 90° about Z axis (longest dimension 0.213m along world X)
  - Native orientation has longest dimension along Y; 90° Z rotation swaps to X
- **Colors**: USD asset default (no color override)

## Target Objects

- **Type**: Hidden marker
- **Count**: 1 (single marker at bin center — base of destination stack)
- **Position**: `[BIN_X_COORD, BIN_Y_COORD, bin_floor_z]` (settled KLT bin floor ≈ 0.0573 + 0.005m)
- **Scale**: 0.05m × 0.05m × 0.001m (thin, invisible)
- **Deferred generation**: marker is created at pairing time (all items share one destination)

## PickPlace Pairing and Sequencing

- **Pairing**: SingleStackStrategy — all picks map to the single bin marker; stacking_map determines layered ordering
- **Ordering**: Top-down — upper layer boxes picked before lower layer (stacking_map computed from source positions via `compute_stacking_map`)
- **Destination stacking**: Each deposited box becomes the new stack top; strategy dynamically extends target list

## Success Condition

All 10 cracker boxes are picked from the circle and stacked into a single column inside the bin.

## Success Checks

1. Each box is contained within the bin footprint (containment check)
2. Each box rests on top of the previous one in the growing stack (is_on_top)

## Implementation Notes

- `ee_height_for_move`: 0.28m — transport height must clear bin walls (~0.20m) plus carried cracker box rest height (~0.036m) plus margin
- `stacking_enabled=True` — enforces top-down pick ordering from the 2-layer source
- `bin_geometry` passed to strategy for spatial verification (floor_z uses settled bin value)
- Cracker box horizontal footprint after 90° Z rotation: 0.213m (X) × 0.164m (Y) × 0.072m (Z)
- Adjacent box center distance at r=0.22m with 5 positions: 0.259m — sufficient clearance for 0.213×0.164 footprint
