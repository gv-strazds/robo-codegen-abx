# TableTaskLayeredCircle2

## Task Overview

Pick horizontal sugar boxes from a circular arrangement stacked 2 layers high (5 positions × 2 layers = 10 total) and restack them into a single growing column inside the KLT pick bin on the cart.

## Concise Task Description

Pick sugar boxes lying flat from a layered circle on the dropzone and stack them into a single column in the bin.

## Pick Items

- **Asset type**: `sugar_box` (USD asset)
- **Count**: 10 (5 per layer × 2 layers)
- **Arrangement**: Circle (radius=0.18m, 5 evenly-spaced positions) on the dropzone, stacked 2 layers high
  - Layer height: 0.045m (horizontal sugar box Z thickness)
- **Orientation**: Horizontal (native orientation, no rotation — boxes lie flat)
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

All 10 sugar boxes are picked from the circle and stacked into a single column inside the bin.

## Success Checks

1. Each box is contained within the bin footprint (containment check)
2. Each box rests on top of the previous one in the growing stack (is_on_top)

## Implementation Notes

- `ee_height_for_move`: 0.27m — transport height must clear bin walls (~0.20m) plus carried item rest height (~0.023m) plus margin
- `stacking_enabled=True` — enforces top-down pick ordering from the 2-layer source
- `bin_geometry` passed to strategy for spatial verification (floor_z uses settled bin value)
