# TableTaskMixedCircle

## Task Overview

Pick 8 mixed items (randomly sampled from 8 asset types: cube, ball, cone, cylinder, soup_can, cracker_box, sugar_box, mustard_bottle) arranged in a circle on the conveyor/dropzone and place them into a 4×2 grid inside the KLT pick bin on the cart. Items with an elongated vertical axis (USD bottles/boxes/cans plus the cylinder and cone primitives, which MixedScaleStrategy elongates along Z) must remain upright after placement.

## Concise Task Description

Pick a random mix of primitives and USD assets from a circle on the conveyor and place them into the bin on the cart.

## Pick Items

- **Asset types**: Randomly sampled from `[cube, ball, cone, cylinder, soup_can, cracker_box, sugar_box, mustard_bottle]` (8 types)
- **Count**: 8 (default; overridable via `--pick-count`)
- **Arrangement**: Circle (radius=0.15m, 8 evenly-spaced positions) on the dropzone/conveyor
  - z = DROPZONE_Z + 0.15m (safe height for all object types)
- **Colors**: RandomChoice(['red', 'green', 'blue', 'yellow'])
- **Orientation**: Mixed — upright (-90° X rotation) for USD assets; default for primitives (MixedOrientationStrategy)
- **Scale**: Mixed — per-asset-type scaling (MixedScaleStrategy)

## Target Objects

- **Type**: Hidden markers
- **Count**: 8 (default; overridable via `--target-count`)
- **Arrangement**: 4×2 grid in the KLT bin on the cart
  - spacing_x=0.08m, spacing_y=0.08m
  - z ≈ bin_floor_z + marker_half_height
- **Deferred generation**: markers created at pairing time

## PickPlace Pairing and Sequencing

- **Pairing**: Sequential (MultiPickStrategy default) — items placed into bin slots in order, no type or color matching
- **Strategy class**: MultiPickStrategy (default, no factory override)

## Success Condition

All 8 items are placed into the bin, with the verticality-required asset types (USD bottles/boxes/cans plus cylinder/cone primitives) remaining upright.

## Success Checks

1. Each item is contained within the bin footprint (`box_verification_info` containment check)
2. Verticality-required asset types (cracker_box, sugar_box, soup_can, mustard_bottle, cylinder, cone) are vertical/upright (`is_vertical`)
3. Isotropic primitives (cube, ball) have no orientation requirement

## Implementation Notes

- Asset types are pre-sampled once at construction time so that `MixedOrientationStrategy` and `MixedScaleStrategy` stay consistent with `SequentialChoice` for asset types
- `containment_check=True` and `box_verification_info` use centralized bin containment verification
- `placement_constraints_fn` applies `is_vertical` to asset types with a clearly identifiable vertical axis (USD bottles/boxes/cans plus elongated cylinder/cone primitives); isotropic primitives are exempt
