# TableTaskGreenCubesRowToYellowGrid

## User Request

Pick green cubes pre-arranged in a single row on the cart and place them onto yellow rectangles arranged in a 2x3 grid on the dropzone.

## Task Overview

- Workspace: `setup_two_tables` is called with `standard_objs=False, add_bin=False` so the cart starts empty (no default cracker box / sugar box / soup can / mustard bottle / KLT bin). Only the green cubes occupy the cart.
- Pick items: 6 green cubes (asset_type `"cube"`) pre-arranged in a single row along the cart's long axis (Y).
- Targets: 6 thin yellow rectangle markers (asset_type `"rect"`, color `"yellow"`) laid out in a 2 (along Y) × 3 (along X) grid on the dropzone.
- Generators: `GridPositionGenerator` for both — `rows=6, cols=1` for the pick row, `rows=2, cols=3` for the dropzone grid.
- Pairing strategy: default `MultiPickStrategy` with sequential pairing. All cubes are identical and all targets are identical, so no color/type matching is needed.

## Concise Task Description

Pick 6 green cubes pre-arranged in a single row on the cart and place them onto yellow rectangles arranged in a 2x3 grid on the dropzone.

## Pick Items

- **Type**: cube
- **Arrangement**: Single row along the cart's Y axis, at an X offset away from the bin so cubes rest on bare cart surface
- **Count**: 6
- **Color/Appearance**: Green (all identical)

## Target Objects

- **Type**: Thin yellow rectangular markers (`asset_type="rect"`, scale ~0.10 × 0.10 × 0.002 m)
- **Arrangement**: 2 × 3 grid on the dropzone (2 rows along Y, 3 columns along X)
- **Markers**: Markers are the targets themselves — they are visible rectangles on the dropzone surface, not hidden virtual markers inside containers
- **Color/Appearance**: Yellow

## PickPlace Pairing and Sequencing

Sequential pairing: `pick[i]` → `target[i]` for i = 0..5, picks consumed in the order produced by the row generator, targets consumed in the order produced by the grid generator. Default `MultiPickStrategy` (no `create_strategy` override).

## Success Condition

Every green cube has been placed onto a distinct yellow rectangle in the dropzone grid.

## Success Checks

- Each cube rests on its paired yellow rectangle (default `is_on_top`: XY-footprint overlap + cube bottom within 2 cm of marker top).
- No additional posture / orientation checks — cubes are symmetric, so a `is_vertical` check is unnecessary.
