# TableTaskRgbyCubesGridRearrange

## User Request

Create a new task: pick red, green, blue and yellow blocks from a 4 x 3 grid on the drop zone and place them to form a more compact 3 x 4 grid on the cart, which should initally be empty of other objects

## Task Overview

- Workspace: `setup_two_tables` is called with `standard_objs=False, add_bin=False` so the cart starts empty (no default cracker box / sugar box / soup can / mustard bottle / KLT bin). The placement grid uses bare cart surface (see Issue 14).
- Pick items: 12 cubes (asset_type `"cube"`) arranged in a 4×3 grid on the dropzone, with **3 each of red, green, blue, yellow** organized one color per row.
- Targets: 12 hidden virtual markers (`asset_type="marker"`, `hidden_strategy=FixedValue(True)`) in a 3×4 grid on the cart surface. Per Issue 15, the user described the placement arrangement but not the target geometry, so virtual hidden targets are used via `TaskImplementationSpec.virtual_target_generation_strategy`.
- Generators: `GridPositionGenerator` for both — `rows=4, cols=3, spacing=0.075m` for the dropzone picks; `rows=3, cols=4, spacing=0.060m` for the cart targets (tighter spacing → "more compact").
- Pairing strategy: default `MultiPickStrategy` with sequential pairing. The task is a rearrangement, not a color sort, so cubes do not need to match specific target positions.

## Concise Task Description

Pick 12 cubes (3 each of red, green, blue, yellow) pre-arranged in a 4x3 grid on the dropzone and place them to form a more compact 3x4 grid on the cart.

## Pick Items

- **Type**: cube
- **Arrangement**: 4×3 grid on the dropzone (4 rows along Y, 3 cols along X), spacing 0.075 m
- **Count**: 12
- **Color/Appearance**: 3 red + 3 green + 3 blue + 3 yellow, assigned one color per row of the source grid

## Target Objects

- **Type**: Hidden virtual markers (`asset_type="marker"`, `hidden_strategy=FixedValue(True)`)
- **Arrangement**: 3×4 grid on the cart surface (3 rows along Y, 4 cols along X), spacing 0.060 m — more compact than the pick spacing of 0.075 m
- **Markers**: No visible markers placed in the scene; the placement positions are virtual (generated at pairing time as `LightweightObj` instances)
- **Color/Appearance**: Hidden (the cubes themselves form the visual 3×4 grid on the cart)

## PickPlace Pairing and Sequencing

Sequential pairing: `pick[i]` → `target[i]` for i = 0..11. Picks are consumed in the order produced by the source grid generator (row-major) and targets are consumed in the order produced by the target grid generator (also row-major). Default `MultiPickStrategy` (no `create_strategy` override).

## Success Condition

Every cube has been picked from the dropzone and placed at a distinct virtual target position in the 3×4 cart grid.

## Success Checks

- Each cube rests on its paired virtual marker (default `is_on_top`: XY-footprint overlap + cube bottom within 2 cm of marker top).
- No additional posture / orientation checks — cubes are symmetric.
