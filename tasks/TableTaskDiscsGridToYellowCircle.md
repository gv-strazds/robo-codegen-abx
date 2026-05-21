# TableTaskDiscsGridToYellowCircle

## User Request

> Pick discs (with default coloring) from originally arranged in a grid in the bin (as many as will comfortably fit) and place them onto thin yellow rectangles arranged in a circle on the drop zone.

## Task Overview

The UR10 picks dynamic discs (`asset_type="disc"`, the `DynamicCylinder` primitive) from a 2x4 grid (2 cols × 4 rows along Y) in the pick bin and places them onto a circle of 8 thin yellow square markers (`asset_type="marker"`, the `VisualCuboid` primitive, scaled to 10 cm × 10 cm × 2 mm — slightly larger than the ~9 cm disc diameter so each disc fits within its marker footprint) on the dropzone. Discs use the default per-instance coloring assigned by the system (`color_strategy=None` — colors vary per disc and are not specified by this task). Pick and target counts match 1:1, so the default sequential `MultiPickStrategy` pairs `pick[i] -> target[i]` with no overflow. The two-table workspace (`setup_two_tables`) is used. Discs spawn flat (Z-up) natively, so no orientation override is needed.

## Concise Task Description

Pick discs from a 2x4 grid in the bin and place each one onto a thin yellow square marker in a circle of 8 markers on the dropzone.

## Pick Items

- **Type**: `disc` (dynamic `DynamicCylinder`).
- **Arrangement**: 2x4 grid (2 cols along X, 4 rows along Y) in the pick bin, `spacing_x=0.10`, `spacing_y=0.095`.
- **Count**: 8.
- **Color/Appearance**: Default per-instance coloring via `color_strategy=None` (colors vary per disc, assigned by the system, not specified by this task). Scale `[0.045, 0.045, 0.045]` (≈ 9 cm diameter, ≈ 4.5 cm thick).

## Target Objects

- **Type**: Thin yellow square markers (`asset_type="marker"`, `VisualCuboid`).
- **Arrangement**: Circle of 8 markers on the dropzone, `radius=0.20 m`, evenly spaced (`CircularPositionGenerator`, `randomize=False`).
- **Markers**: Visible, flat (no containment / no virtual targets).
- **Color/Appearance**: Yellow (`FixedValue("yellow")`). Scale `[0.10, 0.10, 0.002]` (10 cm × 10 cm × 2 mm — slightly larger than the ~9 cm disc diameter so a placed disc fits within the marker footprint).

## PickPlace Pairing and Sequencing

Default sequential pairing via the base `MultiPickStrategy`: `pick[0] -> target[0]`, `pick[1] -> target[1]`, …, `pick[7] -> target[7]`. All picks are identical (same asset type and color), all targets are identical, and the counts match 1:1 — so no custom pairing logic is needed.

## Success Condition

All 8 discs are placed on top of their paired yellow rectangle markers on the dropzone.

## Success Checks

- Each placed disc rests on top of its paired target marker (default `is_on_top` spatial check).
- No orientation check is needed: the `disc` asset has `AssetSymmetry(kind="continuous_axis", axis_local=[0,0,1])` (continuous rotation about Z), so the disc has no meaningful "upright" pose constraint after placement.
