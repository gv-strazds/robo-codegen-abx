# TableTask3c2

## User Request

Very similar to TableTask3b, but it also spawns 2 additional red balls on the cart surface outside the pick bin. After placing all the balls from the bin, if at least 4 were placed, the robot places one of the red balls into the gap between the 4 placed balls. And if 6 balls from the bin were originally placed, then the 2nd red ball gets placed into the 2nd gap in the grid of 6 placed balls.

## Task Overview

Extends TableTask3b with a second phase. First, 3-6 balls are picked from the bin and placed into pocket gaps between discs in a tight 3x4 grid on the dropzone (same as 3b). Then, depending on how many bin balls were placed, 0-2 red balls from the cart surface are placed into gaps between the placed balls:

- 3 bin balls → 0 red balls (no complete 2x2 ball group)
- 4-5 bin balls → 1 red ball placed at center of first 2x2 ball group
- 6 bin balls → 2 red balls placed at centers of both 2x2 ball groups

The 2 red balls always spawn on the cart surface outside the bin but are only picked if enough bin balls were placed to form complete 2x2 groups. The "ball-gap" positions coincide with disc centers (col=1, row=1) and (col=1, row=2) in the disc grid — the red balls nestle on top of 4 placed balls, analogous to how the placed balls nestle on top of 4 discs.

## Concise Task Description

Pick balls from the bin into disc-gap pockets, then place red cart balls into gaps between the placed balls.

## Pick Items

- **Phase 1**: 3-6 balls (random colors) from 3x2 grid in pick bin
- **Phase 2**: 0-2 red balls from cart surface positions outside the bin
- **Total count**: 3-8 (varies by phase 1 count)
- **Scale**: [0.0515, 0.0515, 0.0515]

## Target Objects

- **Phase 1 targets**: Hidden markers at disc-gap positions (2x3 grid of midpoints between 2x2 disc groups), count matches phase 1 ball count
- **Phase 2 targets**: Hidden markers at ball-gap positions (center of 2x2 placed-ball groups), count matches phase 2 ball count
- **Visual objects**: 12 discs in 3x4 grid (workspace setup, not targets)
- **Disc grid spacing**: ~0.105m center-to-center (2mm edge gap)
- **Disc colors**: cycling purple/cyan/black/yellow

## PickPlace Pairing and Sequencing

Sequential (default): picks generated in order (bin balls first, then red balls), targets in matching order. pick[i] → target[i]. Bin balls fill disc gaps, red balls fill ball gaps.

## Success Condition

All generated balls are at their target gap positions: bin balls in disc pockets, red balls in ball pockets (resting on 4 placed balls).

## Success Checks

- Custom spatial check with two Z levels:
  - Bin balls: XY proximity to disc-gap center + Z near disc-pocket height
  - Red balls: XY proximity to ball-gap center + Z near ball-pocket height
- XY tolerance: disc_spacing × 0.4 (~42mm)
- Z tolerance: ball_radius (51.5mm)

## Geometry Notes

- Ball radius = disc radius = 0.0515m
- Disc spacing = 0.105m (2mm edge gap)
- Disc-pocket (ball on 4 discs): d_contact = 0.02275m, pocket_height = 0.04620m
- Ball-pocket (ball on 4 balls): d_horiz = disc_spacing × √2/2 = 0.07425m, dz = √((2R)² - d²) = 0.07138m
- Red ball center Z = ball_center_z_pocket + 0.07138m ≈ 0.170m
- Red balls on cart surface at Y=0.25 (well in front of bin Y range 0.40-0.80)
