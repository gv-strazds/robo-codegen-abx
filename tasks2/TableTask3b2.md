# TableTask3b2

## User Request

Similar to TableTask3 but with differences: 1) space the grid of discs closer together so that they are almost touching each other; 2) place the balls not onto the discs, but onto the gaps between 4 adjacent discs in the grid. Since the balls are bigger than the gaps, the robot will not be able to place them all the way to the dropzone surface, so calculate a good height for releasing the balls. Task success checking will also no longer be based on whether the balls are directly on top of the discs, but will need to check that they are in correct positions at the grid points between the discs. Since the balls should end up resting on 4 neighboring discs, this placement should be stable.

## Task Overview

Balls (`ball`) are picked from a 3x2 grid in the pick bin (3-6 balls, randomly chosen) and placed into the pocket gaps formed between 4 adjacent discs in a tight 3x4 disc grid on the dropzone. The disc grid uses spacing of ~0.105m (2mm gap between disc edges) so balls nestle stably between 4 neighboring discs rather than balancing on top of a single disc. Target markers are invisible virtual `LightweightObj` at the gap positions (2x3 grid = 6 positions). Sequential pairing maps each ball to a gap position.

## Concise Task Description

Pick balls from the bin and place them into gaps between disc targets arranged in a tight 3x4 grid on the dropzone table.

## Pick Items

- **Type**: ball
- **Arrangement**: 3x2 grid in the pick bin (spacing_x=0.08m, spacing_y=0.08m)
- **Count**: random 3-6 (from 6-capacity grid)
- **Color/Appearance**: Random (default)

## Target Objects

- **Type**: hidden markers (virtual `LightweightObj`) at gap positions between disc grid
- **Visual objects**: 12 discs in a 3x4 grid (created in workspace setup, not targets)
- **Arrangement**: 2x3 grid of gap positions (midpoints of each 2x2 disc group)
- **Count**: 6 gap positions (3 used by balls)
- **Disc grid spacing**: ~0.105m center-to-center (discs almost touching, 2mm edge gap)
- **Disc colors**: SequentialChoice(["purple", "cyan", "black", "yellow"], loop=True)

## PickPlace Pairing and Sequencing

Sequential (default): pick[i] -> gap_marker[i]. With 3-6 balls and 6 gap positions, all balls get a target.

## Success Condition

All generated balls (3-6) are placed at their target gap positions, resting stably in pockets formed by 4 neighboring discs.

## Success Checks

- Each ball rests on top of its paired gap marker (default `is_on_top` check).
- The marker Z is computed so the ball release height matches the pocket geometry (ball center height above disc tops, calculated from contact geometry).

## Geometry Notes

- Ball radius = disc radius = 0.0515m (at scale [0.0515, 0.0515, 0.0515])
- Disc diameter = 0.103m, disc height = 0.0515m
- Disc spacing = 0.105m (2mm gap between edges)
- Distance from gap center to nearest disc rim: d = spacing * sqrt(2)/2 - disc_radius = 0.02275m
- Ball center height above disc top: h = sqrt(ball_radius^2 - d^2) = 0.04619m
- Ball sinks ~5.3mm below disc top surface (stable pocket)
