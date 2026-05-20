# TableTask1

## User Request

Create a new task that takes red, green, blue and orange cubes from the pick bin (4 of each color, randomly arranged in terms of color) and places them in a tightly packed square on the conveyor, so that the colors form horizontal stripes.

## Task Overview

16 cubes (4 red, 4 green, 4 blue, 4 orange) are spawned in a 4x4 grid in the pick bin with randomly shuffled colors. The robot picks them and places them onto a 4x4 grid of hidden markers on the dropzone (conveyor area), tightly packed so cubes nearly touch. Each row of 4 markers shares a single color label, so ColorMatchStrategy routes each cube to the row matching its color, forming horizontal stripes.

## Concise Task Description

Pick colored cubes from the bin and arrange them in a tightly packed 4x4 grid on the conveyor with horizontal color stripes.

## Pick Items

- **Type**: cube
- **Arrangement**: 4x4 grid in the pick bin, randomized slot order
- **Count**: 16 (4 red, 4 green, 4 blue, 4 orange)
- **Color/Appearance**: 4 each of red, green, blue, orange; randomly shuffled

## Target Objects

- **Type**: Hidden markers (invisible placement targets)
- **Arrangement**: 4x4 tightly packed grid on the dropzone
- **Markers**: 4 rows of 4 markers; each row labeled with one color (red, green, blue, orange top to bottom)
- **Color/Appearance**: Hidden (invisible), with semantic color labels for pairing

## PickPlace Pairing and Sequencing

- ColorMatchStrategy with color_palette=["red", "green", "blue", "orange"]
- Each cube is paired to a target marker of the same color
- Pick order is determined by the strategy (first available unpaired cube)

## Success Condition

All 16 cubes are placed on their color-matched target markers, forming four horizontal stripes of solid color.

## Success Checks

- Each pick cube rests on its paired hidden marker target (is_on_top).
- Each row contains only cubes of the designated color (enforced by ColorMatchStrategy pairing).
