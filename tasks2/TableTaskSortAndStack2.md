# TableTaskSortAndStack2

## User Request

Sort colored cubes from a stacked source grid into matching color-coded boxes on the cart, while relocating yellow (distractor) cubes to dedicated stacks on the drop zone — make 6 stacks to the right of and a bit closer to the robot than the source pile — so no cubes are permanently blocked.

## Task Overview

- **Pick items**: 90 cubes (6 cols x 5 rows x 3 layers) on the dropzone floor, stacked 3 layers high. Colors randomly assigned from ["red", "green", "blue", "yellow"].
- **Target objects**: 3 open-top color-coded boxes (red, green, blue) on the cart, plus 6 floor markers on the dropzone for yellow cube stacks (2x3 grid, to the right (+X) and closer to the robot (-Y) than the source pile).
- **Generation**: Custom generator producing cubes top-down (layer 2 first) so picking order naturally follows top-down sequence.
- **Pairing**: Custom ColorSortRelocateStackStrategy (extends ColorSortStackStrategy) — classifies by color, routes R/G/B to matching boxes, routes yellow to dropzone stacks. No cubes skipped.
- **BT variant**: Cortex-style MotionCommand BT (`make_cortex_task_controller_tree`).

## Concise Task Description

Pick red, green, and blue cubes from a 6x5x3 stacked grid and sort them into matching color-coded boxes on the cart, stacking cubes on top of previously placed cubes. Yellow cubes are relocated to 6 stacks on the dropzone floor.

## Pick Items

- **Type**: cube (0.0515m per side)
- **Arrangement**: 6x5 grid on the dropzone floor, 3 layers high (90 cubes total)
- **Count**: 90 (all cubes picked — yellow cubes relocated rather than skipped)
- **Color/Appearance**: RandomChoice(["red", "green", "blue", "yellow"])
- **Generation order**: Top-down (layer 2, 1, 0) for natural top-down pick sequencing

## Target Objects

- **Type**: 3 open-top boxes on the cart + hidden markers (bottom-layer only for R/G/B) + 6 floor markers on the dropzone (for yellow stacks)
- **Arrangement**: 3 boxes in a row along Y on the cart (red, green, blue). 6 yellow stack markers in a 2x3 grid on the dropzone floor, to the right (+X) and closer to the robot (-Y) than the source pile, non-overlapping in both X and Y.
- **Markers**: 4 hidden bottom-layer markers per R/G/B box (12 total) + 6 hidden floor markers for yellow stacks in a 2x3 grid (18 total base markers). Upper-layer targets are dynamically added — each placed cube becomes the target for the next cube in that stack.
- **Color/Appearance**: Boxes tinted to match their color (muted red, green, blue). Yellow markers are hidden (no visual box).

## PickPlace Pairing and Sequencing

- **Pairing**: ColorSortRelocateStackStrategy — classifies each pick by color. R/G/B cubes routed to matching boxes (4 stacks per box, round-robin). Yellow cubes routed to 6 dropzone stacks (round-robin). Bottom-layer picks target base markers; upper-layer picks target the previously placed cube (dynamic stacking).
- **Sequencing**: Picks generated top-down from source. Picking order is layer-by-layer at destination: all bottom-layer placements first (across all 4 color groups), then all second-layer, etc. Runtime scan with wrap-around finds picks that satisfy both source and destination constraints.
- **Source stacking constraints**: `stacking_map` computed from pick positions enforces top-down pick ordering — a cube cannot be picked until all cubes directly above it at the source are completed.
- **Destination stacking constraints**: A pick targeting a dynamically added cube (upper-layer stacking target) is blocked until that target cube has been completed (placed).
- **No permanently blocked cubes**: Since yellow cubes are now moved (not skipped), no sort-color cubes are permanently blocked. All cubes in the grid are reachable.
- **Task completion**: The task completes when all 90 cubes have been picked and placed.

## Success Condition

All 90 cubes from the grid are placed in their correct destination: red, green, and blue cubes in their matching color-coded boxes on the cart; yellow cubes in the dropzone stacks region.

## Success Checks

- Each R/G/B pick cube is inside its matching color box (box_verification_info + containment_check=True, match_labels={"color": color}).
- Each yellow cube is within the yellow stacks region on the dropzone (virtual region spec with match_labels={"color": "yellow"}).
- Each color box/region contains only cubes of the matching color.
