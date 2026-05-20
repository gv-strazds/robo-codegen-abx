# TableTaskSortAndStack

## User Request

On the conveyor is a grid of red, green, blue and yellow cubes, 6 x 5 x 3 layers deep, randomly interleaved in terms of color. Sort the red, green and blue cubes into 3 color coded boxes on the cart, each big enough to hold 4 stacks of cubes.

## Task Overview

- **Pick items**: 90 cubes (6 cols x 5 rows x 3 layers) on the dropzone floor, stacked 3 layers high. Colors randomly assigned from ["red", "green", "blue", "yellow"].
- **Target objects**: 3 open-top color-coded boxes (red, green, blue) on the cart. Each box has 4 bottom-layer hidden markers (2x2 grid). Upper-layer targets are the previously placed cubes themselves (dynamic stacking like SingleStackStrategy).
- **Generation**: Custom generator producing cubes top-down (layer 2 first) so picking order naturally follows top-down sequence.
- **Pairing**: Custom ColorSortStackStrategy — classifies by color, routes to matching box, builds round-robin stacks within each box. Yellow cubes skipped.

## Concise Task Description

Pick red, green, and blue cubes from a 6x5x3 stacked grid and sort them into matching color-coded boxes on the cart, stacking cubes on top of previously placed cubes in 2x2 grids within each box.

## Pick Items

- **Type**: cube (0.0515m per side)
- **Arrangement**: 6x5 grid on the dropzone floor, 3 layers high (90 cubes total)
- **Count**: 90 (effectively ~67 picked; ~23 yellow cubes skipped on average)
- **Color/Appearance**: RandomChoice(["red", "green", "blue", "yellow"])
- **Generation order**: Top-down (layer 2, 1, 0) for natural top-down pick sequencing

## Target Objects

- **Type**: 3 open-top boxes on the cart + 4 virtual hidden markers per box (bottom-layer only)
- **Arrangement**: 3 boxes in a row along Y on the cart (red, green, blue)
- **Markers**: 4 hidden bottom-layer markers per box in a 2x2 grid (12 total). Upper-layer targets are dynamically added — each placed cube becomes the target for the next cube in that stack.
- **Color/Appearance**: Boxes tinted to match their color (muted red, green, blue)

## PickPlace Pairing and Sequencing

- **Pairing**: ColorSortStackStrategy — classifies each pick by color. R/G/B cubes routed to matching boxes. Within each box, picks assigned round-robin to 4 stack positions. Bottom-layer picks target base markers; upper-layer picks target the previously placed cube (dynamic stacking).
- **Sequencing**: Picks generated top-down from source. Picking order is layer-by-layer at destination: all bottom-layer placements first (across all 3 boxes), then all second-layer, etc. Runtime scan with wrap-around finds picks that satisfy both source and destination constraints.
- **Source stacking constraints**: `stacking_map` computed from pick positions enforces top-down pick ordering — a cube cannot be picked until all cubes directly above it at the source are completed.
- **Destination stacking constraints**: A pick targeting a dynamically added cube (upper-layer stacking target) is blocked until that target cube has been completed (placed). This prevents placing a cube on top of a cube that hasn't been moved to its destination yet.
- **Permanently blocked cubes**: Sort-color cubes transitively blocked by yellow cubes above them are excluded from pairings entirely (they can never be picked in real physics).
- **Yellow cubes**: Skipped (paired with None, excluded from picking order).
- **Task completion**: The task stops when no more picks satisfy both source and destination stacking constraints. This naturally handles yellow-blocked cubes.

## Success Condition

All accessible red, green, and blue cubes from the grid are inside their matching color-coded box on the cart. Cubes permanently blocked by yellow cubes above them are excluded.

## Success Checks

- Each paired pick cube is inside its matching color box (box_verification_info + containment_check=True).
- Each color box contains only cubes of the matching color (match_labels={"color": color}).
- Yellow cubes are left unpicked (skipped by strategy).
