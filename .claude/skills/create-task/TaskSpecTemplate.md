# TableTask<Name>

## User Request

<!-- The user's original request, verbatim. -->

## Task Overview

<!-- Summary of your understanding of the task. Be specific about:
- Which asset_type(s) for pick items (e.g., "soup_can", "cracker_box", "cube")
- Which asset_type(s) for targets (e.g., "marker", "rect", "madara_pad", or custom boxes)
- Which generation strategies (GridPositionGenerator, CircularPositionGenerator, custom)
- Which pairing strategy (sequential, ColorMatchStrategy, TypeBasedStrategy, BottlePickStrategy)
-->

## Concise Task Description

<!-- One imperative sentence describing what the robot should do. This becomes the task_description string. Example:
"Pick soup cans from the bin and place them upright onto red rectangles in the dropzone."
-->

## Pick Items

- **Type**: <!-- e.g., cube, soup_can, cracker_box, madara_bottle -->
- **Arrangement**: <!-- e.g., 3x3 Grid in the pick bin; Circle on the conveyor; Rows on the drop zone -->
- **Count**: <!-- e.g., 9; 2 cracker_box + 8 soup_can -->
- **Color/Appearance**: <!-- e.g., Random [red, green, blue]; USD Asset default -->

## Target Objects

- **Type**: <!-- e.g., Thin red rectangles; Madara pads; Collection boxes with virtual hidden markers -->
- **Arrangement**: <!-- e.g., 3x4 grid in the dropzone; 2 boxes on the cart -->
- **Markers**: <!-- If using containers: describe hidden marker layout inside each container. For box-packing tasks, use virtual target generation (LightweightObj markers generated at pairing time, not spawned in scene). -->
- **Color/Appearance**: <!-- e.g., Red; Hidden; Cardboard-colored boxes -->

## PickPlace Pairing and Sequencing

<!-- How picks map to targets:
- Sequential (pick[0] -> target[0], etc.)
- Color-matched (red cube -> red target)
- Type-based (cubes -> box A, balls -> box B)
- Custom (describe logic)

How sequencing works:
- Sequential from start
- Interleaved (1 box then 4 cans, repeat)
- Random
-->

## Success Condition

<!-- One sentence: what must be true when the task is complete.
Example: "All soup cans are placed upright on their target rectangles."
-->

## Success Checks

<!-- Specific verifiable checks. Examples:
- Each pick object rests on its paired target (is_on_top).
- Each pick object is inside its paired container (is_within).
- Each pick object is inside the correct box (centralized box_verification_info + containment_check=True) — DEFAULT for box/bin tasks.
- Placed soup cans are vertical (placement_constraints_fn with is_vertical).
- Placed cracker boxes are vertical (placement_constraints_fn with is_vertical).
- Each color box contains only items of the matching color (match_labels in box specs).
-->
