# TableTaskMixedBoxPacking

## User Request (verbatim)

Cracker boxes, soup cans, and mustard bottles are arranged in alternating rows on the conveyor (a row of cracker boxes, a row of soup cans, a row of bottles, repeated twice). The conveyor is not moving, and all items are initially vertical. Pack three boxes on the cart so each box contains one cracker box, two soup cans, and two bottles. All items must be vertical inside their boxes.

## Task Overview

Pick 15 items (3 cracker_box, 6 soup_can, 6 mustard_bottle) from 6 alternating rows on the stationary conveyor and pack them into 3 open-top boxes on the cart. Each box receives exactly 1 cracker box + 2 soup cans + 2 mustard bottles. All items must remain upright (vertical) after placement. The conveyor has 18 total items (6 rows × 3), with 3 extra cracker boxes that are not picked.

## Concise Task Description

Pick cracker boxes, soup cans, and mustard bottles from alternating rows on the conveyor and pack them into three boxes on the cart (one cracker, two soups, two bottles per box), all upright.

## Pick Items

| Attribute       | Value |
|----------------|-------|
| Asset types    | cracker_box, soup_can, mustard_bottle |
| Total count    | 18 (6 per type), 15 picked (3 cracker + 6 soup + 6 bottle) |
| Arrangement    | 6 rows on stationary conveyor, alternating: cracker, soup, bottle (×2) |
| Items per row  | 3 |
| Row spacing    | 0.10m along X |
| Item spacing   | 0.12m along Y within each row |
| Orientation    | Upright (-90° X rotation for all USD assets) |
| Colors         | USD asset defaults |

## Target Objects

| Attribute       | Value |
|----------------|-------|
| Type           | 3 open-top boxes on cart |
| Box inner size | [0.22, 0.30] m |
| Box height     | 0.10 m |
| Box arrangement| Along Y, offsets -0.34, 0, +0.34 from cart center |
| Targets per box| 5 hidden virtual markers (1 cracker + 2 soup + 2 bottle positions) |
| Total targets  | 15 |
| Marker layout  | 2D grid: cracker at center-Y, soups and bottles in two rows at +Y |

## PickPlace Pairing and Sequencing

- **Strategy**: TypeBasedStrategy
- **Routing**: cracker_box → cracker markers (1 per box), soup_can → soup markers (2 per box), mustard_bottle → bottle markers (2 per box)
- **Pick order**: 3 paired crackers → 6 soups → 6 bottles → 3 extra crackers (skipped)
- **Name-prefix auto-detection**: items named `cracker_box_N`, `soup_can_N`, `mustard_bottle_N`

## Success Condition

All 15 picked items are inside their target boxes and upright.

## Success Checks

1. Each of the 15 placed items is contained within a box (containment_check)
2. Each placed item passes `is_vertical(max_tilt_deg=15)`
3. The 3 unpicked cracker boxes on the conveyor are treated as overflow picks (auto-pass)
