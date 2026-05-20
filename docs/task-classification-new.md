# Task Classification by Complexity (tasks2/)

This document classifies the 30 task classes registered under `tasks2/`
into 5 complexity categories.  The classification is derived from the
features each task uses in its `TaskSpec` (scene/description side) and
its nested `TaskImplementationSpec` (execution/policy side).

The supporting data lives in `docs/classification_data/`:

- `task_features.json` — per-task raw feature inventory (corrected from
  the initial extraction by auditing parent files in `tasks/`).
- `features_by_task.json` — derived per-task feature sets, scores, and
  category assignments.
- `task_graph.json` — partial-order graph over outer `TaskSpec` features
  (with equivalence-class collapse for readability).
- `impl_graph.json` — partial-order graph over `TaskImplementationSpec`
  features (with equivalence-class collapse).
- `categories.json` — category thresholds and per-task assignments.

## Methodology

For each task we extracted the full `TaskSpec` + `TaskImplementationSpec`
by reading the `tasks2/` wrapper and merging it with its parent class in
`tasks/` (most wrappers are thin overrides on `_customize_spec`; a few
— `TableTaskSoupCans2`, `TableTaskBottlesToConveyor2[x]` — define their
own spec from scratch).

We then projected each task onto two binary feature vectors:

1. **Outer-task feature vector** (37 features): pick source, target
   destination, asset families, generator types, conveyor configuration
   (only "moving" if the parent or wrapper sets `conveyor_speed`),
   stacking, verification semantics, incremental/spatial-trigger
   spawning, and randomization.  The verification axis carries an
   umbrella `verify:custom_spatial_check` flag (weight 2) that fires
   whenever the task supplies a non-default `spatial_check_fn`, plus
   finer-grained sub-flags (`verify:vertical_check`, `verify:is_within`,
   `verify:proximity_check`) when the custom check actually invokes
   those library helpers.
2. **Implementation feature vector** (19 features): pairing-strategy
   family (sequential → attribute-routing → stacking-aware → dynamic),
   BT tree factory (default → cortex → specialized: cuRobo or
   LowerToPlace experiment), policy hooks (virtual targets, reachability
   gates), and tuning overrides (timeouts, EE-height, posture, hover,
   approach thresholds, etc.).

Two partial-order graphs are constructed by drawing an edge `A → B`
whenever B's feature set is a strict superset of A's on the relevant
axis (transitive reduction is then applied).  Equivalence classes of
tasks with identical feature sets collapse into a single node.

Each feature is assigned a hand-picked complexity weight reflecting
roughly how much new framework machinery it touches (e.g. switching to
a cortex BT is weight 1; switching to a dynamic-selection strategy is
weight 5; a moving conveyor is weight 3; a custom generator is weight
3).  Per-task scores are the sums on each axis; the combined score is
the sum of both.  Categories are obtained by bucketing combined scores
at the natural gaps in the score distribution.

The full weight tables are inside `task_graph.json` and `impl_graph.json`.

**A note on "conveyor" tasks.**  Many tasks position items along the
conveyor surface as a flat layout zone but do *not* set
`TaskSpec.conveyor_speed`, so the belt is stationary.  Only three task
parents actually configure a moving belt:
`tasks/table_task_bottles_to_conveyor.py`,
`tasks/table_task_conveyor_type_sort.py`, and the from-scratch spec in
`tasks2/table_task_soup_cans_2.py`.  The score "moving conveyor" weight
fires only for those (and their tasks2 variants).

## At-a-glance summary

| Cat | Name                                | Count | Combined-score range |
|----:|-------------------------------------|------:|---------------------:|
|  1  | Basic Sequential                    |   7   | 1 – 4                |
|  2  | Attribute, Custom Source, or Specialised BT |   7   | 5 – 8                |
|  3  | Sorting, Stacking & Containers      |   7   | 12 – 19              |
|  4  | Multi-feature Packing & Routing     |   6   | 20 – 25              |
|  5  | Advanced Dynamic / JIT              |   3   | 31 – 35              |

There is a clean natural gap between scores 8 and 12 (no tasks land
there) — this corresponds to the qualitative jump from "single
specialised extension" to "multi-feature task with
sorting/stacking/containers".

---

## Implementation-side partial order

The implementation features form a smaller, more highly-structured space
(19 features) because most v2 wrappers reuse the same execution policy
machinery.  Equivalence-class collapse produces 15 nodes:

| Class | Tasks | Key features added (vs. parent classes) |
|-------|-------|-----------------------------------------|
| I-C00 | TableTask2v2, TableTask3v2, TableTask4v2, TableTask5v2, TableTaskIncrementalTargets2, TableTaskLayeredCubes2 | sequential strategy + cortex BT (baseline) |
| I-C01 | TableTask3b2, TableTask3c2, TableTaskColorCircle2, TableTaskMixedCircle2, TableTaskSoupCanPacking2 | + `virtual_target_generation_strategy` |
| I-C02 | TableTaskBottles2, TableTaskColorBinSort2, TableTaskColorShapes2, TableTaskColors2, TableTaskShapeSortBoxes2 | + attribute-routing strategy (Color/Type/Bottle) |
| I-C03 | TableTaskCrackerBoxes2 | + `ee_height_for_move` override |
| I-C04 | TableTask1v2, TableTaskConveyorSort2 | I-C01 ∪ I-C02 (virtual targets + attribute routing) |
| I-C05 | TableTaskCartToConveyor2, TableTaskMixedPacking2 | I-C01 ∪ I-C03 (virtual targets + ee_height) |
| I-C06 | TableTaskConveyorColorStacks2 | virtual targets + stacking-aware strategy |
| I-C07 | TableTaskLayeredCircle2 | stacking-aware strategy + ee_height |
| I-C08 | TableTask3Curobo | + cuRobo plan-and-stream tree, timeout overrides |
| I-C09 | TableTaskBottlesToConveyor2 | + dynamic-selection strategy + reachability fn + pick approach tuning |
| I-C10 | TableTaskConveyorTypeSort2 | attribute routing + ee_height + pick approach tuning |
| I-C11 | TableTaskSortAndStack2 | virtual targets + stacking-aware + ee_height |
| I-C12 | TableTaskBottlesToConveyor2x | I-C09 + specialised tree (LowerToPlace) |
| I-C13 | TableTaskSoupCans2 | dynamic-selection + reachability + pick/place posture overrides + descent threshold tuning |
| I-C14 | TableTaskCrackerCircle2 | virtual + stacking-aware + ee_height + place hover/approach + startup delay |

Direct-extension edges (Hasse diagram):

```
I-C00 ──┬─► I-C01 ──┬─► I-C04
        │           ├─► I-C05 ──┐
        │           └─► I-C06 ──┤
        ├─► I-C02 ──┬─► I-C04   │
        │           └─► I-C10   │
        ├─► I-C03 ──┬─► I-C05 ──┤
        │           ├─► I-C07 ──┤
        │           └─► I-C10   │
        ├─► I-C08                ▼
        ├─► I-C09 ─────────► I-C12
        └─► I-C13              I-C11 ─► I-C14
```

The tree branches by feature family:

- **virtual-targets branch** (I-C01) is the most common single extension
  on top of the baseline.
- **attribute-routing branch** (I-C02) covers color and type strategies.
- **stacking-aware branch** is reached either from virtual-targets
  (I-C06) or from `ee_height` (I-C07), and the most decorated stacking
  task (I-C14, CrackerCircle) sits at the top of this branch.
- **dynamic-selection branch** (I-C09 → I-C12) is the most exotic
  pairing logic; only the conveyor-bottles tasks use it.
- **specialised tree branch** (I-C08 = cuRobo, I-C12 = LowerToPlace
  experiment) is orthogonal to the strategy axis.

The full edge list and per-task class mapping is in `impl_graph.json`.

---

## Outer-task partial order

The outer `TaskSpec` feature space has 35 features and is much sparser
— most tasks have a unique feature footprint.  After equivalence-class
collapse, 27 classes remain (most singletons; the
`(asset:primitive, dst:grid, pick:bin)` baseline groups three tasks
that differ only on the implementation side, and the
`BottlesToConveyor2 / 2x` pair shares an outer footprint).

Selected direct-extension edges (full graph in `task_graph.json`):

```
{TableTask3v2, TableTaskColors2, TableTask3Curobo}  (bin → grid, primitive)
   ├─► TableTask1v2                  (+ custom shuffled-color pick generator)
   ├─► TableTask3b2                  (+ custom proximity spatial_check_fn + virtual targets)
   └─► TableTaskLayeredCubes2        (+ layered source, + stacking_enabled)

TableTask3b2 ──► TableTask3c2        (+ second pick source: cart)
TableTask4v2 ──► TableTaskColorCircle2 (+ virtual_target_generation)
TableTaskCrackerBoxes2 ──► TableTaskBottles2 (+ custom spatial_check_fn: is_within + is_vertical)
TableTaskLayeredCircle2 ──► TableTaskCrackerCircle2 (+ virtual targets)
TableTaskShapeSortBoxes2 ──► TableTaskConveyorSort2 (+ virtual targets)
TableTaskColorShapes2 ──► TableTaskColorBinSort2 (+ random pick count + incremental spawning)
TableTaskSoupCanPacking2 ──► TableTaskMixedPacking2 (+ multi-asset)
```

Tasks higher up in the partial order (no incoming superset) sit at the
top of their feature branch and tend to combine multiple axes:

- `TableTaskSoupCans2` — moving conveyor as destination + dynamic
  bursts + falloff capture + USD asset
- `TableTaskBottlesToConveyor2[x]` — layered source + stacking +
  moving conveyor destination + spatial-trigger replenishment
- `TableTaskMixedCircle2` — multi-asset (mixed primitive+USD) +
  circular layout + virtual targets + containment + placement constraints
- `TableTaskMixedPacking2`, `TableTaskCartToConveyor2`,
  `TableTaskSoupCanPacking2` — multi-asset packing into containers on
  cart with custom generators

---

## The 5 categories

### Category 1 — Basic Sequential pick-and-place  (7 tasks, score 1 – 4)

Pure cortex-tree variants of the original "Task N" tutorial tasks.
Sequential pairing (`MultiPickStrategy`), single-asset family (primitive
cubes/balls or one USD type), standard generators (`GridPosition` or
`CircularPosition`), no conveyor, no containment, no custom
verification.  The only impl-side change vs. the original tasks is
`tree_factory = make_cortex_task_controller_tree`.

| Task | Notes |
|------|-------|
| `TableTask3v2` | 9 balls bin → disc grid (cortex BT) |
| `TableTask2v2` | 7 cubes dropzone line → blue cube grid |
| `TableTask4v2` | 9 cubes bin → yellow-rect circle |
| `TableTask5v2` | 6 green cubes dropzone-grid → red-rect circle |
| `TableTaskColors2` | 16 colored cubes bin → matching cube grid (`ColorMatchStrategy`, default `is_on_top` verification) |
| `TableTaskCrackerBoxes2` | 12 cracker boxes bin → green-rect grid (first USD asset; `ee_height_for_move=0.45`) |
| `TableTaskColorCircle2` | 9 colored cubes bin → marker circle (virtual targets, sequential pairing) |

### Category 2 — Attribute routing, custom source, or specialised BT  (7 tasks, score 5 – 8)

Tasks that take exactly one step beyond the Cat-1 baseline — either by
supplying a custom `spatial_check_fn`, spawning items incrementally over
time, adding a second pick source, introducing a custom pick generator,
switching to a specialised BT (cuRobo), or pre-stacking the source items.

| Task | Step taken on top of Cat-1 |
|------|----------------------------|
| `TableTaskIncrementalTargets2` | Incremental pick + target spawning |
| `TableTask3b2` | Custom XY+Z proximity `spatial_check_fn` (default `is_on_top` is too strict for ball-in-pocket geometry) + virtual targets |
| `TableTask3c2` | Two pick sources (bin + cart) + the same custom proximity check, with per-target-type expected Z |
| `TableTaskBottles2` | `BottlePickStrategy` (drop-orientation tweak) + bottle/pad assets + custom `spatial_check_fn` (`is_within` + `is_vertical`) |
| `TableTask1v2` | Custom `ShuffledColorPickGenerator` + `ColorMatchStrategy` + virtual targets |
| `TableTask3Curobo` | cuRobo plan-and-stream tree + sim-time timeout overrides (scene side identical to TableTask3) |
| `TableTaskLayeredCubes2` | Layered source (3-layer bin stack) + `stacking_enabled=True` (still sequential pairing for unstack-to-flat-grid) |

### Category 3 — Sorting, Stacking & Containers  (7 tasks, score 12 – 19)

The first big qualitative jump.  Each Cat-3 task combines several Cat-2
features at once: a non-trivial source (conveyor-zone or layered
arrangement) plus either a non-trivial destination (boxes with
containment verification, or a stacking goal) plus an attribute- or
stacking-aware pairing strategy.  Conveyors here are stationary layout
zones (no `conveyor_speed` set on the spec).

| Task | Source | Destination | Strategy / notable feature |
|------|--------|-------------|-----------------------------|
| `TableTaskConveyorColorStacks2` | conveyor-zone (primitives) | color-triplet stacks in bin | `ColorStackStrategy` + virtual targets |
| `TableTaskShapeSortBoxes2` | conveyor-zone (cube+ball) | shape-matching boxes on cart | `TypeBasedStrategy` + containment |
| `TableTaskColorShapes2` | conveyor-zone (4-shape mix) | 3 color-matching boxes | `ColorMatchStrategy` + custom `ColorShapesGenerator` |
| `TableTaskConveyorSort2` | conveyor-zone (cube+ball) | shape-matching boxes on cart | `TypeBasedStrategy` + virtual targets |
| `TableTaskLayeredCircle2` | layered circular sugar-box arrangement | single bin stack | `SingleStackStrategy` + `ee_height` override |
| `TableTaskColorBinSort2` | conveyor-zone + incremental + random count | 3 color bins | `ColorMatchStrategy` + custom `ColorBinSortGenerator` + random pick count (1–5) |
| `TableTaskMixedCircle2` | mixed-asset circle on conveyor-zone | bin container on cart | `MixedOrientationStrategy` + `MixedScaleStrategy` + per-type placement constraints |

### Category 4 — Multi-feature packing & routing  (6 tasks, score 20 – 25)

Tasks that stack 3+ orthogonal features: a non-bin source AND a
container-based destination AND custom generators (or USD multi-asset)
AND something extra (stacking, posture/EE-height tuning, attribute
routing).  They're recognisably "industrial" scenarios.  Aside from
`ConveyorTypeSort2`, the conveyor surfaces are stationary layout zones.

| Task | What stacks up |
|------|----------------|
| `TableTaskSoupCanPacking2` | Conveyor-zone + custom `ConveyorRowsGenerator` + 24 cans + 4 cart boxes + containment + per-type vertical + virtual targets |
| `TableTaskCartToConveyor2` | Cart source + custom `CartPickGenerator` + 4 USD asset types + boxes on conveyor + containment + per-type vertical constraints + `ee_height=0.45` + virtual targets |
| `TableTaskCrackerCircle2` | Layered circular cracker-box source + `SingleStackStrategy` + virtual targets + `ee_height=0.28` + tightened place-hover/approach + 1 s startup delay |
| `TableTaskMixedPacking2` | Conveyor-zone + custom `MixedRowGenerator` + mixed USD (box+can) + 2 cart boxes + containment + per-type vertical + virtual targets + `ee_height=0.35` |
| `TableTaskSortAndStack2` | 90-cube 3-layer source + `ColorStackStrategy` + virtual targets + ee_height + dual-destination (cart boxes + dropzone stacks) |
| `TableTaskConveyorTypeSort2` | **Moving** conveyor + mixed-asset + `TypeBasedStrategy` + cart boxes + random pick count + incremental spawning + `ee_height` + pick approach threshold tuning |

### Category 5 — Advanced Dynamic / JIT  (3 tasks, score 31 – 35)

Tasks at the top of the partial order on **both** axes.  All three add
a dynamic-selection pairing strategy (`DynamicTopPickStrategy` or
`ConveyorProximityStrategy`) on top of an already complex outer
TaskSpec, and all three involve a moving conveyor on at least one side.

| Task | What pushes it past Cat-4 |
|------|---------------------------|
| `TableTaskSoupCans2` | `ConveyorProximityStrategy` JIT pairing + moving-conveyor *targets* (dynamic red rectangles) + custom `BurstRectTargetGenerator` + falloff capture + `down_to_insert_z_thresh` tuning + pick/place posture overrides + per-step belt-velocity priming + physics-material damping |
| `TableTaskBottlesToConveyor2` | `DynamicTopPickStrategy` JIT top-Z pick + layered source + bottles → moving conveyor pads + spatial-trigger target replenishment + falloff capture + `target_reachable_fn` + tight pick approach |
| `TableTaskBottlesToConveyor2x` | Same as above + experimental `cortex_lowertoplace_experiment` tree (FK-anchored descent) |

---

## How this differs from the earlier `docs/task_classification.md`

The earlier document classified 19 of the original-set tasks into 4
categories.  The new classification:

- Covers all 30 tasks2 task variants (the cortex-tree, cuRobo-tree, and
  experiment-tree wrappers — plus tasks not present in the earlier doc
  such as `TableTaskIncrementalTargets2`, `TableTaskSortAndStack2`,
  `TableTaskShapeSortBoxes2`, `TableTaskConveyorTypeSort2`).
- Splits the framework into two orthogonal feature axes (outer
  TaskSpec vs. TaskImplementationSpec) and builds an explicit partial
  order on each.
- Uses five categories rather than four — the new split separates the
  "single-extension" tasks (Cat 2, e.g. `TableTask1v2`,
  `TableTask3Curobo`, `TableTaskLayeredCubes2`) from the
  multi-attribute "Sorting/Stacking/Containers" tasks (Cat 3).  That
  gap was where the old Category-2 ("Attribute Routing") and
  Category-3 ("Sorting & Stacking") boundary used to sit; the new
  scheme makes it a sharper qualitative break.
- Corrects the older notion that every "Conveyor*" task uses a moving
  belt — most are stationary layout zones; only the three Cat-5
  tasks (`SoupCans2`, `BottlesToConveyor2[x]`) and Cat-4's
  `ConveyorTypeSort2` actually run with `conveyor_speed > 0`.
- Re-ranks several tasks: e.g. `TableTaskBottles2` drops to Cat 2
  (single extension on Cat 1 — only the BottlePick strategy);
  `TableTaskCrackerCircle2` rises into Cat 4 because of its full stack
  of impl-side tuning overrides.

The mapping back to the older categories is roughly:

| Old | New |
|-----|-----|
| Basic              | Cat 1 (mostly) + parts of Cat 2 |
| Attribute Routing  | Cat 2 (TableTask1v2, Bottles2) + Cat 3 (ColorShapes2, ConveyorSort2, ColorBinSort2) |
| Sorting & Stacking | Cat 3 (LayeredCircle2, ConveyorColorStacks2) + Cat 4 (CrackerCircle2, SortAndStack2) |
| Complex Scenarios  | Cat 4 (Packing tasks) + Cat 5 (BottlesToConveyor, SoupCans2) |
