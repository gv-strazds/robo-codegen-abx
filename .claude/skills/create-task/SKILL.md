---
name: create-task
description: Generate a new pick-and-place task for the Isaac Sim UR10 robot. Use when the user asks to create, generate, or design a new task, scenario, or TableTask subclass. Also trigger when the user describes a pick-and-place scenario they want (e.g., "sort bottles by color", "pack cans into boxes", "arrange cubes in a circle", "pick items from the conveyor") even without explicitly asking for a "task class".
argument-hint: [description of what the task should do]
user-invocable: true
---

# Task Creator

Create a new `UR10MultiPickPlaceTask` subclass in the `tasks/` directory. Follow these phases in order.

## Phase 1: Understand Requirements

If the user's request is unambiguous and maps directly to known patterns (e.g., "create a task that picks 6 red cubes and places them on markers"), proceed directly to Phase 2 without asking questions. Only ask for clarification when the request is genuinely underspecified — for example, no indication of object types, conflicting requirements, or multiple plausible interpretations. When you do need to clarify, focus on:

1. **Pick items**: What objects? How many? What arrangement (grid, circle, conveyor rows)?
2. **Target objects**: Markers on a surface? Pads? Collection boxes on the cart? How arranged?
3. **Pairing logic**: Sequential (default)? Color-matched? Type-based? Custom?
4. **Success criteria**: What must be true for the task to count as successful? Are there orientation constraints (e.g., bottles/cans must be upright)?

Reference: [reference/assets-and-workspace.md](reference/assets-and-workspace.md) for available asset types and workspace coordinates.

## Phase 2: Write the Task Specification

Create `tasks/TableTask<Name>.md` using the template below. This documents the task design before implementation.

Template: [TaskSpecTemplate.md](TaskSpecTemplate.md)

The spec must include:
- **User Request** (verbatim)
- **Task Overview** (your interpretation, naming specific asset types and strategies)
- **Concise Task Description** (one imperative sentence)
- **Pick Items** (types, arrangement, count, colors)
- **Target Objects** (types, arrangement, markers, colors)
- **PickPlace Pairing and Sequencing** (how picks pair to targets, what order)
- **Success Condition** (one sentence)
- **Success Checks** (specific verifiable checks — e.g., "placed cracker boxes are vertical")

## Phase 3: Implement the Task Class

Create `tasks/table_task_<name>.py` with a class `TableTask<Name>` inheriting from `UR10MultiPickPlaceTask`.

**Before writing code**, read 1-2 existing task files that are closest to the requested task as implementation references:
- Simple marker placement: `tasks/table_task_cracker_boxes_1.py` or `tasks/table_task_soup_cans_1.py`
- Color matching with boxes: `tasks/table_task_color_shapes.py`
- Box-packing with virtual targets: `tasks/table_task_soup_can_packing.py`
- Stacked/layered items (pick from stacks): `tasks/table_task_layered_cubes.py`
- Stacking into bin (SingleStackStrategy): `tasks/table_task_layered_circle.py`
- Color-layered stacking (LayeredStackStrategy): `tasks/table_task_conveyor_color_stacks.py`
- Color-sorted stacking with distractor relocation (ColorSortRelocateStackStrategy): `tasks/table_task_sort_and_stack.py`
- Color-sorted stacking into boxes with skipped distractors (ColorSortStackStrategy, deprecated): `flawed_tasks/table_task_sort_and_stack.py`
- Items into KLT bin (box containment): `tasks/table_task_mixed_circle.py`
- Mixed object types: `tasks/table_task_mixed_packing.py`
- Type-based sorting (cubes/balls into separate boxes): `tasks/table_task_conveyor_sort.py`
- Bottle placement (upright into sockets/pads): `tasks/table_task_bottles_to_conveyor.py`

Template: [task_template.py](task_template.py)

Tasks create a `TaskSpec` object (from `task_spec.py`) that bundles scene-side configuration: generation strategies, workspace setup, verification semantics (`spatial_check_fn`, `placement_constraints_fn`, `containment_check`, `box_verification_info`), and human-readable scene metadata (`scenario`, `pick_description`, `target_description`, `verification_description`, `rationale`). Execution-policy fields — the pairing strategy factory (`create_strategy`), BT tree factory (`tree_factory`), virtual-target generator, postures, hover heights, watchdog timeouts, reachability gates, cuRobo flags, plus `strategy_description` metadata — live on a nested `TaskImplementationSpec`, assigned via `implementation=TaskImplementationSpec(...)`. The full `TaskSpec` is passed to `super().__init__(task_spec=spec, offset=offset, **kwargs)`. v2 / cortex / cuRobo subclasses override `_customize_spec(spec)` and use `spec.with_impl(tree_factory=..., ...)` to swap policy fields without touching the description side.

For implementation details (position generators, attribute strategies, pairing strategies, virtual targets, USD orientation, workspace setup, verification, and TaskSpec metadata), read: [reference/implementation-guide.md](reference/implementation-guide.md)

For specific API details, read the relevant reference files as needed:
- [reference/generation-patterns.md](reference/generation-patterns.md) — position generators, attribute strategies, custom generators, virtual targets
- [reference/strategies.md](reference/strategies.md) — pairing strategy classes
- [reference/verification.md](reference/verification.md) — verification patterns and `spatial_check_fn`
- [reference/assets-and-workspace.md](reference/assets-and-workspace.md) — asset types, workspace coordinates, `spawn_open_box`

## Phase 4: Test with Mock Runner

Run the task through the mock py_trees executor (no Isaac Sim needed):

```bash
mamba run -n env_isaacsim51 env PYTHONPATH=$(pwd)/extsMock:$(pwd) python run_mock_task.py --task TableTask<Name>
```

Check the output for:
- `Completed successfully.` — verification passed
- `Verification checks reported UNSUCCESSFUL completion.` — verification failed; check failure messages
- Traceback — code error; fix and re-run
Add `--show-status` for detailed py_trees tree state per tick (useful for debugging).
Add --seed <random_seed> to specify a random seed for the task.
If the task setup includes some randomization, run the mock task multiple times with different seeds to verify that the task is robust to randomization.

## Phase 5: Agent Self-Check (Snapshot-Based, No GUI Required)

Once the mock and unit tests pass, your *first* full-sim validation is non-interactive — you do this yourself before bothering the user. Run the task headless with teleport (skips motion planning, so it's fast) and `--snapshot-errors` (writes PNG + JSON only on BT failure events, watchdog timeouts, verification failures, plus a guaranteed task-final frame):

```bash
mamba run -n env_isaacsim51 python run_task.py --task TableTask<Name> --headless --teleport --snapshot-errors --auto-exit
```

The run prints the output directory, something like `_results/snapshots/<task>_YYYYmmdd_HHMMSS/`. Inspect what landed there:

1. **Verify the run completed without errors** (return code 0; no traceback in the log).
2. **Read the task-final frame's sidecar JSON.** The file is `task_verified_pickNN_t*.json` next to its PNG. Check `task_successful: true` and an empty `verification_failures: []`.
3. **Look at the task-final PNG.** Confirm the end-state matches intent: items in the right places, upright if required, no items toppled or stuck. Treat this as the canonical "does the final state look reasonable" check.
4. **If any failure-event frames are present** (e.g., `pick_unreachable_*.png`, `grasp_slipped_*.png`, `timeout_descent_*.png`, `verify_fail_*.png`) — open the PNG and read the matching `.json` for `event`, `sim_time`, `pick_index`, `pick_name`, `target_name`. Diagnose and fix before moving on.
5. **If the task-final image is ambiguous or a failure has unclear cause, escalate to cadence snapshots.** Drop `--teleport` (so you see physics-driven motion) and use `--snapshots` instead of `--snapshot-errors` — this adds a 3 Hz simulation-time cadence of wide-camera frames:
   ```bash
   mamba run -n env_isaacsim51 python run_task.py --task TableTask<Name> --headless --snapshots --auto-exit
   ```
   Sort the snapshot directory by filename (sim-time-ordered) and inspect the frames immediately before and after the failure event to see the run-up.

The reason this self-check exists: the user is a scarce resource, and snapshot artifacts (PNG + JSON) are agent-readable without an Isaac Sim GUI session. Catch obvious problems here so Phase 6 is the user reviewing a working task, not debugging a broken one. For the full workflow reference, see `### Visual Debugging with Snapshots and Video (No GUI Required)` in `docs/mock-system-and-testing-design.md` (section 10).

If you fix anything based on the self-check, re-run from Phase 4 (mock) before re-running this phase.

## Phase 6: User Approval

Only after Phase 5 passes (no failure events; task-final image looks reasonable) — and *BEFORE OFFERING TO COMMIT CHANGES* — invite the user to approve the task. Offer them two options:

1. **Look at the task-final image directly.** Give them the path printed in Phase 5 — `_results/snapshots/<task>_<ts>/task_verified_pickNN_t*.png` — and ask them to confirm it matches their intent.
2. **Run interactively with `--teleport`** (optionally `--pause`) if they want to drive it themselves in the Isaac Sim GUI:
   ```bash
   mamba run -n env_isaacsim51 python run_task.py --task TableTask<Name> --teleport --pause
   ```
   `--teleport` skips motion planning; `--pause` stops after each cycle so the user can inspect the scene.

DO NOT consider the task complete or offer to commit the changes until the user has confirmed the task setup and completion state match their intent. If they request adjustments (spacing, positions, box sizes, pick order, etc.), iterate on the implementation and re-test from Phase 4.

If the task setup involves randomization, after the initial approval, ask whether the user wants additional runs with different seeds. For each additional seed, do the Phase 5 self-check first, then surface the new task-final image to the user. Repeat until they're satisfied.

## Phase 7: Check saved learnings
If the user reports issues during or after Phase 6 (from their interactive testing or seed-variant runs), then, in preparation for attempting to diagnose and fix them, do two things:
1. Reproduce the failure under `--snapshot-errors` (or `--snapshots` if the failure-only frames don't show enough context) and inspect the relevant failure-event PNG + JSON before forming a hypothesis — the visual symptom often constrains the search space dramatically.
2. Check `learnings.md` to see if any similar issues have been reported and resolved in the past. If you find entries about similar issues, also read the corresponding sections of `lessons-learned-details.md` for more detail.

## Phase 8: Learn from User Interactions

If the user makes adjustments to the task based on Phase 6, or reports issues based on interactive testing of the full simulated task, then, after fixing the issues (possibly with assistance or guidance from the user), make a record of learnings learned, as follows:

Extract any lessons or tips that can be learned from the issues that were encountered and how they were resolved. Summarize the learnings and save them into learnings.md and lessons-learned-details.md, for future reference (to avoid similar mistakes or to help deal with similar errors). These two files are at differing levels of detail: learnings.md is high-level only, with just Symptom and General Rule subsections for each issue, while lessons-learned-details.md should provide more details about how the issues were identified and resolved. Check exiting entries in these file for examples, and then make edits to add new entries without modifying the exiting ones.

## Naming Conventions

- **File**: `tasks/table_task_<name>.py` (snake_case)
- **Class**: `TableTask<Name>` (PascalCase)
- **task_name arg**: `"table_task_<name>"` (snake_case, matches file)
- **Spec doc**: `tasks/TableTask<Name>.md` (matches class name)
