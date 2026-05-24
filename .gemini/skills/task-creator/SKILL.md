---
name: task-creator
description: Generate and configure new simulation tasks for the Isaac Sim UR10 robot. Use when creating new pick-and-place scenarios in the `tasks/` directory.
---

# Task Creator

This skill guides the creation of new pick-and-place tasks for the Isaac Sim UR10 environment.

## Workflow

1.  **Understand Requirements**: Identify the objects to be picked, their arrangement, the target locations, and any pairing logic (e.g., color sorting, type-based routing).  If some aspects of the task are ambiguous or underspecified, ask the user for clarification.
2.  **Generate a Task Specification artifact** Document the task requirements and implementation strategies in a markdown file in the `tasks/` directory named `TableTask<name>.md`. The Task Specification should include, at least, the following sections: 1) 'User Request' - the user's original request, verbatim; 2) 'Task Overview' - a summary of your understanding of the task; be specific about asset_types and configuration strategies for pick items and target objects; 3) 'Concise Task Description' - a very concise task description (one imperative sentence); 4) 'Pick Items' - types and initial arrangement of items to be picked and placed; 5) 'Target Objects' - types and arrangements of objects that provide the placement targets for the pick items; 6) 'Success Condition' - A one sentence statement of what needs to be true for the task to have been successfully completed; 7) 'Success Checks' - A detailed list of specific checks that should be run to verify that the Success Condition has been completely achieved. Taken together (if all report success), these should verify the overall completion criteria, and also that all constraints specified by the user or implicit in the task goals are satisfied.
3.  **Template for Task Specification**: Use the `assets/TaskSpecTemplate.md` as a starting point.
4.  **Define the Task Class**: Create a new file in the `tasks/` directory named `table_task_<name>.py`.
5.  **Template Initialization**: Use the `assets/task_template.py` as a starting point. The TaskSpec includes human-readable metadata fields that document the task design alongside the executable config.
5b. **Set TaskSpec Metadata and Rationale**: Scene-side metadata lives on the outer `TaskSpec`; implementation-side metadata lives on the nested `TaskImplementationSpec` (assigned via `implementation=`):
    - `scenario` (outer): dict with source/destination/workspace
    - `pick_description` (outer): dict with asset_types, count, arrangement, colors
    - `target_description` (outer): dict with type, arrangement, count (and containers dict for box-packing tasks)
    - `strategy_description` (inner — on `TaskImplementationSpec`): dict with class name, pairing type, details
    - `verification_description` (outer): dict with spatial_check, placement_constraints, containment_check (only for non-default verification)
    - `rationale` (split): scene-related keys on outer `TaskSpec.rationale`; impl-related keys on `TaskImplementationSpec.rationale`. The `create_strategy` rationale (impl-side) is required when a non-default strategy is configured. Other keys are only needed for non-default choices.

    Only include non-None/non-default fields in description sub-dicts. Pass `task_spec=spec` to `super().__init__()`. See the template for the complete pattern.

6.  **Configure Generation Strategies**: Use `item_generation.py` to define how picks and targets are spawned. If the task involves placing items into the pick bin or a box, use `virtual_target_generation_strategy` on `TaskImplementationSpec` (NOT the outer `target_generation_strategy`) to generate hidden marker objects as targets (LightweightObj instances generated at pairing time, not spawned in the USD scene — they're policy helpers). These should be arranged in a grid on the floor of the box (just slightly above the surface on which the bin or box will be resting). Marker objects should be of asset_type 'marker' and scaled to be very thin in the z axis (e.g. 0.001 units). For box containment verification, set `box_verification_info={"box_specs": box_specs}` and `containment_check=True` directly on the outer `TaskSpec`, where each box spec includes `name`, `center_xy`, `floor_z`, `inner_size`, `height`, and optionally `match_labels` (e.g., `{"color": "red"}`). The base class will automatically handle box containment verification. See `TableTaskSoupCanPacking` for a complete example.
7.  **Set Up Pairing Logic**: If the task requires more than sequential pairing, set `create_strategy` inside the `TaskImplementationSpec` block (e.g., `implementation=TaskImplementationSpec(create_strategy=lambda picks, targets: ColorMatchStrategy(picks, targets, ...))`). v2 / cortex / cuRobo subclasses use `spec.with_impl(tree_factory=..., ...)` in `_customize_spec()` to swap policy fields.
8.  **Verify**: Run the full task with `python run_task.py --task <TaskClassName>`. Use `--teleport` for fast scene validation before attempting a full physics run.

## Key Resources

-   **[Task Structure](references/task_structure.md)**: Explains the components of a task class and how to define strategies.
-   **[Strategies](references/strategies.md)**: Details on `MultiPickStrategy` subclasses for complex pairing logic.
-   **[Template](assets/task_template.py)**: A boilerplate Python file for new tasks.

## Code Style & Conventions

-   **File Name**: `table_task_<name>.py`.
-   **Class Name**: `TableTask<Name>` (PascalCase).
-   **Imports**: Use lazy imports within `__init__` for Isaac Sim dependencies.
-   **Description**: Provide a clear `task_description` string in the constructor.
-   **Coordinates**: Use `BIN_X_COORD`, `BIN_Y_COORD`, `DROPZONE_X`, `DROPZONE_Y`, `DROPZONE_Z`, and `CART_SURFACE_CENTER` for relative positioning. These constants live in `env_config_values.py` (Isaac-Sim-free) and are also re-exported from `table_setup` for back-compat.
