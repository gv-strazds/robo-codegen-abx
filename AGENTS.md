# Repository Guidelines

## Project Structure & Module Organization
- Core Python scripts: `multi_pickplace_task.py` (task base class), `run_task.py` (entrypoint), `table_setup.py` (Isaac Sim scene helpers; re-exports constants from `env_config_values.py`), `env_config_values.py` (Isaac-Sim-free workspace geometry constants), `asset_utils.py` (Isaac-Sim-dependent scene/prim operations), `asset_data_utils.py` (Isaac-Sim-free asset metadata and geometry math).
- Task architecture modules (extracted from `multi_pickplace_task.py`):
  - `task_spec.py` — `TaskSpec` (scene/description) plus a nested `TaskImplementationSpec` (policy: pairing strategy factory, BT tree factory, postures, hover heights, watchdog timeouts, cuRobo flags). Task classes build a `TaskSpec` with `implementation=TaskImplementationSpec(...)` and pass it to `super().__init__(task_spec=spec, ...)`.
  - `simulation_configurator.py` — `SimulationConfigurator` manages scene objects, geometry cache, and verification.
  - `task_controller.py` — `TaskController` policy layer connecting strategy, `TaskContext`, and the BT controller.
- Isaac Sim snapshot (read-only reference) under `exts/isaacsim/`:
  - `core/` — utilities, prims, API layers, and tests under `core/**/tests`.
  - `robot/manipulators/` — controllers, grippers, examples, and OGN nodes.
  - `robot/manipulators/tests` and `core/api/**/tests` — example/unit tests.
- Assets/examples: icons (`*.svg`) and sample USD files for tests.
- Scripts in this project need to be run with a specific conda (mamba) environment activated: env_isaacsim51. Note that this agent is usually run with this environment already activated.

## Build, Test, and Development Commands
- Run demo: `python run_task.py --task TableTask3` (or any other task class like `TableTaskColors1`, `TableTaskSoupCans1`, etc.).
- Run tests for the copied Isaac modules: `pytest exts/isaacsim -q`.
- Run project tests: `mamba run -n env_isaacsim51 env PYTHONPATH=$(pwd)/extsMock:$(pwd) python -m pytest tests/` (This ensures mock extensions shadow real Isaac Sim packages and project modules can be imported).
- Notes: The repo is pure Python; no build step. Running demos assumes a working Isaac Sim Python environment on your machine.

## Coding Style & Naming Conventions
- Python 3.x, PEP 8, 4-space indentation, max line length ~100–120.
- Filenames/modules: `snake_case.py`; Classes: `PascalCase`; Functions/vars: `snake_case`.
- Prefer explicit imports from `isaacsim/...` for clarity.
- Add docstrings for public functions; include brief rationale for non-obvious math/transforms.

## Testing Guidelines
- Use `pytest`; place tests alongside modules in `tests` folders or `ogn/tests` where applicable.
- `pytest` is normally installed as a bash command in `env_isaacsim51`; if it isn't on a given machine, fall back to `python -m pytest`.
- Test names: files `test_*.py`; functions `test_<unit_of_behavior>`.
- Keep demos smoke-testable (e.g., fast paths, flags to reduce runtime). Aim to keep unit tests <1s each.

## Commit & Pull Request Guidelines
- Commits: imperative mood (“Add…”, “Fix…”); group logical changes. Prefix types when helpful (e.g., `feat:`, `fix:`, `refactor:`) — this matches prior history.
- PRs: include a concise description, steps to reproduce/run, and screenshots or short clips if behavior changes. Link related issues. Note any Isaac Sim version assumptions.

## Agent-Specific Tips
- Validate imports using repository-relative paths used by IDEs (see prior commits improving import resolution).
- Avoid diverging from the reference `exts/isaacsim` snapshot; prefer adapters/wrappers in top-level scripts. DO NOT edit reference copies in the exts/isaacsim subtree.

