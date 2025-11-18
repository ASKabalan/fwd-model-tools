# Repository Guidelines

## Project Structure & Module Organization
- `src/fwd_model_tools/`: core package modules.
  - `config.py`: simulation configs and shared dataclasses.
  - `fields.py`: Gaussian/lognormal field generators (JAXPM-based).
  - `lensing_model.py`: forward models, likelihoods, wrappers.
  - `sampling.py` and `solvers/`: MCMC orchestration and ODE integrators.
- `scripts/`: runnable examples (`run_lensing_model.py`, `run_simple_sampling.py`).
- `tests/`: pytest suite; stash reproduced artifacts under `output/`.

## API Rewrite Status
- The library is currently being rewritten to provide a cleaner, more user-friendly API.
- The new API lives primarily in the following modules and directories:
  - `src/fwd_model_tools/field.py`
  - `src/fwd_model_tools/normal.py`
  - `src/fwd_model_tools/pm.py`
  - `src/fwd_model_tools/sampling.py`
  - `src/fwd_model_tools/utils.py`
  - `src/fwd_model_tools/kappa.py`
  - `src/fwd_model_tools/lensing.py`
  - everything in `src/fwd_model_tools/solvers`, `src/fwd_model_tools/power`, and `src/fwd_model_tools/_src`
- The rest of the code, in particular `src/fwd_model_tools/lensing_model.py`, has not been updated yet and should be treated as legacy until the rewrite is complete.

## Build, Test, and Development Commands
- `pip install -e .[dev]`: editable install with pytest/ruff extras.
- `pytest`: run tests with coverage (`--cov=fwd_model_tools --cov-report=term-missing`).
- `pytest tests/test_sampling.py::test_run_blackjax`: targeted reproducer.
- `ruff check src tests` (add `--fix` to apply safe rewrites).
- `python -m build`: create sdist/wheel for packaging checks.

## Coding Style & Naming Conventions
- Python 3.9+, four-space indent; type public APIs and keep docstrings current.
- Naming: `snake_case` for funcs/modules, `PascalCase` for dataclasses, `ALL_CAPS` for constants (e.g., `Planck18`).
- Ruff config: `line-length = 120`, double quotes; silence with targeted `# noqa` only.
- Pure JAX: pass explicit `jax.random.PRNGKey`, avoid hidden globals, keep static shapes, no side effects.

## Testing Guidelines
- Pytest discovers `tests/test_*.py`, classes `Test*`, functions `test_*` (see `pyproject.toml`).
- Prefer deterministic keys: `jax.random.PRNGKey(seed)` fixtures.
- Watch coverage in the term-missing report; extend tests for uncovered branches.
- Gate long-running/distributed tests behind marks so CI skips by default.

## Commit & Pull Request Guidelines
- Commits: concise, present tense (e.g., "Allow using max redshift as parameter", "format").
- Bundle by concern; include context when touching performance/GPU paths.
- PRs: problem statement, summary of changes, validation notes (commands run, hardware), and linked issues.
- Attach plots/logs from `output/` when results change; flag breaking API changes early.

## Distributed & JAX Tips
- Set env early when sharing repros: `JAX_PLATFORM_NAME=gpu`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
- Use sharding-aware helpers in `distributed.py` and `sampling.py`; document expected mesh sizes/partitioning in PRs.
