# Contributing

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11+
pip install -e ".[dev]"
pip install pre-commit && pre-commit install         # runs ruff on every commit
```

## Before opening a PR

```bash
ruff check .
pytest -q                     # mocked suite, ~2 min on CPU
pytest -m slow -q             # real YOLO11n-OBB checkpoint (downloads ~6 MB of weights)
```

CI runs the same commands on Python 3.11 and 3.12 and fails below the coverage threshold set in
`.github/workflows/ci.yml`. New logic comes with a test; a bug fix comes with the test that failed.

## Conventions

- Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat:`, `fix:`, `docs:`, `test:`, `chore:`, `ci:`), one logical change per PR.
- PR description: **Problem / Change / How verified / Not verified**.
- Results are claims: any number added to the README must come from a JSON or report committed
  under `results/`, produced by a script in `scripts/` that a reader can rerun. State the split,
  the seed and the command.
- Hold-out discipline: method choices use the `select` split; the `test` split is touched once,
  for the final comparison.
- Negative results are kept and documented, not deleted.
- Record user-visible changes under `## [Unreleased]` in `CHANGELOG.md`.
