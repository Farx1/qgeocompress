# Changelog

All notable changes to this project are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer](https://semver.org/).
No release is tagged yet; `0.1.0` is the version in `pyproject.toml` and the state of `main` on 2026-08-10.

## [Unreleased]

### Added
- `CITATION.cff`, `CONTRIBUTING.md`, `SECURITY.md`, this changelog, `MODEL_CARD.md`,
  `datasets/dota128_power/DATASET_CARD.md`.
- pre-commit (ruff), Dependabot for pip and GitHub Actions.
- CI: Python 3.11/3.12 matrix, coverage threshold, actions pinned by commit SHA, read-only token.

## [0.1.0] - 2026-08-10

### Added
- Data-aware (activation-weighted) low-rank factorization, Lagrangian rank allocation under a
  parameter budget, transfer-coefficient layer weighting, and a bootstrapped frontier study
  (`scripts/frontier_study.py`) (2026-08-10).
- Reproducible head-to-head runner (`scripts/run_headtohead.sh`); Phase 3C regenerated on real runs (2026-08-10).
- Guard against QAOA state vectors too large for the simulator (2026-08-10).
- Quantum layer selection: QUBO encoding solved by QAOA on a CPU simulator, with an exact classical
  reference (2026-07-02).
- CPU hold-out experiments (SVD, QAOA), QAOA-simulator hold-out compression (2026-07-02).
- Phase 3C report, CI, local hold-out pipeline, GPU runbook, multi-format export stack (2026-07-02).
- Valohai post-training DAG, pareto-gain layer selection, export gate (2026-06-24).
- Phase 3B hold-out validation (train/select/test) and post-training quality gate (2026-06-24).
- Initial Phase 2–3B code: low-rank compression, GT-matched calibration (ECE, CER), structural
  low-rank, layer sensitivity probe (2026-06-23).

### Changed
- README and `docs/QUANTUM.md` aligned with measured numbers; hypothesis report regenerated and the
  sensitivity-weighting claim corrected (2026-08-10).
- Rank budget priced in transfer coefficients instead of raw sensitivity (2026-08-10).

### Fixed
- GT matching: predictions were never matched to their ground truth (2026-08-10).
- Broken activation probe; lint now enforced in CI (2026-08-10).
- `.gitignore` excluded the `src/qgeocompress/data` module (2026-07-02).

### Removed
- CPU latency speed-up claim: it did not survive repetition (2026-08-10).
- ECE/CER in `train_baseline`: computed from a proxy that labelled every detection correct, so CER was always 0 (2026-08-10).
- Ultralytics label cache and personal tooling files from version control (2026-07-02, 2026-08-10).

[Unreleased]: https://github.com/Farx1/qgeocompress/compare/2b1c31b...HEAD
[0.1.0]: https://github.com/Farx1/qgeocompress/tree/2b1c31b
