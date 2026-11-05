# Changelog

All notable changes are documented here (Keep a Changelog, semantic versioning).

## [Unreleased]

- `oaqe-label-frames` (`oaqe.labeling`): frames with distance ticks from the calibration, an empty label template and `--finalize`; `docs/REAL_VIDEO.md`.

## [0.1.0]

### Added
- Occlusion-aware lane-level queue estimator: homography localisation, ray-cast occlusion model, hidden-gap and tail inference, ghost vehicles, gated Kalman filtering, confidence and pressure/fuzzy interfaces.
- Synthetic renderer, paired benchmark with ablations and bootstrap intervals, SUMO/TraCI bridge and example scenario.
- Data-driven animation (`oaqe-gif`), static typing (`mypy`), continuous integration.
