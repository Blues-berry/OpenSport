# Project structure

## Version-controlled source

| Path | Purpose |
| --- | --- |
| `imu_analysis/` | Offline cleaning, feature extraction, binary and multiclass training, replay, and strategy code |
| `tests/` | Unit and contract tests |
| `scripts/` | Dataset-label conversion and preparation utilities |
| `assets/` | Small version-controlled UI and document assets |
| `docs/` | Architecture and project documentation |
| Repository-root Python/HTML files | Current BLE/WitMotion live dashboard entry points |

## Local data and generated output

These trees are intentionally excluded from Git.

| Path | Purpose |
| --- | --- |
| `data/raw/<batch>/` | Immutable source recordings grouped by collection batch |
| `data/raw/variants/` | Same-session source variants that differ at the byte level and must be reviewed before deduplication |
| `data/training/activity/` | Flat, filename-labelled CSV inputs for activity and posture training |
| `imu_output/<run>/` | Versioned feature caches, reports, candidate models, and training logs |
| `imu_output/head_posture/` | Current head-posture candidate and its feature cache |
| `imu_output/live_*` | Runtime captures and status files; disposable after required sessions are archived |

Raw data must not be edited in place. Cleaning and resampling always write to
`imu_output/`. A model is not a source-data file and must never be stored under
`data/`.

## Model routes

- `train_logistic.py` produces the deployed two-class `运动 / 非运动` baseline.
- `train_activity_model.py` produces the LightGBM multiclass action candidate.
- `train_head_posture_model.py` produces the independent head-posture model.
- Candidate models remain under a versioned `imu_output/` run until their
  metrics are reviewed and promotion is explicitly requested.

## Git and push rules

`master` is the default integration branch and `origin/master` is the default
remote target. Work is committed locally first. Pushing is an explicit remote
write and requires user authorization; no collaborator should push
automatically or force-push `master`.
