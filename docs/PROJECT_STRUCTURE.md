# Project structure

## Version-controlled source

| Path | Purpose |
| --- | --- |
| `src/opensport/` | Versioned public contracts, ingestion, feature, model registry, runtime, storage, device and API layers |
| `schemas/` | Machine-readable session, activity-label, posture-label and quality-report contracts |
| `templates/` | Fillable JSON examples matching the acquisition contracts |
| `config/legacy_capture_corrections.json` | Reviewed device/person and capture overrides for reproducible legacy migration |
| `imu_analysis/` | Backward-compatible CLI/runtime wrappers and current LightGBM trainers |
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
| `data/training/activity/<dataset_version>/` | Validated per-device captures; semantics always come from labels |
| `data/quarantine/` | Invalid or legacy labels removed from active training inputs |
| `imu_output/runs/<run>/` | Versioned feature caches, reports, candidate models, and training logs |
| `imu_output/models/{activity,posture}/{candidate,champion}/` | Runtime model registry |
| `imu_output/head_posture/` | Current head-posture candidate and its feature cache |
| `imu_output/live_*` | Runtime captures and status files; disposable after required sessions are archived |

Raw data must not be edited in place. Cleaning and resampling always write to
`imu_output/`. A model is not a source-data file and must never be stored under
`data/`.

## Model routes

- `train_logistic.py` remains a legacy two-class compatibility baseline.
- `train_activity_model.py` produces hierarchical activity candidates. Legacy
  evidence requires `--allow-legacy-training` and can never pass the formal gate.
- `train_head_posture_model.py` produces the independent head-posture model.
- Candidate models remain under a versioned `imu_output/` run until their
metrics are reviewed and promotion is explicitly requested.

The live product treats each device stream independently in v1. A session
manifest associates the left and right streams with one trial, but no model
feature silently fuses the two ears.

## Git and push rules

`master` is the default integration branch and `origin/master` is the default
remote target. Work is committed locally first. Pushing is an explicit remote
write and requires user authorization; no collaborator should push
automatically or force-push `master`.
