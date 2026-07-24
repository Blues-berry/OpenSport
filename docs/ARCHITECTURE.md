# OpenSport IMU architecture

## Contract boundary

`src/opensport` is the stable implementation boundary. Existing scripts in
`imu_analysis` and the repository root remain callable, but adapt their inputs
and outputs to the versioned public contracts.

```text
raw trial + session_manifest + reviewed labels
                 |
          strict audit/import
                 |
       per-device SI time series
          /              \
 activity windows       posture orientation
       |                       |
 hierarchical model      binary posture policy
       |                       |
 workout state machine   30 s alert / 5 s recovery
          \              /
        versioned API + idempotent SQLite events
```

## Data eligibility

- `gold`: operator events, reviewed video or precise manual timeline. Eligible
  for training and formal subject-held-out evaluation.
- `legacy_reviewed`: pre-standard recordings reviewed from their complete old
  names. Available only behind an explicit experimental-training flag.
- `session_weak`: ordered activity and set totals only; never creates windows.
- `rejected`: missing identity, invalid wear, schema failure or failed quality
  audit; never trains a model.

The importer verifies stream identity, ear side, SHA-256, monotonic timestamps,
sequence continuity, observed rate, label intervals and units. Raw files are
never rewritten.

## Runtime behavior

The activity model predicts motion, family and a supported detailed activity.
All non-motion content is one activity-model class; posture directions belong
to the posture model. General activity requires 30 seconds to become a product
exercise session, while strength requires 3 seconds. Twenty seconds of low
activity ends a valid five-second strength set, and 240 seconds finalizes the
session.

The posture model reports `normal` or `poor` plus deviation directions and
calibration-relative angles. Hardware orientation is preferred. Six-axis Yaw
is marked degraded. Outputs are ergonomic reminders, not medical diagnoses.

## Model lifecycle

Every model bundle records feature, taxonomy, label, dataset and code versions.
Runtime first looks in `imu_output/models/<kind>/champion`, then permits a
candidate only with an experimental warning. A candidate cannot be promoted
without gold evaluation and all fixed acceptance gates.
