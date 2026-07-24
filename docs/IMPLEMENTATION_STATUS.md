# IMU v1.0 implementation status

## Historical migration

- 359 Schema v2 labels now carry an explicit evidence tier.
- 23 long sessions remain `session_weak`.
- One Schema v1 label was removed from the active labels directory after a
  byte-verified copy to `data/quarantine/legacy_labels`.
- Historical short labels are `legacy_reviewed`; none are formal evaluation
  evidence.
- The paired-person device order and the confirmed 0721/0722 exceptions are
  versioned in `config/legacy_capture_corrections.json`.

## Experimental candidates

Activity candidate:

- test motion Macro-F1: 0.937
- test motion recall: 0.947
- test non-motion specificity: 0.927
- test detailed-action Macro-F1: 0.561
- formal evaluation: unavailable

Posture candidate:

- historical binary normal/poor Macro-F1: 1.000
- historical poor recall: 1.000
- detailed deviation Macro-F1: 0.352
- formal evaluation: unavailable

Both bundles are registered under `imu_output/models/*/candidate` and are
explicitly experimental. Neither was promoted to champion. The perfect
historical posture binary score must not be treated as product validation:
normal and deliberately deviated captures are strongly separated and the
labels predate the new protocol.

## Verification

- 45 automated tests pass, including schema/import failures, subject leakage,
  missing and incompatible models, timestamp-aware offline/live feature
  equivalence, workout timing, reconnect behavior, and posture alert timing.
- Both registered candidate bundles load through the version compatibility
  checks; the web application starts without a champion or legacy model.
