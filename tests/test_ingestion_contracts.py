from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from opensport.data import audit_session
from opensport.data.ingestion import TrialImporter, validate_activity_label
from opensport.features.resample import uniform_resample


FIELDS = [
    "timestamp_monotonic_ns",
    "sequence_id",
    "ear_side",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
]


def write_stream(path: Path, side: str, rate: int = 50) -> str:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index in range(rate * 4 + 1):
            writer.writerow(
                {
                    "timestamp_monotonic_ns": index * 1_000_000_000 // rate,
                    "sequence_id": index,
                    "ear_side": side,
                    "acc_x": 0,
                    "acc_y": 0,
                    "acc_z": 1,
                    "gyro_x": 0,
                    "gyro_y": 0,
                    "gyro_z": 0,
                }
            )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_dual_stream_trial_is_audited_and_resampled(tmp_path: Path) -> None:
    trial = tmp_path / "T0001"
    trial.mkdir()
    left_hash = write_stream(trial / "left.csv", "left")
    right_hash = write_stream(trial / "right.csv", "right")
    label = {
        "schema_version": "2.0",
        "taxonomy_version": "test",
        "date": "2026-07-25",
        "participant": "S001",
        "device": ["D01", "D02"],
        "csv_file": ["left.csv", "right.csv"],
        "annotation_scope": "full_recording",
        "window_trainable": True,
        "evidence_tier": "gold",
        "recording": {
            "start_time": "2026-07-25T10:00:00+08:00",
            "end_time": "2026-07-25T10:00:04+08:00",
            "duration_seconds": 4.0,
            "row_count": 201,
        },
        "annotation_quality": {"status": "reviewed", "reason": "operator"},
        "segments": [
            {
                "start_s": 0,
                "end_s": 4,
                "activity_id": "squat",
                "motion_state": "motion",
                "wear_state": "valid",
                "phase": "active",
                "window_trainable": True,
                "label_source": "operator_event",
                "confidence": "high",
                "review_note": "test",
            }
        ],
    }
    (trial / "labels.json").write_text(json.dumps(label), encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "protocol_version": "v1.0",
        "date": "2026-07-25",
        "session_id": "session-1",
        "subject_id": "S001",
        "collector_id": "C001",
        "app_version": "1.0.0",
        "trials": [
            {
                "trial_id": "T0001",
                "label_file": "T0001/labels.json",
                "streams": [
                    {
                        "device_id": "D01",
                        "ear_side": "left",
                        "csv_file": "T0001/left.csv",
                        "sha256": left_hash,
                        "sample_rate_configured_hz": 50,
                        "acceleration_unit": "g",
                        "angular_velocity_unit": "deg/s",
                        "acceleration_range": 16,
                        "angular_velocity_range": 2000,
                        "filter_config": "default",
                        "firmware_version": "1.0.0",
                    },
                    {
                        "device_id": "D02",
                        "ear_side": "right",
                        "csv_file": "T0001/right.csv",
                        "sha256": right_hash,
                        "sample_rate_configured_hz": 50,
                        "acceleration_unit": "g",
                        "angular_velocity_unit": "deg/s",
                        "acceleration_range": 16,
                        "angular_velocity_range": 2000,
                        "filter_config": "default",
                        "firmware_version": "1.0.0",
                    },
                ],
            }
        ],
    }
    manifest_path = tmp_path / "session_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    importer = TrialImporter(manifest_path, "T0001")
    report, streams = importer.audit()

    assert report.status == "accepted"
    assert report.evidence_tier == "gold"
    assert set(streams) == {"D01", "D02"}
    assert abs(streams["D01"][1][0, 2] - 9.80665) < 1e-6
    written_report, written = importer.write_training_captures(
        tmp_path / "training"
    )
    assert written_report.status == "accepted"
    assert len(written) == 5
    derived = json.loads(
        next(
            path for path in written
            if path.name.endswith(".labels.json")
        ).read_text(encoding="utf-8")
    )
    assert isinstance(derived["device"], str)
    assert derived["derived_from"]["units"]["acceleration"] == "m/s2"
    batch = audit_session(manifest_path)
    assert batch["accepted"]
    assert batch["violation_count"] == 0

    label["device"] = "D01"
    (trial / "labels.json").write_text(json.dumps(label), encoding="utf-8")
    manifest["trials"][0]["streams"] = manifest["trials"][0]["streams"][:1]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    single_report, _ = TrialImporter(manifest_path, "T0001").audit()
    assert single_report.status == "recollect_required"
    assert "dual_ear_stream_missing" in single_report.issues


def test_label_validation_rejects_long_supervised_and_bad_wear() -> None:
    payload = {
        "schema_version": "2.0",
        "taxonomy_version": "x",
        "date": "2026-07-25",
        "participant": "S001",
        "device": "D01",
        "csv_file": "anything.csv",
        "annotation_scope": "full_recording",
        "window_trainable": True,
        "evidence_tier": "gold",
        "recording": {"duration_seconds": 181},
        "annotation_quality": {},
        "segments": [
            {
                "start_s": 0,
                "end_s": 10,
                "wear_state": "removed",
                "motion_state": "motion",
                "phase": "active",
                "window_trainable": True,
                "label_source": "operator_event",
                "confidence": "high",
            }
        ],
    }
    errors = validate_activity_label(payload)
    assert any("over 180" in error for error in errors)
    assert any("invalid wear" in error for error in errors)


def test_resampling_requires_monotonic_timestamps() -> None:
    with np.testing.assert_raises(ValueError):
        uniform_resample(
            np.asarray([0, 2, 1], dtype=np.int64),
            np.zeros((3, 6)),
            50,
        )
