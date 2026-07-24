"""Strict import and quality audit for acquisition-standard trial folders."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from opensport.features.resample import uniform_resample
from opensport.types import (
    DeviceStream,
    QualityReport,
    StreamQuality,
    TrialManifest,
)


G_TO_MPS2 = 9.80665
DEG_TO_RAD = math.pi / 180.0
REQUIRED_SAMPLE_COLUMNS = {
    "timestamp_monotonic_ns",
    "sequence_id",
    "ear_side",
    "acc_x",
    "acc_y",
    "acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
}
GOLD_LABEL_SOURCES = {"operator_event", "video_review", "manual_timeline"}


def _require(payload: dict[str, Any], keys: set[str], context: str) -> None:
    missing = sorted(keys - payload.keys())
    if missing:
        raise ValueError(f"{context} missing required fields: {missing}")


def load_trial_manifest(manifest_path: Path | str, trial_id: str) -> TrialManifest:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    _require(
        payload,
        {
            "schema_version",
            "protocol_version",
            "date",
            "session_id",
            "subject_id",
            "collector_id",
            "app_version",
            "trials",
        },
        "session_manifest",
    )
    if payload["schema_version"] != "1.0":
        raise ValueError("session_manifest schema_version must be 1.0")
    matching = [item for item in payload["trials"] if item.get("trial_id") == trial_id]
    if len(matching) != 1:
        raise ValueError(f"trial_id {trial_id!r} must occur exactly once")
    trial = matching[0]
    _require(trial, {"trial_id", "label_file", "streams"}, "trial")
    streams: list[DeviceStream] = []
    identities: set[tuple[str, str]] = set()
    for raw in trial["streams"]:
        _require(
            raw,
            {
                "device_id",
                "ear_side",
                "csv_file",
                "sha256",
                "sample_rate_configured_hz",
                "acceleration_unit",
                "angular_velocity_unit",
                "acceleration_range",
                "angular_velocity_range",
                "filter_config",
                "firmware_version",
            },
            "device stream",
        )
        identity = (str(raw["device_id"]), str(raw["ear_side"]))
        if identity in identities:
            raise ValueError(f"duplicate device stream {identity}")
        identities.add(identity)
        streams.append(
            DeviceStream(
                device_id=identity[0],
                ear_side=identity[1],
                csv_file=str(raw["csv_file"]),
                sha256=str(raw["sha256"]).lower(),
                sample_rate_configured_hz=float(raw["sample_rate_configured_hz"]),
                acceleration_unit=str(raw["acceleration_unit"]),
                angular_velocity_unit=str(raw["angular_velocity_unit"]),
                acceleration_range=float(raw["acceleration_range"]),
                angular_velocity_range=float(raw["angular_velocity_range"]),
                filter_config=str(raw["filter_config"]),
                firmware_version=str(raw["firmware_version"]),
                axis_sign=tuple(int(value) for value in raw.get("axis_sign", (1, 1, 1))),
            )
        )
    return TrialManifest(
        schema_version="1.0",
        protocol_version=str(payload["protocol_version"]),
        date=str(payload["date"]),
        session_id=str(payload["session_id"]),
        subject_id=str(payload["subject_id"]),
        trial_id=str(trial["trial_id"]),
        collector_id=str(payload["collector_id"]),
        app_version=str(payload["app_version"]),
        label_file=str(trial["label_file"]),
        streams=tuple(streams),
        replaces_trial_id=trial.get("replaces_trial_id"),
    )


def validate_activity_label(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = {
        "schema_version",
        "taxonomy_version",
        "date",
        "participant",
        "device",
        "csv_file",
        "annotation_scope",
        "recording",
        "window_trainable",
        "annotation_quality",
        "evidence_tier",
        "segments",
    }
    if missing := sorted(required - payload.keys()):
        errors.append(f"missing required fields: {missing}")
    if payload.get("schema_version") != "2.0":
        errors.append("schema_version must be 2.0")
    tier = payload.get("evidence_tier")
    if tier not in {"gold", "legacy_reviewed", "session_weak", "rejected"}:
        errors.append("invalid evidence_tier")
    duration = float(payload.get("recording", {}).get("duration_seconds", -1))
    scope = payload.get("annotation_scope")
    segments = payload.get("segments", [])
    if duration > 180 and scope != "session_weak":
        errors.append("recordings over 180 seconds must be session_weak")
    if scope == "session_weak":
        if segments:
            errors.append("session_weak must not have segments")
        if payload.get("window_trainable") is not False:
            errors.append("session_weak must not be window trainable")
    previous_end = 0.0
    for index, segment in enumerate(segments):
        start = float(segment.get("start_s", -1))
        end = float(segment.get("end_s", -1))
        if start < previous_end:
            errors.append(f"segments[{index}] overlaps or is out of order")
        if start < 0 or end <= start or (duration >= 0 and end > duration + 1e-3):
            errors.append(f"segments[{index}] has invalid bounds")
        previous_end = max(previous_end, end)
        wear = segment.get("wear_state")
        motion = segment.get("motion_state")
        if wear not in {"valid", "removed", "asymmetric", "invalid"}:
            errors.append(f"segments[{index}] has invalid wear_state")
        if wear != "valid" and segment.get("window_trainable"):
            errors.append(f"segments[{index}] invalid wear is trainable")
        if wear != "valid" and motion is not None:
            errors.append(f"segments[{index}] invalid wear must have motion_state=null")
        if segment.get("phase") not in {"active", "rest", "transition", "artifact"}:
            errors.append(f"segments[{index}] has invalid phase")
        if tier == "gold" and segment.get("window_trainable"):
            if segment.get("label_source") not in GOLD_LABEL_SOURCES:
                errors.append(f"segments[{index}] gold label has unsupported source")
            if segment.get("confidence") not in {"high", "medium"}:
                errors.append(f"segments[{index}] gold label confidence is too low")
    return errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_stream(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV has no header")
        if missing := REQUIRED_SAMPLE_COLUMNS - set(reader.fieldnames):
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no samples")
    fields = [
        "timestamp_monotonic_ns",
        "sequence_id",
        "acc_x",
        "acc_y",
        "acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    ]
    return [str(row["ear_side"]) for row in rows], np.asarray(
        [[float(row[field]) for field in fields] for row in rows], dtype=float
    )


def _quality_for_stream(root: Path, stream: DeviceStream) -> tuple[StreamQuality, np.ndarray]:
    path = (root / stream.csv_file).resolve()
    if root.resolve() not in path.parents:
        raise ValueError("stream path escapes the trial directory")
    sides, data = _read_stream(path)
    issues: list[str] = []
    if set(sides) != {stream.ear_side}:
        issues.append("ear_side_mismatch")
    timestamps = data[:, 0].astype(np.int64)
    sequence = data[:, 1].astype(np.int64)
    diffs = np.diff(timestamps)
    reorder_count = int(np.sum(diffs <= 0))
    positive = diffs[diffs > 0]
    duration = max(0.0, (timestamps[-1] - timestamps[0]) / 1_000_000_000)
    observed = (len(timestamps) - 1) / duration if duration > 0 else 0.0
    if reorder_count:
        issues.append("timestamp_not_strictly_increasing")
    if not 0.95 * stream.sample_rate_configured_hz <= observed <= 1.05 * stream.sample_rate_configured_hz:
        issues.append("sample_rate_outside_5_percent")
    expected_step = 1_000_000_000 / max(observed, 1e-9)
    long_gaps = int(np.sum(positive > max(100_000_000, expected_step * 3)))
    if long_gaps:
        issues.append("long_timestamp_gap")
    sequence_delta = (np.diff(sequence) % 65536).astype(int)
    sequence_gaps = int(np.sum(np.maximum(sequence_delta - 1, 0)))
    sequence_reorders = int(np.sum(sequence_delta == 0))
    if sequence_gaps:
        issues.append("sequence_gap")
    if sequence_reorders:
        issues.append("sequence_duplicate_or_reorder")
    sha_valid = _sha256(path) == stream.sha256
    if not sha_valid:
        issues.append("sha256_mismatch")
    sensor = data[:, 2:].astype(float)
    non_finite = float(np.mean(~np.isfinite(sensor)))
    finite = np.nan_to_num(sensor, nan=0.0, posinf=0.0, neginf=0.0)
    clipped = float(
        np.mean(
            np.column_stack(
                [
                    np.abs(finite[:, :3])
                    >= 0.995 * stream.acceleration_range,
                    np.abs(finite[:, 3:])
                    >= 0.995 * stream.angular_velocity_range,
                ]
            )
        )
    )
    if non_finite > 0:
        issues.append("non_finite_sensor_value")
    if clipped > 0.002:
        issues.append("sensor_saturation")
    quality = StreamQuality(
        device_id=stream.device_id,
        ear_side=stream.ear_side,
        row_count=len(data),
        duration_seconds=duration,
        configured_rate_hz=stream.sample_rate_configured_hz,
        observed_rate_hz=observed,
        sequence_gap_count=sequence_gaps,
        sequence_reorder_count=reorder_count + sequence_reorders,
        long_gap_count=long_gaps,
        clipped_ratio=clipped,
        sha256_valid=sha_valid,
        issues=tuple(issues),
    )
    return quality, data


def _to_si(stream: DeviceStream, data: np.ndarray) -> np.ndarray:
    sensor = data[:, 2:].astype(float).copy()
    sensor[:, :3] *= np.asarray(stream.axis_sign, dtype=float)
    sensor[:, 3:] *= np.asarray(stream.axis_sign, dtype=float)
    if stream.acceleration_unit == "g":
        sensor[:, :3] *= G_TO_MPS2
    elif stream.acceleration_unit not in {"m/s2", "m/s²"}:
        raise ValueError(f"unsupported acceleration unit {stream.acceleration_unit!r}")
    if stream.angular_velocity_unit in {"deg/s", "°/s"}:
        sensor[:, 3:] *= DEG_TO_RAD
    elif stream.angular_velocity_unit != "rad/s":
        raise ValueError(f"unsupported angular velocity unit {stream.angular_velocity_unit!r}")
    return sensor


class TrialImporter:
    def __init__(self, manifest_path: Path | str, trial_id: str) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = load_trial_manifest(self.manifest_path, trial_id)

    def audit(self) -> tuple[QualityReport, dict[str, tuple[np.ndarray, np.ndarray]]]:
        label_path = (self.root / self.manifest.label_file).resolve()
        if self.root not in label_path.parents:
            raise ValueError("label path escapes the session directory")
        label = json.loads(label_path.read_text(encoding="utf-8-sig"))
        label_errors = validate_activity_label(label)
        if str(label.get("participant")) != self.manifest.subject_id:
            label_errors.append("label participant does not match manifest")
        if str(label.get("date")) != self.manifest.date:
            label_errors.append("label date does not match manifest")
        labelled_devices = label.get("device")
        expected_devices = {
            stream.device_id for stream in self.manifest.streams
        }
        if isinstance(labelled_devices, list):
            device_set = {str(value) for value in labelled_devices}
        else:
            device_set = {str(labelled_devices)}
        if device_set != expected_devices:
            label_errors.append("label devices do not match manifest streams")
        qualities: list[StreamQuality] = []
        normalized: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        raw_starts: list[int] = []
        for stream in self.manifest.streams:
            quality, raw = _quality_for_stream(self.root, stream)
            qualities.append(quality)
            raw_starts.append(int(raw[0, 0]))
            if not quality.issues:
                timestamps = raw[:, 0].astype(np.int64)
                target_time, values = uniform_resample(
                    timestamps,
                    _to_si(stream, raw),
                    stream.sample_rate_configured_hz,
                )
                normalized[stream.device_id] = (target_time, values)
        issues = list(label_errors)
        if len(self.manifest.streams) != 2:
            issues.append("dual_ear_stream_missing")
        sides = {stream.ear_side for stream in self.manifest.streams}
        if sides != {"left", "right"}:
            issues.append("left_right_pair_incomplete")
        if len(raw_starts) == 2:
            offset_ms = abs(raw_starts[0] - raw_starts[1]) / 1_000_000
            if offset_ms > 100:
                issues.append(f"left_right_sync_offset_ms={offset_ms:.3f}")
        issues.extend(
            f"{quality.device_id}:{issue}"
            for quality in qualities
            for issue in quality.issues
        )
        evidence = str(label.get("evidence_tier", "rejected"))
        hard = bool(label_errors) or any(
            issue.endswith(
                (
                    "sha256_mismatch",
                    "timestamp_not_strictly_increasing",
                    "sample_rate_outside_5_percent",
                    "non_finite_sensor_value",
                )
            )
            for issue in issues
        )
        hard = hard or any(
            issue.startswith("left_right_sync_offset_ms=")
            and float(issue.split("=", 1)[1]) > 500
            for issue in issues
        ) or any(
            quality.clipped_ratio > 0.02 for quality in qualities
        )
        hard = hard or any(
            issue in {"dual_ear_stream_missing", "left_right_pair_incomplete"}
            for issue in issues
        )
        status = (
            "recollect_required"
            if hard
            else ("accepted_with_note" if issues else "accepted")
        )
        if status == "recollect_required":
            evidence = "rejected"
        report = QualityReport(
            schema_version="1.0",
            session_id=self.manifest.session_id,
            trial_id=self.manifest.trial_id,
            status=status,
            evidence_tier=evidence,
            streams=tuple(qualities),
            issues=tuple(issues),
        )
        return report, normalized

    def write_report(self, destination: Path | str) -> QualityReport:
        report, _ = self.audit()
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return report

    def write_training_captures(
        self, output_dir: Path | str
    ) -> tuple[QualityReport, list[Path]]:
        report, normalized = self.audit()
        if report.status == "recollect_required":
            raise ValueError(
                "trial failed quality audit and cannot enter the training tree"
            )
        destination = Path(output_dir)
        labels_dir = destination / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        source_label_path = (self.root / self.manifest.label_file).resolve()
        source_label = json.loads(
            source_label_path.read_text(encoding="utf-8-sig")
        )
        date_code = self.manifest.date[5:7] + self.manifest.date[8:10]
        written: list[Path] = []
        streams_by_id = {
            stream.device_id: stream for stream in self.manifest.streams
        }
        for device_id, (timestamps, values) in normalized.items():
            stream = streams_by_id[device_id]
            stem = (
                f"{date_code}-{self.manifest.subject_id}-"
                f"{self.manifest.trial_id}-{device_id}-{stream.ear_side}"
            )
            csv_path = destination / f"{stem}.csv"
            temporary_csv = csv_path.with_suffix(".csv.partial")
            with temporary_csv.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                fieldnames = [
                    "timestamp_monotonic_ns",
                    "sequence_id",
                    "device_id",
                    "ear_side",
                    "acc_x_mps2",
                    "acc_y_mps2",
                    "acc_z_mps2",
                    "gyro_x_rad_s",
                    "gyro_y_rad_s",
                    "gyro_z_rad_s",
                ]
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for index, (timestamp, row) in enumerate(
                    zip(timestamps, values)
                ):
                    writer.writerow(
                        {
                            "timestamp_monotonic_ns": int(timestamp),
                            "sequence_id": index,
                            "device_id": device_id,
                            "ear_side": stream.ear_side,
                            **{
                                name: float(value)
                                for name, value in zip(
                                    fieldnames[4:], row
                                )
                            },
                        }
                    )
            temporary_csv.replace(csv_path)
            derived_label = {
                **source_label,
                "device": device_id,
                "csv_file": f"../{csv_path.name}",
                "derived_from": {
                    "session_id": self.manifest.session_id,
                    "trial_id": self.manifest.trial_id,
                    "source_csv": stream.csv_file,
                    "source_sha256": stream.sha256,
                    "resampled_rate_hz": stream.sample_rate_configured_hz,
                    "units": {
                        "acceleration": "m/s2",
                        "angular_velocity": "rad/s",
                    },
                },
                "recording": {
                    **source_label.get("recording", {}),
                    "row_count": len(timestamps),
                    "duration_seconds": (
                        float(timestamps[-1] - timestamps[0])
                        / 1_000_000_000
                    ),
                },
            }
            label_path = labels_dir / f"{stem}.labels.json"
            temporary_label = label_path.with_suffix(
                label_path.suffix + ".partial"
            )
            temporary_label.write_text(
                json.dumps(derived_label, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary_label.replace(label_path)
            written.extend((csv_path, label_path))
        report_path = destination / f"{self.manifest.trial_id}.quality.json"
        report_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(report_path)
        return report, written
