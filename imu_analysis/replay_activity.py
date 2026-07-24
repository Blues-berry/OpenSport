"""Replay a logger CSV through the production inference and policy path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from activity_features import SENSOR_KEYS, TARGET_RATE_HZ, uniform_resample
from activity_runtime import RuntimeCoordinator
from imu_common import ACC_COLS, GYRO_COLS, elapsed_seconds, read_imu_csv
from train_activity_model import recording_duration
from workout_store import WorkoutStore
from label_schema import load_label_document
from weak_session import evaluate_weak_session


def replay_file(
    source: Path,
    model_path: Path,
    database_path: Path | str,
    base_timestamp: float | None = None,
) -> dict:
    frame = read_imu_csv(source)
    duration = recording_duration(frame)
    matrix = frame.reindex(columns=ACC_COLS + GYRO_COLS).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    matrix = pd.DataFrame(matrix).interpolate(limit_direction="both").fillna(0.0).to_numpy()
    sample_seconds, _ = elapsed_seconds(frame)
    values = uniform_resample(
        matrix,
        duration,
        TARGET_RATE_HZ,
        source_timestamps_s=sample_seconds,
    )
    runtime = RuntimeCoordinator(model_path, database_path)
    base = float(base_timestamp if base_timestamp is not None else time.time())
    inference_count = 0
    for index, row in enumerate(values):
        sample = {
            "timestamp": base + index / TARGET_RATE_HZ,
            "sequence_id": index & 0xFFFF,
            **{key: float(value) for key, value in zip(SENSOR_KEYS, row)},
        }
        if runtime.update(sample):
            inference_count += 1
    runtime.flush(base + max(0, len(values) - 1) / TARGET_RATE_HZ)
    if runtime.store is None:
        raise RuntimeError("Replay requires a workout store")
    summary = runtime.store.daily_summary(now=base)
    summary.update(
        {
            "filename": source.name,
            "source_duration_seconds": round(duration),
            "inference_windows": inference_count,
            "last_inference": runtime.last_result,
        }
    )
    label_path = source.parent / "labels" / f"{source.stem}.labels.json"
    if label_path.exists():
        label = load_label_document(label_path)
        if label.get("annotation_scope") == "session_weak":
            predicted = [
                activity
                for session in summary.get("sessions", [])
                for activity in session.get("activities", [])
            ]
            summary["weak_validation"] = evaluate_weak_session(
                label.get("weak_targets", {}), predicted
            )
    return summary


def analyze_temporary(source: Path, model_path: Path) -> dict:
    return replay_file(source, model_path, ":memory:")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--database",
        type=Path,
        default=":memory:",
        help="Optional persistent replay database; defaults to an isolated in-memory store.",
    )
    args = parser.parse_args()
    print(json.dumps(replay_file(args.source, args.model, args.database), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
