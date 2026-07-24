"""Compare two IMU actions across people using fixed-duration windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from imu_common import ACC_COLS, ANGLE_COLS, GYRO_COLS, elapsed_seconds, read_imu_csv, sampling_info


def identity(path: Path) -> tuple[str, str]:
    parts = path.stem.split("-", 1)
    return (parts[0], parts[1] if len(parts) > 1 else path.parent.name)


def resample(frame: pd.DataFrame, fs: float) -> tuple[np.ndarray, np.ndarray]:
    time, _ = elapsed_seconds(frame)
    time = np.asarray(time, dtype=float)
    valid = np.isfinite(time)
    values = frame.loc[valid, ACC_COLS + GYRO_COLS + ANGLE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    time = time[valid]
    order = np.argsort(time)
    time, values = time[order], values[order]
    unique = np.r_[True, np.diff(time) > 1e-9]
    time, values = time[unique], values[unique]
    uniform = np.arange(time[0], time[-1], 1.0 / fs)
    output = np.column_stack([np.interp(uniform, time, values[:, i]) for i in range(values.shape[1])])
    return uniform, output


def dominant_frequency(values: np.ndarray, fs: float) -> float:
    centered = values - np.mean(values)
    power = np.abs(np.fft.rfft(centered)) ** 2
    frequency = np.fft.rfftfreq(len(values), 1.0 / fs)
    valid = (frequency >= 0.15) & (frequency <= 3.0)
    return float(frequency[valid][np.argmax(power[valid])]) if valid.any() else np.nan


def features(values: np.ndarray, fs: float) -> dict[str, float]:
    acc, gyro, angle = values[:, :3], values[:, 3:6], values[:, 6:9]
    acc_mag, gyro_mag = np.linalg.norm(acc, axis=1), np.linalg.norm(gyro, axis=1)
    result = {
        "acc_mag_std": float(np.std(acc_mag)),
        "acc_dynamic_p95": float(np.percentile(np.abs(acc_mag - np.median(acc_mag)), 95)),
        "gyro_mag_mean": float(np.mean(gyro_mag)),
        "gyro_mag_std": float(np.std(gyro_mag)),
        "gyro_mag_p95": float(np.percentile(gyro_mag, 95)),
        "acc_dominant_hz": dominant_frequency(acc_mag, fs),
        "gyro_dominant_hz": dominant_frequency(gyro_mag, fs),
    }
    for index, axis in enumerate("xyz"):
        result[f"acc_{axis}_std"] = float(np.std(acc[:, index]))
        result[f"gyro_{axis}_std"] = float(np.std(gyro[:, index]))
        result[f"angle_{axis}_range_p90"] = float(np.percentile(angle[:, index], 95) - np.percentile(angle[:, index], 5))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, default=50.0)
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    windows: list[dict[str, object]] = []
    for path in args.files:
        person, action = identity(path)
        frame = read_imu_csv(path)
        time, values = resample(frame, args.sample_rate)
        summaries.append({
            "person": person,
            "action": action,
            "file": str(path),
            "rows": len(frame),
            "duration_s": float(time[-1] - time[0]),
            "raw_sample_rate_hz": float(sampling_info(frame)["sample_rate_hz"]),
            "full_duplicate_rows": int(frame.duplicated().sum()),
            **features(values, args.sample_rate),
        })
        length = max(16, round(args.window_seconds * args.sample_rate))
        step = max(1, round(length * (1 - args.overlap)))
        for start in range(0, len(values) - length + 1, step):
            windows.append({
                "person": person,
                "action": action,
                "file": str(path),
                "window_start_s": start / args.sample_rate,
                **features(values[start : start + length], args.sample_rate),
            })

    summary = pd.DataFrame(summaries)
    window = pd.DataFrame(windows)
    summary.to_csv(args.output_dir / "recording_summary.csv", index=False, encoding="utf-8-sig")
    window.to_csv(args.output_dir / "window_features.csv", index=False, encoding="utf-8-sig")

    feature_cols = [c for c in window.select_dtypes(include=[np.number]) if c != "window_start_s"]
    actions = sorted(window["action"].unique())
    people = sorted(window["person"].unique())
    comparisons: list[dict[str, object]] = []
    if len(actions) == 2:
        for feature in feature_cols:
            item: dict[str, object] = {"feature": feature}
            signs = []
            for person in people:
                med = window.loc[window["person"].eq(person)].groupby("action")[feature].median()
                if all(action in med for action in actions):
                    difference = float(med[actions[0]] - med[actions[1]])
                    item[f"{person}_difference_{actions[0]}_minus_{actions[1]}"] = difference
                    signs.append(np.sign(difference))
            item["direction_consistent"] = bool(len(signs) >= 2 and len(set(signs)) == 1)
            comparisons.append(item)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(args.output_dir / "feature_comparison.csv", index=False, encoding="utf-8-sig")

    # Exploratory leave-one-person-out nearest-centroid validation. This is
    # deliberately dependency-free and is not a claim of production accuracy.
    validation: list[dict[str, object]] = []
    if len(actions) == 2 and len(people) >= 2:
        for test_person in people:
            train = window["person"].ne(test_person)
            test = ~train
            x_train = window.loc[train, feature_cols].to_numpy(float)
            x_test = window.loc[test, feature_cols].to_numpy(float)
            median = np.nanmedian(x_train, axis=0)
            x_train = np.where(np.isfinite(x_train), x_train, median)
            x_test = np.where(np.isfinite(x_test), x_test, median)
            scale = np.nanstd(x_train, axis=0)
            scale[scale < 1e-9] = 1.0
            x_train, x_test = (x_train - median) / scale, (x_test - median) / scale
            y_train = window.loc[train, "action"].to_numpy()
            y_test = window.loc[test, "action"].to_numpy()
            centers = {action: x_train[y_train == action].mean(axis=0) for action in actions}
            predicted = np.array([min(actions, key=lambda action: np.sum((row - centers[action]) ** 2)) for row in x_test])
            recalls = {action: float(np.mean(predicted[y_test == action] == action)) for action in actions}
            validation.append({
                "test_person": test_person,
                "windows": len(y_test),
                "balanced_accuracy": float(np.mean(list(recalls.values()))),
                **{f"recall_{action}": value for action, value in recalls.items()},
            })
    pd.DataFrame(validation).to_csv(args.output_dir / "cross_person_validation.csv", index=False, encoding="utf-8-sig")
    print(json.dumps({"recordings": len(summary), "windows": len(window), "validation": validation}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
