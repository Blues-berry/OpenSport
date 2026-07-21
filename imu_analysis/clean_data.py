"""Clean IMU CSVs without overwriting raw data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from imu_common import ACC_COLS, GYRO_COLS, activity_from_folder, capture_group, device_id_from_path, elapsed_seconds, find_imu_files, read_imu_file, recording_name, sampling_info, session_name_from_path, source_name, subject_group_from_path


def isolated_zero_mask(x: pd.Series, max_run: int) -> pd.Series:
    """Flag short zero runs surrounded by clearly non-zero samples."""
    zero = x.eq(0) & x.notna()
    group = zero.ne(zero.shift(fill_value=False)).cumsum()
    run_size = zero.groupby(group).transform("sum")
    left, right = x.shift(1), x.shift(-1)
    baseline = (left + right) / 2
    same_sign = np.sign(left) == np.sign(right)
    neighbors_close = (left - right).abs() <= (0.2 * baseline.abs()).clip(lower=0.01)
    # Restrict repair to a single impossible-looking dropout, not normal sensor
    # quantisation around zero or a genuine zero crossing.
    return zero & run_size.eq(1) & same_sign & neighbors_close & baseline.abs().gt(0.02)


def hampel_mask(x: pd.Series, window: int, n_sigma: float) -> pd.Series:
    width = max(5, int(window) | 1)
    median = x.rolling(width, center=True, min_periods=max(3, width // 3)).median()
    abs_dev = (x - median).abs()
    mad = abs_dev.rolling(width, center=True, min_periods=max(3, width // 3)).median()
    threshold = n_sigma * 1.4826 * mad
    neighbors_close = (x.shift(1) - x.shift(-1)).abs() <= (0.25 * abs_dev).clip(lower=1e-9)
    return abs_dev.gt(threshold) & threshold.gt(1e-12) & neighbors_close


def clean_channel(
    x: pd.Series,
    fs: float,
    hampel_seconds: float,
    smooth_seconds: float,
    max_zero_run: int,
) -> tuple[pd.Series, dict[str, int]]:
    y = pd.to_numeric(x, errors="coerce").astype(float)
    zero_bad = isolated_zero_mask(y, max_zero_run)
    y.loc[zero_bad] = np.nan
    hampel_window = max(5, round(fs * hampel_seconds)) if np.isfinite(fs) else 7
    spike_bad = hampel_mask(y, hampel_window, 8.0)
    y.loc[spike_bad] = np.nan
    missing_before = int(y.isna().sum())
    # Only interpolate short gaps. Long dropouts remain missing and visible.
    gap_limit = max(1, round(fs * 0.25)) if np.isfinite(fs) else 5
    y = y.interpolate(method="linear", limit=gap_limit, limit_direction="both")
    smooth_window = max(1, round(fs * smooth_seconds)) if np.isfinite(fs) else 3
    if smooth_window > 1:
        y = y.rolling(smooth_window, center=True, min_periods=1).mean()
    return y, {
        "isolated_zero_replaced": int(zero_bad.sum()),
        "spike_replaced": int(spike_bad.sum()),
        "missing_after_detection": missing_before,
        "missing_after_cleaning": int(y.isna().sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("processed/cleaned"))
    parser.add_argument("--hampel-seconds", type=float, default=0.15)
    parser.add_argument("--smooth-seconds", type=float, default=0.06)
    parser.add_argument("--max-zero-run", type=int, default=2)
    args = parser.parse_args()
    files = find_imu_files(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = []
    for path in files:
        df = read_imu_file(path)
        recording = recording_name(path)
        duplicate_rows_removed = int(df.duplicated().sum())
        if duplicate_rows_removed:
            df = df.drop_duplicates().reset_index(drop=True)
        info = sampling_info(df)
        fs = float(info["sample_rate_hz"])
        sec, source = elapsed_seconds(df)
        df.insert(0, "analysis_time_s", sec)
        df.insert(1, "_capture_group", capture_group(path, df))
        df.insert(2, "_source_format", path.suffix.lower().lstrip("."))
        df.insert(3, "_activity", activity_from_folder(source_name(path)))
        df.insert(4, "_subject_group", subject_group_from_path(path))
        df.insert(5, "_session_name", session_name_from_path(path))
        df.insert(6, "_device_id", device_id_from_path(path))
        # Euler angles are already fused/filtered and quaternions must not be
        # independently smoothed (that breaks unit-norm geometry). Clean only
        # raw acceleration and angular velocity channels.
        clean_cols = [c for c in ACC_COLS + GYRO_COLS if c in df]
        for col_index, col in enumerate(clean_cols):
            cleaned, counts = clean_channel(
                df[col], fs, args.hampel_seconds, args.smooth_seconds, args.max_zero_run
            )
            df[col] = cleaned
            logs.append({"recording": recording, "source_format": path.suffix.lower().lstrip("."), "channel": col, "duplicate_rows_removed": duplicate_rows_removed if col_index == 0 else 0, **counts})
        if all(c in df for c in ["加速度X(g)", "加速度Y(g)", "加速度Z(g)"]):
            df["acc_magnitude_g"] = np.sqrt(sum(df[c] ** 2 for c in ["加速度X(g)", "加速度Y(g)", "加速度Z(g)"]))
            df["dynamic_acc_g"] = (df["acc_magnitude_g"] - 1.0).abs()
        if all(c in df for c in ["角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)"]):
            df["gyro_magnitude_dps"] = np.sqrt(sum(df[c] ** 2 for c in ["角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)"]))
        target_dir = args.output_dir / recording
        target_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_dir / "data.csv", index=False, encoding="utf-8-sig")
    log_df = pd.DataFrame(logs)
    log_df.to_csv(args.output_dir / "cleaning_log.csv", index=False, encoding="utf-8-sig")
    totals = {c: int(log_df[c].sum()) for c in log_df.columns if c.endswith(("replaced", "cleaning"))}
    totals["duplicate_rows_removed"] = int(log_df["duplicate_rows_removed"].sum())
    (args.output_dir / "cleaning_summary.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
