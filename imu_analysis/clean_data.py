"""Clean IMU CSVs without overwriting raw data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from imu_common import ACC_COLS, GYRO_COLS, activity_from_folder, capture_group, device_id_from_path, elapsed_seconds, find_imu_files, read_imu_file, recording_name, robust_jump_mask, sampling_info, session_name_from_path, source_name, subject_group_from_path


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
) -> tuple[pd.Series, dict[str, int], dict[str, pd.Series]]:
    original = pd.to_numeric(x, errors="coerce").astype(float)
    y = original.copy()
    missing_input = original.isna()
    zero_bad = isolated_zero_mask(y, max_zero_run)
    y.loc[zero_bad] = np.nan
    hampel_window = max(5, round(fs * hampel_seconds)) if np.isfinite(fs) else 7
    spike_bad = hampel_mask(y, hampel_window, 8.0)
    jump_candidate = robust_jump_mask(original)
    y.loc[spike_bad] = np.nan
    missing_before = int(y.isna().sum())
    # Only interpolate short gaps. Long dropouts remain missing and visible.
    gap_limit = max(1, round(fs * 0.25)) if np.isfinite(fs) else 5
    y = y.interpolate(method="linear", limit=gap_limit, limit_direction="both")
    smooth_window = max(1, round(fs * smooth_seconds)) if np.isfinite(fs) else 3
    if smooth_window > 1:
        y = y.rolling(smooth_window, center=True, min_periods=1).mean()
    counts = {
        "missing_input": int(missing_input.sum()),
        "isolated_zero_replaced": int(zero_bad.sum()),
        "spike_replaced": int(spike_bad.sum()),
        "jump_candidate": int(jump_candidate.sum()),
        "missing_after_detection": missing_before,
        "missing_after_cleaning": int(y.isna().sum()),
    }
    masks = {
        "missing_input": missing_input,
        "isolated_zero": zero_bad,
        "spike": spike_bad,
        "jump_candidate": jump_candidate,
    }
    return y, counts, masks


def anomaly_events(
    df: pd.DataFrame,
    mask: pd.Series,
    channel: str,
    anomaly_type: str,
    action_taken: str,
    original_values: pd.Series,
) -> pd.DataFrame:
    selected = mask.fillna(False).to_numpy(dtype=bool)
    if not selected.any():
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "source_row": df.loc[selected, "_source_row"].to_numpy(),
            "analysis_time_s": df.loc[selected, "analysis_time_s"].to_numpy(),
            "channel": channel,
            "anomaly_type": anomaly_type,
            "original_value": original_values.loc[selected].to_numpy(),
            "action_taken": action_taken,
        }
    )


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
        df.insert(0, "_source_row", np.arange(len(df), dtype=int))
        duplicate_mask = df.drop(columns="_source_row").duplicated()
        duplicate_rows_removed = int(duplicate_mask.sum())
        duplicate_events = pd.DataFrame(
            {
                "source_row": df.loc[duplicate_mask, "_source_row"].to_numpy(),
                "analysis_time_s": np.nan,
                "channel": "*",
                "anomaly_type": "duplicate_row",
                "original_value": np.nan,
                "action_taken": "removed",
            }
        )
        if duplicate_rows_removed:
            df = df.loc[~duplicate_mask].reset_index(drop=True)
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
        event_frames = [duplicate_events]
        row_repaired = pd.Series(False, index=df.index)
        row_anomaly_candidate = pd.Series(False, index=df.index)
        for col_index, col in enumerate(clean_cols):
            original_values = pd.to_numeric(df[col], errors="coerce").astype(float)
            cleaned, counts, masks = clean_channel(
                df[col], fs, args.hampel_seconds, args.smooth_seconds, args.max_zero_run
            )
            physical_limit = original_values.abs().gt(16 if col in ACC_COLS else 2000)
            counts["physical_limit_candidate"] = int(physical_limit.sum())
            masks["physical_limit_candidate"] = physical_limit
            df[col] = cleaned
            logs.append({"recording": recording, "source_format": path.suffix.lower().lstrip("."), "channel": col, "duplicate_rows_removed": duplicate_rows_removed if col_index == 0 else 0, **counts})
            row_repaired |= masks["isolated_zero"] | masks["spike"]
            row_anomaly_candidate |= pd.concat(list(masks.values()), axis=1).any(axis=1)
            for anomaly_type, action_taken in [
                ("missing_input", "short_gap_interpolated_if_possible"),
                ("isolated_zero", "interpolated"),
                ("spike", "interpolated"),
                ("jump_candidate", "flag_only"),
                ("physical_limit_candidate", "flag_only"),
            ]:
                event_frames.append(anomaly_events(df, masks[anomaly_type], col, anomaly_type, action_taken, original_values))
        df.insert(7, "_row_repaired", row_repaired.to_numpy())
        df.insert(8, "_row_anomaly_candidate", row_anomaly_candidate.to_numpy())
        if all(c in df for c in ["加速度X(g)", "加速度Y(g)", "加速度Z(g)"]):
            df["acc_magnitude_g"] = np.sqrt(sum(df[c] ** 2 for c in ["加速度X(g)", "加速度Y(g)", "加速度Z(g)"]))
            df["dynamic_acc_g"] = (df["acc_magnitude_g"] - 1.0).abs()
        if all(c in df for c in ["角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)"]):
            df["gyro_magnitude_dps"] = np.sqrt(sum(df[c] ** 2 for c in ["角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)"]))
        target_dir = args.output_dir / recording
        target_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(target_dir / "data.csv", index=False, encoding="utf-8-sig")
        events = pd.concat([frame for frame in event_frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in event_frames) else pd.DataFrame(columns=["source_row", "analysis_time_s", "channel", "anomaly_type", "original_value", "action_taken"])
        events.insert(0, "recording", recording)
        events.insert(1, "activity", activity_from_folder(source_name(path)))
        events.insert(2, "device_id", device_id_from_path(path))
        events.to_csv(target_dir / "anomalies.csv", index=False, encoding="utf-8-sig")
    log_df = pd.DataFrame(logs)
    log_df.to_csv(args.output_dir / "cleaning_log.csv", index=False, encoding="utf-8-sig")
    totals = {c: int(log_df[c].sum()) for c in log_df.columns if c.endswith(("replaced", "cleaning"))}
    totals["duplicate_rows_removed"] = int(log_df["duplicate_rows_removed"].sum())
    for col in ["missing_input", "jump_candidate", "physical_limit_candidate"]:
        totals[col] = int(log_df[col].sum())
    (args.output_dir / "cleaning_summary.json").write_text(
        json.dumps(totals, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
