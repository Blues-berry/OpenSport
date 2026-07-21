"""Inspect all IMU CSV files and produce machine-readable and Markdown reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from imu_common import (
    ACC_COLS,
    ANGLE_COLS,
    CORE_COLS,
    GYRO_COLS,
    QUAT_COLS,
    activity_from_folder,
    capture_group,
    device_id_from_path,
    find_imu_files,
    longest_run,
    longest_constant_run,
    markdown_table,
    parse_time_column,
    read_imu_file,
    recording_name,
    source_name,
    subject_group_from_path,
    session_name_from_path,
    robust_jump_mask,
    robust_outlier_mask,
    sampling_info,
    trial_from_folder,
)


def inspect_file(path: Path, root: Path) -> tuple[dict, list[dict]]:
    df = read_imu_file(path)
    info = sampling_info(df)
    parsed_time = parse_time_column(df, "时间")
    first_time = parsed_time.dropna().iloc[0] if parsed_time.notna().any() else pd.NaT
    folder = source_name(path)
    recording = recording_name(path)
    full_duplicate_count = int(df.duplicated().sum())
    host_duplicate_ratio = float(df["时间"].duplicated().mean()) if "时间" in df else np.nan
    onboard_duplicate_ratio = float(df["片上时间()"].duplicated().mean()) if "片上时间()" in df else np.nan

    core_present = [c for c in CORE_COLS if c in df]
    missing_core = [c for c in CORE_COLS if c not in df]
    summary = {
        "activity": activity_from_folder(folder),
        "trial": trial_from_folder(folder),
        "recording": recording,
        "capture_group": capture_group(path, df),
        "source_format": path.suffix.lower().lstrip("."),
        "subject_group": subject_group_from_path(path),
        "session_name": session_name_from_path(path),
        "device_id": device_id_from_path(path),
        "file": str(path.relative_to(root)),
        "rows": len(df),
        **info,
        "columns": len(df.columns),
        "missing_core_columns": ";".join(missing_core),
        "full_duplicate_rows": full_duplicate_count,
        "full_duplicate_ratio": full_duplicate_count / max(len(df), 1),
        "host_timestamp_duplicate_ratio": host_duplicate_ratio,
        "onboard_timestamp_duplicate_ratio": onboard_duplicate_ratio,
        "device_count": int(df["设备名称"].nunique(dropna=True)) if "设备名称" in df else 0,
        "start_time": first_time.isoformat() if pd.notna(first_time) else "",
        "start_second_of_day": (first_time.hour * 3600 + first_time.minute * 60 + first_time.second + first_time.microsecond / 1e6) if pd.notna(first_time) else np.nan,
    }
    orientation_cols = [c for c in ANGLE_COLS + QUAT_COLS if c in df]
    summary["orientation_valid"] = bool(
        len(orientation_cols) == 7
        and all(df[c].nunique(dropna=True) > 2 for c in ANGLE_COLS)
        and not all(df[c].abs().max() == 0 for c in QUAT_COLS)
    )

    channel_rows: list[dict] = []
    for col in core_present:
        x = pd.to_numeric(df[col], errors="coerce")
        finite = x[np.isfinite(x)]
        zero_mask = x.eq(0).fillna(False).to_numpy()
        physical = pd.Series(False, index=x.index)
        if col in ACC_COLS:
            physical = x.abs() > 16
        elif col in GYRO_COLS:
            physical = x.abs() > 2000
        channel_rows.append(
            {
                "activity": summary["activity"],
                "trial": summary["trial"],
                "recording": recording,
                "channel": col,
                "count": int(x.notna().sum()),
                "missing_ratio": float(x.isna().mean()),
                "zero_ratio": float(x.eq(0).mean()),
                "longest_zero_run": longest_run(zero_mask),
                "longest_constant_run": longest_constant_run(x),
                "unique_values": int(x.nunique(dropna=True)),
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
                "min": float(finite.min()) if len(finite) else np.nan,
                "max": float(finite.max()) if len(finite) else np.nan,
                "robust_outlier_ratio": float(robust_outlier_mask(x).mean()),
                "jump_ratio": float(robust_jump_mask(x).mean()),
                "physical_limit_ratio": float(physical.mean()),
            }
        )
    return summary, channel_rows


def paired_session_summary(files: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (capture, device), part in files.groupby(["capture_group", "device_id"]):
        if set(part["source_format"]) != {"csv", "txt"} or len(part) != 2:
            continue
        indexed = part.set_index("source_format")
        rows.append(
            {
                "capture_group": capture,
                "device_id": device,
                "activity": indexed.loc["csv", "activity"],
                "csv_recording": indexed.loc["csv", "recording"],
                "txt_recording": indexed.loc["txt", "recording"],
                "start_offset_s": abs(indexed.loc["csv", "start_second_of_day"] - indexed.loc["txt", "start_second_of_day"]),
                "duration_abs_diff_s": abs(indexed.loc["csv", "duration_s"] - indexed.loc["txt", "duration_s"]),
                "sample_rate_abs_diff_hz": abs(indexed.loc["csv", "sample_rate_hz"] - indexed.loc["txt", "sample_rate_hz"]),
                "row_count_abs_diff": abs(indexed.loc["csv", "rows"] - indexed.loc["txt", "rows"]),
            }
        )
    return pd.DataFrame(rows)


def write_markdown(files: pd.DataFrame, channels: pd.DataFrame, pairs: pd.DataFrame, output: Path) -> None:
    total_rows = int(files["rows"].sum())
    fs = files["sample_rate_hz"].replace([np.inf, -np.inf], np.nan)
    missing = channels.groupby("channel")["missing_ratio"].mean().sort_values(ascending=False)
    zero = channels.groupby("channel")["zero_ratio"].mean().sort_values(ascending=False)
    outlier = channels.groupby("channel")["robust_outlier_ratio"].mean().sort_values(ascending=False)
    parallel_groups = files.groupby("capture_group").size()
    parallel_groups = parallel_groups[parallel_groups > 1]
    equivalent_exports = bool(
        len(pairs)
        and pairs["start_offset_s"].median() < 0.1
        and pairs["duration_abs_diff_s"].median() < 0.2
        and pairs["sample_rate_abs_diff_hz"].median() < 0.1
    )
    analysis_files = files[files["source_format"] == "csv"].copy() if equivalent_exports else files.copy()
    activity_overview = analysis_files.groupby("activity", as_index=False).agg(
        sessions=("capture_group", "nunique"),
        device_samples=("recording", "size"),
        duration_median_s=("duration_s", "median"),
        sample_rate_median_hz=("sample_rate_hz", "median"),
    )
    subject_overview = analysis_files.groupby("subject_group", as_index=False).agg(
        sessions=("capture_group", "nunique"), device_samples=("recording", "size")
    )
    device_overview = analysis_files.groupby("device_id", as_index=False).agg(
        samples=("recording", "size"),
        sample_rate_median_hz=("sample_rate_hz", "median"),
        exact_duplicate_ratio_median=("full_duplicate_ratio", "median"),
    )
    format_finding = (
        "CSV/TXT 对同一设备高度符合‘同一数据流的两种导出格式’，特征统计只选 CSV 一份；两种格式仍分别保留在质量审计和清洗输出中。"
        if equivalent_exports
        else "CSV/TXT 按不同样本保留；动作和起始时刻接近的同期样本共享 capture_group，交叉验证时整组划分，避免同期动作泄漏。"
    )
    invalid_orientation = files.loc[~files["orientation_valid"], "recording"].tolist()
    orientation_finding = (
        f"姿态输出无效的采集共有 {len(invalid_orientation)} 段：" + "、".join(invalid_orientation) + "。这些段不能用于头颈姿态分析。"
        if invalid_orientation
        else "所有采集的欧拉角和四元数均有变化，未发现整段冻结或全零的姿态输出。"
    )
    lines = [
        "# IMU 数据质量检查报告",
        "",
        f"- 文件数：{len(files)}；总样本数：{total_rows:,}",
        f"- 采样率中位数：{fs.median():.2f} Hz；范围：{fs.min():.2f}–{fs.max():.2f} Hz",
        f"- 单次时长范围：{files['duration_s'].min():.2f}–{files['duration_s'].max():.2f} 秒",
        f"- 完全重复行：{int(files['full_duplicate_rows'].sum()):,}",
        f"- 样本文件：{len(files)}；按动作与起始时刻识别到的采集会话：{files['capture_group'].nunique()}；含 CSV/TXT 同期样本的会话：{len(parallel_groups)}",
        "",
        "## 主要发现",
        "",
        "1. 时间戳存在批量重复时，不应使用逐行时间差作为采样率；脚本用总样本数/总时长估算有效采样率并重建均匀时间轴。",
        "2. 位移和速度等整列为空的派生字段不参与质量评分；核心检查集中在加速度、角速度、欧拉角和四元数。",
        "3. " + orientation_finding,
        "4. robust_outlier_ratio 与 jump_ratio 是候选异常比例，不等同于应删除比例；剧烈运动的真实峰值也可能被统计为候选点。",
        "5. " + format_finding,
        "",
        "## CSV/TXT 同期采集依据",
        "",
        (f"识别到 {len(pairs)} 组同设备、同动作同期 CSV/TXT 文件。起始时刻差中位数 {pairs['start_offset_s'].median():.3f} 秒、最大 {pairs['start_offset_s'].max():.3f} 秒，"
         f"{(pairs['start_offset_s'] <= 2).mean():.1%} 的组在 2 秒内；采样率绝对差中位数 {pairs['sample_rate_abs_diff_hz'].median():.3f} Hz，"
         f"时长绝对差中位数 {pairs['duration_abs_diff_s'].median():.3f} 秒，行数差中位数 {pairs['row_count_abs_diff'].median():.0f}。"
         + ("这高度支持二者是同一设备数据流的两种导出格式，建模不应重复计权。" if equivalent_exports else "这支持将二者放在同一采集会话分组中。") if len(pairs) else "没有识别到同期 CSV/TXT 样本。"),
        "",
        markdown_table(pairs.round(4)) if len(pairs) else "",
        "",
        "## 动作、人员组与设备概览（CSV 去重口径）",
        "",
        markdown_table(activity_overview.round(4)),
        "",
        markdown_table(subject_overview.round(4)),
        "",
        markdown_table(device_overview.round(4)),
        "",
        "## 各次采集概览",
        "",
        markdown_table(files[["recording", "rows", "duration_s", "sample_rate_hz", "full_duplicate_ratio", "host_timestamp_duplicate_ratio", "onboard_timestamp_duplicate_ratio", "orientation_valid"]].round(4)),
        "",
        "## 缺失率最高通道（均值）",
        "",
        markdown_table(missing.head(10).rename("missing_ratio").round(6).to_frame(), include_index=True),
        "",
        "## 零值比例最高通道（均值）",
        "",
        markdown_table(zero.head(10).rename("zero_ratio").round(6).to_frame(), include_index=True),
        "",
        "## 候选异常值比例最高通道（均值）",
        "",
        markdown_table(outlier.head(10).rename("robust_outlier_ratio").round(6).to_frame(), include_index=True),
        "",
        "完整数据见 `quality_files.csv`、`quality_channels.csv` 和 `quality_summary.json`。",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/quality"))
    args = parser.parse_args()
    files = find_imu_files(args.data_dir)
    if not files:
        raise SystemExit(f"No CSV/TXT files found under {args.data_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    file_rows, channel_rows = [], []
    for path in files:
        summary, details = inspect_file(path, args.data_dir)
        file_rows.append(summary)
        channel_rows.extend(details)
    file_df = pd.DataFrame(file_rows)
    channel_df = pd.DataFrame(channel_rows)
    pair_df = paired_session_summary(file_df)
    file_df.to_csv(args.output_dir / "quality_files.csv", index=False, encoding="utf-8-sig")
    channel_df.to_csv(args.output_dir / "quality_channels.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(args.output_dir / "csv_txt_session_pairs.csv", index=False, encoding="utf-8-sig")
    payload = {
        "file_count": len(file_df),
        "total_rows": int(file_df["rows"].sum()),
        "sample_rate_hz_median": float(file_df["sample_rate_hz"].median()),
        "sample_rate_hz_min": float(file_df["sample_rate_hz"].min()),
        "sample_rate_hz_max": float(file_df["sample_rate_hz"].max()),
        "total_duplicate_rows": int(file_df["full_duplicate_rows"].sum()),
    }
    (args.output_dir / "quality_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_markdown(file_df, channel_df, pair_df, args.output_dir / "quality_report.md")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
