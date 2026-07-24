"""Extract windowed IMU features and compare actions and exercise states."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from imu_common import ACC_COLS, ANGLE_COLS, GYRO_COLS, activity_from_folder, find_csv_files, markdown_table, read_imu_file, state_from_activity, trial_from_folder


def spectral_features(x: np.ndarray, fs: float) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 8 or not np.isfinite(fs) or fs <= 0:
        return np.nan, np.nan
    x = x - np.mean(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    freq = np.fft.rfftfreq(len(x), 1.0 / fs)
    valid = (freq >= 0.2) & (freq <= min(15.0, fs / 2))
    if not valid.any() or power[valid].sum() <= 0:
        return 0.0, 0.0
    p = power[valid] / power[valid].sum()
    dominant = float(freq[valid][np.argmax(power[valid])])
    entropy = float(-(p * np.log2(p + 1e-15)).sum() / np.log2(len(p))) if len(p) > 1 else 0.0
    return dominant, entropy


def stats(prefix: str, x: np.ndarray, fs: float) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if len(finite) < 2:
        return {f"{prefix}_{k}": np.nan for k in ["mean", "std", "rms", "range", "iqr", "mad", "jerk_rms", "dom_freq_hz", "spec_entropy"]}
    diff = np.diff(finite) * fs
    dom, entropy = spectral_features(finite, fs)
    return {
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_std": float(np.std(finite, ddof=1)),
        f"{prefix}_rms": float(np.sqrt(np.mean(finite**2))),
        f"{prefix}_range": float(np.ptp(finite)),
        f"{prefix}_iqr": float(np.percentile(finite, 75) - np.percentile(finite, 25)),
        f"{prefix}_mad": float(np.median(np.abs(finite - np.median(finite)))),
        f"{prefix}_jerk_rms": float(np.sqrt(np.mean(diff**2))) if len(diff) else 0.0,
        f"{prefix}_dom_freq_hz": dom,
        f"{prefix}_spec_entropy": entropy,
    }


def windows_for_file(path: Path, window_s: float, overlap: float) -> list[dict]:
    df = read_imu_file(path)
    if "analysis_time_s" in df:
        time = pd.to_numeric(df["analysis_time_s"], errors="coerce").to_numpy()
        duration = float(time[-1] - time[0]) if len(time) > 1 else 0
        fs = (len(time) - 1) / duration if duration > 0 else np.nan
    else:
        # Cleaned files should normally include analysis_time_s.
        fs = 50.0
        time = np.arange(len(df)) / fs
    # Some audit-only CSVs contain an analysis_time_s column but no valid
    # sampling interval.  Treat those files like raw logger exports instead
    # of attempting to construct a window with a NaN length.
    if not np.isfinite(fs) or fs <= 0:
        fs = 50.0
        time = np.arange(len(df), dtype=float) / fs
    n = max(16, round(window_s * fs))
    step = max(1, round(n * (1 - overlap)))
    activity = str(df["_activity"].iloc[0]) if "_activity" in df else activity_from_folder(path.parent.name)
    capture = str(df["_capture_group"].iloc[0]) if "_capture_group" in df else path.parent.name
    source_format = str(df["_source_format"].iloc[0]) if "_source_format" in df else "csv"
    device_id = str(df["_device_id"].iloc[0]) if "_device_id" in df else "unknown-device"
    subject_group = str(df["_subject_group"].iloc[0]) if "_subject_group" in df else "unknown"
    orientation_valid = all(c in df and df[c].nunique(dropna=True) > 2 for c in ANGLE_COLS)
    rows = []
    for start in range(0, max(0, len(df) - n + 1), step):
        part = df.iloc[start : start + n]
        row = {
            "activity": activity,
            "state": state_from_activity(activity),
            "recording": path.parent.name,
            "capture_group": capture,
            "source_format": source_format,
            "device_id": device_id,
            "subject_group": subject_group,
            "trial": trial_from_folder(path.parent.name),
            "window_start_s": float(time[start]),
            "window_end_s": float(time[start + n - 1]),
            "sample_rate_hz": float(fs),
            "orientation_valid": orientation_valid,
        }
        for col, prefix in zip(ACC_COLS + GYRO_COLS + ANGLE_COLS, ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z", "angle_x", "angle_y", "angle_z"]):
            if col in part and (col not in ANGLE_COLS or orientation_valid):
                row.update(stats(prefix, part[col].to_numpy(), fs))
        if all(c in part for c in ACC_COLS):
            acc = part[ACC_COLS].to_numpy(dtype=float)
            mag = np.linalg.norm(acc, axis=1)
            row.update(stats("acc_mag", mag, fs))
            row.update(stats("dynamic_acc", np.abs(mag - 1.0), fs))
            row["acc_sma"] = float(np.nanmean(np.sum(np.abs(acc - np.nanmean(acc, axis=0)), axis=1)))
        if all(c in part for c in GYRO_COLS):
            gyro = part[GYRO_COLS].to_numpy(dtype=float)
            row.update(stats("gyro_mag", np.linalg.norm(gyro, axis=1), fs))
        rows.append(row)
    return rows


def standardized_effect(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna().to_numpy(), b.dropna().to_numpy()
    if len(a) < 2 or len(b) < 2:
        return np.nan
    pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0


def nearest_centroid_loro(features: pd.DataFrame, feature_cols: list[str], target: str = "state", group_col: str | None = None) -> pd.DataFrame:
    """Leave-one-recording-out estimate, avoiding overlap-window leakage."""
    predictions = []
    group_col = group_col or ("capture_group" if "capture_group" in features else "recording")
    for held in features[group_col].unique():
        train = features[features[group_col] != held]
        test = features[features[group_col] == held]
        if train[target].nunique() < 2:
            continue
        mean = train[feature_cols].mean()
        std = train[feature_cols].std().replace(0, 1).fillna(1)
        ztrain = (train[feature_cols] - mean) / std
        ztest = (test[feature_cols] - mean) / std
        centers = ztrain.groupby(train[target]).mean()
        known = test[target].isin(centers.index).to_numpy()
        if not known.any():
            # A class recorded only once (currently 爬楼) has no training
            # example when its sole recording is held out, so it is not a
            # meaningful LORO test case.
            continue
        test = test.iloc[np.flatnonzero(known)]
        ztest = ztest.iloc[np.flatnonzero(known)]
        distances = np.stack([np.nansum((ztest.to_numpy() - centers.loc[label].to_numpy()) ** 2, axis=1) for label in centers.index], axis=1)
        pred = np.asarray(centers.index)[np.argmin(distances, axis=1)]
        predictions.extend(
            {"recording": held, "actual": actual, "predicted": guess}
            for actual, guess in zip(test[target], pred)
        )
    return pd.DataFrame(predictions)


def pairwise_action_distances(features: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Standardized centroid distances; larger usually means easier separation."""
    scaled = (features[feature_cols] - features[feature_cols].mean()) / features[feature_cols].std().replace(0, 1)
    centers = scaled.groupby(features["activity"]).mean()
    rows = []
    labels = list(centers.index)
    for i, left in enumerate(labels):
        for right in labels[i + 1 :]:
            distance = float(np.sqrt(np.nanmean((centers.loc[left] - centers.loc[right]) ** 2)))
            rows.append({"activity_a": left, "activity_b": right, "standardized_centroid_distance": distance})
    if not rows:
        return pd.DataFrame(columns=["activity_a", "activity_b", "standardized_centroid_distance"])
    return pd.DataFrame(rows).sort_values("standardized_centroid_distance")


def write_analysis(features: pd.DataFrame, group: pd.DataFrame, effects: pd.DataFrame, cv: pd.DataFrame, subject_cv: pd.DataFrame, action_cv: pd.DataFrame, pairwise: pd.DataFrame, output: Path) -> None:
    accuracy = float((cv["actual"] == cv["predicted"]).mean()) if len(cv) else np.nan
    balanced_accuracy = float(cv.assign(correct=cv["actual"] == cv["predicted"]).groupby("actual")["correct"].mean().mean()) if len(cv) else np.nan
    per_recording = cv.assign(correct=cv["actual"] == cv["predicted"]).groupby("recording")["correct"].mean().sort_values() if len(cv) else pd.Series(dtype=float)
    action_accuracy = float((action_cv["actual"] == action_cv["predicted"]).mean()) if len(action_cv) else np.nan
    subject_accuracy = float((subject_cv["actual"] == subject_cv["predicted"]).mean()) if len(subject_cv) else np.nan
    subject_balanced_accuracy = float(subject_cv.assign(correct=subject_cv["actual"] == subject_cv["predicted"]).groupby("actual")["correct"].mean().mean()) if len(subject_cv) else np.nan
    confusion = pd.crosstab(action_cv["actual"], action_cv["predicted"], normalize="index").round(3) if len(action_cv) else pd.DataFrame()
    reliable_effects = effects[~effects["feature"].str.startswith(("angle_", "quat_"))]
    top = reliable_effects.reindex(reliable_effects["abs_effect"].sort_values(ascending=False).index).head(12)
    selected = [c for c in ["dynamic_acc_std", "gyro_mag_mean", "gyro_mag_std", "acc_mag_std", "angle_x_mean", "angle_y_mean", "angle_z_mean"] if c in group.columns]
    activity_sessions = features.groupby("activity")["capture_group"].nunique().sort_values()
    singleton_actions = activity_sessions[activity_sessions < 2].index.tolist()
    state_sessions = features.groupby("state")["capture_group"].nunique().rename("sessions").reset_index()
    posture_rows = group[group["activity"].astype(str).str.startswith(("坐姿", "低头", "偏头", "弯腰取物"))]
    posture_cols = [c for c in ["activity", "angle_x_mean", "angle_y_mean", "angle_z_mean", "gyro_mag_mean", "dynamic_acc_std"] if c in posture_rows]
    lines = [
        "# IMU 动作差异与初步策略",
        "",
        "## 数据边界",
        "",
        "当前严格按“是否正在健身”而不是“是否发生运动”解释标签：跑步机、深蹲、弓步蹲、卷腹暂列为健身；坐姿、站姿、说话、咀嚼、低头、偏头、弯腰取物暂列为非健身；自由行走、爬楼/下楼、坐起缺少场景语义，列为 ambiguous；不对称佩戴、佩戴取下列为 wear_artifact。后两类不参与健身二分类初步验证。坐姿1/2/3没有明确的良好/不良含义，因此不能从这批数据验证坐姿好坏二分类准确率。",
        "",
        markdown_table(state_sessions),
        "",
        "## 运动/非运动差异最大的窗口特征",
        "",
        markdown_table(top[["feature", "exercise_mean", "non_exercise_mean", "effect_size", "abs_effect"]].round(5)),
        "",
        "## 各动作关键特征均值",
        "",
        markdown_table(group[["activity"] + selected].round(5)),
        "",
        "## 头颈姿态信号的可用性",
        "",
        "低头、偏头与坐姿在运动强度上都可能很低，必须使用佩戴后中立姿态的相对角度。本批数据中低头的 Y 角、偏头的 X/Y 角与多数坐姿存在差异，说明姿态方向有潜力；但弯腰取物也会产生很大的俯仰变化，必须先确认处于持续坐姿并设置持续时间条件。Z 角受设备朝向和角度环绕影响明显，不宜直接使用跨人的绝对阈值。",
        "",
        markdown_table(posture_rows[posture_cols].round(5)) if len(posture_rows) else "没有姿态相关标签。",
        "",
        "## 最容易混淆的动作对",
        "",
        "下表是基于通用运动特征的标准化质心距离；数值越小代表两类整体特征越接近，并不等于最终模型的错误率。",
        "",
        markdown_table(pairwise.head(12).round(4)),
        "",
        "## 初步验证",
        "",
        f"使用少量通用特征的最近质心模型，仅对明确的健身/非健身标签按整段采集做留一验证（避免重叠窗口泄漏），窗口级准确率为 {accuracy:.1%}、平衡准确率为 {balanced_accuracy:.1%}。类别不均衡时应优先看平衡准确率。这只是少量人员、单日、小样本探索结果，不是产品准确率。",
        "",
        markdown_table(per_recording.rename("window_accuracy").round(4).to_frame(), include_index=True) if len(per_recording) else "无可用验证结果。",
        "",
        f"按人员组合整组留出的跨人员窗口准确率为 {subject_accuracy:.1%}、平衡准确率为 {subject_balanced_accuracy:.1%}。当前只有 3 个人员组合，其中一组只有咀嚼且没有健身动作；这个结果仅用于检查人员依赖，不能视为正式泛化准确率。",
        "",
        f"同样按整段留一验证、且至少有两个会话的动作窗口准确率为 {action_accuracy:.1%}。只有一个会话而无法做留一验证的动作包括：{'、'.join(singleton_actions) if singleton_actions else '无'}；它们不计入该准确率。混淆矩阵按真实类别逐行归一化：",
        "",
        markdown_table(confusion, include_index=True) if len(confusion) else "无可用动作验证结果。",
        "",
        "## 推荐判别框架",
        "",
        "1. 2 秒窗口、50% 重叠，先用 `dynamic_acc_std / acc_mag_std / gyro_mag_mean / gyro_mag_std` 判断动态活动。",
        "2. 对动态活动，再用主频、谱熵、轴向角速度与加速度范围区分跑步机、深蹲、弓步蹲、卷腹与日常弯腰等动作。",
        "3. 动态不等于健身：自由行走、上下楼和坐起必须结合场景标签或持续模式；在业务定义确认前保持 ambiguous，不要硬塞入正负类。",
        "4. 输出需做时序滞回：例如连续 3 个窗口满足才切换状态，连续 5 个窗口不满足才退出，减少状态闪烁。",
        "5. 坐姿质量应以佩戴后的中立姿态校准为基线，使用相对俯仰/横滚及持续时间；必须新增明确的良好坐姿、低头、前伸头、侧倾、后仰标签。",
        "6. 先做佩戴有效性门控：检测到佩戴取下或不对称佩戴时，不输出健身或姿态结论，避免把设备移动误判成人体动作。",
        "",
        "## 采集建议",
        "",
        "- 固定并记录采样率，建议 50 Hz（动作细分可用 100 Hz）；时间戳必须单调且每样本有明确序号。",
        "- 每段保存 subject_id、device_id、session_id、raw_action、fitness_state、motion_state、posture_state、wear_state、context、起止时间、设备方向、采样率和备注，标签不要只依赖中文文件夹名。",
        "- 每人每动作至少 3–5 段、每段 1–3 分钟；纳入不同人、不同天、不同耳机松紧度。训练/验证必须按人划分。",
        "- 加入真实负样本：办公时转头、说话、喝水、刷手机、乘车、普通走路；这些比安静坐姿更容易造成误报。",
        "- 每次佩戴先静止 5–10 秒做重力和零偏校准；保存原始数据，清洗结果另存，禁止覆盖原始数据。",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cleaned_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/features"))
    parser.add_argument("--window-seconds", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in find_csv_files(args.cleaned_dir):
        if path.name in {"cleaning_log.csv", "anomalies.csv"}:
            continue
        rows.extend(windows_for_file(path, args.window_seconds, args.overlap))
    all_features = pd.DataFrame(rows)
    if all_features.empty:
        raise SystemExit("No valid windows were extracted")
    # In the 0720 logger layout CSV/TXT from one device are alternate exports
    # of the same stream. Keep both in the audit table but select one format per
    # physical device sample for statistics. Older data without device metadata
    # naturally falls back to its recording identifier.
    all_features["physical_sample"] = all_features["capture_group"].astype(str) + "__" + all_features["device_id"].astype(str)
    source_choice = (
        all_features[["physical_sample", "source_format"]]
        .drop_duplicates()
        .assign(priority=lambda frame: frame["source_format"].map({"csv": 0, "txt": 1}).fillna(2))
        .sort_values(["physical_sample", "priority"])
        .drop_duplicates("physical_sample")
        .set_index("physical_sample")["source_format"]
    )
    features = all_features[all_features.apply(lambda row: row["source_format"] == source_choice[row["physical_sample"]], axis=1)].copy()
    feature_cols = [c for c in features.select_dtypes(include=[np.number]).columns if c not in {"trial", "window_start_s", "window_end_s", "sample_rate_hz", "orientation_valid"}]
    group = features.groupby("activity")[feature_cols].mean().reset_index()
    effects = []
    for col in feature_cols:
        ex = features.loc[features["state"] == "exercise", col]
        non = features.loc[features["state"] == "non_exercise", col]
        effect = standardized_effect(ex, non)
        effects.append({"feature": col, "exercise_mean": ex.mean(), "non_exercise_mean": non.mean(), "effect_size": effect, "abs_effect": abs(effect)})
    effects_df = pd.DataFrame(effects).sort_values("abs_effect", ascending=False)
    model_features = [c for c in ["dynamic_acc_std", "gyro_mag_mean", "gyro_mag_std", "acc_mag_std", "dynamic_acc_dom_freq_hz", "gyro_mag_dom_freq_hz"] if c in features]
    model_data = features[features["state"].isin(["exercise", "non_exercise"])].dropna(subset=model_features)
    cv = nearest_centroid_loro(model_data, model_features, target="state")
    subject_cv = nearest_centroid_loro(model_data, model_features, target="state", group_col="subject_group")
    action_features = [c for c in [
        "dynamic_acc_std", "acc_mag_std", "acc_sma", "gyro_mag_mean", "gyro_mag_std",
        "acc_x_std", "acc_y_std", "acc_z_std", "gyro_x_std", "gyro_y_std", "gyro_z_std",
        "dynamic_acc_dom_freq_hz", "gyro_mag_dom_freq_hz", "acc_mag_spec_entropy",
    ] if c in features]
    action_data = features.dropna(subset=action_features)
    action_cv = nearest_centroid_loro(action_data, action_features, target="activity")
    pairwise = pairwise_action_distances(action_data, action_features)
    all_features.to_csv(args.output_dir / "window_features_all_sources.csv", index=False, encoding="utf-8-sig")
    features.to_csv(args.output_dir / "window_features.csv", index=False, encoding="utf-8-sig")
    group.to_csv(args.output_dir / "activity_feature_means.csv", index=False, encoding="utf-8-sig")
    effects_df.to_csv(args.output_dir / "state_effect_sizes.csv", index=False, encoding="utf-8-sig")
    cv.to_csv(args.output_dir / "loro_predictions.csv", index=False, encoding="utf-8-sig")
    subject_cv.to_csv(args.output_dir / "subject_group_predictions.csv", index=False, encoding="utf-8-sig")
    action_cv.to_csv(args.output_dir / "action_loro_predictions.csv", index=False, encoding="utf-8-sig")
    pairwise.to_csv(args.output_dir / "action_pairwise_distances.csv", index=False, encoding="utf-8-sig")
    write_analysis(features, group, effects_df, cv, subject_cv, action_cv, pairwise, args.output_dir / "analysis_report.md")
    print(f"windows={len(features)}, activities={features['activity'].nunique()}, recordings={features['recording'].nunique()}")


if __name__ == "__main__":
    main()
