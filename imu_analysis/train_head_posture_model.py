"""Train a calibrated seated head-posture model from labelled IMU captures."""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

from head_posture_features import (
    POSTURE_HOP_SECONDS,
    POSTURE_RATE_HZ,
    POSTURE_WINDOW_SECONDS,
    extract_posture_features,
    posture_baseline,
)
from imu_common import ACC_COLS, ANGLE_COLS, GYRO_COLS, QUAT_COLS, elapsed_seconds, read_imu_file


POSTURE_CLASSES = ("normal", "head_down", "head_up", "head_tilt", "head_turn")
CLASS_NAMES_ZH = {
    "normal": "正常坐姿",
    "head_down": "低头",
    "head_up": "抬头",
    "head_tilt": "歪头",
    "head_turn": "偏头看向一侧",
}


def capture_label(path: Path) -> str | None:
    text = path.stem
    if "挂脖子" in text or "站起" in text or "有头动" in text:
        return None
    if "左右偏头" in text or "偏头" in text:
        return "head_tilt"
    if "低头" in text:
        return "head_down"
    if "抬头" in text:
        return "head_up"
    if "左看" in text:
        return "head_turn"
    if "右看" in text:
        return "head_turn"
    if re.search(r"-坐姿(?:\d+)?$", text):
        return "normal"
    return None


def capture_identity(path: Path) -> tuple[str, str]:
    parts = path.stem.split("-", 2)
    if len(parts) < 3:
        return "unknown-date", path.stem
    return parts[0], parts[1]


def read_posture_capture(path: Path) -> tuple[np.ndarray, float]:
    frame = read_imu_file(path)
    columns = ACC_COLS + GYRO_COLS + ANGLE_COLS + QUAT_COLS
    matrix = frame.reindex(columns=columns).apply(pd.to_numeric, errors="coerce")
    matrix = matrix.interpolate(limit_direction="both").dropna().to_numpy(dtype=float)
    seconds, _ = elapsed_seconds(frame)
    duration = float(seconds[-1] - seconds[0]) if len(seconds) > 1 else 0.0
    if len(matrix) < 2 or duration <= 0:
        raise ValueError(f"No usable posture samples: {path}")
    output_count = max(2, int(round(duration * POSTURE_RATE_HZ)) + 1)
    source_time = np.linspace(0.0, duration, len(matrix))
    target_time = np.arange(output_count, dtype=float) / POSTURE_RATE_HZ
    target_time = target_time[target_time <= duration + 1e-9]
    resampled = np.column_stack(
        [np.interp(target_time, source_time, matrix[:, index]) for index in range(matrix.shape[1])]
    )
    return resampled, duration


def build_dataset(data_dir: Path, max_windows_per_capture: int = 160) -> tuple[pd.DataFrame, pd.DataFrame]:
    captures: list[dict] = []
    for path in sorted((*data_dir.glob("*.csv"), *data_dir.glob("*.txt"))):
        label = capture_label(path)
        if label is None:
            continue
        date, subject = capture_identity(path)
        try:
            samples, duration = read_posture_capture(path)
        except (OSError, ValueError):
            continue
        captures.append(
            {"path": path, "label": label, "date": date, "subject": subject, "samples": samples, "duration": duration}
        )

    neutral: dict[tuple[str, str], dict] = {}
    for key in {(row["date"], row["subject"]) for row in captures}:
        normal = [row["samples"] for row in captures if (row["date"], row["subject"]) == key and row["label"] == "normal"]
        if normal:
            selected = np.concatenate([values[int(0.15 * len(values)):int(0.85 * len(values))] for values in normal])
            neutral[key] = posture_baseline(selected)

    rows: list[dict] = []
    manifest: list[dict] = []
    window = int(round(POSTURE_WINDOW_SECONDS * POSTURE_RATE_HZ))
    hop = int(round(POSTURE_HOP_SECONDS * POSTURE_RATE_HZ))
    for capture in captures:
        key = (capture["date"], capture["subject"])
        baseline = neutral.get(key)
        included = baseline is not None
        manifest.append(
            {
                "source_file": capture["path"].name,
                "date": capture["date"],
                "subject_id": capture["subject"],
                "posture": capture["label"],
                "duration_s": round(capture["duration"], 3),
                "included": included,
                "excluded_reason": "" if included else "missing_normal_seated_baseline",
            }
        )
        if not included:
            continue
        starts = list(range(0, max(0, len(capture["samples"]) - window + 1), hop))
        if len(starts) > max_windows_per_capture:
            indices = np.linspace(0, len(starts) - 1, max_windows_per_capture).round().astype(int)
            starts = [starts[index] for index in indices]
        for start in starts:
            features = extract_posture_features(capture["samples"][start:start + window], baseline)
            # Action captures include the movement into/out of a pose. The model
            # is intentionally trained on sustained, stable poses because the
            # runtime policy handles duration separately and must not alert on
            # a brief glance or transition.
            stable = features["gyro_mean_dps"] <= 12.0 and features["dynamic_acc_std_g"] <= 0.08
            deviation = features.get("rotation_degrees_mean", features["gravity_change_degrees"])
            if not stable:
                continue
            if capture["label"] == "normal" and deviation > 12.0:
                continue
            if capture["label"] != "normal" and deviation < 15.0:
                continue
            rows.append(
                {
                    **features,
                    "posture": capture["label"],
                    "subject_id": capture["subject"],
                    "capture_id": capture["path"].stem,
                    "source_file": capture["path"].name,
                    "window_start_s": round(start / POSTURE_RATE_HZ, 3),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(manifest)


def _subject_split(data: pd.DataFrame, seed: int) -> pd.DataFrame:
    from sklearn.model_selection import GroupShuffleSplit

    result = data.copy()
    groups = result["subject_id"].astype(str)
    splitter = GroupShuffleSplit(n_splits=200, test_size=0.25, random_state=seed)
    required = set(result["posture"])
    for train_index, test_index in splitter.split(result, result["posture"], groups):
        if set(result.iloc[train_index]["posture"]) == required and set(result.iloc[test_index]["posture"]) == required:
            result["split"] = "train"
            result.iloc[test_index, result.columns.get_loc("split")] = "test"
            return result
    raise ValueError("Could not create a subject-independent split containing every posture class")


def train_model(dataset: pd.DataFrame, output_dir: Path, seed: int = 20260723) -> dict:
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    metadata = {"posture", "subject_id", "capture_id", "source_file", "window_start_s", "split"}
    features = [column for column in dataset.columns if column not in metadata]
    classes = [label for label in POSTURE_CLASSES if label in set(dataset["posture"])]
    class_index = {label: index for index, label in enumerate(classes)}
    train = dataset[dataset["split"].eq("train")]
    test = dataset[dataset["split"].eq("test")]
    capture_counts = train["capture_id"].value_counts()
    class_counts = train["posture"].value_counts()
    weights = np.asarray(
        [1.0 / capture_counts[capture] / class_counts[label] for capture, label in zip(train["capture_id"], train["posture"])],
        dtype=float,
    )
    weights /= weights.mean()
    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=300,
        learning_rate=0.04,
        num_leaves=25,
        max_depth=6,
        min_child_samples=30,
        reg_lambda=1.5,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(train[features], train["posture"].map(class_index), sample_weight=weights)
    actual = test["posture"].map(class_index).to_numpy()
    probability = model.predict_proba(test[features])
    predicted = probability.argmax(axis=1)
    labels = list(range(len(classes)))
    metrics = {
        "samples": int(len(test)),
        "subjects": sorted(test["subject_id"].unique().tolist()),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "classification_report": classification_report(
            actual, predicted, labels=labels, target_names=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix_rows_actual": confusion_matrix(actual, predicted, labels=labels).tolist(),
    }
    per_class_recall = [metrics["classification_report"][label]["recall"] for label in classes]
    demo_ready = metrics["macro_f1"] >= 0.80 and min(per_class_recall) >= 0.70
    payload = {
        "format_version": 1,
        "model_family": "calibrated_lightgbm_head_posture",
        "model": model,
        "features": features,
        "classes": classes,
        "class_names_zh": {key: CLASS_NAMES_ZH[key] for key in classes},
        "sample_rate_hz": POSTURE_RATE_HZ,
        "window_seconds": POSTURE_WINDOW_SECONDS,
        "hop_seconds": POSTURE_HOP_SECONDS,
        "calibration_seconds": 10.0,
        "probability_threshold": 0.60,
        "abnormal_on_seconds": 3.0,
        "normal_off_seconds": 2.0,
        "demo_ready": demo_ready,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "head_posture_model.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    model.booster_.save_model(str(output_dir / "head_posture_model.txt"))
    importance = pd.DataFrame(
        {"feature": features, "gain": model.booster_.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    report = {
        "classes": classes,
        "class_names_zh": payload["class_names_zh"],
        "metrics": metrics,
        "demo_ready": demo_ready,
        "acceptance_thresholds": {"macro_f1": 0.80, "minimum_class_recall": 0.70},
        "split_counts": dataset.groupby(["split", "posture"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "runtime_policy": {
            "calibration_seconds": 10.0,
            "abnormal_on_seconds": 3.0,
            "normal_off_seconds": 2.0,
            "note": "Seated context must be asserted during calibration; a head IMU alone cannot separate still sitting from still standing.",
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    model_card = f"""# 坐姿头部姿态模型

- 状态：{"通过离线验收" if demo_ready else "未通过离线验收，仅用于链路联调"}
- 类别：{", ".join(CLASS_NAMES_ZH[label] for label in classes)}
- 输入：佩戴后 10 秒正常坐姿校准；50 Hz IMU、姿态角和四元数；3 秒窗口、0.5 秒步长
- 整人留出测试 Macro-F1：{metrics["macro_f1"]:.3f}
- 整人留出测试 Accuracy：{metrics["accuracy"]:.3f}
- 异常触发：同一异常稳定持续 3 秒；恢复：正常姿态稳定持续 2 秒

## 限制

- 当前“歪头”数据在不同受试者间动作定义不一致，尚不能泛化，模型不得用于正式提醒。
- 头戴 IMU 无法可靠区分静止坐姿和静止站姿，必须由坐姿校准流程或外部场景信号确认监测上下文。
- 实时输入必须包含四元数；仅有六轴数据时需先增加姿态融合，不应直接补零。
"""
    (output_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("imu_output/head_posture"))
    parser.add_argument("--features-csv", type=Path)
    parser.add_argument("--max-windows-per-capture", type=int, default=160)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.features_csv and args.features_csv.exists():
        dataset = pd.read_csv(args.features_csv, encoding="utf-8-sig")
        manifest = pd.DataFrame()
    else:
        dataset, manifest = build_dataset(args.data_dir, args.max_windows_per_capture)
        dataset.to_csv(args.output_dir / "window_features.csv", index=False, encoding="utf-8-sig")
        manifest.to_csv(args.output_dir / "data_manifest.csv", index=False, encoding="utf-8-sig")
    dataset = _subject_split(dataset, 20260723)
    dataset.to_csv(args.output_dir / "window_features_with_split.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(train_model(dataset, args.output_dir / "model"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
