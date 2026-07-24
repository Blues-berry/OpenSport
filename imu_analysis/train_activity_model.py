"""Build 50 Hz windows and train the multiclass LightGBM demo model."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from activity_features import HOP_SECONDS, TARGET_RATE_HZ, WINDOW_SECONDS, iter_feature_windows, uniform_resample
from activity_taxonomy import ACTION_NAMES_ZH, TARGET_ACTIONS, capture_identity
from imu_common import ACC_COLS, GYRO_COLS, read_imu_csv


CLASSES = ("non_exercise", "unknown_motion", *TARGET_ACTIONS)


def recording_duration(frame: pd.DataFrame) -> float:
    values = frame.get("时间")
    if values is None or len(values) < 2:
        return max(0.0, (len(frame) - 1) / 100.0)
    parsed = pd.to_timedelta(values.astype(str).str.strip(), errors="coerce").dt.total_seconds()
    parsed = parsed.dropna()
    if len(parsed) < 2:
        return max(0.0, (len(frame) - 1) / 100.0)
    duration = float(parsed.iloc[-1] - parsed.iloc[0])
    if duration < 0:
        duration += 86400.0
    return duration


def read_timeline(path: Path | None) -> pd.DataFrame:
    columns = ["source_file", "start_s", "end_s", "action_id", "phase", "set_id"]
    if path is None or not path.exists():
        return pd.DataFrame(columns=columns)
    timeline = pd.read_csv(path, encoding="utf-8-sig")
    missing = set(columns[:-1]) - set(timeline.columns)
    if missing:
        raise ValueError(f"Timeline is missing columns: {sorted(missing)}")
    if "set_id" not in timeline:
        timeline["set_id"] = ""
    timeline["source_file"] = timeline["source_file"].astype(str).map(lambda value: Path(value).name)
    return timeline[columns]


def labelled_intervals(path: Path, duration_s: float, timeline: pd.DataFrame) -> list[dict]:
    selected = timeline[timeline["source_file"].eq(path.name)]
    if not selected.empty:
        intervals = selected.to_dict("records")
    else:
        identity = capture_identity(path)
        if not identity.usable_without_timeline or identity.action_id is None:
            return []
        intervals = [
            {
                "source_file": path.name,
                "start_s": 1.0,
                "end_s": max(1.0, duration_s - 1.0),
                "action_id": identity.action_id,
                "phase": identity.phase,
                "set_id": "",
            }
        ]
    normalized = []
    for row in intervals:
        phase = str(row["phase"]).strip()
        action = str(row["action_id"]).strip()
        if phase in {"calibration", "transition"}:
            continue
        if phase in {"inter_set", "non_exercise"}:
            action = "non_exercise"
        if action not in CLASSES:
            action = "unknown_motion" if phase == "active_set" else "non_exercise"
        start = max(0.0, float(row["start_s"]))
        end = min(duration_s, float(row["end_s"]))
        if end - start >= WINDOW_SECONDS:
            normalized.append({**row, "start_s": start, "end_s": end, "action_id": action, "phase": phase})
    return normalized


def build_feature_dataset(data_dir: Path, timeline_path: Path | None, max_windows_per_capture: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    timeline = read_timeline(timeline_path)
    rows: list[dict] = []
    manifest: list[dict] = []
    for path in sorted(data_dir.glob("*.csv")):
        identity = capture_identity(path)
        frame = read_imu_csv(path)
        duration = recording_duration(frame)
        intervals = labelled_intervals(path, duration, timeline)
        reason = "" if intervals else "mixed_or_unlabelled_without_timeline"
        manifest.append(
            {
                "source_file": path.name,
                "date": identity.date,
                "subject_id": identity.subject_id,
                "raw_action": identity.raw_action,
                "duration_s": round(duration, 3),
                "intervals": len(intervals),
                "included": bool(intervals),
                "excluded_reason": reason,
            }
        )
        if not intervals:
            continue
        matrix = frame.reindex(columns=ACC_COLS + GYRO_COLS).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(matrix).all():
            matrix = pd.DataFrame(matrix).interpolate(limit_direction="both").fillna(0.0).to_numpy()
        resampled = uniform_resample(matrix, duration, TARGET_RATE_HZ)
        capture_rows: list[dict] = []
        for interval_index, interval in enumerate(intervals):
            start_index = int(math.ceil(float(interval["start_s"]) * TARGET_RATE_HZ))
            end_index = int(math.floor(float(interval["end_s"]) * TARGET_RATE_HZ))
            segment = resampled[start_index:end_index]
            for local_start, local_end, features in iter_feature_windows(segment):
                capture_rows.append(
                    {
                        **features,
                        "action_id": interval["action_id"],
                        "phase": interval["phase"],
                        "set_id": interval.get("set_id", ""),
                        "subject_id": identity.subject_id,
                        "capture_id": f"{identity.date}:{identity.subject_id}:{path.stem}:{interval_index}",
                        "source_file": path.name,
                        "window_start_s": round(float(interval["start_s"]) + local_start, 3),
                        "window_end_s": round(float(interval["start_s"]) + local_end, 3),
                    }
                )
        if len(capture_rows) > max_windows_per_capture:
            indices = np.linspace(0, len(capture_rows) - 1, max_windows_per_capture).round().astype(int)
            capture_rows = [capture_rows[index] for index in indices]
        rows.extend(capture_rows)
    return pd.DataFrame(rows), pd.DataFrame(manifest)


def split_by_subject(data: pd.DataFrame, seed: int = 20260723) -> pd.DataFrame:
    result = data.copy()
    result["split"] = "train"
    test_subjects = select_holdout_subjects(result, test_size=0.15, seed=seed)
    result.loc[result["subject_id"].astype(str).isin(test_subjects), "split"] = "test"
    remaining = result[result["split"].eq("train")]
    validation_subjects = select_holdout_subjects(
        remaining, test_size=0.1765, seed=seed + 1
    )
    result.loc[result["subject_id"].astype(str).isin(validation_subjects), "split"] = "validation"
    return result


def select_holdout_subjects(
    data: pd.DataFrame,
    test_size: float,
    seed: int,
    candidates: int = 4096,
) -> set[str]:
    """Choose a subject holdout that maximizes feasible class coverage.

    A plain random group split can omit common actions from validation/test.
    Classes recorded for only one subject must stay in training; all classes
    with at least two subjects are preferred in the holdout.
    """
    frame = data[["subject_id", "action_id"]].copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    subjects = np.asarray(sorted(frame["subject_id"].unique()), dtype=object)
    if len(subjects) < 3:
        raise ValueError("Subject split requires at least three subjects")
    holdout_count = min(len(subjects) - 1, max(1, int(round(len(subjects) * test_size))))
    all_classes = set(frame["action_id"].astype(str))
    class_subjects = frame.groupby("action_id")["subject_id"].nunique()
    feasible_holdout = {label for label, count in class_subjects.items() if int(count) >= 2}
    global_share = frame["action_id"].value_counts(normalize=True)
    rng = np.random.default_rng(seed)
    best_subjects: set[str] | None = None
    best_score: tuple[int, float] | None = None
    for _ in range(candidates):
        shuffled = rng.permutation(subjects)
        holdout = set(str(value) for value in shuffled[:holdout_count])
        train_labels = set(frame.loc[~frame["subject_id"].isin(holdout), "action_id"].astype(str))
        if train_labels != all_classes:
            continue
        held = frame[frame["subject_id"].isin(holdout)]
        held_labels = set(held["action_id"].astype(str))
        coverage = len(held_labels & feasible_holdout)
        held_share = held["action_id"].value_counts(normalize=True)
        distribution_error = float(
            sum(abs(float(held_share.get(label, 0.0)) - float(global_share.get(label, 0.0)))
                for label in feasible_holdout)
        )
        score = (coverage, -distribution_error)
        if best_score is None or score > best_score:
            best_subjects, best_score = holdout, score
    if best_subjects is None:
        raise ValueError("Could not create a subject holdout while preserving every training class")
    return best_subjects


def temperature_scale(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(probabilities, 1e-9, 1.0)
    best_temperature, best_loss = 1.0, float("inf")
    for temperature in np.linspace(0.5, 3.0, 101):
        logits = np.log(clipped) / temperature
        scaled = np.exp(logits - logits.max(axis=1, keepdims=True))
        scaled /= scaled.sum(axis=1, keepdims=True)
        loss = -float(np.mean(np.log(scaled[np.arange(len(labels)), labels] + 1e-12)))
        if loss < best_loss:
            best_temperature, best_loss = float(temperature), loss
    return best_temperature


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
    scaled = np.exp(logits - logits.max(axis=1, keepdims=True))
    return scaled / scaled.sum(axis=1, keepdims=True)


def metric_bundle(actual: np.ndarray, probabilities: np.ndarray, classes: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    predicted = probabilities.argmax(axis=1)
    labels = list(range(len(classes)))
    present_labels = sorted(set(int(value) for value in actual))
    return {
        "samples": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)),
        "macro_f1_present_classes": float(
            f1_score(actual, predicted, labels=present_labels, average="macro", zero_division=0)
        ),
        "classification_report": classification_report(
            actual, predicted, labels=labels, target_names=classes, output_dict=True, zero_division=0
        ),
        "confusion_matrix_rows_actual": confusion_matrix(actual, predicted, labels=labels).tolist(),
    }


def motion_metric(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
    classes: list[str],
) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    eligible = frame["action_id"].eq("non_exercise") | frame["action_id"].isin(TARGET_ACTIONS)
    selected = frame.loc[eligible]
    selected_probability = probabilities[eligible.to_numpy()]
    target_indices = [classes.index(action) for action in TARGET_ACTIONS if action in classes]
    motion_probability = selected_probability[:, target_indices].sum(axis=1)
    actual = selected["action_id"].isin(TARGET_ACTIONS).astype(int).to_numpy()
    predicted = (motion_probability >= 0.5).astype(int)
    return {
        "samples": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
        "precision_exercise": float(precision_score(actual, predicted, zero_division=0)),
        "recall_exercise": float(recall_score(actual, predicted, zero_division=0)),
        "confusion_matrix_rows_actual_nonexercise_exercise": confusion_matrix(actual, predicted, labels=[0, 1]).tolist(),
    }


def train_model(dataset: pd.DataFrame, output_dir: Path, seed: int = 20260723) -> dict:
    try:
        import lightgbm as lgb
    except ImportError as error:
        raise SystemExit("Install training dependencies: pip install lightgbm scikit-learn") from error

    metadata = {
        "action_id", "phase", "set_id", "subject_id", "capture_id", "source_file",
        "window_start_s", "window_end_s", "split",
    }
    features = sorted(column for column in dataset.columns if column not in metadata)
    classes = [label for label in CLASSES if label in set(dataset["action_id"])]
    class_to_index = {label: index for index, label in enumerate(classes)}
    train = dataset[dataset["split"].eq("train")]
    validation = dataset[dataset["split"].eq("validation")]
    test = dataset[dataset["split"].eq("test")]
    if len(classes) < 3 or train.empty or validation.empty or test.empty:
        raise ValueError("Dataset needs at least three classes and non-empty train/validation/test subjects")
    missing_train = set(classes) - set(train["action_id"])
    if missing_train:
        raise ValueError(f"Training split is missing classes: {sorted(missing_train)}")

    capture_counts = train["capture_id"].value_counts()
    sample_weight = np.array(
        [
            1.0 / capture_counts[capture]
            for capture in train["capture_id"]
        ],
        dtype=float,
    )
    train_actions = train["action_id"].to_numpy()
    for action in classes:
        mask = train_actions == action
        sample_weight[mask] /= sample_weight[mask].sum()
    sample_weight /= sample_weight.mean()
    model = lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=350,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=6,
        min_child_samples=25,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )
    fit_kwargs = {
        "sample_weight": sample_weight,
        "callbacks": [lgb.early_stopping(30, verbose=False)],
    }
    validation_y = validation["action_id"].map(class_to_index)
    if "eval_X" in inspect.signature(model.fit).parameters:
        fit_kwargs.update({"eval_X": validation[features], "eval_y": validation_y})
    else:
        fit_kwargs["eval_set"] = [(validation[features], validation_y)]
    model.fit(train[features], train["action_id"].map(class_to_index), **fit_kwargs)
    validation_probability = model.predict_proba(validation[features])
    temperature = temperature_scale(
        validation_probability, validation["action_id"].map(class_to_index).to_numpy()
    )
    metrics = {}
    for name, part in (("validation", validation), ("test", test)):
        known = part["action_id"].isin(classes)
        actual = part.loc[known, "action_id"].map(class_to_index).to_numpy()
        probability = apply_temperature(model.predict_proba(part.loc[known, features]), temperature)
        metrics[name] = metric_bundle(actual, probability, classes)
        metrics[name]["motion"] = motion_metric(part.loc[known], probability, classes)
        metrics[name]["subjects"] = sorted(part.loc[known, "subject_id"].unique().tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    target_test_support = test["action_id"].value_counts()
    subject_coverage = (
        dataset.groupby("action_id")["subject_id"].nunique().astype(int).to_dict()
    )
    single_subject_targets = [
        action for action in TARGET_ACTIONS if int(subject_coverage.get(action, 0)) < 2
    ]
    demo_ready = (
        metrics["validation"]["motion"]["macro_f1"] >= 0.90
        and metrics["test"]["motion"]["macro_f1"] >= 0.90
        and metrics["test"]["macro_f1"] >= 0.80
        and all(int(target_test_support.get(action, 0)) > 0 for action in TARGET_ACTIONS)
    )
    payload = {
        "format_version": 1,
        "model_family": "lightgbm_multiclass",
        "model": model,
        "features": features,
        "classes": classes,
        "class_names_zh": {key: ACTION_NAMES_ZH[key] for key in classes},
        "target_actions": [action for action in TARGET_ACTIONS if action in classes],
        "sample_rate_hz": TARGET_RATE_HZ,
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "temperature": temperature,
        "action_threshold": 0.65,
        "unknown_threshold": 0.50,
        "demo_ready": demo_ready,
    }
    with (output_dir / "activity_model.pkl").open("wb") as handle:
        pickle.dump(payload, handle)
    model.booster_.save_model(str(output_dir / "activity_model.txt"))
    importance = pd.DataFrame(
        {"feature": features, "gain": model.booster_.feature_importance(importance_type="gain")}
    ).sort_values("gain", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False, encoding="utf-8-sig")
    report = {
        "classes": classes,
        "class_names_zh": payload["class_names_zh"],
        "temperature": temperature,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "demo_ready": demo_ready,
        "acceptance_thresholds": {
            "motion_macro_f1": 0.90,
            "action_macro_f1": 0.80,
            "all_target_actions_present_in_test": True,
        },
        "subject_coverage_by_class": subject_coverage,
        "targets_without_two_subjects": single_subject_targets,
        "metrics": metrics,
        "split_counts": dataset.groupby(["split", "action_id"]).size().unstack(fill_value=0).to_dict(orient="index"),
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    test_counts = report["split_counts"]["test"]
    missing_test = [action for action in TARGET_ACTIONS if int(test_counts.get(action, 0)) == 0]
    model_card = f"""# IMU Demo 动作模型卡

- 模型：LightGBM 多分类，{len(features)} 个窗口特征
- 输入：六轴 IMU，{TARGET_RATE_HZ:g} Hz，{WINDOW_SECONDS:g} 秒窗口，{HOP_SECONDS:g} 秒步长
- 类别：{", ".join(classes)}
- 验收状态：{"通过" if demo_ready else "未通过（仅用于链路联调）"}

## 固定跨用户评估

| 指标 | 验证集 | 测试集 | 目标 |
| --- | ---: | ---: | ---: |
| 运动 Macro-F1 | {metrics["validation"]["motion"]["macro_f1"]:.3f} | {metrics["test"]["motion"]["macro_f1"]:.3f} | ≥ 0.900 |
| 动作 Macro-F1 | {metrics["validation"]["macro_f1"]:.3f} | {metrics["test"]["macro_f1"]:.3f} | ≥ 0.800 |
| 测试集已有类别 Macro-F1 | {metrics["validation"]["macro_f1_present_classes"]:.3f} | {metrics["test"]["macro_f1_present_classes"]:.3f} | 仅诊断 |

测试集未覆盖目标动作：{", ".join(missing_test) if missing_test else "无"}。
少于两名受试者、无法同时进入训练和测试的目标动作：{", ".join(single_subject_targets) if single_subject_targets else "无"}。

## 已知限制

- 混合动作文件没有逐时刻标签时不参与训练。
- 组间状态由时序状态机判断，不是单窗口模型类别。
- 普通走路不计入正式锻炼。
- 平板支撑以及头部运动很弱的器械动作容易与静止休息混淆。
- 未通过验收门槛的模型不得自动提升为冠军版本。
"""
    (output_dir / "MODEL_CARD.md").write_text(model_card, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--timeline", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("imu_output/demo_activity"))
    parser.add_argument("--features-csv", type=Path)
    parser.add_argument("--max-windows-per-capture", type=int, default=300)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.features_csv and args.features_csv.exists():
        dataset = pd.read_csv(args.features_csv, encoding="utf-8-sig")
        manifest = pd.DataFrame()
    else:
        dataset, manifest = build_feature_dataset(
            args.data_dir, args.timeline, max_windows_per_capture=args.max_windows_per_capture
        )
        dataset.to_csv(args.output_dir / "window_features.csv", index=False, encoding="utf-8-sig")
        manifest.to_csv(args.output_dir / "data_manifest.csv", index=False, encoding="utf-8-sig")
    dataset = split_by_subject(dataset)
    dataset.to_csv(args.output_dir / "window_features_with_split.csv", index=False, encoding="utf-8-sig")
    report = train_model(dataset, args.output_dir / "model")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
