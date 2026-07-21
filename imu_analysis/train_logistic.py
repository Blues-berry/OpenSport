"""Train and honestly validate an L2-regularized exercise-state model."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = [
    "dynamic_acc_std",
    "dynamic_acc_rms",
    "dynamic_acc_dom_freq_hz",
    "dynamic_acc_spec_entropy",
    "acc_mag_std",
    "acc_mag_spec_entropy",
    "acc_sma",
    "gyro_mag_mean",
    "gyro_mag_std",
    "gyro_mag_dom_freq_hz",
    "gyro_mag_spec_entropy",
    "acc_x_std",
    "acc_y_std",
    "acc_z_std",
    "gyro_x_std",
    "gyro_y_std",
    "gyro_z_std",
]


def sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-value))


def fit_model(x: pd.DataFrame, y: np.ndarray, weights: np.ndarray, c_value: float) -> dict:
    """Fit standardized weighted logistic regression with an L2 penalty."""
    values = x.to_numpy(dtype=float)
    median = np.nanmedian(values, axis=0)
    values = np.where(np.isfinite(values), values, median)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (values - mean) / scale
    design = np.column_stack([np.ones(len(z)), z])
    beta = np.zeros(design.shape[1], dtype=float)
    penalty = np.r_[0.0, np.full(z.shape[1], 1.0 / c_value)]
    weight_sum = weights.sum()
    for _ in range(200):
        probability = sigmoid(design @ beta)
        gradient = design.T @ (weights * (probability - y)) / weight_sum + penalty * beta
        curvature = weights * probability * (1.0 - probability)
        hessian = (design.T @ (design * curvature[:, None])) / weight_sum + np.diag(penalty)
        hessian.flat[:: hessian.shape[0] + 1] += 1e-8
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return {"median": median, "mean": mean, "scale": scale, "intercept": float(beta[0]), "coefficients": beta[1:], "C": c_value}


def predict_probability(model: dict, x: pd.DataFrame) -> np.ndarray:
    values = x.to_numpy(dtype=float)
    values = np.where(np.isfinite(values), values, model["median"])
    z = (values - model["mean"]) / model["scale"]
    return sigmoid(model["intercept"] + z @ model["coefficients"])


def stratified_group_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int):
    group_frame = pd.DataFrame({"group": groups, "y": y}).drop_duplicates("group")
    fold_groups: list[list[str]] = [[] for _ in range(n_splits)]
    rng = np.random.default_rng(seed)
    for _, part in group_frame.groupby("y"):
        labels = part["group"].astype(str).to_numpy().copy()
        rng.shuffle(labels)
        for index, label in enumerate(labels):
            fold_groups[index % n_splits].append(label)
    for held_groups in fold_groups:
        test = np.isin(groups, held_groups)
        yield np.flatnonzero(~test), np.flatnonzero(test)


def balanced_accuracy(y: np.ndarray, predicted: np.ndarray) -> float:
    recalls = [np.mean(predicted[y == label] == label) for label in (0, 1) if np.any(y == label)]
    return float(np.mean(recalls))


def roc_auc(y: np.ndarray, probability: np.ndarray) -> float:
    positives = y == 1
    n_pos, n_neg = int(positives.sum()), int((~positives).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = pd.Series(probability).rank(method="average").to_numpy()
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def balanced_group_weights(y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Give each session equal total weight, then balance the two classes."""
    group_counts = pd.Series(groups).value_counts()
    weights = np.array([1.0 / group_counts[g] for g in groups], dtype=float)
    for label in np.unique(y):
        mask = y == label
        weights[mask] /= weights[mask].sum()
    return weights / weights.mean()


def group_predictions(y: np.ndarray, probability: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": y, "probability": probability, "capture_group": groups})
    grouped = frame.groupby("capture_group", as_index=False).agg(actual=("actual", "first"), probability=("probability", "mean"))
    grouped["predicted"] = (grouped["probability"] >= 0.5).astype(int)
    return grouped


def choose_c(x: pd.DataFrame, y: np.ndarray, groups: np.ndarray, grid: list[float]) -> tuple[float, pd.DataFrame]:
    group_labels = pd.DataFrame({"group": groups, "y": y}).drop_duplicates()
    min_groups = int(group_labels.groupby("y").size().min())
    n_splits = min(4, min_groups)
    if n_splits < 2:
        return 1.0, pd.DataFrame({"C": grid, "mean_group_balanced_accuracy": np.nan})
    rows = []
    for c_value in grid:
        scores = []
        for train_idx, valid_idx in stratified_group_splits(y, groups, n_splits, seed=314):
            weights = balanced_group_weights(y[train_idx], groups[train_idx])
            model = fit_model(x.iloc[train_idx], y[train_idx], weights, c_value)
            probability = predict_probability(model, x.iloc[valid_idx])
            grouped = group_predictions(y[valid_idx], probability, groups[valid_idx])
            scores.append(balanced_accuracy(grouped["actual"].to_numpy(), grouped["predicted"].to_numpy()))
        rows.append({"C": c_value, "mean_group_balanced_accuracy": float(np.mean(scores))})
    results = pd.DataFrame(rows)
    # Prefer stronger regularization when scores tie.
    best = results.sort_values(["mean_group_balanced_accuracy", "C"], ascending=[False, True]).iloc[0]
    return float(best["C"]), results


def metric_bundle(y: np.ndarray, predicted: np.ndarray, probability: np.ndarray) -> dict[str, float | list[list[int]]]:
    true_positive = int(((y == 1) & (predicted == 1)).sum())
    false_positive = int(((y == 0) & (predicted == 1)).sum())
    true_negative = int(((y == 0) & (predicted == 0)).sum())
    false_negative = int(((y == 1) & (predicted == 0)).sum())
    return {
        "accuracy": float(np.mean(y == predicted)),
        "balanced_accuracy": balanced_accuracy(y, predicted),
        "roc_auc": roc_auc(y, probability),
        "precision_exercise": true_positive / max(1, true_positive + false_positive),
        "recall_exercise": true_positive / max(1, true_positive + false_negative),
        "specificity_non_exercise": true_negative / max(1, true_negative + false_positive),
        "confusion_matrix_rows_actual_0_1": [[true_negative, false_positive], [false_negative, true_positive]],
    }


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "无"
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("features_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("imu_output/model"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.features_csv)
    feature_cols = [column for column in FEATURES if column in data.columns]
    required = {"state", "capture_group", "activity", "source_format"}
    if missing := required - set(data.columns):
        raise SystemExit(f"Missing required columns: {sorted(missing)}")
    if len(feature_cols) < 8:
        raise SystemExit("Too few model features are available")
    # Do not silently turn protocol-ambiguous daily activities or wear
    # artifacts into negative fitness labels.
    data = data[data["state"].isin(["exercise", "non_exercise"])].copy()
    data = data.reset_index(drop=True)
    x = data[feature_cols]
    y = (data["state"] == "exercise").astype(int).to_numpy()
    groups = data["capture_group"].astype(str).to_numpy()
    group_labels = data[["capture_group", "state"]].drop_duplicates()
    if group_labels.groupby("capture_group")["state"].nunique().max() != 1:
        raise SystemExit("A capture_group contains conflicting state labels")

    c_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
    oof_probability = np.full(len(data), np.nan)
    fold_rows = []
    tuning_rows = []
    for fold, (train_idx, test_idx) in enumerate(stratified_group_splits(y, groups, 5, seed=42), start=1):
        best_c, tuning = choose_c(x.iloc[train_idx].reset_index(drop=True), y[train_idx], groups[train_idx], c_grid)
        tuning.insert(0, "outer_fold", fold)
        tuning_rows.append(tuning)
        weights = balanced_group_weights(y[train_idx], groups[train_idx])
        model = fit_model(x.iloc[train_idx], y[train_idx], weights, best_c)
        probability = predict_probability(model, x.iloc[test_idx])
        oof_probability[test_idx] = probability
        grouped = group_predictions(y[test_idx], probability, groups[test_idx])
        fold_rows.append(
            {
                "fold": fold,
                "selected_C": best_c,
                "train_sessions": int(pd.Series(groups[train_idx]).nunique()),
                "test_sessions": int(pd.Series(groups[test_idx]).nunique()),
                "test_windows": int(len(test_idx)),
                "group_balanced_accuracy": balanced_accuracy(grouped["actual"].to_numpy(), grouped["predicted"].to_numpy()),
            }
        )

    oof_predicted = (oof_probability >= 0.5).astype(int)
    window_metrics = metric_bundle(y, oof_predicted, oof_probability)
    grouped_oof = group_predictions(y, oof_probability, groups)
    group_metrics = metric_bundle(
        grouped_oof["actual"].to_numpy(),
        grouped_oof["predicted"].to_numpy(),
        grouped_oof["probability"].to_numpy(),
    )
    _, final_tuning = choose_c(x, y, groups, c_grid)
    # The median outer-fold choice is more stable than selecting the weakest
    # penalty from a very small difference on all sessions.
    final_c = float(np.median([row["selected_C"] for row in fold_rows]))
    final_weights = balanced_group_weights(y, groups)
    final_model = fit_model(x, y, final_weights, final_c)

    predictions = data[["activity", "state", "recording", "capture_group", "source_format", "window_start_s", "window_end_s"]].copy()
    predictions["actual"] = y
    predictions["probability_exercise"] = oof_probability
    predictions["predicted"] = oof_predicted
    predictions["correct"] = predictions["actual"] == predictions["predicted"]
    by_activity = predictions.groupby("activity", as_index=False).agg(
        windows=("actual", "size"),
        sessions=("capture_group", "nunique"),
        actual=("actual", "first"),
        accuracy=("correct", "mean"),
        mean_exercise_probability=("probability_exercise", "mean"),
    )
    by_source = predictions.groupby("source_format", as_index=False).agg(
        windows=("actual", "size"), accuracy=("correct", "mean")
    )
    coefficient = pd.DataFrame(
        {
            "feature": feature_cols,
            "standardized_coefficient": final_model["coefficients"],
        }
    )
    coefficient["absolute_coefficient"] = coefficient["standardized_coefficient"].abs()
    coefficient = coefficient.sort_values("absolute_coefficient", ascending=False)

    predictions.to_csv(args.output_dir / "oof_window_predictions.csv", index=False, encoding="utf-8-sig")
    grouped_oof.to_csv(args.output_dir / "oof_session_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "outer_fold_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(tuning_rows, ignore_index=True).to_csv(args.output_dir / "nested_tuning.csv", index=False, encoding="utf-8-sig")
    final_tuning.to_csv(args.output_dir / "final_tuning.csv", index=False, encoding="utf-8-sig")
    coefficient.to_csv(args.output_dir / "coefficients.csv", index=False, encoding="utf-8-sig")
    by_activity.to_csv(args.output_dir / "metrics_by_activity.csv", index=False, encoding="utf-8-sig")
    with (args.output_dir / "l2_logistic_model.pkl").open("wb") as handle:
        pickle.dump({"model": final_model, "features": feature_cols, "threshold": 0.5, "positive_label": "exercise"}, handle)
    metrics = {
        "model": "L2-regularized logistic regression",
        "label_rule": "Only explicit exercise/non_exercise protocol labels; ambiguous and wear_artifact excluded",
        "window_seconds": 2.0,
        "threshold": 0.5,
        "samples_files": int(data["recording"].nunique()),
        "capture_sessions": int(data["capture_group"].nunique()),
        "windows": int(len(data)),
        "features": feature_cols,
        "selected_final_C": final_c,
        "window_level_nested_group_cv": window_metrics,
        "session_level_nested_group_cv": group_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    report = [
        "# L2 正则化逻辑回归结果",
        "",
        "## 任务定义",
        "",
        "当前标签中没有独立的 fitness_state 字段，因此采用协议动作的保守临时规则；只使用明确的 exercise/non_exercise，ambiguous 与 wear_artifact 不进入训练。该规则仍需采集人员确认。",
        "",
        "CSV 与 TXT 均作为不同样本保留。动作标签和起始时刻接近的同期样本共享 capture_group，并在五折验证中整组进入训练或测试，防止同期动作信息泄漏。正则强度 C 在每个外层训练折内再次分组验证选择。",
        "",
        "## 验证结果",
        "",
        f"窗口级：准确率 {window_metrics['accuracy']:.1%}，平衡准确率 {window_metrics['balanced_accuracy']:.1%}，ROC AUC {window_metrics['roc_auc']:.3f}，运动召回率 {window_metrics['recall_exercise']:.1%}，非运动召回率 {window_metrics['specificity_non_exercise']:.1%}。",
        "",
        f"采集会话级（窗口概率取均值）：准确率 {group_metrics['accuracy']:.1%}，平衡准确率 {group_metrics['balanced_accuracy']:.1%}，ROC AUC {group_metrics['roc_auc']:.3f}。最终全量模型采用外层各折选择值的中位数 C={final_c:g}（偏向更稳定的正则化），阈值为 0.5。",
        "",
        "这些结果来自单人、单日、动作预先分段的数据，只表示内部验证表现，不能保证对新用户、新佩戴位置或自由生活场景达到同样准确率。",
        "",
        "## 各动作外层验证表现",
        "",
        markdown_table(by_activity.round(4)),
        "",
        "## TXT/CSV 分格式表现",
        "",
        markdown_table(by_source.round(4)),
        "",
        "## 影响最大的标准化特征",
        "",
        markdown_table(coefficient.head(12).round(5)),
        "",
        "正系数推动模型判断为运动，负系数推动判断为非运动；相关特征之间会分摊系数，不能把单个系数解释成因果关系。部署时应先按相同清洗与 2 秒、50% 重叠窗口提取同名特征，再调用模型并对连续窗口做滞回。",
    ]
    (args.output_dir / "model_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
