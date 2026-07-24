#!/usr/bin/env python3
"""Audit Schema v2 labels against the dataset-level acceptance rules."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "imu_analysis"))

from label_schema import (  # noqa: E402
    SCHEMA_VERSION,
    SHORT_RECORDING_MAX_SECONDS,
    load_label_document,
    validate_v2_document,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    csv_files = sorted(args.training_dir.glob("*.csv"))
    labels = sorted(args.labels_dir.glob("*.labels.json"))
    csv_by_stem = {path.stem: path for path in csv_files}
    rows = []
    violations = []
    for label_path in labels:
        stem = label_path.name[: -len(".labels.json")]
        document = load_label_document(label_path)
        errors = validate_v2_document(document)
        duration = float(document.get("recording", {}).get("duration_seconds", 0))
        segments = document.get("segments", [])
        trainable_segments = [
            segment for segment in segments if segment.get("window_trainable")
        ]
        raw_action = str(document.get("raw_action") or stem.split("-", 2)[-1])
        row = {
            "label_file": label_path.name,
            "source_csv_exists": stem in csv_by_stem,
            "schema_version": document.get("schema_version"),
            "annotation_scope": document.get("annotation_scope"),
            "duration_seconds": duration,
            "window_trainable": document.get("window_trainable"),
            "activity_ids": [segment.get("activity_id") for segment in segments],
            "motion_states": [segment.get("motion_state") for segment in segments],
            "wear_states": [segment.get("wear_state") for segment in segments],
            "validation_errors": errors,
        }
        rows.append(row)
        if errors:
            violations.append({"label_file": label_path.name, "rule": "schema", "details": errors})
        if duration > SHORT_RECORDING_MAX_SECONDS and trainable_segments:
            violations.append({"label_file": label_path.name, "rule": "long_has_training_windows"})
        if any(
            segment.get("wear_state") != "valid" and segment.get("window_trainable")
            for segment in segments
        ):
            violations.append({"label_file": label_path.name, "rule": "wear_artifact_trainable"})
        if raw_action == "跑步机走路" and any(
            segment.get("activity_id") == "run" for segment in segments
        ):
            violations.append({"label_file": label_path.name, "rule": "treadmill_walk_is_run"})
        if raw_action.startswith("爬坡") and any(
            segment.get("motion_state") != "motion" for segment in trainable_segments
        ):
            violations.append({"label_file": label_path.name, "rule": "incline_not_motion"})
        if duration > SHORT_RECORDING_MAX_SECONDS and (
            "3组" in raw_action or "6组" in raw_action
        ) and segments:
            violations.append({"label_file": label_path.name, "rule": "long_group_record_segmented"})
        for segment in trainable_segments:
            if not segment.get("activity_id") or segment.get("motion_state") not in {
                "motion", "non_motion",
            }:
                violations.append({"label_file": label_path.name, "rule": "trainable_missing_dual_label"})

    duplicate_labels = sorted(
        name for name in {row["label_file"] for row in rows}
        if sum(item["label_file"] == name for item in rows) > 1
    )
    missing_labels = sorted(
        path.name for path in csv_files
        if f"{path.stem}.labels.json" not in {row["label_file"] for row in rows}
    )
    orphan_labels = sorted(
        row["label_file"] for row in rows if not row["source_csv_exists"]
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "csv_count": len(csv_files),
        "label_count": len(rows),
        "inventory_count": len(set(csv_by_stem) | {
            row["label_file"][: -len(".labels.json")] for row in rows
        }),
        "short_count": sum(row["duration_seconds"] <= SHORT_RECORDING_MAX_SECONDS for row in rows),
        "long_session_count": sum(row["duration_seconds"] > SHORT_RECORDING_MAX_SECONDS for row in rows),
        "window_trainable_count": sum(
            bool(row["window_trainable"]) and bool(row["source_csv_exists"])
            for row in rows
        ),
        "missing_labels": missing_labels,
        "orphan_labels": orphan_labels,
        "duplicate_labels": duplicate_labels,
        "violation_count": len(violations),
        "acceptance_passed": (
            not missing_labels and not duplicate_labels and not violations
        ),
    }
    report = {"summary": summary, "violations": violations, "files": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if summary["acceptance_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
