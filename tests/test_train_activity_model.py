from __future__ import annotations

import json

import pandas as pd

from train_activity_model import (
    collapse_rare_classes,
    label_trainability,
    read_timeline,
    recording_duration,
    split_by_subject,
)


def test_subject_split_covers_every_feasible_class_and_preserves_training_classes():
    rows = []
    coverage = {
        "sitting": ["s1", "s2", "s3", "s4", "s5", "s6"],
        "run": ["s1", "s2", "s3", "s4"],
        "squat": ["s3", "s4", "s5", "s6"],
        "elliptical": ["s1"],
    }
    for action, subjects in coverage.items():
        for subject in subjects:
            rows.append({"subject_id": subject, "action_id": action})

    split = split_by_subject(pd.DataFrame(rows), seed=7)

    assert set(split.loc[split["split"].eq("train"), "action_id"]) == set(coverage)
    test_actions = set(split.loc[split["split"].eq("test"), "action_id"])
    assert {"sitting", "run", "squat"} <= test_actions
    assert "elliptical" not in test_actions


def test_label_trainability_reads_optional_audit_flag(tmp_path):
    csv_path = tmp_path / "0722-测试用户-站姿.csv"
    csv_path.touch()
    assert label_trainability(csv_path) is None

    labels = tmp_path / "labels"
    labels.mkdir()
    label_path = labels / "0722-测试用户-站姿.labels.json"
    label_path.write_text(
        json.dumps(
            {
                "schema_version": "2.0",
                "annotation_scope": "full_recording",
                "window_trainable": False,
                "recording": {"duration_seconds": 30},
                "segments": [
                    {
                        "activity_id": "removed_wear",
                        "motion_state": None,
                        "wear_state": "removed",
                        "phase": "artifact",
                        "window_trainable": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert label_trainability(csv_path) is False


def test_rare_exact_classes_are_preserved_but_grouped_for_training():
    source = pd.DataFrame(
        [
            {"subject_id": "s1", "exact_activity_id": "bench_press", "action_id": "bench_press", "motion_state": "motion"},
            {"subject_id": "s2", "exact_activity_id": "bench_press", "action_id": "bench_press", "motion_state": "motion"},
            {"subject_id": "s1", "exact_activity_id": "sitting", "action_id": "sitting", "motion_state": "non_motion"},
            {"subject_id": "s2", "exact_activity_id": "sitting", "action_id": "sitting", "motion_state": "non_motion"},
            {"subject_id": "s3", "exact_activity_id": "sitting", "action_id": "sitting", "motion_state": "non_motion"},
        ]
    )
    grouped, support = collapse_rare_classes(source)
    assert support["bench_press"] == 2
    assert set(grouped.loc[grouped["exact_activity_id"].eq("bench_press"), "action_id"]) == {"other_motion"}
    assert set(grouped.loc[grouped["exact_activity_id"].eq("sitting"), "action_id"]) == {"sitting"}


def test_recording_duration_handles_full_datetime_without_treating_date_as_delta():
    frame = pd.DataFrame(
        {"时间": ["2026-7-17 15:50:21.856", "2026-7-17 15:52:24.986"]}
    )
    assert abs(recording_duration(frame) - 123.13) < 1e-6


def test_schema_v2_timeline_template_fields_are_read_without_inference(tmp_path):
    timeline_path = tmp_path / "timeline.csv"
    timeline_path.write_text(
        "source_file,start_s,end_s,motion_state,activity_id,wear_state,"
        "phase,set_id,window_trainable,label_source,confidence,notes\n"
        "0725-S001-T0007.csv,3,45,motion,squat,valid,active,1,true,"
        "operator_event,high,现场事件\n",
        encoding="utf-8",
    )

    timeline = read_timeline(timeline_path)

    assert timeline.loc[0, "motion_state"] == "motion"
    assert timeline.loc[0, "activity_id"] == "squat"
    assert timeline.loc[0, "wear_state"] == "valid"
    assert bool(timeline.loc[0, "window_trainable"]) is True
