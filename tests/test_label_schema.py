from __future__ import annotations

from label_schema import (
    CONTINUOUS_SESSION_TARGETS,
    SCHEMA_VERSION,
    reviewed_short_label,
    validate_v2_document,
)


def test_wear_artifact_has_no_binary_label_and_is_not_trainable():
    label = reviewed_short_label("佩戴取下")
    assert label is not None
    assert label["motion_state"] is None
    assert label["wear_state"] == "removed"
    assert not label["window_trainable"]


def test_long_session_rejects_supervised_segments():
    document = {
        "schema_version": SCHEMA_VERSION,
        "annotation_scope": "session_weak",
        "window_trainable": False,
        "recording": {"duration_seconds": 181},
        "segments": [
            {
                "activity_id": "run",
                "motion_state": "motion",
                "wear_state": "valid",
                "phase": "active",
                "window_trainable": True,
            }
        ],
    }
    errors = validate_v2_document(document)
    assert any("must not contain supervised segments" in error for error in errors)


def test_treadmill_walk_and_incline_walk_are_distinct_motion_classes():
    treadmill = reviewed_short_label("跑步机走路")
    incline = reviewed_short_label("爬坡")
    assert treadmill and treadmill["activity_id"] == "treadmill_walk"
    assert incline and incline["activity_id"] == "incline_walk"
    assert treadmill["motion_state"] == incline["motion_state"] == "motion"


def test_continuous_session_targets_structure_counts_and_process_notes():
    assert CONTINUOUS_SESSION_TARGETS["周翼天"]["ordered_activities"][1] == {
        "activity_id": "lat_pulldown",
        "sets": 4,
        "repetitions_per_set": 12,
    }
    assert len(CONTINUOUS_SESSION_TARGETS["黄诗敏"]["process"]) == 5
