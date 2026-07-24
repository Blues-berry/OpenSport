from __future__ import annotations

import pandas as pd

from train_logistic import LABEL_SOURCE_INPUT, LABEL_SOURCE_TAXONOMY, apply_label_policy


def test_training_recomputes_stale_activity_labels_from_taxonomy():
    source = pd.DataFrame(
        {
            "activity": ["开合跳", "平板支撑1", "坐姿"],
            "state": ["non_exercise", "non_exercise", "non_exercise"],
            "capture_group": ["jump", "plank", "sit"],
        }
    )

    labelled, audit = apply_label_policy(source, LABEL_SOURCE_TAXONOMY)

    assert labelled["state"].tolist() == ["exercise", "exercise", "non_exercise"]
    assert labelled["label_changed"].tolist() == [True, True, False]
    assert int(audit["windows"].sum()) == 3


def test_training_can_explicitly_preserve_input_labels():
    source = pd.DataFrame(
        {
            "activity": ["开合跳"],
            "state": ["non_exercise"],
            "capture_group": ["jump"],
        }
    )

    labelled, _ = apply_label_policy(source, LABEL_SOURCE_INPUT)

    assert labelled.loc[0, "state"] == "non_exercise"
    assert bool(labelled.loc[0, "label_changed"])


def test_legacy_binary_trainer_prefers_schema_v2_motion_state():
    source = pd.DataFrame(
        {
            "exact_activity_id": ["free_walk", "speaking", "removed_wear"],
            "motion_state": ["motion", "non_motion", None],
            "state": ["non_exercise", "exercise", "non_exercise"],
            "capture_group": ["walk", "speak", "wear"],
        }
    )
    labelled, _ = apply_label_policy(source, LABEL_SOURCE_TAXONOMY)
    assert labelled["state"].tolist() == ["exercise", "non_exercise", "unlabelled"]
