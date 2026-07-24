from __future__ import annotations

import pandas as pd

from train_activity_model import split_by_subject


def test_subject_split_covers_every_feasible_class_and_preserves_training_classes():
    rows = []
    coverage = {
        "non_exercise": ["s1", "s2", "s3", "s4", "s5", "s6"],
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
    assert {"non_exercise", "run", "squat"} <= test_actions
    assert "elliptical" not in test_actions
