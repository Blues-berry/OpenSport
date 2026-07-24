from pathlib import Path

from activity_taxonomy import capture_identity, normalize_action


def test_filename_taxonomy_handles_targets_and_negatives():
    assert normalize_action("深蹲1")[0] == "squat"
    assert normalize_action("跑步机跑步")[0] == "run"
    assert normalize_action("自由走路")[0] == "non_exercise"
    assert normalize_action("卧推三组")[0] == "unknown_motion"


def test_mixed_recording_requires_timeline():
    action, phase, usable = normalize_action("深蹲+一个卷腹")
    assert action is None
    assert phase == "transition"
    assert not usable


def test_flat_filename_identity():
    identity = capture_identity(Path("0722-测试用户-开合跳.csv"))
    assert identity.date == "0722"
    assert identity.subject_id == "测试用户"
    assert identity.action_id == "jumping_jack"
