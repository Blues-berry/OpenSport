from pathlib import Path

from activity_taxonomy import capture_identity, normalize_action


def test_reviewed_taxonomy_uses_exact_dual_label_actions():
    assert normalize_action("深蹲1")[0] == "squat"
    assert normalize_action("跑步机跑步")[0] == "run"
    assert normalize_action("跑步机走路")[0] == "treadmill_walk"
    assert normalize_action("自由走路")[0] == "free_walk"
    assert normalize_action("爬坡")[0] == "incline_walk"
    assert normalize_action("工字蹲")[0] == "squat"
    assert normalize_action("弯腰")[0] == "bending"
    assert normalize_action("卧推三组")[0] == "bench_press"


def test_unknown_name_never_falls_back_to_substring_matching():
    assert normalize_action("跑步机走路新协议") == (None, "transition", False)


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
