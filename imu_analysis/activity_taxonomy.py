"""Canonical action labels and filename parsing for the IMU demo."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


TARGET_ACTIONS = (
    "run",
    "elliptical",
    "squat",
    "lunge",
    "crunch",
    "pushup",
    "jumping_jack",
    "plank",
)

ACTION_NAMES_ZH = {
    "non_exercise": "非运动",
    "unknown_motion": "未识别动作",
    "run": "跑步",
    "elliptical": "椭圆机",
    "squat": "深蹲",
    "lunge": "弓步蹲",
    "crunch": "卷腹",
    "pushup": "俯卧撑",
    "jumping_jack": "开合跳",
    "plank": "平板支撑",
}

CARDIO_ACTIONS = frozenset({"run", "elliptical"})
STRENGTH_ACTIONS = frozenset(set(TARGET_ACTIONS) - CARDIO_ACTIONS)

NON_EXERCISE_TERMS = (
    "坐姿", "站姿", "坐起", "站起", "坐下起立", "自由走", "自由行走", "走路",
    "低头", "抬头", "偏头", "左看", "右看", "左右看", "说话", "咀嚼", "喝水",
    "弯腰取物", "佩戴取下", "不对称佩戴", "王者荣耀", "下楼", "爬楼", "爬坡",
)

UNKNOWN_EXERCISE_TERMS = (
    "卧推", "蹬腿", "腿间", "硬拉", "高位下拉", "推肩", "引体向上",
)


@dataclass(frozen=True)
class CaptureIdentity:
    date: str
    subject_id: str
    raw_action: str
    action_id: str | None
    phase: str
    usable_without_timeline: bool


def normalize_action(raw_action: str) -> tuple[str | None, str, bool]:
    """Return action_id, default phase and whether file-level labelling is safe."""
    text = re.sub(r"\s+", "", raw_action)
    if ("深蹲" in text and "卷腹" in text) or ("椭圆机" in text and "卷腹" in text):
        return None, "transition", False
    if "推肩" in text and "高位下拉" in text:
        return None, "transition", False
    if "跑步" in text and "走" not in text and "3.0" not in text:
        return "run", "active_set", True
    if text.startswith("跑步机") and "跑步" in text:
        return "run", "active_set", True
    if "椭圆机" in text:
        return "elliptical", "active_set", True
    if "弓步蹲" in text:
        return "lunge", "active_set", True
    if "深蹲" in text or "蹲起" in text:
        return "squat", "active_set", True
    if "卷腹" in text:
        return "crunch", "active_set", True
    if "俯卧撑" in text:
        return "pushup", "active_set", True
    if "开合跳" in text:
        return "jumping_jack", "active_set", True
    if "平板支撑" in text:
        return "plank", "active_set", True
    if any(term in text for term in UNKNOWN_EXERCISE_TERMS):
        return "unknown_motion", "active_set", True
    if any(term in text for term in NON_EXERCISE_TERMS):
        return "non_exercise", "non_exercise", True
    return None, "transition", False


def capture_identity(path: Path) -> CaptureIdentity:
    """Parse ``date-subject-action.csv`` while retaining unknown files safely."""
    parts = path.stem.split("-", 2)
    if len(parts) == 3 and re.fullmatch(r"\d{4}", parts[0]):
        date, subject, raw_action = parts
    else:
        date, subject, raw_action = "unknown-date", "unknown-subject", path.stem
    action_id, phase, usable = normalize_action(raw_action)
    return CaptureIdentity(date, subject, raw_action, action_id, phase, usable)
