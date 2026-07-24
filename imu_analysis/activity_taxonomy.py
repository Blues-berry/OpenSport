"""Canonical activity taxonomy backed by reviewed Schema v2 labels."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from label_schema import (
    ACTIVITIES,
    REVIEWED_SHORT_EXCLUSIONS,
    activity_name_zh,
    parse_capture_name,
    reviewed_short_label,
)


ACTION_NAMES_ZH = {
    activity_id: definition.name_zh for activity_id, definition in ACTIVITIES.items()
}
ACTION_NAMES_ZH.update(
    {
        "other_motion": "其他运动",
        "other_non_motion": "其他非运动",
        "unknown_motion": "未识别运动",
        "unknown_non_motion": "未识别非运动",
    }
)

MOTION_ACTIONS = tuple(
    activity_id
    for activity_id, definition in ACTIVITIES.items()
    if definition.motion_state == "motion"
)
NON_MOTION_ACTIONS = tuple(
    activity_id
    for activity_id, definition in ACTIVITIES.items()
    if definition.motion_state == "non_motion"
    and activity_id not in {"removed_wear", "asymmetric_wear"}
)
TARGET_ACTIONS = MOTION_ACTIONS
CARDIO_ACTIONS = frozenset(
    {
        "run",
        "treadmill_walk",
        "treadmill_unspecified",
        "incline_walk",
        "elliptical",
        "free_walk",
        "stairs_up",
        "stairs_down",
        "interval_walk_run",
    }
)
STRENGTH_ACTIONS = frozenset(set(MOTION_ACTIONS) - CARDIO_ACTIONS)


@dataclass(frozen=True)
class CaptureIdentity:
    date: str
    subject_id: str
    raw_action: str
    action_id: str | None
    phase: str
    usable_without_timeline: bool


def normalize_action(raw_action: str) -> tuple[str | None, str, bool]:
    """Return only labels present in the explicit reviewed action table."""
    text = re.sub(r"\s+", "", raw_action)
    reviewed = reviewed_short_label(text)
    if reviewed is None or text in REVIEWED_SHORT_EXCLUSIONS:
        return None, "transition", False
    if not reviewed["window_trainable"]:
        return None, reviewed["phase"], False
    return reviewed["activity_id"], reviewed["phase"], True


def capture_identity(path: Path) -> CaptureIdentity:
    """Parse ``date-subject-action.csv`` without inferring unknown semantics."""
    date, subject, raw_action = parse_capture_name(path)
    action_id, phase, usable = normalize_action(raw_action)
    return CaptureIdentity(date, subject, raw_action, action_id, phase, usable)


def action_name_zh(action_id: str) -> str:
    return ACTION_NAMES_ZH.get(action_id, activity_name_zh(action_id))
