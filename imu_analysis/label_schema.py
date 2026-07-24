"""Authoritative dual-level activity labels and legacy compatibility.

The reviewed table in this module is intentionally keyed by the complete raw
action text.  It is not a filename substring classifier: a new action name is
unlabelled until a reviewer adds an explicit entry.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "2.0"
SHORT_RECORDING_MAX_SECONDS = 180.0
MIN_SUBJECTS_FOR_MODEL_CLASS = 5
EVIDENCE_TIERS = frozenset(
    {"gold", "legacy_reviewed", "session_weak", "rejected"}
)
GOLD_LABEL_SOURCES = frozenset(
    {"operator_event", "video_review", "manual_timeline"}
)
MOTION_STATES = frozenset({"motion", "non_motion"})
WEAR_STATES = frozenset({"valid", "removed", "asymmetric", "invalid"})
PHASES = frozenset({"active", "rest", "transition", "artifact"})


@dataclass(frozen=True)
class ActivityDefinition:
    name_zh: str
    motion_state: str


ACTIVITIES: dict[str, ActivityDefinition] = {
    "asymmetric_wear": ActivityDefinition("不对称佩戴", "non_motion"),
    "bench_press": ActivityDefinition("卧推", "motion"),
    "bending": ActivityDefinition("弯腰", "motion"),
    "chewing": ActivityDefinition("咀嚼", "non_motion"),
    "chest_fly": ActivityDefinition("飞鸟", "motion"),
    "chest_press": ActivityDefinition("坐姿推胸", "motion"),
    "crunch": ActivityDefinition("卷腹", "motion"),
    "deadlift": ActivityDefinition("硬拉", "motion"),
    "dip": ActivityDefinition("双杠臂屈伸", "motion"),
    "drinking": ActivityDefinition("喝水", "non_motion"),
    "elliptical": ActivityDefinition("椭圆机", "motion"),
    "free_walk": ActivityDefinition("自由走路", "motion"),
    "gaming": ActivityDefinition("游戏", "non_motion"),
    "head_down": ActivityDefinition("低头", "non_motion"),
    "head_left": ActivityDefinition("左看", "non_motion"),
    "head_right": ActivityDefinition("右看", "non_motion"),
    "head_side": ActivityDefinition("偏头", "non_motion"),
    "head_turn": ActivityDefinition("左右看", "non_motion"),
    "head_up": ActivityDefinition("抬头", "non_motion"),
    "back_extension": ActivityDefinition("山羊挺身", "motion"),
    "biceps_curl": ActivityDefinition("肱二头肌训练", "motion"),
    "hip_adduction": ActivityDefinition("腿间训练", "motion"),
    "incline_walk": ActivityDefinition("爬坡走", "motion"),
    "interval_walk_run": ActivityDefinition("走跑交替", "motion"),
    "jumping_jack": ActivityDefinition("开合跳", "motion"),
    "lat_pulldown": ActivityDefinition("高位下拉", "motion"),
    "leg_press": ActivityDefinition("蹬腿", "motion"),
    "leg_raise": ActivityDefinition("举腿", "motion"),
    "lunge": ActivityDefinition("弓步蹲", "motion"),
    "plank": ActivityDefinition("平板支撑", "motion"),
    "pull_up": ActivityDefinition("引体向上", "motion"),
    "pushup": ActivityDefinition("俯卧撑", "motion"),
    "removed_wear": ActivityDefinition("佩戴取下", "non_motion"),
    "run": ActivityDefinition("跑步", "motion"),
    "rowing": ActivityDefinition("划船", "motion"),
    "sitting": ActivityDefinition("坐姿", "non_motion"),
    "sit_to_stand": ActivityDefinition("坐起站起", "motion"),
    "speaking": ActivityDefinition("说话", "non_motion"),
    "shoulder_press": ActivityDefinition("推肩", "motion"),
    "squat": ActivityDefinition("深蹲", "motion"),
    "stairs_down": ActivityDefinition("下楼", "motion"),
    "stairs_up": ActivityDefinition("爬楼", "motion"),
    "standing": ActivityDefinition("站姿", "non_motion"),
    "treadmill_unspecified": ActivityDefinition("跑步机（速度未知）", "motion"),
    "treadmill_walk": ActivityDefinition("跑步机走路", "motion"),
    "triceps_extension": ActivityDefinition("肱三头肌训练", "motion"),
    "warmup": ActivityDefinition("热身", "motion"),
}


def _aliases(activity_id: str, *raw_actions: str) -> dict[str, str]:
    return {raw: activity_id for raw in raw_actions}


# Every short-recording action currently present in data/training/activity is
# represented explicitly here.  Numeric suffixes are listed, not stripped.
REVIEWED_SHORT_ACTIONS: dict[str, str] = {
    **_aliases("asymmetric_wear", "不对称佩戴"),
    **_aliases("head_down", "低头"),
    **_aliases("squat", "蹲起", "工字蹲", "深蹲", "深蹲1", "深蹲2组"),
    **_aliases("pushup", "俯卧撑"),
    **_aliases("lunge", "弓步蹲"),
    **_aliases("drinking", "喝水"),
    **_aliases("chewing", "咀嚼"),
    **_aliases("crunch", "卷腹", "卷腹1", "卷腹2"),
    **_aliases("jumping_jack", "开合跳", "开合跳1", "开合跳2"),
    **_aliases("stairs_up", "爬楼"),
    **_aliases(
        "incline_walk",
        "爬坡",
        "爬坡20",
        "爬坡20-2",
        "爬坡20-3",
        "爬坡20-4",
        "爬坡20-5",
        "爬坡20-6",
        "爬坡20-7",
        "爬坡20-8",
        "爬坡20-9",
        "爬坡20-10",
    ),
    **_aliases("run", "跑步", "跑步2", "跑步9-0", "跑步机跑步"),
    **_aliases("treadmill_unspecified", "跑步机"),
    **_aliases("treadmill_walk", "跑步机3.0", "跑步机走路", "走路3.0", "走路5.0"),
    **_aliases("removed_wear", "佩戴取下"),
    **_aliases("head_side", "偏头", "左右偏头"),
    **_aliases("plank", "平板支撑", "平板支撑1", "平板支撑2"),
    **_aliases("interval_walk_run", "时走时跑"),
    **_aliases("speaking", "说话", "说话1", "说话2"),
    **_aliases("head_up", "抬头", "抬头2"),
    **_aliases("elliptical", "椭圆机", "椭圆机-3"),
    **_aliases("bending", "弯腰", "弯腰取物"),
    **_aliases("bench_press", "卧推", "卧推三组"),
    **_aliases("stairs_down", "下楼"),
    **_aliases("pull_up", "引体向上", "引体向上1", "引体向上2"),
    **_aliases("head_right", "右看"),
    **_aliases("sit_to_stand", "站起", "坐起", "坐下起立"),
    **_aliases("standing", "站姿", "站姿1", "站姿2", "站姿3", "站姿3-1", "站姿4"),
    **_aliases("free_walk", "自由行走", "自由走动", "自由走路", "走路"),
    **_aliases("head_left", "左看"),
    **_aliases("head_turn", "左右看"),
    **_aliases("sitting", "坐姿", "坐姿1", "坐姿2", "坐姿3", "坐姿-余有头动"),
}


# These short captures contain a documented transition, bad execution or wear
# artifact without a trustworthy boundary.  They are reviewed but not used.
REVIEWED_SHORT_EXCLUSIONS: dict[str, tuple[str, str]] = {
    "跑步-王后期提前终止": ("run", "记录后期提前终止，缺少可信切点"),
    "深蹲+一个卷腹": ("squat", "同一短记录包含深蹲和卷腹"),
    "站起-张子怡有耳机掉落的情况，一次": ("sit_to_stand", "记录中发生耳机掉落"),
    "站姿-李浩宇挂脖子上": ("standing", "设备挂在脖子上，佩戴无效"),
    "坐起-余有不规范的地方": ("sit_to_stand", "动作执行不规范且无可信边界"),
    "坐姿-李浩宇把耳机挂脖子上": ("sitting", "设备挂在脖子上，佩戴无效"),
    "坐姿1-许歆儿最后几秒有站起走动": ("sitting", "最后几秒站起走动且无可信切点"),
}


LONG_ACTION_HINTS: dict[str, list[dict[str, Any]]] = {
    "王者荣耀": [{"activity_id": "gaming"}],
    "蹬腿3组": [{"activity_id": "leg_press", "sets": 3}],
    "爬坡": [{"activity_id": "incline_walk"}],
    "爬坡2": [{"activity_id": "incline_walk"}],
    "椭圆机-卷腹10、平板支撑1分钟、蹲起10、开合跳20+、卷腹10、卷腹10、开合跳20": [
        {"activity_id": "elliptical"},
        {"activity_id": "crunch", "repetitions": 10},
        {"activity_id": "plank", "duration_seconds": 60},
        {"activity_id": "squat", "repetitions": 10},
        {"activity_id": "jumping_jack", "repetitions": 20},
        {"activity_id": "crunch", "repetitions": 10},
        {"activity_id": "crunch", "repetitions": 10},
        {"activity_id": "jumping_jack", "repetitions": 20},
    ],
    "椭圆机-2": [{"activity_id": "elliptical"}],
    "蹬腿6组": [{"activity_id": "leg_press", "sets": 6}],
    "推肩2、高位下拉几个": [
        {"activity_id": "other_motion", "raw_label": "推肩", "sets": 2},
        {"activity_id": "lat_pulldown", "sets": None},
    ],
    "卧推一组": [{"activity_id": "bench_press", "sets": 1}],
    "硬拉3组": [{"activity_id": "deadlift", "sets": 3}],
    "高位下拉2组": [{"activity_id": "lat_pulldown", "sets": 2}],
    "高位下拉3组": [{"activity_id": "lat_pulldown", "sets": 3}],
    "腿间3组": [{"activity_id": "hip_adduction", "sets": 3}],
    "连续健身": [],
}


CONTINUOUS_SESSION_TARGETS: dict[str, dict[str, Any]] = {
    "丁俊夫": {
        "ordered_activities": [
            {"activity_id": "warmup"},
            {"activity_id": "bench_press", "sets": 5},
            {"activity_id": "chest_press", "sets": 4},
        ],
        "process": [{"phase": "inter_set", "approximate_duration_seconds": 120}],
    },
    "樊华": {
        "ordered_activities": [
            {"activity_id": "incline_walk", "approximate_duration_seconds": 360},
            {"activity_id": "leg_press", "sets": 8, "raw_note": "左右各四组"},
            {"activity_id": "leg_raise", "sets": 3, "raw_label": "收缩举腿"},
            {"activity_id": "leg_raise", "sets": 3, "repetitions_per_set": 15, "raw_label": "仰卧举腿"},
            {"activity_id": "other_motion", "sets": 4, "repetitions_per_set": 16, "raw_label": "左右转髋"},
            {"activity_id": "crunch", "sets": 3, "repetitions_per_set": 15},
            {"activity_id": "lat_pulldown", "sets": 3, "repetitions_per_set": 15},
            {"activity_id": "treadmill_walk", "approximate_duration_seconds": 120},
        ],
        "process": [
            {"phase": "rest", "description": "自由走、坐着、站着", "approximate_duration_seconds": 180},
            {"phase": "artifact", "description": "仰卧蹬腿时耳机晃动"},
            {"phase": "artifact", "description": "休息时耳机掉落一次"},
            {"phase": "rest", "description": "走走停停", "approximate_duration_seconds": 120},
        ],
    },
    "黄诗敏": {
        "ordered_activities": [
            {"activity_id": "warmup", "approximate_duration_seconds": 60},
            {"activity_id": "pull_up", "repetitions": 20, "weight_kg": 45, "raw_label": "正手辅助引体"},
            {"activity_id": "pull_up", "repetitions": 10, "weight_kg": 45, "raw_label": "反手辅助引体"},
            {"activity_id": "dip", "repetitions": 15, "raw_label": "辅助双杠臂屈伸"},
            {"activity_id": "chest_fly", "repetitions": 15, "weight_kg": 2.5},
            {"activity_id": "back_extension", "repetitions": 20},
            {"activity_id": "biceps_curl", "repetitions": 20, "weight_kg": 5, "raw_note": "左右各10个"},
            {"activity_id": "biceps_curl", "repetitions": 40, "weight_kg": 2.5, "raw_note": "左右各20个"},
            {"activity_id": "back_extension", "repetitions": 20},
            {"activity_id": "other_motion", "repetitions": 15, "weight_kg": 30, "raw_label": "髋外展"},
            {"activity_id": "hip_adduction", "repetitions": 15, "weight_kg": 15},
            {"activity_id": "leg_press", "repetitions": 20, "weight_kg": 30, "raw_label": "坐姿腿推举"},
            {"activity_id": "pull_up", "repetitions": 15, "raw_label": "辅助引体"},
            {"activity_id": "dip", "repetitions": 15, "raw_label": "辅助双杠臂屈伸"},
            {"activity_id": "back_extension", "repetitions": 15},
            {"activity_id": "crunch", "repetitions": 20, "raw_label": "仰卧起坐"},
            {"activity_id": "pull_up", "repetitions": 15, "raw_label": "反手引体"},
            {"activity_id": "pull_up", "repetitions": 20, "raw_label": "正手引体"},
            {"activity_id": "leg_press", "repetitions": 20, "weight_kg": 30, "raw_label": "坐姿腿推举"},
            {"activity_id": "dip", "repetitions": 15},
            {"activity_id": "chest_fly", "repetitions": 20},
        ],
        "process": [
            {"phase": "rest", "clock_range": "15:14-15:16", "duration_seconds": 120},
            {"phase": "rest", "clock_range": "15:21-15:23", "duration_seconds": 120},
            {"phase": "rest", "clock_range": "15:28-15:31", "duration_seconds": 180},
            {"phase": "rest", "clock_range": "15:35-15:38", "duration_seconds": 180},
            {"phase": "rest", "clock_range": "15:44-15:46", "duration_seconds": 120},
        ],
    },
    "王昱东": {
        "ordered_activities": [
            {"activity_id": "warmup"},
            {"activity_id": "shoulder_press", "sets": 4, "repetitions": [10, 10, 10, 10], "weight_kg": 7.5},
            {"activity_id": "chest_fly", "sets": 4, "repetitions_per_set": 12, "weight_kg": 5},
            {"activity_id": "biceps_curl", "sets": 4, "repetitions_per_set": 12, "weight_kg": 20},
            {"activity_id": "triceps_extension", "sets": 4, "repetitions_per_set": 12, "weight_kg": 15},
        ],
        "process": [{"phase": "inter_set", "rest_minutes": [2, 2, 2, 3]}],
    },
    "周翼天": {
        "ordered_activities": [
            {"activity_id": "warmup"},
            {"activity_id": "lat_pulldown", "sets": 4, "repetitions_per_set": 12},
            {"activity_id": "shoulder_press", "sets": 5, "repetitions_per_set": 12},
            {"activity_id": "rowing", "sets": 4, "repetitions_per_set": 12},
        ],
        "process": [
            {"phase": "inter_set", "activity_id": "lat_pulldown", "rest_minutes_range": [2, 3]},
            {"phase": "inter_set", "activity_id": "shoulder_press", "rest_minutes": 3},
            {"phase": "inter_set", "activity_id": "rowing", "rest_minutes_range": [3, 4]},
        ],
    },
}


LEGACY_LABEL_ALIASES: dict[str, str] = {
    "仰卧蹬腿": "leg_press",
    "卷腹": "crunch",
    "高位下拉": "lat_pulldown",
    "跑步机走路": "treadmill_walk",
    "爬坡": "incline_walk",
    "卧推": "bench_press",
    "引体": "pull_up",
    "辅助引体": "pull_up",
    "双杠臂屈伸": "dip",
    "飞鸟": "chest_fly",
    "山羊挺身": "back_extension",
    "肱二头": "biceps_curl",
    "肱三头": "triceps_extension",
    "髋外展": "other_motion",
    "髋内收": "other_motion",
    "坐姿腿推举": "leg_press",
    "仰卧起坐": "crunch",
    "推肩": "shoulder_press",
    "划船": "rowing",
    "收缩举腿": "leg_raise",
    "仰卧举腿": "leg_raise",
    "左右转髋": "other_motion",
}


def taxonomy_version() -> str:
    payload = {
        "schema": SCHEMA_VERSION,
        "activities": {key: vars(value) for key, value in ACTIVITIES.items()},
        "short": REVIEWED_SHORT_ACTIONS,
        "excluded": REVIEWED_SHORT_EXCLUSIONS,
        "long": LONG_ACTION_HINTS,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def parse_capture_name(path: Path) -> tuple[str, str, str]:
    parts = path.stem.split("-", 2)
    if len(parts) == 3 and re.fullmatch(r"\d{4}", parts[0]):
        return parts[0], parts[1], parts[2]
    if len(parts) >= 2:
        return "unknown-date", parts[0], "-".join(parts[1:])
    return "unknown-date", "unknown-subject", path.stem


def reviewed_short_label(raw_action: str) -> dict[str, Any] | None:
    if raw_action in REVIEWED_SHORT_EXCLUSIONS:
        activity_id, reason = REVIEWED_SHORT_EXCLUSIONS[raw_action]
        invalid_wear = any(term in raw_action for term in ("挂脖", "掉落"))
        return {
            "activity_id": activity_id,
            "motion_state": None if invalid_wear else ACTIVITIES[activity_id].motion_state,
            "wear_state": "invalid" if invalid_wear else "valid",
            "phase": "artifact" if invalid_wear else "transition",
            "window_trainable": False,
            "confidence": "high",
            "review_note": reason,
        }
    activity_id = REVIEWED_SHORT_ACTIONS.get(raw_action)
    if activity_id is None:
        return None
    if activity_id == "removed_wear":
        wear_state = "removed"
    elif activity_id == "asymmetric_wear":
        wear_state = "asymmetric"
    else:
        wear_state = "valid"
    invalid = wear_state != "valid"
    return {
        "activity_id": activity_id,
        "motion_state": None if invalid else ACTIVITIES[activity_id].motion_state,
        "wear_state": wear_state,
        "phase": "artifact" if invalid else (
            "active" if ACTIVITIES[activity_id].motion_state == "motion" else "rest"
        ),
        "window_trainable": not invalid,
        "confidence": "high",
        "review_note": "已按完整原始动作名称逐项审核",
    }


def activity_name_zh(activity_id: str) -> str:
    if activity_id == "other_motion":
        return "其他运动"
    if activity_id == "other_non_motion":
        return "其他非运动"
    return ACTIVITIES.get(
        activity_id, ActivityDefinition(activity_id, "non_motion")
    ).name_zh


def motion_state_for_activity(activity_id: str) -> str | None:
    if activity_id == "other_motion":
        return "motion"
    if activity_id == "other_non_motion":
        return "non_motion"
    definition = ACTIVITIES.get(activity_id)
    return definition.motion_state if definition else None


def legacy_label_to_activity(raw_label: str) -> str | None:
    for term, activity_id in LEGACY_LABEL_ALIASES.items():
        if term in raw_label:
            return activity_id
    return None


def validate_v2_document(document: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append("schema_version must be 2.0")
    required = {
        "taxonomy_version",
        "date",
        "participant",
        "device",
        "csv_file",
        "annotation_scope",
        "recording",
        "window_trainable",
        "annotation_quality",
        "evidence_tier",
        "segments",
    }
    if missing := sorted(required - document.keys()):
        errors.append(f"missing required fields: {missing}")
    evidence_tier = document.get("evidence_tier")
    if evidence_tier not in EVIDENCE_TIERS:
        errors.append("evidence_tier is invalid")
    duration = float(document.get("recording", {}).get("duration_seconds", -1))
    annotation_scope = document.get("annotation_scope")
    if duration > SHORT_RECORDING_MAX_SECONDS and annotation_scope != "session_weak":
        errors.append("recordings over 180 seconds must use session_weak")
    if annotation_scope == "session_weak":
        if document.get("segments"):
            errors.append("session_weak must not contain supervised segments")
        if document.get("window_trainable") is not False:
            errors.append("session_weak must set window_trainable=false")
    previous_end = 0.0
    for index, segment in enumerate(document.get("segments", [])):
        prefix = f"segments[{index}]"
        activity_id = segment.get("activity_id")
        motion_state = segment.get("motion_state")
        wear_state = segment.get("wear_state")
        start_s = float(segment.get("start_s", -1))
        end_s = float(segment.get("end_s", -1))
        if start_s < previous_end:
            errors.append(f"{prefix} overlaps or is out of order")
        if start_s < 0 or end_s <= start_s or (
            duration >= 0 and end_s > duration + 1e-3
        ):
            errors.append(f"{prefix} has invalid bounds")
        previous_end = max(previous_end, end_s)
        if activity_id not in ACTIVITIES:
            errors.append(f"{prefix}.activity_id is unknown: {activity_id}")
        if motion_state is not None and motion_state not in MOTION_STATES:
            errors.append(f"{prefix}.motion_state is invalid")
        if wear_state not in WEAR_STATES:
            errors.append(f"{prefix}.wear_state is invalid")
        if wear_state != "valid" and segment.get("window_trainable"):
            errors.append(f"{prefix} invalid wear cannot be window_trainable")
        if wear_state != "valid" and motion_state is not None:
            errors.append(f"{prefix} invalid wear must set motion_state=null")
        if segment.get("phase") not in PHASES:
            errors.append(f"{prefix}.phase is invalid")
        if evidence_tier == "gold" and segment.get("window_trainable"):
            if segment.get("label_source") not in GOLD_LABEL_SOURCES:
                errors.append(f"{prefix} gold label source is not accepted")
            if segment.get("confidence") not in {"high", "medium"}:
                errors.append(f"{prefix} gold label confidence is too low")
    return errors


def load_label_document(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") == SCHEMA_VERSION:
        errors = validate_v2_document(payload)
        if errors:
            raise ValueError(f"{path.name}: " + "; ".join(errors))
        return payload
    return convert_legacy_document(payload, path)


def convert_legacy_document(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    """Read a legacy file conservatively without inventing window labels."""
    recording = payload.get("recording", {})
    start = recording.get("start_time")
    end = recording.get("end_time")
    duration = 0.0
    if start and end:
        from datetime import datetime

        duration = (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds()
    _, _, raw_action = parse_capture_name(Path(payload.get("csv_file", path.stem)))
    if duration > SHORT_RECORDING_MAX_SECONDS:
        return {
            **payload,
            "schema_version": SCHEMA_VERSION,
            "evidence_tier": "session_weak",
            "annotation_scope": "session_weak",
            "window_trainable": False,
            "recording": {**recording, "duration_seconds": duration},
            "segments": [],
            "weak_targets": {"ordered_activities": LONG_ACTION_HINTS.get(raw_action, [])},
            "compatibility_source": "legacy",
        }
    reviewed = reviewed_short_label(raw_action)
    if reviewed is None:
        return {
            **payload,
            "schema_version": SCHEMA_VERSION,
            "evidence_tier": "rejected",
            "annotation_scope": "review_required",
            "window_trainable": False,
            "recording": {**recording, "duration_seconds": duration},
            "segments": [],
            "compatibility_source": "legacy_unreviewed",
        }
    segment = {
        **reviewed,
        "start_s": 0.0,
        "end_s": duration,
        "label_source": "reviewed_exact_action_compatibility",
    }
    return {
        **payload,
        "schema_version": SCHEMA_VERSION,
        "evidence_tier": "legacy_reviewed",
        "annotation_scope": "full_recording",
        "window_trainable": reviewed["window_trainable"],
        "recording": {**recording, "duration_seconds": duration},
        "segments": [segment],
        "compatibility_source": "legacy",
    }
