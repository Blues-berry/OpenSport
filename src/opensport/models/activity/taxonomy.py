"""Product-level activity family mapping."""

from __future__ import annotations


CARDIO = frozenset(
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
NON_MOTION = frozenset(
    {
        "sitting",
        "standing",
        "speaking",
        "chewing",
        "drinking",
        "gaming",
        "head_down",
        "head_up",
        "head_left",
        "head_right",
        "head_side",
        "head_turn",
        "other_non_motion",
    }
)


def activity_family(activity_id: str) -> str:
    if activity_id in CARDIO:
        return "cardio"
    if activity_id in NON_MOTION:
        return "other"
    if activity_id in {"removed_wear", "asymmetric_wear"}:
        return "other"
    return "strength" if activity_id not in {"unknown_motion", "other_motion"} else "other"
