"""Session-level validation for long recordings with no window boundaries."""

from __future__ import annotations

from typing import Any


def _lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def evaluate_weak_session(
    weak_targets: dict[str, Any],
    predicted_activities: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare an ordered workout summary with weak session targets."""
    expected = [
        str(item["activity_id"])
        for item in weak_targets.get("ordered_activities", [])
        if item.get("activity_id")
    ]
    predicted = [
        str(item.get("activity_id") or item.get("action_id"))
        for item in predicted_activities
        if item.get("activity_id") or item.get("action_id")
    ]
    expected_unique = set(expected)
    coverage = (
        len(expected_unique & set(predicted)) / len(expected_unique)
        if expected_unique
        else None
    )
    order_score = _lcs_length(expected, predicted) / len(expected) if expected else None
    expected_sets = weak_targets.get("total_sets")
    predicted_sets = sum(int(item.get("sets") or 0) for item in predicted_activities)
    return {
        "expected_activities": expected,
        "predicted_activities": predicted,
        "activity_coverage": coverage,
        "order_score": order_score,
        "expected_total_sets": expected_sets,
        "predicted_total_sets": predicted_sets,
        "set_count_error": (
            predicted_sets - int(expected_sets) if expected_sets is not None else None
        ),
    }
