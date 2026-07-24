"""Convert compatibility runtime dictionaries into stable public types."""

from __future__ import annotations

from opensport.types import ActivityPrediction


def activity_prediction_from_runtime(result: dict) -> ActivityPrediction:
    strategy = result.get("strategy", {})
    return ActivityPrediction(
        schema_version="1.0",
        timestamp=float(result["timestamp"]),
        wear_state=str(result.get("wear_state", "invalid")),
        signal_quality=str(result.get("signal_quality", "poor")),
        motion_state=result.get("motion_state"),
        motion_probability=float(result.get("motion_probability", 0.0)),
        exercise_state=str(
            result.get(
                "exercise_state",
                strategy.get("exercise_state", "not_exercising"),
            )
        ),
        activity_family=str(result.get("activity_family", "other")),
        activity_id=str(result.get("activity_id", "other_non_motion")),
        confidence=float(result.get("action_probability", 0.0)),
        workout_phase=str(
            result.get("workout_phase", strategy.get("state", "idle"))
        ),
        set_count=int(
            result.get("set_count", strategy.get("sets_in_session", 0))
        ),
        session_id=result.get("session_id", strategy.get("session_id")),
        finalized=bool(result.get("finalized", False)),
        experimental=bool(result.get("experimental", True)),
        warning=result.get("warning"),
    )
