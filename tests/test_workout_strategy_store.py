from datetime import datetime

from activity_runtime import RuntimeCoordinator
from activity_taxonomy import TARGET_ACTIONS
from workout_store import WorkoutStore
from workout_strategy import WorkoutStrategy


def probabilities(action=None, value=0.9):
    output = {name: 0.005 for name in TARGET_ACTIONS}
    output["non_exercise"] = 0.9 if action is None else 0.02
    output["unknown_motion"] = 0.02
    if action:
        output[action] = value
    return output


def feed(strategy, store, start, end, action):
    events = []
    timestamp = start
    while timestamp <= end:
        _, emitted = strategy.update(timestamp, probabilities(action))
        store.apply_events(emitted)
        events.extend(emitted)
        timestamp += 0.5
    return events


def test_two_sets_become_one_ordered_activity():
    base = datetime(2026, 7, 23, 9, 10).timestamp()
    store = WorkoutStore(":memory:")
    strategy = WorkoutStrategy()
    feed(strategy, store, base, base + 12, "squat")
    feed(strategy, store, base + 12.5, base + 33, None)
    feed(strategy, store, base + 33.5, base + 45.5, "squat")
    feed(strategy, store, base + 46, base + 67, None)
    _, events = strategy.flush(base + 308)
    store.apply_events(events)

    summary = store.daily_summary("2026-07-23")
    assert summary["session_count"] == 1
    assert summary["sessions"][0]["activities"][0]["action"] == "深蹲"
    assert summary["sessions"][0]["activities"][0]["sets"] == 2
    assert summary["sessions"][0]["active_seconds"] == 25


def test_general_motion_requires_30_seconds_but_strength_starts_after_3():
    base = datetime(2026, 7, 23, 9, 10).timestamp()
    strength = WorkoutStrategy()
    for offset in range(3):
        strength.update(base + offset, probabilities("squat"))
    assert strength.snapshot.state == "idle"
    strength.update(base + 3, probabilities("squat"))
    assert strength.snapshot.state == "active"

    cardio = WorkoutStrategy()
    for offset in range(30):
        cardio.update(base + offset, probabilities("free_walk"))
    assert cardio.snapshot.state == "idle"
    cardio.update(base + 30, probabilities("free_walk"))
    assert cardio.snapshot.state == "active"


def test_store_is_idempotent_for_duplicate_events():
    store = WorkoutStore(":memory:")
    event = {
        "type": "workout_started",
        "session_id": "same",
        "timestamp": 100.0,
    }
    store.apply_events([event, event])
    with store._connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM strategy_events").fetchone()[0] == 1


def test_runtime_flush_reports_consistent_finalized_snapshot():
    base = datetime(2026, 7, 23, 9, 10).timestamp()
    coordinator = RuntimeCoordinator.__new__(RuntimeCoordinator)
    coordinator.strategy = WorkoutStrategy()
    coordinator.store = None
    coordinator.last_result = {"timestamp": base}
    for offset in range(4):
        coordinator.strategy.update(base + offset, probabilities("squat"))

    events = coordinator.flush(base + 10)

    assert any(event["type"] == "workout_ended" for event in events)
    assert coordinator.last_result["finalized"] is True
    assert coordinator.last_result["strategy"]["finalized"] is True
