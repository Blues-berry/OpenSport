from datetime import datetime

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
    feed(strategy, store, base, base + 9.5, "squat")
    feed(strategy, store, base + 10, base + 14, None)
    feed(strategy, store, base + 15, base + 24.5, "squat")
    feed(strategy, store, base + 25, base + 29, None)
    _, events = strategy.flush(base + 30)
    store.apply_events(events)

    summary = store.daily_summary("2026-07-23")
    assert summary["session_count"] == 1
    assert summary["sessions"][0]["activities"][0]["action"] == "深蹲"
    assert summary["sessions"][0]["activities"][0]["sets"] == 2
    assert summary["sessions"][0]["active_seconds"] == 20
