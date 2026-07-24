"""Deterministic workout/session policy on top of model probabilities."""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass

from activity_taxonomy import CARDIO_ACTIONS, TARGET_ACTIONS


@dataclass(frozen=True)
class StrategyConfig:
    start_threshold: float = 0.65
    continue_threshold: float = 0.45
    start_hold_s: float = 2.0
    end_hold_s: float = 3.0
    minimum_set_s: float = 4.0
    workout_timeout_s: float = 600.0


@dataclass
class StrategySnapshot:
    state: str = "idle"
    session_id: str | None = None
    action: str | None = None
    motion_probability: float = 0.0
    action_probability: float = 0.0
    sets_in_session: int = 0
    active_seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class WorkoutStrategy:
    """Convert noisy window probabilities into sets, rests and workout sessions."""

    def __init__(
        self,
        config: StrategyConfig | None = None,
        motion_actions: list[str] | tuple[str, ...] | None = None,
    ):
        self.config = config or StrategyConfig()
        self.motion_actions = tuple(motion_actions or TARGET_ACTIONS)
        self.snapshot = StrategySnapshot()
        self._candidate_action: str | None = None
        self._candidate_since: float | None = None
        self._low_since: float | None = None
        self._active_start: float | None = None
        self._session_start: float | None = None
        self._last_active_end: float | None = None
        self._confidence_sum = 0.0
        self._confidence_count = 0

    def _start_action(self, timestamp: float, action: str, probability: float) -> list[dict]:
        events: list[dict] = []
        start = self._candidate_since if self._candidate_since is not None else timestamp
        if self.snapshot.session_id is None:
            self.snapshot.session_id = uuid.uuid4().hex
            self._session_start = start
            events.append(
                {
                    "type": "workout_started",
                    "session_id": self.snapshot.session_id,
                    "timestamp": start,
                }
            )
        self.snapshot.state = "active_set"
        self.snapshot.action = action
        self.snapshot.action_probability = probability
        self._active_start = start
        self._low_since = None
        self._confidence_sum = probability
        self._confidence_count = 1
        events.append(
            {
                "type": "cardio_started" if action in CARDIO_ACTIONS else "set_started",
                "session_id": self.snapshot.session_id,
                "timestamp": start,
                "action": action,
                "confidence": probability,
            }
        )
        return events

    def _end_action(self, timestamp: float) -> list[dict]:
        if self._active_start is None or self.snapshot.action is None or self.snapshot.session_id is None:
            return []
        end = max(timestamp, self._active_start)
        duration = end - self._active_start
        action = self.snapshot.action
        confidence = self._confidence_sum / max(1, self._confidence_count)
        events: list[dict] = []
        if duration >= self.config.minimum_set_s:
            event_type = "cardio_ended" if action in CARDIO_ACTIONS else "set_ended"
            events.append(
                {
                    "type": event_type,
                    "session_id": self.snapshot.session_id,
                    "timestamp": end,
                    "start_timestamp": self._active_start,
                    "action": action,
                    "duration_seconds": duration,
                    "confidence": confidence,
                }
            )
            self.snapshot.active_seconds += duration
            if action not in CARDIO_ACTIONS:
                self.snapshot.sets_in_session += 1
        self._last_active_end = end
        self.snapshot.state = "inter_set"
        self.snapshot.action = None
        self.snapshot.action_probability = 0.0
        self._active_start = None
        self._low_since = None
        self._confidence_sum = 0.0
        self._confidence_count = 0
        return events

    def _end_workout_if_due(self, timestamp: float) -> list[dict]:
        if (
            self.snapshot.session_id
            and self.snapshot.state == "inter_set"
            and self._last_active_end is not None
            and timestamp - self._last_active_end >= self.config.workout_timeout_s
        ):
            event = {
                "type": "workout_ended",
                "session_id": self.snapshot.session_id,
                "timestamp": self._last_active_end,
                "start_timestamp": self._session_start,
                "active_seconds": self.snapshot.active_seconds,
                "sets": self.snapshot.sets_in_session,
            }
            self.snapshot = StrategySnapshot()
            self._candidate_action = None
            self._candidate_since = None
            self._session_start = None
            self._last_active_end = None
            return [event]
        return []

    def update(self, timestamp: float, probabilities: dict[str, float], signal_quality: str = "good") -> tuple[StrategySnapshot, list[dict]]:
        target = {
            action: float(probabilities.get(action, 0.0))
            for action in self.motion_actions
        }
        if not target:
            self.snapshot.motion_probability = 0.0
            return self.snapshot, []
        action = max(target, key=target.get)
        probability = target[action]
        motion_probability = min(1.0, sum(target.values()))
        if signal_quality == "poor":
            probability = 0.0
            motion_probability = 0.0
        self.snapshot.motion_probability = motion_probability
        events: list[dict] = []

        if self.snapshot.state == "active_set" and self.snapshot.action:
            current_probability = target.get(self.snapshot.action, 0.0)
            self.snapshot.action_probability = current_probability
            self._confidence_sum += current_probability
            self._confidence_count += 1
            if current_probability >= self.config.continue_threshold:
                self._low_since = None
            else:
                self._low_since = self._low_since if self._low_since is not None else timestamp
                if timestamp - self._low_since >= self.config.end_hold_s:
                    events.extend(self._end_action(self._low_since))
        else:
            if probability >= self.config.start_threshold:
                if action != self._candidate_action:
                    self._candidate_action = action
                    self._candidate_since = timestamp
                elif self._candidate_since is not None and timestamp - self._candidate_since >= self.config.start_hold_s:
                    events.extend(self._start_action(timestamp, action, probability))
                    self._candidate_action = None
                    self._candidate_since = None
            else:
                self._candidate_action = None
                self._candidate_since = None
            events.extend(self._end_workout_if_due(timestamp))
        return self.snapshot, events

    def flush(self, timestamp: float) -> tuple[StrategySnapshot, list[dict]]:
        events = []
        if self.snapshot.state == "active_set":
            events.extend(self._end_action(timestamp))
        if self.snapshot.session_id and self._last_active_end is not None:
            events.extend(self._end_workout_if_due(self._last_active_end + self.config.workout_timeout_s))
        return self.snapshot, events
