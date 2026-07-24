"""Streaming LightGBM inference and strategy coordination."""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from activity_features import SENSOR_KEYS, extract_window_features, signal_quality
from workout_store import WorkoutStore
from workout_strategy import WorkoutStrategy


def calibrated_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / max(temperature, 1e-6)
    scaled = np.exp(logits - np.max(logits))
    return scaled / scaled.sum()


class StreamingActivityClassifier:
    """Maintain a causal window and emit model probabilities every hop."""

    def __init__(self, model_path: Path):
        with Path(model_path).open("rb") as handle:
            self.payload = pickle.load(handle)
        if self.payload.get("model_family") != "lightgbm_multiclass":
            raise ValueError("Expected a lightgbm_multiclass activity model")
        self.model = self.payload["model"]
        self.features = self.payload["features"]
        self.classes = self.payload["classes"]
        self.sample_rate_hz = float(self.payload["sample_rate_hz"])
        self.window_seconds = float(self.payload["window_seconds"])
        self.hop_seconds = float(self.payload["hop_seconds"])
        self.temperature = float(self.payload.get("temperature", 1.0))
        self.samples: deque[dict] = deque()
        self.last_inference = -float("inf")
        self.last_sequence: int | None = None
        self.sequence_gaps = 0

    def _uniform_window(self, now: float) -> np.ndarray:
        rows = list(self.samples)
        timestamps = np.asarray([float(row["timestamp"]) for row in rows])
        values = np.asarray([[float(row[key]) for key in SENSOR_KEYS] for row in rows])
        target = np.arange(
            now - self.window_seconds + 1.0 / self.sample_rate_hz,
            now + 0.5 / self.sample_rate_hz,
            1.0 / self.sample_rate_hz,
        )
        return np.column_stack([np.interp(target, timestamps, values[:, index]) for index in range(6)])

    def update(self, sample: dict) -> dict | None:
        now = float(sample["timestamp"])
        sequence = sample.get("sequence_id")
        if sequence is not None:
            sequence = int(sequence) & 0xFFFF
            if self.last_sequence is not None and sequence != ((self.last_sequence + 1) & 0xFFFF):
                self.sequence_gaps += (sequence - self.last_sequence - 1) & 0xFFFF
            self.last_sequence = sequence
        self.samples.append(sample)
        while self.samples and now - float(self.samples[0]["timestamp"]) > self.window_seconds + 0.25:
            self.samples.popleft()
        if now - self.last_inference < self.hop_seconds:
            return None
        if len(self.samples) < 16 or now - float(self.samples[0]["timestamp"]) < self.window_seconds * 0.95:
            return None
        self.last_inference = now
        window = self._uniform_window(now)
        quality = signal_quality(window)
        features = extract_window_features(window, self.sample_rate_hz)
        vector = pd.DataFrame(
            [[features.get(name, 0.0) for name in self.features]],
            columns=self.features,
        )
        raw = np.asarray(self.model.predict_proba(vector)[0], dtype=float)
        probability = calibrated_probabilities(raw, self.temperature)
        probabilities = {label: float(value) for label, value in zip(self.classes, probability)}
        target_actions = self.payload.get("target_actions", [])
        motion_probability = min(1.0, sum(probabilities.get(action, 0.0) for action in target_actions))
        action = max(target_actions, key=lambda label: probabilities.get(label, 0.0)) if target_actions else "unknown_motion"
        action_probability = probabilities.get(action, 0.0)
        if action_probability < float(self.payload.get("action_threshold", 0.65)):
            action = "unknown_motion"
        return {
            "timestamp": now,
            "motion_probability": motion_probability,
            "action": action,
            "action_probability": action_probability,
            "unknown_probability": probabilities.get("unknown_motion", 0.0),
            "signal_quality": quality.state,
            "sequence_gaps": self.sequence_gaps,
            "probabilities": probabilities,
        }


class RuntimeCoordinator:
    """Keep inference pure while forwarding its outputs to policy and storage."""

    def __init__(self, model_path: Path, database_path: Path | str | None = None):
        self.classifier = StreamingActivityClassifier(model_path)
        self.strategy = WorkoutStrategy()
        self.store = WorkoutStore(database_path) if database_path is not None else None
        self.last_result: dict | None = None

    def update(self, sample: dict) -> dict | None:
        inference = self.classifier.update(sample)
        if inference is None:
            return None
        snapshot, events = self.strategy.update(
            float(inference["timestamp"]),
            inference["probabilities"],
            str(inference["signal_quality"]),
        )
        if self.store:
            self.store.apply_events(events)
        self.last_result = {**inference, "strategy": snapshot.to_dict(), "events": events}
        return self.last_result

    def flush(self, timestamp: float) -> list[dict]:
        _, events = self.strategy.flush(timestamp)
        if self.store:
            self.store.apply_events(events)
        return events
