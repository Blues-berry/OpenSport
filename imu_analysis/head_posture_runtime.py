"""Streaming calibrated posture inference with continuous abnormal-duration tracking."""

from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from head_posture_features import POSTURE_SENSOR_KEYS, extract_posture_features, posture_baseline


def _quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def _quaternion_from_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = source / max(float(np.linalg.norm(source)), 1e-9)
    target = target / max(float(np.linalg.norm(target)), 1e-9)
    dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if dot < -0.999999:
        axis = np.cross(source, np.asarray([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(source, np.asarray([0.0, 1.0, 0.0]))
        axis /= max(float(np.linalg.norm(axis)), 1e-9)
        return np.asarray([0.0, *axis])
    quaternion = np.asarray([1.0 + dot, *np.cross(source, target)], dtype=float)
    return quaternion / max(float(np.linalg.norm(quaternion)), 1e-9)


def _euler_from_quaternion(quaternion: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quaternion / max(float(np.linalg.norm(quaternion)), 1e-9)
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(float(np.rad2deg(value)) for value in (roll, pitch, yaw))


class SixAxisRelativeOrientation:
    """Create calibration-relative orientation fields from the existing six-axis stream."""

    def __init__(self) -> None:
        self.gravity: np.ndarray | None = None
        self.baseline_gravity: np.ndarray | None = None
        self.gyro_bias = np.zeros(3, dtype=float)
        self.bias_samples: list[np.ndarray] = []
        self.yaw_radians = 0.0
        self.last_timestamp: float | None = None

    def reset(self) -> None:
        self.gravity = None
        self.baseline_gravity = None
        self.gyro_bias[:] = 0.0
        self.bias_samples.clear()
        self.yaw_radians = 0.0
        self.last_timestamp = None

    def finish_calibration(self) -> None:
        if self.gravity is not None:
            self.baseline_gravity = self.gravity / max(float(np.linalg.norm(self.gravity)), 1e-9)
        if self.bias_samples:
            self.gyro_bias = np.median(np.asarray(self.bias_samples), axis=0)
        self.yaw_radians = 0.0
        self.last_timestamp = None

    def update(self, sample: dict, calibrating: bool) -> dict:
        result = dict(sample)
        acceleration = np.asarray([sample["ax_g"], sample["ay_g"], sample["az_g"]], dtype=float)
        gyroscope = np.asarray([sample["gx_dps"], sample["gy_dps"], sample["gz_dps"]], dtype=float)
        norm = float(np.linalg.norm(acceleration))
        if norm > 1e-6:
            direction = acceleration / norm
            self.gravity = direction if self.gravity is None else 0.94 * self.gravity + 0.06 * direction
            self.gravity /= max(float(np.linalg.norm(self.gravity)), 1e-9)
        if self.gravity is None:
            self.gravity = np.asarray([0.0, 0.0, 1.0])

        timestamp = float(sample["timestamp"])
        if calibrating:
            self.bias_samples.append(gyroscope)
            quaternion = np.asarray([1.0, 0.0, 0.0, 0.0])
        else:
            if self.baseline_gravity is None:
                self.finish_calibration()
            dt = 0.0 if self.last_timestamp is None else float(np.clip(timestamp - self.last_timestamp, 0.0, 0.1))
            vertical_rate_dps = float(np.dot(gyroscope - self.gyro_bias, self.gravity))
            self.yaw_radians += np.deg2rad(vertical_rate_dps) * dt
            tilt = _quaternion_from_vectors(self.baseline_gravity, self.gravity)
            half_yaw = self.yaw_radians / 2.0
            yaw = np.asarray(
                [
                    np.cos(half_yaw),
                    *(self.baseline_gravity * np.sin(half_yaw)),
                ]
            )
            quaternion = _quaternion_multiply(yaw, tilt)
            quaternion /= max(float(np.linalg.norm(quaternion)), 1e-9)
        self.last_timestamp = timestamp
        roll, pitch, yaw_degrees = _euler_from_quaternion(quaternion)
        result.update(
            {
                "roll_deg": roll,
                "pitch_deg": pitch,
                "yaw_deg": yaw_degrees,
                "q0": float(quaternion[0]),
                "q1": float(quaternion[1]),
                "q2": float(quaternion[2]),
                "q3": float(quaternion[3]),
            }
        )
        return result


class StreamingHeadPostureClassifier:
    """Calibrate while seated, classify stable windows, and time abnormal episodes."""

    def __init__(self, model_path: Path):
        with Path(model_path).open("rb") as handle:
            self.payload = pickle.load(handle)
        if self.payload.get("model_family") != "calibrated_lightgbm_head_posture":
            raise ValueError("Expected a calibrated_lightgbm_head_posture model")
        self.model = self.payload["model"]
        self.features = self.payload["features"]
        self.classes = self.payload["classes"]
        self.sample_rate_hz = float(self.payload["sample_rate_hz"])
        self.window_seconds = float(self.payload["window_seconds"])
        self.hop_seconds = float(self.payload["hop_seconds"])
        self.calibration_seconds = float(self.payload.get("calibration_seconds", 10.0))
        self.threshold = float(self.payload.get("probability_threshold", 0.60))
        self.abnormal_on_seconds = float(self.payload.get("abnormal_on_seconds", 3.0))
        self.normal_off_seconds = float(self.payload.get("normal_off_seconds", 2.0))
        self.samples: deque[dict] = deque()
        self.calibration_samples: list[dict] = []
        self.baseline: dict | None = None
        self.last_inference = -float("inf")
        self.candidate_label = "normal"
        self.candidate_since: float | None = None
        self.active_label = "normal"
        self.active_since: float | None = None
        self.last_abnormal_duration = 0.0
        self.orientation = SixAxisRelativeOrientation()

    def start_calibration(self) -> None:
        self.calibration_samples.clear()
        self.baseline = None
        self.samples.clear()
        self.active_label = "normal"
        self.active_since = None
        self.orientation.reset()

    @staticmethod
    def _matrix(rows: list[dict]) -> np.ndarray:
        matrix = []
        for row in rows:
            values = [float(row[key]) for key in POSTURE_SENSOR_KEYS[:9]]
            if all(key in row for key in POSTURE_SENSOR_KEYS[9:]):
                values.extend(float(row[key]) for key in POSTURE_SENSOR_KEYS[9:])
            matrix.append(values)
        widths = {len(row) for row in matrix}
        if len(widths) != 1:
            raise ValueError("Do not mix samples with and without quaternions")
        return np.asarray(matrix, dtype=float)

    def _uniform_window(self, now: float) -> np.ndarray:
        rows = list(self.samples)
        timestamps = np.asarray([float(row["timestamp"]) for row in rows])
        values = self._matrix(rows)
        target = np.arange(
            now - self.window_seconds + 1.0 / self.sample_rate_hz,
            now + 0.5 / self.sample_rate_hz,
            1.0 / self.sample_rate_hz,
        )
        return np.column_stack([np.interp(target, timestamps, values[:, index]) for index in range(values.shape[1])])

    def _update_episode(self, now: float, label: str) -> list[dict]:
        events: list[dict] = []
        if label != self.candidate_label:
            self.candidate_label = label
            self.candidate_since = now
        if self.candidate_since is None:
            self.candidate_since = now

        required = self.normal_off_seconds if label == "normal" else self.abnormal_on_seconds
        if label != self.active_label and now - self.candidate_since >= required:
            if self.active_label != "normal" and self.active_since is not None:
                ended_at = self.candidate_since
                duration = max(0.0, ended_at - self.active_since)
                self.last_abnormal_duration = duration
                events.append(
                    {
                        "type": "posture_ended",
                        "posture": self.active_label,
                        "ended_at": ended_at,
                        "duration_seconds": duration,
                    }
                )
            self.active_label = label
            self.active_since = None if label == "normal" else self.candidate_since
            if label != "normal":
                events.append({"type": "posture_started", "posture": label, "started_at": self.active_since})
        return events

    def update(self, sample: dict) -> dict | None:
        sample = self.orientation.update(sample, calibrating=self.baseline is None)
        now = float(sample["timestamp"])
        if self.baseline is None:
            self.calibration_samples.append(sample)
            elapsed = now - float(self.calibration_samples[0]["timestamp"])
            if elapsed < self.calibration_seconds:
                return {"state": "calibrating", "calibration_progress": min(1.0, elapsed / self.calibration_seconds)}
            matrix = self._matrix(self.calibration_samples)
            gyro_mean = float(np.mean(np.linalg.norm(matrix[:, 3:6], axis=1)))
            if gyro_mean > 8.0:
                self.calibration_samples.clear()
                self.orientation.reset()
                return {"state": "calibration_failed", "reason": "head_moved_during_calibration"}
            self.baseline = posture_baseline(matrix)
            self.orientation.finish_calibration()
            self.samples.clear()
            return {
                "state": "calibrated",
                "calibration_progress": 1.0,
                "orientation": {
                    "roll_degrees": sample["roll_deg"],
                    "pitch_degrees": sample["pitch_deg"],
                    "yaw_degrees": sample["yaw_deg"],
                },
            }

        self.samples.append(sample)
        while self.samples and now - float(self.samples[0]["timestamp"]) > self.window_seconds + 0.25:
            self.samples.popleft()
        if now - self.last_inference < self.hop_seconds:
            return None
        if len(self.samples) < 16 or now - float(self.samples[0]["timestamp"]) < self.window_seconds * 0.95:
            return None
        self.last_inference = now
        window = self._uniform_window(now)
        features = extract_posture_features(window, self.baseline, self.sample_rate_hz)
        vector = pd.DataFrame([[features[name] for name in self.features]], columns=self.features)
        probability = np.asarray(self.model.predict_proba(vector)[0], dtype=float)
        probabilities = {label: float(value) for label, value in zip(self.classes, probability)}
        predicted = self.classes[int(np.argmax(probability))]
        confidence = probabilities[predicted]
        stable = features["gyro_mean_dps"] <= 12.0 and features["dynamic_acc_std_g"] <= 0.08
        observed = predicted if confidence >= self.threshold and stable else self.active_label
        events = self._update_episode(now, observed)
        active_duration = (
            max(0.0, now - self.active_since)
            if self.active_label != "normal" and self.active_since is not None
            else 0.0
        )
        return {
            "state": "monitoring",
            "posture": self.active_label,
            "posture_name_zh": self.payload.get("class_names_zh", {}).get(self.active_label, self.active_label),
            "predicted_posture": predicted,
            "confidence": confidence,
            "stable": stable,
            "abnormal": self.active_label != "normal",
            "continuous_seconds": active_duration,
            "probabilities": probabilities,
            "events": events,
            "orientation": {
                "roll_degrees": float(window[-1, 6]),
                "pitch_degrees": float(window[-1, 7]),
                "yaw_degrees": float(window[-1, 8]),
            },
        }

    def flush(self, timestamp: float) -> list[dict]:
        if self.active_label == "normal" or self.active_since is None:
            return []
        duration = max(0.0, float(timestamp) - self.active_since)
        event = {
            "type": "posture_ended",
            "posture": self.active_label,
            "ended_at": float(timestamp),
            "duration_seconds": duration,
        }
        self.active_label = "normal"
        self.active_since = None
        return [event]
