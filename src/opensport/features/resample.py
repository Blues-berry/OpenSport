"""Timestamp-aware resampling shared by offline and live paths."""

from __future__ import annotations

import numpy as np


FEATURE_VERSION = "imu-si-resample-v1"


def interpolate_to_grid(
    source_timestamps_seconds: np.ndarray,
    values: np.ndarray,
    target_timestamps_seconds: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source_timestamps_seconds, dtype=float)
    matrix = np.asarray(values, dtype=float)
    target = np.asarray(target_timestamps_seconds, dtype=float)
    if source.ndim != 1 or matrix.ndim != 2 or len(source) != len(matrix):
        raise ValueError("timestamps and values must describe the same 2-D stream")
    if len(source) < 2 or np.any(np.diff(source) <= 0):
        raise ValueError("source timestamps must be strictly increasing")
    if target.ndim != 1 or np.any(np.diff(target) <= 0):
        raise ValueError("target timestamps must be strictly increasing")
    return np.column_stack(
        [
            np.interp(target, source, matrix[:, index])
            for index in range(matrix.shape[1])
        ]
    )


def uniform_resample(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    target_rate_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    matrix = np.asarray(values, dtype=float)
    if timestamps.ndim != 1 or matrix.ndim != 2 or len(timestamps) != len(matrix):
        raise ValueError("timestamps and values must describe the same 2-D stream")
    if len(timestamps) < 2 or np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    if target_rate_hz <= 0:
        raise ValueError("target_rate_hz must be positive")
    step_ns = int(round(1_000_000_000 / target_rate_hz))
    target = np.arange(timestamps[0], timestamps[-1] + 1, step_ns, dtype=np.int64)
    seconds = (timestamps - timestamps[0]).astype(float) / 1_000_000_000
    target_seconds = (target - timestamps[0]).astype(float) / 1_000_000_000
    output = interpolate_to_grid(seconds, matrix, target_seconds)
    return target, output
