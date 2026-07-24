"""Shared 50 Hz preprocessing and window features for training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from opensport.features import interpolate_to_grid


TARGET_RATE_HZ = 50.0
WINDOW_SECONDS = 4.0
HOP_SECONDS = 1.0
SENSOR_KEYS = ("ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps")


def _spectral(values: np.ndarray, fs: float) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    x = x - np.mean(x)
    power = np.abs(np.fft.rfft(x)) ** 2
    frequency = np.fft.rfftfreq(len(x), 1.0 / fs)
    valid = (frequency >= 0.2) & (frequency <= min(15.0, fs / 2))
    if not valid.any() or power[valid].sum() <= 1e-15:
        return 0.0, 0.0
    selected = power[valid]
    probability = selected / selected.sum()
    entropy = -(probability * np.log2(probability + 1e-15)).sum()
    entropy /= max(np.log2(len(probability)), 1.0)
    return float(frequency[valid][np.argmax(selected)]), float(entropy)


def _autocorrelation_period(values: np.ndarray, fs: float) -> float:
    x = np.asarray(values, dtype=float)
    x = x - np.mean(x)
    if np.std(x) < 1e-8:
        return 0.0
    correlation = np.correlate(x, x, mode="full")[len(x) - 1 :]
    low = max(1, int(fs / 4.0))
    high = min(len(correlation), int(fs / 0.3))
    if high <= low:
        return 0.0
    lag = low + int(np.argmax(correlation[low:high]))
    return float(lag / fs)


def _stats(prefix: str, values: np.ndarray, fs: float) -> dict[str, float]:
    x = np.asarray(values, dtype=float)
    dominant, entropy = _spectral(x, fs)
    derivative = np.diff(x) * fs
    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_rms": float(np.sqrt(np.mean(x**2))),
        f"{prefix}_range": float(np.ptp(x)),
        f"{prefix}_iqr": float(np.percentile(x, 75) - np.percentile(x, 25)),
        f"{prefix}_jerk_rms": float(np.sqrt(np.mean(derivative**2))) if len(derivative) else 0.0,
        f"{prefix}_dominant_hz": dominant,
        f"{prefix}_spectral_entropy": entropy,
        f"{prefix}_period_s": _autocorrelation_period(x, fs),
    }


def extract_window_features(samples: np.ndarray, fs: float = TARGET_RATE_HZ) -> dict[str, float]:
    """Extract the exact feature contract shared by offline and live paths."""
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] != 6 or len(values) < 16:
        raise ValueError("A feature window must contain at least 16 six-axis samples")
    result: dict[str, float] = {}
    prefixes = ("acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z")
    for prefix, column in zip(prefixes, values.T):
        result.update(_stats(prefix, column, fs))
    acceleration = values[:, :3]
    gyroscope = values[:, 3:]
    acc_magnitude = np.linalg.norm(acceleration, axis=1)
    gyro_magnitude = np.linalg.norm(gyroscope, axis=1)
    dynamic_acceleration = np.abs(acc_magnitude - np.median(acc_magnitude))
    result.update(_stats("acc_mag", acc_magnitude, fs))
    result.update(_stats("dynamic_acc", dynamic_acceleration, fs))
    result.update(_stats("gyro_mag", gyro_magnitude, fs))
    result["acc_sma"] = float(np.mean(np.sum(np.abs(acceleration - np.mean(acceleration, axis=0)), axis=1)))
    gravity = np.mean(acceleration, axis=0)
    gravity_norm = max(float(np.linalg.norm(gravity)), 1e-9)
    for axis, component in zip("xyz", gravity / gravity_norm):
        result[f"gravity_{axis}"] = float(component)
    def pair_correlation(left: np.ndarray, right: np.ndarray) -> float:
        left_centered = left - np.mean(left)
        right_centered = right - np.mean(right)
        denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
        return float(np.dot(left_centered, right_centered) / denominator) if denominator > 1e-12 else 0.0

    for family, matrix in (("acc", acceleration), ("gyro", gyroscope)):
        result[f"{family}_corr_xy"] = pair_correlation(matrix[:, 0], matrix[:, 1])
        result[f"{family}_corr_xz"] = pair_correlation(matrix[:, 0], matrix[:, 2])
        result[f"{family}_corr_yz"] = pair_correlation(matrix[:, 1], matrix[:, 2])
    return result


def uniform_resample(
    values: np.ndarray,
    duration_s: float,
    target_hz: float = TARGET_RATE_HZ,
    source_timestamps_s: np.ndarray | None = None,
) -> np.ndarray:
    """Resample a batched logger stream using recording duration, not duplicate timestamps."""
    source = np.asarray(values, dtype=float)
    if source.ndim != 2 or source.shape[1] != 6:
        raise ValueError("Expected a six-axis matrix")
    if len(source) < 2 or duration_s <= 0:
        return source.copy()
    output_count = max(2, int(round(duration_s * target_hz)) + 1)
    source_time = (
        np.asarray(source_timestamps_s, dtype=float)
        if source_timestamps_s is not None
        else np.linspace(0.0, duration_s, len(source))
    )
    source_time = source_time - source_time[0]
    if np.any(np.diff(source_time) < 0):
        raise ValueError("Source timestamps must be monotonic")
    unique_time, inverse = np.unique(source_time, return_inverse=True)
    if len(unique_time) != len(source_time):
        totals = np.zeros((len(unique_time), source.shape[1]), dtype=float)
        counts = np.zeros(len(unique_time), dtype=float)
        np.add.at(totals, inverse, source)
        np.add.at(counts, inverse, 1.0)
        source = totals / counts[:, None]
        source_time = unique_time
    target_time = np.arange(output_count, dtype=float) / target_hz
    target_time = target_time[target_time <= duration_s + 1e-9]
    return interpolate_to_grid(source_time, source, target_time)


def iter_feature_windows(
    samples: np.ndarray,
    fs: float = TARGET_RATE_HZ,
    window_seconds: float = WINDOW_SECONDS,
    hop_seconds: float = HOP_SECONDS,
) -> Iterable[tuple[float, float, dict[str, float]]]:
    window = max(16, int(round(window_seconds * fs)))
    hop = max(1, int(round(hop_seconds * fs)))
    for start in range(0, max(0, len(samples) - window + 1), hop):
        end = start + window
        yield start / fs, (end - 1) / fs, extract_window_features(samples[start:end], fs)


@dataclass(frozen=True)
class SignalQuality:
    state: str
    missing_ratio: float
    clipped_ratio: float


def signal_quality(samples: np.ndarray) -> SignalQuality:
    values = np.asarray(samples, dtype=float)
    missing = float(1.0 - np.isfinite(values).mean())
    clipped = float(
        np.mean(
            np.column_stack(
                [
                    np.abs(values[:, :3]) >= 15.9,
                    np.abs(values[:, 3:]) >= 1990.0,
                ]
            )
        )
    )
    if missing > 0.02 or clipped > 0.02:
        state = "poor"
    elif missing > 0 or clipped > 0.002:
        state = "fair"
    else:
        state = "good"
    return SignalQuality(state, missing, clipped)
