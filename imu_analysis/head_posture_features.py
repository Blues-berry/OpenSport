"""Feature contract for calibrated, seated head-posture classification."""

from __future__ import annotations

import numpy as np


POSTURE_RATE_HZ = 50.0
POSTURE_WINDOW_SECONDS = 3.0
POSTURE_HOP_SECONDS = 0.5
POSTURE_SENSOR_KEYS = (
    "ax_g", "ay_g", "az_g",
    "gx_dps", "gy_dps", "gz_dps",
    "roll_deg", "pitch_deg", "yaw_deg",
    "q0", "q1", "q2", "q3",
)


def circular_mean_degrees(values: np.ndarray) -> float:
    radians = np.deg2rad(np.asarray(values, dtype=float))
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))))


def angle_difference_degrees(values: np.ndarray, baseline: float) -> np.ndarray:
    return (np.asarray(values, dtype=float) - float(baseline) + 180.0) % 360.0 - 180.0


def _normalized_quaternions(values: np.ndarray) -> np.ndarray:
    quaternions = np.asarray(values, dtype=float)
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    return quaternions / np.maximum(norms, 1e-9)


def _mean_quaternion(values: np.ndarray) -> np.ndarray:
    quaternions = _normalized_quaternions(values)
    matrix = quaternions.T @ quaternions
    _, vectors = np.linalg.eigh(matrix)
    result = vectors[:, -1]
    return result if result[0] >= 0 else -result


def _relative_rotation_vectors(values: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    current = _normalized_quaternions(values)
    base = np.asarray(baseline, dtype=float)
    base /= max(float(np.linalg.norm(base)), 1e-9)
    # WitMotion exports q0 as scalar w. q_rel = inverse(q_baseline) * q_current.
    bw, bx, by, bz = base
    inverse = np.asarray([bw, -bx, -by, -bz])
    aw, ax, ay, az = np.broadcast_arrays(
        inverse[0], inverse[1], inverse[2], inverse[3], subok=True
    )
    cw, cx, cy, cz = current.T
    relative = np.column_stack(
        [
            aw * cw - ax * cx - ay * cy - az * cz,
            aw * cx + ax * cw + ay * cz - az * cy,
            aw * cy - ax * cz + ay * cw + az * cx,
            aw * cz + ax * cy - ay * cx + az * cw,
        ]
    )
    relative[relative[:, 0] < 0] *= -1
    vector = relative[:, 1:]
    magnitude = np.linalg.norm(vector, axis=1)
    angle = 2.0 * np.arctan2(magnitude, np.clip(relative[:, 0], -1.0, 1.0))
    scale = np.divide(angle, magnitude, out=np.full_like(angle, 2.0), where=magnitude > 1e-9)
    return np.rad2deg(vector * scale[:, None])


def posture_baseline(samples: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] not in {9, 13} or len(values) < 16:
        raise ValueError("Posture calibration needs at least 16 orientation samples")
    acceleration = np.nanmedian(values[:, :3], axis=0)
    acceleration /= max(float(np.linalg.norm(acceleration)), 1e-9)
    angles = [circular_mean_degrees(values[:, index]) for index in range(6, 9)]
    result = {"gravity": acceleration.tolist(), "angles": angles}
    if values.shape[1] == 13:
        result["quaternion"] = _mean_quaternion(values[:, 9:13]).tolist()
    return result


def extract_posture_features(samples: np.ndarray, baseline: dict, fs: float = POSTURE_RATE_HZ) -> dict[str, float]:
    values = np.asarray(samples, dtype=float)
    if values.ndim != 2 or values.shape[1] not in {9, 13} or len(values) < 16:
        raise ValueError("A posture window needs at least 16 orientation samples")
    if not np.isfinite(values).all():
        raise ValueError("Posture feature input contains non-finite values")

    result: dict[str, float] = {}
    relative_angles = np.column_stack(
        [
            angle_difference_degrees(values[:, 6 + index], baseline["angles"][index])
            for index in range(3)
        ]
    )
    for name, column in zip(("roll", "pitch", "yaw"), relative_angles.T):
        result[f"relative_{name}_mean"] = float(np.mean(column))
        result[f"relative_{name}_median"] = float(np.median(column))
        result[f"relative_{name}_std"] = float(np.std(column))
        result[f"relative_{name}_range"] = float(np.ptp(column))
        result[f"relative_{name}_last"] = float(column[-1])

    if values.shape[1] == 13 and "quaternion" in baseline:
        rotation_vectors = _relative_rotation_vectors(values[:, 9:13], np.asarray(baseline["quaternion"]))
        for name, column in zip(("x", "y", "z"), rotation_vectors.T):
            result[f"rotation_vector_{name}_mean"] = float(np.mean(column))
            result[f"rotation_vector_{name}_median"] = float(np.median(column))
            result[f"rotation_vector_{name}_std"] = float(np.std(column))
            result[f"rotation_vector_{name}_last"] = float(column[-1])
        result["rotation_degrees_mean"] = float(np.mean(np.linalg.norm(rotation_vectors, axis=1)))
        result["rotation_degrees_p95"] = float(np.percentile(np.linalg.norm(rotation_vectors, axis=1), 95))

    acceleration = values[:, :3]
    gyroscope = values[:, 3:6]
    gravity = np.mean(acceleration, axis=0)
    gravity /= max(float(np.linalg.norm(gravity)), 1e-9)
    baseline_gravity = np.asarray(baseline["gravity"], dtype=float)
    baseline_gravity /= max(float(np.linalg.norm(baseline_gravity)), 1e-9)
    result["gravity_change_degrees"] = float(
        np.rad2deg(np.arccos(np.clip(np.dot(gravity, baseline_gravity), -1.0, 1.0)))
    )
    for axis, value in zip("xyz", gravity - baseline_gravity):
        result[f"relative_gravity_{axis}"] = float(value)

    gyro_magnitude = np.linalg.norm(gyroscope, axis=1)
    acc_magnitude = np.linalg.norm(acceleration, axis=1)
    result["gyro_mean_dps"] = float(np.mean(gyro_magnitude))
    result["gyro_p95_dps"] = float(np.percentile(gyro_magnitude, 95))
    result["gyro_rms_dps"] = float(np.sqrt(np.mean(gyro_magnitude**2)))
    result["dynamic_acc_std_g"] = float(np.std(acc_magnitude))
    result["dynamic_acc_range_g"] = float(np.ptp(acc_magnitude))
    result["window_seconds"] = float((len(values) - 1) / fs)
    return result
