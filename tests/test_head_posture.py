from __future__ import annotations

import numpy as np

from head_posture_features import angle_difference_degrees, extract_posture_features, posture_baseline
from head_posture_runtime import SixAxisRelativeOrientation, StreamingHeadPostureClassifier


def test_angle_difference_wraps_at_180_degrees() -> None:
    values = np.asarray([179.0, -179.0, -175.0])
    difference = angle_difference_degrees(values, 178.0)
    assert np.allclose(difference, [1.0, 3.0, 7.0])


def test_calibrated_features_are_relative_to_neutral() -> None:
    neutral = np.zeros((150, 9), dtype=float)
    neutral[:, 2] = 1.0
    neutral[:, 6:9] = [10.0, 20.0, 170.0]
    baseline = posture_baseline(neutral)
    tilted = neutral.copy()
    tilted[:, 6:9] += [15.0, -25.0, 20.0]
    features = extract_posture_features(tilted, baseline)
    assert abs(features["relative_roll_mean"] - 15.0) < 1e-6
    assert abs(features["relative_pitch_mean"] + 25.0) < 1e-6
    assert abs(features["relative_yaw_mean"] - 20.0) < 1e-6


def test_gravity_change_is_orientation_relative() -> None:
    neutral = np.zeros((150, 9), dtype=float)
    neutral[:, 2] = 1.0
    baseline = posture_baseline(neutral)
    changed = neutral.copy()
    changed[:, 0] = 1.0
    changed[:, 2] = 0.0
    features = extract_posture_features(changed, baseline)
    assert abs(features["gravity_change_degrees"] - 90.0) < 1e-6


def test_relative_quaternion_features_handle_angle_wrapping() -> None:
    neutral = np.zeros((150, 13), dtype=float)
    neutral[:, 2] = 1.0
    neutral[:, 9] = 1.0
    baseline = posture_baseline(neutral)
    rotated = neutral.copy()
    half_angle = np.deg2rad(15.0)
    rotated[:, 9] = np.cos(half_angle)
    rotated[:, 10] = np.sin(half_angle)
    features = extract_posture_features(rotated, baseline)
    assert abs(features["rotation_vector_x_mean"] - 30.0) < 1e-6
    assert abs(features["rotation_vector_y_mean"]) < 1e-6


def test_abnormal_episode_records_continuous_duration() -> None:
    runtime = StreamingHeadPostureClassifier.__new__(StreamingHeadPostureClassifier)
    runtime.abnormal_on_seconds = 3.0
    runtime.normal_off_seconds = 2.0
    runtime.candidate_label = "normal"
    runtime.candidate_since = None
    runtime.active_label = "normal"
    runtime.active_since = None
    runtime.last_abnormal_duration = 0.0

    assert runtime._update_episode(0.0, "head_down") == []
    events = runtime._update_episode(3.0, "head_down")
    assert events == [{"type": "posture_started", "posture": "head_down", "started_at": 0.0}]
    assert runtime._update_episode(10.0, "normal") == []
    events = runtime._update_episode(12.0, "normal")
    assert events[0]["type"] == "posture_ended"
    assert events[0]["ended_at"] == 10.0
    assert events[0]["duration_seconds"] == 10.0


def test_six_axis_stream_is_augmented_with_relative_orientation() -> None:
    orientation = SixAxisRelativeOrientation()
    for index in range(50):
        orientation.update(
            {
                "timestamp": index / 50,
                "ax_g": 0.0,
                "ay_g": 0.0,
                "az_g": 1.0,
                "gx_dps": 0.1,
                "gy_dps": -0.2,
                "gz_dps": 0.05,
            },
            calibrating=True,
        )
    orientation.finish_calibration()
    output = None
    for index in range(50, 100):
        output = orientation.update(
            {
                "timestamp": index / 50,
                "ax_g": 0.0,
                "ay_g": 1.0,
                "az_g": 0.0,
                "gx_dps": 0.1,
                "gy_dps": -0.2,
                "gz_dps": 0.05,
            },
            calibrating=False,
        )
    assert output is not None
    assert all(key in output for key in ("roll_deg", "pitch_deg", "yaw_deg", "q0", "q1", "q2", "q3"))
    assert abs(output["roll_deg"]) + abs(output["pitch_deg"]) > 20.0
