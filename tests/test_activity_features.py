import numpy as np

from activity_features import extract_window_features, signal_quality, uniform_resample


def test_features_capture_periodicity_and_gravity():
    fs = 50.0
    time = np.arange(200) / fs
    samples = np.column_stack(
        [
            0.2 * np.sin(2 * np.pi * 1.5 * time),
            np.zeros_like(time),
            np.ones_like(time),
            20 * np.sin(2 * np.pi * 1.5 * time),
            np.zeros_like(time),
            np.zeros_like(time),
        ]
    )
    features = extract_window_features(samples, fs)
    assert features["acc_x_dominant_hz"] == 1.5
    assert features["gravity_z"] > 0.98
    assert features["acc_sma"] > 0
    assert signal_quality(samples).state == "good"


def test_resample_uses_duration_not_duplicate_logger_timestamps():
    source = np.column_stack([np.linspace(0, 1, 101)] * 6)
    result = uniform_resample(source, duration_s=2.0, target_hz=50.0)
    assert result.shape == (101, 6)
    assert result[-1, 0] == 1.0
