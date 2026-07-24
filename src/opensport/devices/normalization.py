"""Adapters from existing BLE dictionaries to the public SI sample contract."""

from __future__ import annotations

import math

from opensport.types import ImuSample


def normalize_runtime_sample(
    sample: dict,
    device_id: str,
    ear_side: str = "unknown",
) -> ImuSample:
    timestamp_seconds = float(
        sample.get("timestamp", sample.get("timestamp_monotonic_s"))
    )
    quaternion = (
        tuple(float(sample[key]) for key in ("q0", "q1", "q2", "q3"))
        if all(key in sample for key in ("q0", "q1", "q2", "q3"))
        else None
    )
    euler = (
        tuple(float(sample[key]) for key in ("roll_deg", "pitch_deg", "yaw_deg"))
        if all(key in sample for key in ("roll_deg", "pitch_deg", "yaw_deg"))
        else None
    )
    return ImuSample(
        timestamp_monotonic_ns=int(round(timestamp_seconds * 1_000_000_000)),
        sequence_id=int(sample.get("sequence_id", 0)),
        device_id=device_id,
        ear_side=ear_side,
        acc_x_mps2=float(sample["ax_g"]) * 9.80665,
        acc_y_mps2=float(sample["ay_g"]) * 9.80665,
        acc_z_mps2=float(sample["az_g"]) * 9.80665,
        gyro_x_rad_s=float(sample["gx_dps"]) * math.pi / 180.0,
        gyro_y_rad_s=float(sample["gy_dps"]) * math.pi / 180.0,
        gyro_z_rad_s=float(sample["gz_dps"]) * math.pi / 180.0,
        quaternion_wxyz=quaternion,
        euler_degrees=euler,
        orientation_source=sample.get("orientation_source"),
    )
