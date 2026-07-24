"""Stable public types shared by ingestion, training and live inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


EvidenceTier = Literal["gold", "legacy_reviewed", "session_weak", "rejected"]
QualityStatus = Literal["accepted", "accepted_with_note", "recollect_required"]
EarSide = Literal["left", "right", "unknown"]


@dataclass(frozen=True)
class ImuSample:
    timestamp_monotonic_ns: int
    sequence_id: int
    device_id: str
    ear_side: EarSide
    acc_x_mps2: float
    acc_y_mps2: float
    acc_z_mps2: float
    gyro_x_rad_s: float
    gyro_y_rad_s: float
    gyro_z_rad_s: float
    mag_x_ut: float | None = None
    mag_y_ut: float | None = None
    mag_z_ut: float | None = None
    temperature_c: float | None = None
    quaternion_wxyz: tuple[float, float, float, float] | None = None
    euler_degrees: tuple[float, float, float] | None = None
    orientation_source: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DeviceStream:
    device_id: str
    ear_side: EarSide
    csv_file: str
    sha256: str
    sample_rate_configured_hz: float
    acceleration_unit: str
    angular_velocity_unit: str
    acceleration_range: float
    angular_velocity_range: float
    filter_config: str
    firmware_version: str
    axis_sign: tuple[int, int, int] = (1, 1, 1)


@dataclass(frozen=True)
class TrialManifest:
    schema_version: str
    protocol_version: str
    date: str
    session_id: str
    subject_id: str
    trial_id: str
    collector_id: str
    app_version: str
    label_file: str
    streams: tuple[DeviceStream, ...]
    replaces_trial_id: str | None = None


@dataclass(frozen=True)
class StreamQuality:
    device_id: str
    ear_side: EarSide
    row_count: int
    duration_seconds: float
    configured_rate_hz: float
    observed_rate_hz: float
    sequence_gap_count: int
    sequence_reorder_count: int
    long_gap_count: int
    clipped_ratio: float
    sha256_valid: bool
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class QualityReport:
    schema_version: str
    session_id: str
    trial_id: str
    status: QualityStatus
    evidence_tier: EvidenceTier
    streams: tuple[StreamQuality, ...]
    issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActivityPrediction:
    schema_version: str
    timestamp: float
    wear_state: str
    signal_quality: str
    motion_state: str | None
    motion_probability: float
    exercise_state: str
    activity_family: str
    activity_id: str
    confidence: float
    workout_phase: str
    set_count: int
    session_id: str | None
    finalized: bool
    experimental: bool
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkoutEvent:
    event_id: str
    event_type: str
    timestamp: float
    session_id: str | None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PosturePrediction:
    schema_version: str
    timestamp: float
    wear_state: str
    posture_state: Literal["normal", "poor", "calibrating", "unavailable"]
    deviations: tuple[str, ...]
    relative_angles_degrees: dict[str, float]
    continuous_seconds: float
    alert: bool
    calibrated: bool
    yaw_reliability: Literal["reliable", "degraded", "unavailable"]
    confidence: float
    experimental: bool = True
    warning: str | None = None
    medical_diagnostic: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelBundleManifest:
    schema_version: str
    model_kind: Literal["activity", "posture"]
    model_family: str
    dataset_version: str
    feature_version: str
    label_schema_version: str
    taxonomy_version: str
    sample_rate_hz: float
    window_seconds: float
    hop_seconds: float
    classes: tuple[str, ...]
    metrics: dict[str, Any]
    code_version: str
    experimental: bool
