"""Binary posture policy on calibration-relative head orientation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PostureThresholds:
    pitch_degrees: float = 15.0
    roll_degrees: float = 15.0
    yaw_degrees: float = 15.0


class PosturePolicy:
    def __init__(self, thresholds: PostureThresholds | None = None) -> None:
        self.thresholds = thresholds or PostureThresholds()

    def classify(
        self,
        roll_degrees: float,
        pitch_degrees: float,
        yaw_degrees: float,
        yaw_reliable: bool,
    ) -> tuple[str, tuple[str, ...]]:
        deviations: list[str] = []
        if pitch_degrees <= -self.thresholds.pitch_degrees:
            deviations.append("head_down")
        elif pitch_degrees >= self.thresholds.pitch_degrees:
            deviations.append("head_up")
        if roll_degrees <= -self.thresholds.roll_degrees:
            deviations.append("head_tilt_left")
        elif roll_degrees >= self.thresholds.roll_degrees:
            deviations.append("head_tilt_right")
        if yaw_reliable:
            if yaw_degrees <= -self.thresholds.yaw_degrees:
                deviations.append("head_turn_left")
            elif yaw_degrees >= self.thresholds.yaw_degrees:
                deviations.append("head_turn_right")
        return ("poor" if deviations else "normal"), tuple(deviations)
