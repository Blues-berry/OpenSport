"""Versioned candidate/champion model discovery without import-time loading."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ModelUnavailableError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class ResolvedModel:
    path: Path
    role: str
    experimental: bool
    warning: str | None


class ModelRegistry:
    def __init__(self, root: Path | str = Path("imu_output/models")) -> None:
        self.root = Path(root)

    def resolve(self, kind: str, allow_experimental: bool = True) -> ResolvedModel | None:
        filename = "activity_model.pkl" if kind == "activity" else "head_posture_model.pkl"
        champion = self.root / kind / "champion" / filename
        if champion.exists():
            return ResolvedModel(champion, "champion", False, None)
        candidate = self.root / kind / "candidate" / filename
        if allow_experimental and candidate.exists():
            return ResolvedModel(
                candidate,
                "candidate",
                True,
                "实验模型/低于正式验收标准",
            )
        return None

    def load(self, kind: str, allow_experimental: bool = True) -> tuple[dict[str, Any], ResolvedModel]:
        resolved = self.resolve(kind, allow_experimental)
        if resolved is None:
            raise ModelUnavailableError(f"No {kind} model is registered")
        with resolved.path.open("rb") as handle:
            payload = pickle.load(handle)
        self._validate_payload(kind, payload)
        return payload, resolved

    @staticmethod
    def _validate_payload(kind: str, payload: dict[str, Any]) -> None:
        required = {
            "model",
            "features",
            "classes",
            "feature_version",
            "label_schema_version",
            "taxonomy_version",
            "sample_rate_hz",
            "window_seconds",
            "hop_seconds",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ValueError(f"Incompatible {kind} model bundle; missing {missing}")
        expected = "lightgbm_multiclass" if kind == "activity" else "calibrated_lightgbm_head_posture"
        if payload.get("model_family") != expected:
            raise ValueError(f"Incompatible {kind} model_family")
        supported_feature = {
            "activity": "activity-window-v2",
            "posture": "head-posture-relative-v2",
        }[kind]
        if payload.get("feature_version") != supported_feature:
            raise ValueError(
                f"Incompatible {kind} feature_version: "
                f"{payload.get('feature_version')!r}"
            )

    def status(self) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for kind in ("activity", "posture"):
            resolved = self.resolve(kind)
            output[kind] = (
                {
                    "available": True,
                    "path": str(resolved.path),
                    "role": resolved.role,
                    "experimental": resolved.experimental,
                    "warning": resolved.warning,
                }
                if resolved
                else {"available": False, "role": "missing", "experimental": False}
            )
        return output
