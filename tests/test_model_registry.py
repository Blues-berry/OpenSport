from __future__ import annotations

from pathlib import Path
import pickle

import pytest

from opensport.models.registry import ModelRegistry


def test_missing_registry_is_nonfatal(tmp_path: Path) -> None:
    registry = ModelRegistry(tmp_path / "models")
    assert registry.resolve("activity") is None
    assert registry.status()["activity"]["available"] is False


def test_candidate_is_explicitly_experimental(tmp_path: Path) -> None:
    candidate = tmp_path / "models" / "activity" / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "activity_model.pkl").write_bytes(b"placeholder")
    resolved = ModelRegistry(tmp_path / "models").resolve("activity")
    assert resolved is not None
    assert resolved.experimental
    assert "实验模型" in str(resolved.warning)


def test_registry_rejects_incompatible_feature_contract(tmp_path: Path) -> None:
    candidate = tmp_path / "models" / "activity" / "candidate"
    candidate.mkdir(parents=True)
    with (candidate / "activity_model.pkl").open("wb") as handle:
        pickle.dump(
            {
                "model": object(),
                "model_family": "lightgbm_multiclass",
                "features": [],
                "classes": [],
                "feature_version": "wrong",
                "label_schema_version": "2.0",
                "taxonomy_version": "x",
                "sample_rate_hz": 50,
                "window_seconds": 4,
                "hop_seconds": 1,
            },
            handle,
        )
    with pytest.raises(ValueError, match="feature_version"):
        ModelRegistry(tmp_path / "models").load("activity")
