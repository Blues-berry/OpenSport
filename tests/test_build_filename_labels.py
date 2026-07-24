from __future__ import annotations

import json
import sys

from scripts import build_filename_labels


def test_apply_preserves_existing_reviewed_schema_v2_label(tmp_path, monkeypatch):
    training_dir = tmp_path / "activity"
    labels_dir = training_dir / "labels"
    training_dir.mkdir()
    labels_dir.mkdir()
    csv_path = training_dir / "0725-S001-T0007.csv"
    csv_path.write_text(
        "时间,设备\n"
        "10:00:00.000,D01\n"
        "10:00:10.000,D01\n",
        encoding="utf-8",
    )
    label_path = labels_dir / "0725-S001-T0007.labels.json"
    original = {
        "schema_version": "2.0",
        "taxonomy_version": build_filename_labels.taxonomy_version(),
        "date": "2026-07-25",
        "participant": "S001",
        "device": "D01",
        "csv_file": "../0725-S001-T0007.csv",
        "annotation_scope": "review_required",
        "window_trainable": False,
        "evidence_tier": "rejected",
        "recording": {"duration_seconds": 10.0},
        "annotation_quality": {
            "status": "review_required",
            "reason": "manual review pending",
        },
        "segments": [],
        "review_sentinel": "manual-label-must-survive",
    }
    label_path.write_text(json.dumps(original), encoding="utf-8")
    report_path = tmp_path / "audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_filename_labels.py",
            "--training-dir",
            str(training_dir),
            "--labels-dir",
            str(labels_dir),
            "--report",
            str(report_path),
            "--apply",
        ],
    )

    assert build_filename_labels.main() == 0
    assert json.loads(label_path.read_text(encoding="utf-8")) == original
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["files"][0]["label_origin"] == "preserved_reviewed_v2"
