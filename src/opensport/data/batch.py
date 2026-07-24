"""Batch-level acceptance audit required before a session enters training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from opensport.data.ingestion import TrialImporter


def audit_session(manifest_path: Path | str) -> dict[str, Any]:
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    trials = payload.get("trials", [])
    trial_ids = [str(item.get("trial_id", "")) for item in trials]
    duplicate_labels = sorted(
        {
            trial_id
            for trial_id in trial_ids
            if trial_id and trial_ids.count(trial_id) > 1
        }
    )
    missing_labels: list[str] = []
    reports: list[dict[str, Any]] = []
    violations: list[dict[str, str]] = []
    for trial_id in sorted(set(trial_ids)):
        try:
            report, _ = TrialImporter(path, trial_id).audit()
            reports.append(report.to_dict())
            if report.status == "recollect_required":
                violations.append(
                    {"trial_id": trial_id, "reason": "recollect_required"}
                )
        except FileNotFoundError as error:
            missing_labels.append(trial_id)
            violations.append({"trial_id": trial_id, "reason": str(error)})
        except (OSError, ValueError, json.JSONDecodeError) as error:
            violations.append({"trial_id": trial_id, "reason": str(error)})
    for duplicate in duplicate_labels:
        violations.append(
            {"trial_id": duplicate, "reason": "duplicate_trial_id"}
        )
    return {
        "schema_version": "1.0",
        "session_id": payload.get("session_id"),
        "missing_labels": missing_labels,
        "duplicate_labels": duplicate_labels,
        "violation_count": len(violations),
        "violations": violations,
        "reports": reports,
        "accepted": (
            not missing_labels
            and not duplicate_labels
            and not violations
        ),
    }
