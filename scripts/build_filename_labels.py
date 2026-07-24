#!/usr/bin/env python3
"""Build reviewed Schema v2 labels without inferring semantics from substrings.

The script scans files and timestamps, but semantic labels only come from the
explicit reviewed table in ``imu_analysis/label_schema.py``.  Unknown names are
emitted as non-trainable review items.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "imu_analysis"))

from label_schema import (  # noqa: E402
    CONTINUOUS_SESSION_TARGETS,
    LONG_ACTION_HINTS,
    SCHEMA_VERSION,
    SHORT_RECORDING_MAX_SECONDS,
    activity_name_zh,
    legacy_label_to_activity,
    parse_capture_name,
    reviewed_short_label,
    taxonomy_version,
    validate_v2_document,
)


CHINA_TZ = timezone(timedelta(hours=8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--default-year", type=int, default=2026)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--overwrite-v2",
        action="store_true",
        help=(
            "Regenerate existing Schema v2 labels. By default reviewed v2 labels "
            "are validated and preserved."
        ),
    )
    return parser.parse_args()


def parse_filename(path: Path, default_year: int) -> dict[str, Any]:
    date_code, participant, raw_action = parse_capture_name(path)
    if date_code != "unknown-date":
        date_value = datetime.strptime(f"{default_year}{date_code}", "%Y%m%d").date()
        return {
            "date": date_value.isoformat(),
            "date_source": "filename",
            "participant": participant,
            "raw_action": raw_action,
        }
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    return {
        "date": modified.date().isoformat(),
        "date_source": "file_mtime",
        "participant": participant,
        "raw_action": raw_action,
    }


def parse_time(date_text: str, value: str) -> datetime:
    value = value.strip().replace("\x00", "")
    match = re.search(
        r"(?:(\d{4})-(\d{1,2})-(\d{1,2})\s+)?"
        r"(\d{1,2}):(\d{1,2}):(\d{1,2})(?:\.(\d+))?",
        value,
    )
    if not match:
        raise ValueError(f"无法解析时间：{value!r}")
    if match.group(1):
        year, month, day = map(int, match.group(1, 2, 3))
    else:
        base = datetime.strptime(date_text, "%Y-%m-%d")
        year, month, day = base.year, base.month, base.day
    hour, minute, second = map(int, match.group(4, 5, 6))
    microsecond = int((match.group(7) or "0")[:6].ljust(6, "0"))
    parsed = datetime(year, month, day, hour, minute, second, microsecond)
    return parsed.replace(tzinfo=CHINA_TZ)


def scan_csv(path: Path, date_text: str) -> dict[str, Any]:
    row_count = 0
    first_time: datetime | None = None
    last_time: datetime | None = None
    device = ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader((line.replace("\x00", "") for line in handle))
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV 为空") from exc
        for row in reader:
            if not row or not row[0].strip():
                continue
            current = parse_time(date_text, row[0])
            if first_time is None:
                first_time = current
                device = row[1].strip() if len(row) > 1 else ""
            if last_time is not None and current < last_time:
                if (last_time - current) > timedelta(hours=12):
                    current += timedelta(days=1)
                else:
                    raise ValueError(f"第 {row_count + 2} 行时间倒序")
            last_time = current
            row_count += 1
    if not row_count or first_time is None or last_time is None:
        raise ValueError("CSV 没有数据行")
    return {
        "header": header,
        "row_count": row_count,
        "start": first_time,
        "end": last_time,
        "duration_seconds": (last_time - first_time).total_seconds(),
        "device": device,
    }


def iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def read_existing(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def legacy_evidence(existing: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = dict(existing.get("evidence", {}))
    for key in ("annotation_quality", "planned_sequence", "source_notes"):
        if existing.get(key):
            evidence[key] = existing[key]
    legacy_segments = []
    for segment in existing.get("segments", []):
        legacy_segments.append(
            {
                key: segment.get(key)
                for key in ("label", "label_type", "confidence", "start_time", "end_time")
                if segment.get(key) is not None
            }
        )
    if legacy_segments:
        evidence["legacy_segments"] = legacy_segments
    return evidence


def weak_targets(
    raw_action: str,
    participant: str,
    existing: dict[str, Any],
) -> dict[str, Any]:
    reviewed_session = CONTINUOUS_SESSION_TARGETS.get(participant, {})
    ordered = [
        dict(item)
        for item in reviewed_session.get(
            "ordered_activities", LONG_ACTION_HINTS.get(raw_action, [])
        )
    ]
    if raw_action == "连续健身" and not reviewed_session:
        seen: list[str] = []
        for segment in existing.get("segments", []):
            raw_label = str(segment.get("label", ""))
            activity_id = legacy_label_to_activity(raw_label)
            if activity_id and activity_id not in seen:
                ordered.append(
                    {
                        "activity_id": activity_id,
                        "raw_label": raw_label,
                        "source": "legacy_description",
                    }
                )
                seen.append(activity_id)
        if not ordered:
            for raw_label in existing.get("planned_sequence", []):
                activity_id = legacy_label_to_activity(str(raw_label))
                if activity_id:
                    ordered.append(
                        {
                            "activity_id": activity_id,
                            "raw_label": raw_label,
                            "source": "legacy_description",
                        }
                    )
    total_sets = sum(
        int(item["sets"]) for item in ordered if isinstance(item.get("sets"), int)
    )
    return {
        "ordered_activities": ordered,
        "total_sets": total_sets or None,
        "process": reviewed_session.get("process", []),
        "sequence_trainable": False,
        "window_boundaries_available": False,
        "usage": "session_level_validation_only",
    }


def build_document(
    csv_path: Path,
    labels_dir: Path,
    metadata: dict[str, Any],
    scan: dict[str, Any],
    existing: dict[str, Any],
) -> dict[str, Any]:
    duration = float(scan["duration_seconds"])
    common = {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": taxonomy_version(),
        "date": metadata["date"],
        "date_source": metadata["date_source"],
        "participant": metadata["participant"],
        "device": scan["device"],
        "csv_file": os.path.relpath(csv_path, labels_dir),
        "raw_action": metadata["raw_action"],
        "csv_columns": scan["header"],
        "row_indexing": "zero_based_excluding_header",
        "recording": {
            "start_time": iso(scan["start"]),
            "end_time": iso(scan["end"]),
            "duration_seconds": round(duration, 6),
            "row_count": scan["row_count"],
        },
        "evidence": legacy_evidence(existing),
    }
    if duration > SHORT_RECORDING_MAX_SECONDS:
        return {
            **common,
            "annotation_scope": "session_weak",
            "window_trainable": False,
            "annotation_quality": {
                "status": "reviewed_weak",
                "trainable": False,
                "reason": "记录超过180秒；仅保留会话级动作、数量、顺序和过程，不生成逐窗边界。",
            },
            "segments": [],
            "weak_targets": weak_targets(
                metadata["raw_action"], metadata["participant"], existing
            ),
        }

    reviewed = reviewed_short_label(metadata["raw_action"])
    if reviewed is None:
        return {
            **common,
            "annotation_scope": "review_required",
            "window_trainable": False,
            "annotation_quality": {
                "status": "review_required",
                "trainable": False,
                "reason": "完整原始动作名称尚未进入人工审核表。",
            },
            "segments": [],
            "weak_targets": {},
        }

    segment = {
        "start_time": iso(scan["start"]),
        "end_time": iso(scan["end"]),
        "start_s": 0.0,
        "end_s": round(duration, 6),
        "start_row": 0,
        "end_row": scan["row_count"] - 1,
        "row_count": scan["row_count"],
        "activity_id": reviewed["activity_id"],
        "activity_name_zh": activity_name_zh(reviewed["activity_id"]),
        "motion_state": reviewed["motion_state"],
        "wear_state": reviewed["wear_state"],
        "phase": reviewed["phase"],
        "window_trainable": reviewed["window_trainable"],
        "label_source": "reviewed_exact_action_table",
        "confidence": reviewed["confidence"],
        "review_note": reviewed["review_note"],
    }
    return {
        **common,
        "annotation_scope": "full_recording",
        "window_trainable": reviewed["window_trainable"],
        "annotation_quality": {
            "status": "reviewed",
            "trainable": reviewed["window_trainable"],
            "reason": reviewed["review_note"],
        },
        "segments": [segment],
        "weak_targets": {},
    }


def main() -> int:
    args = parse_args()
    training_dir = Path(args.training_dir).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    report_path = Path(args.report).resolve()
    csv_files = sorted(training_dir.glob("*.csv"))
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "taxonomy_version": taxonomy_version(),
        "mode": "apply" if args.apply else "dry_run",
        "files": [],
        "errors": [],
    }
    if args.apply:
        labels_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_files:
        label_path = labels_dir / f"{csv_path.stem}.labels.json"
        try:
            metadata = parse_filename(csv_path, args.default_year)
            scan = scan_csv(csv_path, metadata["date"])
            existing = read_existing(label_path)
            preserved_existing = (
                existing.get("schema_version") == SCHEMA_VERSION
                and not args.overwrite_v2
            )
            if preserved_existing:
                document = existing
            else:
                backup = labels_dir / "legacy_v1" / label_path.name
                evidence_source = (
                    read_existing(backup)
                    if existing.get("schema_version") == SCHEMA_VERSION
                    and backup.exists()
                    else existing
                )
                document = build_document(
                    csv_path, labels_dir, metadata, scan, evidence_source
                )
            errors = validate_v2_document(document)
            if errors:
                raise ValueError("; ".join(errors))
            if args.apply and not preserved_existing:
                if existing and existing.get("schema_version") != SCHEMA_VERSION:
                    backup = labels_dir / "legacy_v1" / label_path.name
                    backup.parent.mkdir(parents=True, exist_ok=True)
                    if not backup.exists():
                        shutil.copy2(label_path, backup)
                temporary = label_path.with_suffix(label_path.suffix + ".partial")
                temporary.write_text(
                    json.dumps(document, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(temporary, label_path)
            report["files"].append(
                {
                    "source_file": csv_path.name,
                    "label_origin": (
                        "preserved_reviewed_v2"
                        if preserved_existing
                        else "generated_review_skeleton"
                    ),
                    "duration_seconds": document["recording"]["duration_seconds"],
                    "annotation_scope": document["annotation_scope"],
                    "window_trainable": document["window_trainable"],
                    "activity_id": (
                        document["segments"][0]["activity_id"]
                        if document["segments"]
                        else None
                    ),
                    "motion_state": (
                        document["segments"][0]["motion_state"]
                        if document["segments"]
                        else None
                    ),
                    "wear_state": (
                        document["segments"][0]["wear_state"]
                        if document["segments"]
                        else None
                    ),
                    "weak_targets": document.get("weak_targets", {}),
                }
            )
        except Exception as exc:
            report["errors"].append({"source_file": csv_path.name, "error": str(exc)})
    csv_stems = {path.stem for path in csv_files}
    orphan_labels = [
        path.name
        for path in sorted(labels_dir.glob("*.labels.json"))
        if path.name[: -len(".labels.json")] not in csv_stems
    ]
    report["orphan_labels"] = [
        {
            "label_file": name,
            "status": "missing_source_csv",
            "window_trainable": False,
        }
        for name in orphan_labels
    ]
    counts: dict[str, int] = {}
    for row in report["files"]:
        counts[row["annotation_scope"]] = counts.get(row["annotation_scope"], 0) + 1
    report["summary"] = {
        "csv_count": len(csv_files),
        "inventory_count": len(csv_files) + len(orphan_labels),
        "labelled_count": len(report["files"]),
        "orphan_label_count": len(orphan_labels),
        "error_count": len(report["errors"]),
        "scope_counts": counts,
        "window_trainable_count": sum(bool(row["window_trainable"]) for row in report["files"]),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0 if not report["errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
