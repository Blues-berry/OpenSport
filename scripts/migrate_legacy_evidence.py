#!/usr/bin/env python3
"""Annotate historical labels with evidence tiers without touching raw IMU files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def migrate_document(document: dict) -> dict:
    result = dict(document)
    if result.get("schema_version") != "2.0":
        return result
    scope = result.get("annotation_scope")
    if scope == "session_weak":
        tier = "session_weak"
    elif scope == "full_recording":
        sources = {
            segment.get("label_source")
            for segment in result.get("segments", [])
            if segment.get("window_trainable")
        }
        tier = (
            "gold"
            if sources
            and sources <= {"operator_event", "video_review", "manual_timeline"}
            else "legacy_reviewed"
        )
    else:
        tier = "rejected"
    result["evidence_tier"] = tier
    result.setdefault("protocol_version", "legacy-pre-v1.0")
    quality = dict(result.get("annotation_quality", {}))
    if tier == "legacy_reviewed":
        quality["formal_evaluation_eligible"] = False
        quality["reason"] = (
            "历史数据按完整名称人工审核；缺少新规范现场事件，"
            "仅允许实验训练，不进入正式验证或测试。"
        )
    result["annotation_quality"] = quality
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_dir", type=Path)
    parser.add_argument("--quarantine", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    labels = sorted(args.labels_dir.glob("*.labels.json"))
    report = {"v2_migrated": 0, "legacy_quarantined": 0, "unchanged": 0}
    for path in labels:
        document = json.loads(path.read_text(encoding="utf-8-sig"))
        if document.get("schema_version") != "2.0":
            report["legacy_quarantined"] += 1
            if args.apply and args.quarantine:
                args.quarantine.mkdir(parents=True, exist_ok=True)
                destination = args.quarantine / path.name
                shutil.copy2(path, destination)
                if destination.read_bytes() != path.read_bytes():
                    raise RuntimeError(f"quarantine verification failed for {path.name}")
                path.unlink()
            continue
        migrated = migrate_document(document)
        if migrated == document:
            report["unchanged"] += 1
            continue
        report["v2_migrated"] += 1
        if args.apply:
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_text(
                json.dumps(migrated, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
