#!/usr/bin/env python3
"""Create full-recording label JSON files for conventionally named IMU CSVs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


CHINA_TZ = timezone(timedelta(hours=8))
REVIEW_PATTERN = re.compile(
    r"[、+]|耳机|掉落|不规范|提前终止|最后几秒|挂脖子|时走时跑|有头动"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--default-year", type=int, default=2026)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def parse_filename(path: Path, default_year: int) -> dict:
    standard = re.match(r"^(?P<mmdd>\d{4})-(?P<person>[^-]+)-(?P<action>.+)$", path.stem)
    if standard:
        mmdd = standard.group("mmdd")
        date_value = datetime.strptime(f"{default_year}{mmdd}", "%Y%m%d").date()
        return {
            "date": date_value.isoformat(),
            "date_source": "filename",
            "participant": standard.group("person"),
            "action": standard.group("action"),
            "filename_warning": None,
        }

    fallback = re.match(r"^(?P<person>[^-]+)-(?P<action>.+)$", path.stem)
    if not fallback:
        raise ValueError("文件名无法解析为“日期-姓名-动作”或“姓名-动作”")
    modified = datetime.fromtimestamp(path.stat().st_mtime)
    return {
        "date": modified.date().isoformat(),
        "date_source": "file_mtime",
        "participant": fallback.group("person"),
        "action": fallback.group("action"),
        "filename_warning": "文件名缺少日期，使用文件修改日期",
    }


def parse_time(date_text: str, value: str) -> datetime:
    parsed = datetime.strptime(
        f"{date_text} {value.strip()}", "%Y-%m-%d %H:%M:%S.%f"
    )
    return parsed.replace(tzinfo=CHINA_TZ)


def scan_csv(path: Path, date_text: str) -> dict:
    row_count = 0
    first_time = None
    last_time = None
    device = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("CSV 为空") from exc
        if len(header) < 2:
            raise ValueError("CSV 表头字段不足")
        for row in reader:
            if not row or not row[0].strip():
                continue
            if len(row) < 2:
                raise ValueError(f"第 {row_count + 2} 行字段不足")
            current = parse_time(date_text, row[0])
            if first_time is None:
                first_time = current
                device = row[1].strip()
            if last_time is not None and current < last_time:
                if (last_time - current) > timedelta(hours=12):
                    current += timedelta(days=1)
                else:
                    raise ValueError(f"第 {row_count + 2} 行时间倒序")
            last_time = current
            row_count += 1
    if row_count == 0 or first_time is None or last_time is None:
        raise ValueError("CSV 没有数据行")
    return {
        "header": header,
        "row_count": row_count,
        "start": first_time,
        "end": last_time,
        "device": device,
    }


def iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def build_document(
    csv_path: Path,
    labels_dir: Path,
    metadata: dict,
    scan: dict,
) -> dict:
    needs_review = bool(REVIEW_PATTERN.search(metadata["action"]))
    label_type = "mixed_or_annotated_activity" if needs_review else "activity"
    quality = {
        "status": "needs_review" if needs_review else "filename_derived",
        "trainable": not needs_review,
        "reason": (
            "文件名包含组合动作或异常备注，无法保证整个 CSV 是单一动作。"
            if needs_review
            else "文件名符合“日期-姓名-动作”，整段按该单一动作标注。"
        ),
    }
    if metadata["filename_warning"]:
        quality["status"] = "needs_review"
        quality["trainable"] = False
        quality["reason"] = metadata["filename_warning"]

    return {
        "schema_version": "1.0",
        "date": metadata["date"],
        "date_source": metadata["date_source"],
        "participant": metadata["participant"],
        "device": scan["device"],
        "csv_file": os.path.relpath(csv_path, labels_dir),
        "csv_columns": scan["header"],
        "row_indexing": "zero_based_excluding_header",
        "interval_semantics": "single full-recording interval",
        "recording": {
            "start_time": iso(scan["start"]),
            "end_time": iso(scan["end"]),
            "row_count": scan["row_count"],
        },
        "annotation_quality": quality,
        "segments": [
            {
                "start_time": iso(scan["start"]),
                "end_time": iso(scan["end"]),
                "start_row": 0,
                "end_row": scan["row_count"] - 1,
                "row_count": scan["row_count"],
                "label": metadata["action"],
                "label_type": label_type,
                "confidence": "low" if quality["status"] == "needs_review" else "medium",
                "trainable": quality["trainable"],
                "source": "filename",
            }
        ],
    }


def main() -> int:
    args = parse_args()
    training_dir = Path(args.training_dir).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    report_path = Path(args.report).resolve()
    csv_files = sorted(training_dir.glob("*.csv"))
    report = {
        "mode": "Apply" if args.apply else "DryRun",
        "training_dir": str(training_dir),
        "labels_dir": str(labels_dir),
        "existing_labels": [],
        "generated_or_planned": [],
        "skipped": [],
    }

    if args.apply:
        labels_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in csv_files:
        label_path = labels_dir / f"{csv_path.stem}.labels.json"
        if label_path.exists():
            report["existing_labels"].append(str(label_path))
            continue
        try:
            metadata = parse_filename(csv_path, args.default_year)
            scan = scan_csv(csv_path, metadata["date"])
            document = build_document(csv_path, labels_dir, metadata, scan)
            if args.apply:
                temp_path = label_path.with_suffix(label_path.suffix + ".partial")
                with temp_path.open("w", encoding="utf-8") as handle:
                    json.dump(document, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                os.replace(temp_path, label_path)
            report["generated_or_planned"].append(
                {
                    "csv": str(csv_path),
                    "label_json": str(label_path),
                    "participant": metadata["participant"],
                    "action": metadata["action"],
                    "row_count": scan["row_count"],
                    "start_time": iso(scan["start"]),
                    "end_time": iso(scan["end"]),
                    "annotation_quality": document["annotation_quality"],
                }
            )
        except Exception as exc:
            report["skipped"].append({"csv": str(csv_path), "reason": str(exc)})

    report["summary"] = {
        "csv_count": len(csv_files),
        "existing_label_count": len(report["existing_labels"]),
        "generated_or_planned_count": len(report["generated_or_planned"]),
        "skipped_count": len(report["skipped"]),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0 if not report["skipped"] else 2


if __name__ == "__main__":
    sys.exit(main())
