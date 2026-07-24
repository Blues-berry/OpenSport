#!/usr/bin/env python3
"""Convert legacy UTF-8 TSV .txt IMU samples to CSV and row-aligned JSON labels."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--training-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--batch-code", default="0717")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def parse_source_name(path: Path) -> dict:
    match = re.match(
        r"^(?P<action>.+?)\s*\((?P<sample>\d+)\)(?P<suffix>\d+(?:\.\d+)?)?$",
        path.stem,
    )
    if match:
        action = match.group("action").strip()
        suffix = match.group("suffix")
        if suffix:
            action = f"{action}{suffix}"
        sample_index = int(match.group("sample"))
        participant = f"未知受试者{sample_index}"
        participant_source = "原文件名括号编号，仅作为样本编号，真实身份未知"
    else:
        action = path.stem.strip()
        sample_index = None
        participant = "未知受试者"
        participant_source = "原文件名没有姓名或样本编号"
    if not action:
        raise ValueError("无法从文件名提取动作")
    return {
        "action": action,
        "sample_index": sample_index,
        "participant": participant,
        "participant_source": participant_source,
    }


def parse_timestamp(value: str) -> datetime:
    parsed = datetime.strptime(value.strip(), "%Y-%m-%d %H:%M:%S.%f")
    return parsed.replace(tzinfo=CHINA_TZ)


def iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def scan_tsv(path: Path) -> dict:
    first = None
    last = None
    previous = None
    regression_count = 0
    max_regression_seconds = 0.0
    row_count = 0
    device = None
    expected_columns = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as exc:
            raise ValueError("TXT 为空") from exc
        expected_columns = len(header)
        if expected_columns < 2:
            raise ValueError("TXT 表头字段不足")
        for row_number, row in enumerate(reader, start=2):
            if not row or not any(field.strip() for field in row):
                continue
            if len(row) != expected_columns:
                raise ValueError(
                    f"第 {row_number} 行列数为 {len(row)}，预期 {expected_columns}"
                )
            current = parse_timestamp(row[0])
            if first is None or current < first:
                first = current
            if last is None or current > last:
                last = current
            if previous is not None and current < previous:
                regression_count += 1
                max_regression_seconds = max(
                    max_regression_seconds,
                    (previous - current).total_seconds(),
                )
            if device is None:
                device = row[1].strip()
            previous = current
            row_count += 1
    if row_count == 0 or first is None or last is None:
        raise ValueError("TXT 没有数据行")
    return {
        "header": header,
        "columns": expected_columns,
        "row_count": row_count,
        "device": device,
        "start": first,
        "end": last,
        "regression_count": regression_count,
        "max_regression_seconds": round(max_regression_seconds, 6),
    }


def write_csv(source: Path, destination: Path) -> None:
    temp = destination.with_suffix(destination.suffix + ".partial")
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as input_handle:
            reader = csv.reader(input_handle, delimiter="\t")
            header = next(reader)
            indexed_rows = [
                (parse_timestamp(row[0]), original_index, row)
                for original_index, row in enumerate(reader)
                if row and any(field.strip() for field in row)
            ]
            indexed_rows.sort(key=lambda item: (item[0], item[1]))
            with temp.open("w", encoding="utf-8", newline="") as output_handle:
                writer = csv.writer(output_handle, lineterminator="\n")
                writer.writerow(header)
                writer.writerows(item[2] for item in indexed_rows)
        os.replace(temp, destination)
    except Exception:
        temp.unlink(missing_ok=True)
        raise


def build_label(
    source: Path,
    csv_path: Path,
    labels_dir: Path,
    metadata: dict,
    scan: dict,
) -> dict:
    return {
        "schema_version": "1.0",
        "date": scan["start"].date().isoformat(),
        "date_source": "TXT时间列",
        "participant": metadata["participant"],
        "participant_verified": False,
        "participant_source": metadata["participant_source"],
        "sample_index": metadata["sample_index"],
        "device": scan["device"],
        "csv_file": os.path.relpath(csv_path, labels_dir),
        "csv_columns": scan["header"],
        "row_indexing": "zero_based_excluding_header",
        "interval_semantics": "single full-recording interval",
        "conversion": {
            "source_txt": str(source),
            "source_format": "UTF-8 TSV",
            "destination_format": "UTF-8 CSV",
            "column_count": scan["columns"],
            "row_ordering": {
                "sorted_by": "时间列，时间相同时保持原始行序",
                "source_regression_count": scan["regression_count"],
                "source_max_regression_seconds": scan["max_regression_seconds"],
            },
        },
        "recording": {
            "start_time": iso(scan["start"]),
            "end_time": iso(scan["end"]),
            "row_count": scan["row_count"],
        },
        "annotation_quality": {
            "status": "filename_derived",
            "trainable": True,
            "reason": "动作由原TXT文件名提取；受试者真实身份未知，不影响动作分类标签。",
        },
        "segments": [
            {
                "start_time": iso(scan["start"]),
                "end_time": iso(scan["end"]),
                "start_row": 0,
                "end_row": scan["row_count"] - 1,
                "row_count": scan["row_count"],
                "label": metadata["action"],
                "label_type": "activity",
                "confidence": "medium",
                "trainable": True,
                "source": "TXT文件名",
            }
        ],
    }


def main() -> int:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    training_dir = Path(args.training_dir).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    report_path = Path(args.report).resolve()
    sources = sorted(source_dir.glob("*.txt"))
    report = {
        "mode": "Apply" if args.apply else "DryRun",
        "source_dir": str(source_dir),
        "training_dir": str(training_dir),
        "labels_dir": str(labels_dir),
        "converted_or_planned": [],
        "skipped": [],
    }
    plans = []

    for source in sources:
        try:
            metadata = parse_source_name(source)
            scan = scan_tsv(source)
            csv_name = (
                f"{args.batch_code}-{metadata['participant']}-"
                f"{metadata['action']}.csv"
            )
            csv_path = training_dir / csv_name
            label_path = labels_dir / f"{Path(csv_name).stem}.labels.json"
            if csv_path.exists() or label_path.exists():
                raise ValueError("目标 CSV 或标签 JSON 已存在")
            plans.append((source, csv_path, label_path, metadata, scan))
            report["converted_or_planned"].append(
                {
                    "source_txt": str(source),
                    "destination_csv": str(csv_path),
                    "label_json": str(label_path),
                    "participant": metadata["participant"],
                    "sample_index": metadata["sample_index"],
                    "action": metadata["action"],
                    "device": scan["device"],
                    "row_count": scan["row_count"],
                    "start_time": iso(scan["start"]),
                    "end_time": iso(scan["end"]),
                    "source_regression_count": scan["regression_count"],
                    "source_max_regression_seconds": scan["max_regression_seconds"],
                }
            )
        except Exception as exc:
            report["skipped"].append({"source_txt": str(source), "reason": str(exc)})

    created_csvs: list[Path] = []
    created_labels: list[Path] = []
    if args.apply and not report["skipped"]:
        training_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        try:
            for source, csv_path, label_path, metadata, scan in plans:
                write_csv(source, csv_path)
                created_csvs.append(csv_path)
                document = build_label(
                    source, csv_path, labels_dir, metadata, scan
                )
                temp_label = label_path.with_suffix(label_path.suffix + ".partial")
                with temp_label.open("w", encoding="utf-8") as handle:
                    json.dump(document, handle, ensure_ascii=False, indent=2)
                    handle.write("\n")
                os.replace(temp_label, label_path)
                created_labels.append(label_path)
            for source, *_ in plans:
                source.unlink()
            if source_dir.exists() and not any(source_dir.iterdir()):
                source_dir.rmdir()
        except Exception:
            for path in created_csvs + created_labels:
                path.unlink(missing_ok=True)
            raise

    report["summary"] = {
        "source_txt_count": len(sources),
        "converted_or_planned_count": len(report["converted_or_planned"]),
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
