#!/usr/bin/env python3
"""Merge continuous IMU CSV fragments and build row-aligned label JSON files."""

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


CHINA_TZ = timezone(timedelta(hours=8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--csv-output-dir", required=True)
    parser.add_argument("--labels-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def safe_file_part(value: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "、", value).strip().rstrip(".")


def fragment_index(path: Path) -> tuple[int, str]:
    match = re.search(r"_(\d+)\.csv$", path.name, re.IGNORECASE)
    return (int(match.group(1)) if match else 0, path.name)


def parse_host_time(date_text: str, value: str) -> datetime:
    value = value.strip()
    if value in {"recording_start", "recording_end"}:
        raise ValueError("dynamic boundary must be resolved before parsing")
    parsed = datetime.strptime(f"{date_text} {value}", "%Y-%m-%d %H:%M:%S.%f")
    return parsed.replace(tzinfo=CHINA_TZ)


def csv_time_from_line(date_text: str, line: str) -> datetime:
    first_field = line.split(",", 1)[0].strip()
    return parse_host_time(date_text, first_field)


def first_and_last_data_time(path: Path, date_text: str) -> tuple[datetime, datetime]:
    first = None
    last = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        next(handle, None)
        for line in handle:
            if not line.strip():
                continue
            current = csv_time_from_line(date_text, line)
            if first is None:
                first = current
            last = current
    if first is None or last is None:
        raise ValueError(f"CSV 没有数据行：{path}")
    return first, last


def find_fragments(session_dir: Path, device: str) -> list[Path]:
    matches = [
        path
        for path in session_dir.rglob("*.csv")
        if device.lower() in path.name.lower()
    ]
    return sorted(matches, key=fragment_index)


def read_small_notes(session_dir: Path) -> list[dict]:
    notes = []
    for path in sorted(session_dir.rglob("*.txt")):
        if path.stat().st_size > 100_000 or "WT901" in path.name or "WT22222" in path.name:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig").strip()
        except UnicodeDecodeError:
            text = path.read_text(encoding="gb18030").strip()
        notes.append(
            {
                "file": str(path.relative_to(session_dir)),
                "content": text,
            }
        )
    return notes


def unique_destination(root: Path, base_name: str, reserved: set[str]) -> Path:
    candidate = root / f"{base_name}.csv"
    sequence = 2
    while str(candidate).lower() in reserved or candidate.exists():
        candidate = root / f"{base_name}-{sequence}.csv"
        sequence += 1
    reserved.add(str(candidate).lower())
    return candidate


def segment_from_definition(
    definition: dict,
    date_text: str,
    start: datetime,
    end: datetime,
) -> dict:
    segment = dict(definition)
    start_value = segment.pop("start")
    end_value = segment.pop("end")
    segment_start = start if start_value == "recording_start" else parse_host_time(date_text, start_value)
    segment_end = end if end_value == "recording_end" else parse_host_time(date_text, end_value)
    segment["start"] = max(segment_start, start)
    segment["end"] = min(segment_end, end)
    return segment


def build_segments(
    participant: dict,
    date_text: str,
    recording_start: datetime,
    recording_end: datetime,
) -> list[dict]:
    definitions = participant["segments"]
    mode = participant.get("timeline_mode", "absolute")
    segments: list[dict] = []

    if mode == "durations_from_recording_start":
        cursor = recording_start
        for definition in definitions:
            duration = timedelta(seconds=float(definition["duration_seconds"]))
            segment = {key: value for key, value in definition.items() if key != "duration_seconds"}
            segment["start"] = cursor
            segment["end"] = min(cursor + duration, recording_end)
            if segment["start"] < recording_end:
                segments.append(segment)
            cursor += duration
    else:
        for definition in definitions:
            segment = segment_from_definition(
                definition, date_text, recording_start, recording_end
            )
            if segment["start"] < segment["end"]:
                segments.append(segment)

    segments.sort(key=lambda item: item["start"])
    covered: list[dict] = []
    cursor = recording_start
    for segment in segments:
        if segment["start"] > cursor:
            covered.append(
                {
                    "start": cursor,
                    "end": segment["start"],
                    "label": "未标注",
                    "label_type": "unknown",
                    "confidence": "none",
                    "trainable": False,
                    "source": "automatic_gap_fill",
                }
            )
        if segment["start"] < cursor:
            segment["start"] = cursor
        if segment["start"] < segment["end"]:
            covered.append(segment)
            cursor = max(cursor, segment["end"])
    if cursor < recording_end:
        covered.append(
            {
                "start": cursor,
                "end": recording_end,
                "label": "未标注",
                "label_type": "unknown",
                "confidence": "none",
                "trainable": False,
                "source": "automatic_gap_fill",
            }
        )
    return covered


def iso(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds")


def validate_continuity(
    fragments: list[Path], date_text: str, max_gap_seconds: float
) -> tuple[datetime, datetime, list[dict]]:
    bounds = []
    for fragment in fragments:
        start, end = first_and_last_data_time(fragment, date_text)
        bounds.append({"path": fragment, "start": start, "end": end})
    gaps = []
    for previous, current in zip(bounds, bounds[1:]):
        gap = (current["start"] - previous["end"]).total_seconds()
        gaps.append(
            {
                "previous": previous["path"].name,
                "next": current["path"].name,
                "gap_seconds": round(gap, 6),
            }
        )
        if gap < -0.1 or gap > max_gap_seconds:
            raise ValueError(
                f"CSV 续录时间不连续：{previous['path'].name} -> "
                f"{current['path'].name}，间隔 {gap:.3f}s"
            )
    return bounds[0]["start"], bounds[-1]["end"], gaps


def merge_and_index(
    fragments: list[Path],
    temp_csv: Path,
    date_text: str,
    segments: list[dict],
) -> tuple[int, str]:
    for segment in segments:
        segment["start_row"] = None
        segment["end_row"] = None
        segment["row_count"] = 0

    row_index = 0
    segment_index = 0
    header_value = None
    with temp_csv.open("w", encoding="utf-8", newline="") as output:
        for fragment_number, fragment in enumerate(fragments):
            with fragment.open("r", encoding="utf-8-sig", newline="") as source:
                header = source.readline()
                if not header:
                    raise ValueError(f"CSV 为空：{fragment}")
                normalized_header = header.strip()
                if header_value is None:
                    header_value = normalized_header
                    output.write(header if header.endswith(("\n", "\r")) else header + "\n")
                elif normalized_header != header_value:
                    raise ValueError(f"CSV 表头不一致：{fragment}")

                for line in source:
                    if not line.strip():
                        continue
                    current_time = csv_time_from_line(date_text, line)
                    while (
                        segment_index < len(segments) - 1
                        and current_time >= segments[segment_index]["end"]
                    ):
                        segment_index += 1
                    segment = segments[segment_index]
                    if current_time < segment["start"]:
                        raise ValueError(
                            f"CSV 时间早于标签区间：{fragment.name} {current_time}"
                        )
                    if segment["start_row"] is None:
                        segment["start_row"] = row_index
                    segment["end_row"] = row_index
                    segment["row_count"] += 1
                    output.write(line if line.endswith(("\n", "\r")) else line + "\n")
                    row_index += 1
    return row_index, header_value or ""


def serialize_segments(segments: list[dict]) -> list[dict]:
    result = []
    for segment in segments:
        item = dict(segment)
        item["start_time"] = iso(item.pop("start"))
        item["end_time"] = iso(item.pop("end"))
        result.append(item)
    return result


def process_participant(
    root: Path,
    csv_output_dir: Path,
    labels_dir: Path,
    session_dir: Path,
    date_text: str,
    batch_code: str,
    participant: dict,
    notes: list[dict],
    reserved: set[str],
    apply_changes: bool,
) -> dict:
    fragments = find_fragments(session_dir, participant["device"])
    if not fragments:
        raise ValueError(
            f"{participant['name']} 找不到设备 {participant['device']} 的 CSV"
        )

    recording_start, recording_end, gaps = validate_continuity(
        fragments,
        date_text,
        float(participant.get("max_fragment_gap_seconds", 2.0)),
    )
    base_name = (
        f"{batch_code}-{safe_file_part(participant['name'])}-"
        f"{safe_file_part(participant['output_action'])}"
    )
    destination = unique_destination(csv_output_dir, base_name, reserved)
    label_destination = labels_dir / f"{destination.stem}.labels.json"
    segments = build_segments(
        participant, date_text, recording_start, recording_end
    )

    result = {
        "participant": participant["name"],
        "device": participant["device"],
        "source_fragments": [str(path) for path in fragments],
        "destination_csv": str(destination),
        "label_json": str(label_destination),
        "recording_start": iso(recording_start),
        "recording_end": iso(recording_end),
        "fragment_gaps": gaps,
        "status": "planned" if not apply_changes else "completed",
    }
    if not apply_changes:
        return result

    root.mkdir(parents=True, exist_ok=True)
    csv_output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    temp_csv = destination.with_suffix(destination.suffix + ".partial")
    temp_label = label_destination.with_suffix(label_destination.suffix + ".partial")
    try:
        row_count, header = merge_and_index(
            fragments, temp_csv, date_text, segments
        )
        document = {
            "schema_version": "1.0",
            "date": date_text,
            "participant": participant["name"],
            "device": participant["device"],
            "csv_file": os.path.relpath(destination, labels_dir),
            "csv_columns": next(csv.reader([header])),
            "row_indexing": "zero_based_excluding_header",
            "interval_semantics": "[start_time, end_time), final interval includes recording end",
            "recording": {
                "start_time": iso(recording_start),
                "end_time": iso(recording_end),
                "row_count": row_count,
                "source_fragments": [str(path) for path in fragments],
                "fragment_gaps": gaps,
            },
            "annotation_quality": participant["annotation_quality"],
            "planned_sequence": participant.get("planned_sequence", []),
            "source_notes": notes,
            "segments": serialize_segments(segments),
        }
        with temp_label.open("w", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_csv, destination)
        os.replace(temp_label, label_destination)
    except Exception:
        temp_csv.unlink(missing_ok=True)
        temp_label.unlink(missing_ok=True)
        raise
    return result


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    csv_output_dir = Path(args.csv_output_dir).resolve()
    manifest_path = Path(args.manifest).resolve()
    labels_dir = Path(args.labels_dir).resolve()
    report_path = Path(args.report).resolve()
    manifest = read_json(manifest_path)
    batch_dir = root / manifest["source_batch"]
    reserved = {str(path).lower() for path in csv_output_dir.glob("*.csv")}
    report = {
        "mode": "Apply" if args.apply else "DryRun",
        "root": str(root),
        "manifest": str(manifest_path),
        "csv_output_dir": str(csv_output_dir),
        "labels_dir": str(labels_dir),
        "processed_sessions": [],
        "skipped_sessions": [],
        "handled_source_folders": [],
    }

    for session in manifest["sessions"]:
        session_dir = batch_dir / session["folder"]
        if not session_dir.exists():
            report["skipped_sessions"].append(
                {"folder": str(session_dir), "reason": "源文件夹不存在"}
            )
            continue
        notes = read_small_notes(session_dir)
        participant_results = []
        try:
            for participant in session["participants"]:
                participant_results.append(
                    process_participant(
                        root=root,
                        csv_output_dir=csv_output_dir,
                        labels_dir=labels_dir,
                        session_dir=session_dir,
                        date_text=manifest["date"],
                        batch_code=manifest["batch_code"],
                        participant=participant,
                        notes=notes,
                        reserved=reserved,
                        apply_changes=args.apply,
                    )
                )
            if args.apply:
                shutil.rmtree(session_dir)
            report["processed_sessions"].append(
                {
                    "folder": str(session_dir),
                    "participants": participant_results,
                    "status": "completed" if args.apply else "planned",
                }
            )
            report["handled_source_folders"].append(str(session_dir))
        except Exception as exc:
            report["skipped_sessions"].append(
                {"folder": str(session_dir), "reason": str(exc)}
            )

    report["summary"] = {
        "processed_session_count": len(report["processed_sessions"]),
        "participant_csv_count": sum(
            len(item["participants"]) for item in report["processed_sessions"]
        ),
        "skipped_session_count": len(report["skipped_sessions"]),
    }
    if args.apply and batch_dir.exists() and not any(batch_dir.iterdir()):
        batch_dir.rmdir()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(report["summary"], ensure_ascii=False))
    return 0 if not report["skipped_sessions"] else 2


if __name__ == "__main__":
    sys.exit(main())
