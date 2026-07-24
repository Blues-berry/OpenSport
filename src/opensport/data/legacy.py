"""Explicit, reviewable rules for the pre-standard paired-person folders."""

from __future__ import annotations

import json
import re
from pathlib import Path


def load_legacy_corrections(path: Path | str) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if payload.get("schema_version") != "1.0":
        raise ValueError("legacy correction schema_version must be 1.0")
    return payload


def paired_participants(folder_name: str) -> tuple[str, str]:
    match = re.match(
        r"^(?P<first>[\u4e00-\u9fff]{2,4})\+"
        r"(?P<second>[\u4e00-\u9fff]{2,4})",
        folder_name,
    )
    if not match:
        raise ValueError(f"not a paired-person legacy folder: {folder_name}")
    return match.group("first"), match.group("second")


def participant_for_device(
    folder_name: str,
    device_id: str,
    corrections: dict,
) -> str:
    first, second = paired_participants(folder_name)
    order = corrections["participant_device_order"]
    if device_id.startswith(str(order["first"])):
        return first
    if device_id.startswith(str(order["second"])):
        return second
    raise ValueError(f"unreviewed legacy device {device_id!r}")


def capture_override(relative_capture: str, corrections: dict) -> dict | None:
    normalized = relative_capture.replace("\\", "/").strip("/")
    return corrections.get("capture_overrides", {}).get(normalized)
