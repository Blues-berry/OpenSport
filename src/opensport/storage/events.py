"""Idempotent event identifiers shared by storage implementations."""

from __future__ import annotations

import hashlib
import json


def stable_event_id(event: dict) -> str:
    payload = json.dumps(
        event, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
