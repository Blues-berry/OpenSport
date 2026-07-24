"""Serialize public dataclass contracts without leaking model objects."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def public_payload(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): public_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [public_payload(item) for item in value]
    return value
