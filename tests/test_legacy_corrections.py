from __future__ import annotations

from pathlib import Path

from opensport.data import (
    capture_override,
    load_legacy_corrections,
    participant_for_device,
)


ROOT = Path(__file__).resolve().parents[1]


def test_reviewed_device_order_and_capture_overrides_are_stable() -> None:
    corrections = load_legacy_corrections(
        ROOT / "config" / "legacy_capture_corrections.json"
    )
    folder = "李浩宇+车学远-"
    assert participant_for_device(
        folder, "WT901BLE11(f7-36-ca-b7-cb-34)", corrections
    ) == "李浩宇"
    assert participant_for_device(
        folder, "WT22222(f6-b1-93-b5-2b-23)", corrections
    ) == "车学远"
    assert capture_override(
        r"0722\李浩宇+车学远-\14-48-44-727", corrections
    )["activity_id"] == "head_up"
    assert capture_override(
        "0721/毛泽凯+车学远-坐姿/14-56-19-426", corrections
    )["decision"] == "rejected"
