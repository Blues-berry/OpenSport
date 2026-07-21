"""Regression tests for the direct BLE receiver and dual-stream API."""
from __future__ import annotations

import csv
import json
import struct
import tempfile
import unittest
from pathlib import Path

from monitor_server import TARGET_DEVICES, live_payload
from realtime_ble_imu import (
    FRAME_HEADER,
    LIVE_FIELDS,
    FrameDecoder,
    NotificationClock,
    RealtimeNormalizer,
)


def make_frame(seed: int = 0) -> bytes:
    return FRAME_HEADER + struct.pack(
        "<9h", 2048 + seed, -1024, 16384, 100, 200, -300, 1000, -2000, 3000
    )


class FrameDecoderTests(unittest.TestCase):
    def test_recovers_from_noise_and_chunk_boundaries(self) -> None:
        decoder = FrameDecoder()
        first, second = make_frame(), make_frame(1)
        self.assertEqual(list(decoder.feed(b"noise\x55" + first[:1])), [])
        samples = list(decoder.feed(first[1:9]))
        self.assertEqual(samples, [])
        samples = list(decoder.feed(first[9:] + b"ignored\x55\x71" + second))
        self.assertEqual(len(samples), 2)
        self.assertAlmostEqual(samples[0]["ax_g"], 1.0)
        self.assertGreater(decoder.discarded_bytes, 0)

    def test_batched_clock_is_monotonic_and_uses_arrival_rate(self) -> None:
        clock = NotificationClock(100.0)
        first = clock.assign(1, 10.0)
        second = clock.assign(1, 10.00125)
        batch = clock.assign(8, 10.01125)
        points = first + second + batch
        self.assertTrue(all(right > left for left, right in zip(points, points[1:])))
        self.assertGreater(clock.rate_hz, 500.0)

    def test_normalizer_outputs_about_target_rate(self) -> None:
        normalizer = RealtimeNormalizer(100.0, 0.06)
        sample = list(FrameDecoder().feed(make_frame()))[0]
        normalized = []
        for index in range(801):
            raw = sample.copy()
            raw["timestamp_monotonic_s"] = index / 800.0
            normalized.extend(normalizer.feed(raw))
        self.assertGreaterEqual(len(normalized), 99)
        self.assertLessEqual(len(normalized), 102)
        self.assertTrue(all(
            right["timestamp_monotonic_s"] > left["timestamp_monotonic_s"]
            for left, right in zip(normalized, normalized[1:])
        ))


class LiveApiTests(unittest.TestCase):
    def test_api_groups_rows_and_marks_old_stream_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "live.csv"
            status_path = root / "status.json"
            rows = []
            for name in TARGET_DEVICES:
                for index in range(3):
                    row = {field: "" for field in LIVE_FIELDS}
                    row.update({
                        "device": name,
                        "timestamp_unix_s": str(100.0 if name == "WT22222" else 90.0),
                        "ax_g": "0.1", "ay_g": "0.2", "az_g": "1.0",
                    })
                    rows.append(row)
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=LIVE_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            status_path.write_text(json.dumps({"devices": {
                "WT22222": {"state": "live", "last_sample_unix_s": 100.0},
                "WT901BLE11": {"state": "live", "last_sample_unix_s": 90.0},
            }}), encoding="utf-8")

            payload = live_payload(csv_path, status_path, now_unix_s=101.0)
            self.assertEqual(len(payload["streams"]["WT22222"]["rows"]), 3)
            self.assertEqual(len(payload["streams"]["WT901BLE11"]["rows"]), 3)
            self.assertEqual(payload["streams"]["WT22222"]["state"], "live")
            self.assertEqual(payload["streams"]["WT901BLE11"]["state"], "stale")


if __name__ == "__main__":
    unittest.main()
