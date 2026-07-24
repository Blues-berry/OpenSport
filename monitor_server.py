"""Serve the dual-device IMU dashboard and its local live-stream API."""
from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from bleak import BleakScanner


ROOT = Path(__file__).resolve().parent
IMU_OUTPUT = ROOT / "imu_output"
LIVE_CSV_PATH = IMU_OUTPUT / "live_imu.csv"
STATUS_PATH = IMU_OUTPUT / "live_status.json"
BLE_CSV_PATH = IMU_OUTPUT / "live_ble_imu.csv"
BLE_STATUS_PATH = IMU_OUTPUT / "live_ble_status.json"
SOURCES = {
    "witmonitor": (LIVE_CSV_PATH, STATUS_PATH),
    "ble": (BLE_CSV_PATH, BLE_STATUS_PATH),
}
TARGET_DEVICES = {
    "WT22222": "F6:B1:93:B5:2B:23",
    "WT901BLE11": "F7:36:CA:B7:CB:34",
}
STALE_SECONDS = 3.0


async def check_target_devices() -> list[dict]:
    """Report Windows BLE visibility without changing the receiver's device set."""
    found = {device.address.upper(): device for device in await BleakScanner.discover(timeout=5)}
    return [
        {
            "name": name,
            "address": address,
            "available": address in found,
            "rssi": getattr(found[address], "rssi", None) if address in found else None,
            "advertised_name": found[address].name if address in found else None,
        }
        for name, address in TARGET_DEVICES.items()
    ]


def _tail_rows(path: Path) -> list[dict[str, str]]:
    """Read a sufficiently large complete CSV tail while another process appends."""
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        with path.open("rb") as handle:
            header = handle.readline().decode("utf-8-sig")
            data_start = handle.tell()
            size = handle.seek(0, 2)
            start = max(data_start, size - 524288)
            handle.seek(start)
            tail = handle.read().decode("utf-8", errors="ignore")
        if start > data_start:
            tail = tail.split("\n", 1)[-1]
        return list(csv.DictReader(io.StringIO(header + tail)))
    except (OSError, UnicodeDecodeError, csv.Error):
        return []


def latest_rows_by_device(path: Path, limit: int = 180) -> dict[str, list[dict[str, str]]]:
    grouped = {name: [] for name in TARGET_DEVICES}
    for row in _tail_rows(path):
        device = row.get("device", "")
        if device in grouped:
            grouped[device].append(row)
    return {name: rows[-limit:] for name, rows in grouped.items()}


def read_status(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError):
        return {}


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def live_payload(
    csv_path: Path = LIVE_CSV_PATH,
    status_path: Path = STATUS_PATH,
    now_unix_s: float | None = None,
) -> dict[str, Any]:
    """Return two independent stream objects for the browser."""
    now = time.time() if now_unix_s is None else now_unix_s
    rows_by_device = latest_rows_by_device(csv_path)
    raw_status = read_status(status_path).get("devices", {})
    streams = {}
    for name, address in TARGET_DEVICES.items():
        rows = rows_by_device[name]
        status = raw_status.get(name, {}) if isinstance(raw_status, dict) else {}
        status_last_unix = _number(status.get("last_sample_unix_s"))
        last_unix = status_last_unix
        if rows:
            last_unix = max(last_unix, _number(rows[-1].get("timestamp_unix_s")))
        age = now - last_unix if last_unix > 0 else None
        receiver_state = str(status.get("state") or "waiting")
        # A passive WitMotion CSV bridge shares the same normalized output but
        # has no BLE receiver status.  Prefer its fresh sample over a stale
        # status file left by an earlier direct-BLE attempt.
        if rows and last_unix > status_last_unix:
            receiver_state = "file_stream"
        display_state = "stale" if age is None or age > STALE_SECONDS else receiver_state
        streams[name] = {
            "name": name,
            "address": address,
            "state": display_state,
            "last_sample_age_s": round(age, 3) if age is not None else None,
            "rows": rows,
            "stats": status,
        }
    return {"generated_unix_s": now, "streams": streams}


def source_payload(source: str) -> dict[str, Any]:
    """Read one isolated live source without mixing its samples with another."""
    csv_path, status_path = SOURCES.get(source, SOURCES["witmonitor"])
    payload = live_payload(csv_path, status_path)
    payload["source"] = source if source in SOURCES else "witmonitor"
    return payload


class DashboardHandler(SimpleHTTPRequestHandler):
    def send_json(self, body: dict, status: int = 200) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/api/live":
            source = parse_qs(urlparse(self.path).query).get("source", ["witmonitor"])[0]
            self.send_json(source_payload(source))
            return
        if route == "/api/devices":
            self.send_json({"devices": [
                {"name": name, "address": address} for name, address in TARGET_DEVICES.items()
            ]})
            return
        if self.path in {"/", "/index.html"}:
            self.path = "/monitor_dashboard.html"
        return super().do_GET()

    def do_POST(self) -> None:
        if urlparse(self.path).path == "/api/check-configured-devices":
            try:
                self.send_json({"devices": asyncio.run(check_target_devices())})
            except Exception as exc:
                self.send_json({"error": f"检查失败：{type(exc).__name__}"}, 503)
            return
        self.send_json({"error": "未找到接口"}, 404)

    def log_message(self, _format: str, *_args: object) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"双设备监测面板：http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
