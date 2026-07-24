"""Receive legacy or compact BLE IMU packets and run the demo activity stack."""

from __future__ import annotations

import argparse
import asyncio
import json
import struct
import time
from pathlib import Path

from activity_runtime import RuntimeCoordinator
from ble_protocol import MAGIC, decode_batch
from head_posture_runtime import StreamingHeadPostureClassifier


NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"
LEGACY_HEADER = b"\x55\x61"
LEGACY_FRAME_SIZE = 20


class LegacyDecoder:
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data: bytes) -> list[dict]:
        self.buffer.extend(data)
        output = []
        while True:
            start = self.buffer.find(LEGACY_HEADER)
            if start < 0:
                self.buffer[:] = self.buffer[-1:] if self.buffer[-1:] == b"\x55" else b""
                break
            if start:
                del self.buffer[:start]
            if len(self.buffer) < LEGACY_FRAME_SIZE:
                break
            frame = bytes(self.buffer[:LEGACY_FRAME_SIZE])
            del self.buffer[:LEGACY_FRAME_SIZE]
            ax, ay, az, gx, gy, gz, roll, pitch, yaw = struct.unpack(
                "<9h", frame[2:]
            )
            output.append(
                {
                    "ax_g": ax * 16.0 / 32768.0,
                    "ay_g": ay * 16.0 / 32768.0,
                    "az_g": az * 16.0 / 32768.0,
                    "gx_dps": gx * 2000.0 / 32768.0,
                    "gy_dps": gy * 2000.0 / 32768.0,
                    "gz_dps": gz * 2000.0 / 32768.0,
                    "roll_deg": roll * 180.0 / 32768.0,
                    "pitch_deg": pitch * 180.0 / 32768.0,
                    "yaw_deg": yaw * 180.0 / 32768.0,
                    "orientation_source": "hardware_euler",
                }
            )
        return output


def write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


async def run_device(args: argparse.Namespace) -> None:
    try:
        from bleak import BleakClient
    except ImportError as error:
        raise SystemExit("Install BLE support with: pip install bleak") from error

    runtime = RuntimeCoordinator(args.model, args.database)
    posture_runtime = StreamingHeadPostureClassifier(args.posture_model) if args.posture_model else None
    decoder = LegacyDecoder()
    legacy_sequence = 0
    last_assigned = time.monotonic()
    last_status_write = 0.0
    last_command_id = ""
    status = {
        "device": args.name,
        "state": "connecting",
        "last_result": None,
        "last_posture_result": None,
        "last_sample": None,
        "posture_supported": posture_runtime is not None,
        "updated_at": time.time(),
    }
    write_status(args.status, status)
    while True:
        try:
            async with BleakClient(args.address) as client:
                status["state"] = "live"
                write_status(args.status, status)

                def notification(_: object, value: bytearray) -> None:
                    nonlocal legacy_sequence, last_assigned, last_status_write
                    arrival = time.monotonic()
                    samples = []
                    if bytes(value).startswith(MAGIC):
                        decoded = decode_batch(bytes(value))
                        first = max(last_assigned + 1.0 / 50.0, arrival - (len(decoded) - 1) / 50.0)
                        for index, sample in enumerate(decoded):
                            samples.append(sample.runtime_dict(first + index / 50.0))
                    else:
                        decoded = decoder.feed(bytes(value))
                        first = max(last_assigned + 1.0 / args.legacy_rate, arrival - (len(decoded) - 1) / args.legacy_rate)
                        for index, axes in enumerate(decoded):
                            timestamp = first + index / args.legacy_rate
                            samples.append(
                                {
                                    "timestamp": timestamp,
                                    "sequence_id": legacy_sequence,
                                    **axes,
                                }
                            )
                            legacy_sequence = (legacy_sequence + 1) & 0xFFFF
                    for sample in samples:
                        last_assigned = float(sample["timestamp"])
                        result = runtime.update(sample)
                        posture_result = posture_runtime.update(sample) if posture_runtime else None
                        status["last_sample"] = {
                            key: sample[key]
                            for key in ("ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps")
                        }
                        if result is not None:
                            status["last_result"] = result
                        if posture_result is not None:
                            status["last_posture_result"] = posture_result
                        if (result is not None or posture_result is not None) and time.time() - last_status_write >= 0.2:
                            status["updated_at"] = time.time()
                            write_status(args.status, status)
                            last_status_write = status["updated_at"]

                await client.start_notify(args.notify_uuid, notification)
                while client.is_connected:
                    if posture_runtime and args.posture_command.exists():
                        try:
                            command = json.loads(args.posture_command.read_text(encoding="utf-8"))
                            command_id = str(command.get("request_id", ""))
                            if command_id and command_id != last_command_id and command.get("action") == "calibrate":
                                last_command_id = command_id
                                posture_runtime.start_calibration()
                                status["last_posture_result"] = {
                                    "state": "calibrating",
                                    "calibration_progress": 0.0,
                                }
                                status["updated_at"] = time.time()
                                write_status(args.status, status)
                        except (OSError, ValueError, json.JSONDecodeError):
                            pass
                    await asyncio.sleep(0.25)
            status["state"] = "reconnecting"
        except Exception as error:
            status["state"] = "reconnecting"
            status["error"] = f"{type(error).__name__}: {error}"
        status["updated_at"] = time.time()
        write_status(args.status, status)
        await asyncio.sleep(2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="IMU")
    parser.add_argument("--address", required=True)
    parser.add_argument("--notify-uuid", default=NOTIFY_UUID)
    parser.add_argument("--legacy-rate", type=float, default=100.0)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--posture-model", type=Path)
    parser.add_argument("--posture-command", type=Path, default=Path("imu_output/posture_command.json"))
    parser.add_argument("--database", type=Path, default=Path("imu_output/workouts.sqlite3"))
    parser.add_argument("--status", type=Path, default=Path("imu_output/activity_live_status.json"))
    args = parser.parse_args()
    asyncio.run(run_device(args))


if __name__ == "__main__":
    main()
