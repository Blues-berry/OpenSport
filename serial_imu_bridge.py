"""Bridge a WitMotion serial receiver (for example COM3) to the live web UI.

Close WitMonitor before running: Windows gives one process exclusive access to
each COM port.  The bridge expects the same 20-byte ``55 61`` frames used by
the BLE receiver and writes the dashboard's ``live_imu.csv`` stream.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import serial

from realtime_ble_imu import FrameDecoder, LiveExerciseClassifier, configured_devices


FIELDS = [
    "timestamp_monotonic_s", "device", "ax_g", "ay_g", "az_g",
    "gx_dps", "gy_dps", "gz_dps", "roll_deg", "pitch_deg", "yaw_deg",
    "inference_label", "exercise_probability",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM3", help="WitMotion serial receiver port")
    parser.add_argument("--baud", type=int, default=9600)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("imu_output/live_imu.csv"))
    parser.add_argument("--device", help="display name; defaults to the dashboard-selected device")
    parser.add_argument("--wait-for-port", action="store_true", help="when COM is occupied, keep retrying until it is released")
    parser.add_argument("--retry-seconds", type=float, default=3.0, help="retry interval used with --wait-for-port")
    args = parser.parse_args()

    configured = configured_devices()
    device_name = args.device or (next(iter(configured)) if len(configured) == 1 else "COM3-IMU")
    classifier = LiveExerciseClassifier(str(args.model))
    decoder = FrameDecoder()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    last_print = 0.0
    last_diagnostic = time.monotonic()
    bytes_received = 0
    frames_decoded = 0

    while True:
        try:
            port = serial.Serial(args.port, args.baud, timeout=0.05)
            break
        except serial.SerialException as exc:
            if not args.wait_for_port:
                raise SystemExit(
                    f"无法打开 {args.port}: {exc}\n请先关闭 WitMonitor，确认串口号和波特率后重试。"
                )
            print(f"{args.port} 正被占用或不可用，{args.retry_seconds:g} 秒后重试：{exc}")
            time.sleep(args.retry_seconds)

    print(f"已打开 {args.port} @ {args.baud} baud；显示设备：{device_name}")
    try:
        with args.csv.open("w", newline="", encoding="utf-8-sig") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDS)
            writer.writeheader()
            file.flush()
            while True:
                chunk = port.read(512)
                now = time.monotonic()
                if not chunk:
                    if now - last_diagnostic >= 5:
                        print(f"[{device_name}] 串口已打开，等待 IMU 数据帧…")
                        last_diagnostic = now
                    continue
                bytes_received += len(chunk)
                for sample in decoder.feed(chunk):
                    frames_decoded += 1
                    sample["timestamp_monotonic_s"] = time.monotonic()
                    sample["device"] = device_name
                    prediction = classifier.update(sample)
                    if prediction:
                        label, probability = prediction
                        print(f"[{device_name}] 实时判定：{label}（运动概率 {probability:.1%}）")
                    sample["inference_label"] = classifier.stable_label or ""
                    sample["exercise_probability"] = classifier.last_probability if classifier.last_probability is not None else ""
                    writer.writerow(sample)
                    file.flush()
                    if sample["timestamp_monotonic_s"] - last_print >= 0.5:
                        print(
                            f"[{device_name}] a=({sample['ax_g']:.3f}, {sample['ay_g']:.3f}, {sample['az_g']:.3f}) g, "
                            f"rpy=({sample['roll_deg']:.1f}, {sample['pitch_deg']:.1f}, {sample['yaw_deg']:.1f})°"
                        )
                        last_print = sample["timestamp_monotonic_s"]
                if now - last_diagnostic >= 5:
                    print(f"[{device_name}] 串口已收 {bytes_received} 字节，已解码 {frames_decoded} 个 55 61 帧")
                    last_diagnostic = now
    except KeyboardInterrupt:
        print("串口桥接已停止")
    finally:
        port.close()


if __name__ == "__main__":
    main()
