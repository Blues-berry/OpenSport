"""实时读取 WitMotion BLE IMU（FFE5/FFE4）并解码为 CSV 数值。

示例：
  python realtime_ble_imu.py
  python realtime_ble_imu.py --csv live_imu.csv

运行前请关闭 WitMotion 上位机，避免它占用 BLE 连接。
"""
import argparse
import asyncio
import csv
import struct
import time
from pathlib import Path
from typing import Optional

from bleak import BleakClient, BleakScanner

NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"
DEVICES = {
    "WT22222": "F6:B1:93:B5:2B:23",
    "WT901BLE11": "F7:36:CA:B7:CB:34",
}
FRAME_HEADER = b"\x55\x61"
FRAME_SIZE = 20


def decode_frame(frame: bytes) -> dict:
    """解码 55 61 + 9 个 little-endian int16 的实时姿态帧。"""
    if len(frame) != FRAME_SIZE or not frame.startswith(FRAME_HEADER):
        raise ValueError("不是有效的 20 字节 55 61 帧")
    ax, ay, az, gx, gy, gz, roll, pitch, yaw = struct.unpack("<9h", frame[2:])
    return {
        "ax_g": ax * 16.0 / 32768.0,
        "ay_g": ay * 16.0 / 32768.0,
        "az_g": az * 16.0 / 32768.0,
        "gx_dps": gx * 2000.0 / 32768.0,
        "gy_dps": gy * 2000.0 / 32768.0,
        "gz_dps": gz * 2000.0 / 32768.0,
        "roll_deg": roll * 180.0 / 32768.0,
        "pitch_deg": pitch * 180.0 / 32768.0,
        "yaw_deg": yaw * 180.0 / 32768.0,
    }


class FrameDecoder:
    def __init__(self):
        self.buffer = bytearray()

    def feed(self, data: bytes):
        self.buffer.extend(data)
        while True:
            start = self.buffer.find(FRAME_HEADER)
            if start < 0:
                self.buffer[:] = self.buffer[-1:]
                return
            if start:
                del self.buffer[:start]
            if len(self.buffer) < FRAME_SIZE:
                return
            frame = bytes(self.buffer[:FRAME_SIZE])
            del self.buffer[:FRAME_SIZE]
            yield decode_frame(frame)


async def stream_device(name: str, address: str, writer):
    decoder = FrameDecoder()
    last_print_time = 0.0
    while True:
        try:
            device = await BleakScanner.find_device_by_address(address, timeout=15)
            if device is None:
                print(f"[{name}] 未发现，10 秒后重试")
                await asyncio.sleep(10)
                continue

            def on_notification(_sender, data: bytearray):
                nonlocal last_print_time
                for sample in decoder.feed(bytes(data)):
                    sample["timestamp_monotonic_s"] = time.monotonic()
                    sample["device"] = name
                    if sample["timestamp_monotonic_s"] - last_print_time >= 0.5:
                        line = (f"[{name}] a=({sample['ax_g']:.3f}, {sample['ay_g']:.3f}, "
                                f"{sample['az_g']:.3f}) g, "
                                f"w=({sample['gx_dps']:.1f}, {sample['gy_dps']:.1f}, "
                                f"{sample['gz_dps']:.1f}) °/s, "
                                f"rpy=({sample['roll_deg']:.1f}, {sample['pitch_deg']:.1f}, "
                                f"{sample['yaw_deg']:.1f})°")
                        print(line)
                        last_print_time = sample["timestamp_monotonic_s"]
                    if writer:
                        writer.writerow(sample)

            async with BleakClient(device, timeout=20) as client:
                print(f"[{name}] 已连接，开始接收")
                await client.start_notify(NOTIFY_UUID, on_notification)
                while client.is_connected:
                    await asyncio.sleep(1)
        except Exception as exc:
            print(f"[{name}] 连接/接收异常：{type(exc).__name__}: {exc}")
        await asyncio.sleep(3)


async def main(csv_path: Optional[str]):
    file = None
    writer = None
    if csv_path:
        file = Path(csv_path).open("w", newline="", encoding="utf-8-sig")
        writer = csv.DictWriter(file, fieldnames=[
            "timestamp_monotonic_s", "device", "ax_g", "ay_g", "az_g",
            "gx_dps", "gy_dps", "gz_dps", "roll_deg", "pitch_deg", "yaw_deg",
        ])
        writer.writeheader()
    try:
        await asyncio.gather(*(stream_device(n, a, writer) for n, a in DEVICES.items()))
    finally:
        if file:
            file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", help="可选：实时写入的 CSV 文件路径")
    args = parser.parse_args()
    asyncio.run(main(args.csv))
