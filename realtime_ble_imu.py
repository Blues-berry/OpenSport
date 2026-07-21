"""Connect directly to two WitMotion BLE IMUs and run local inference.

Close WitMotion before starting this program: Windows gives one process the
BLE connection.  Both configured devices are always connected independently;
the dashboard selection never changes the capture set.

Example:
  python realtime_ble_imu.py --model imu_output/run_20260721/model/l2_logistic_model.pkl

This is intentionally separate from the WitMotion recording-file bridge.  It
receives GATT notifications directly and writes ``live_ble_imu.csv``.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import pickle
import struct
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Deque, Iterable, Optional

import numpy as np
from bleak import BleakClient, BleakScanner


NOTIFY_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"
DEVICES = {
    "WT22222": "F6:B1:93:B5:2B:23",
    "WT901BLE11": "F7:36:CA:B7:CB:34",
}
FRAME_HEADER = b"\x55\x61"
FRAME_SIZE = 20
TARGET_RATE_HZ = 100.0
SMOOTH_SECONDS = 0.06
LIVE_FIELDS = [
    "timestamp_monotonic_s", "timestamp_unix_s", "device", "ax_g", "ay_g", "az_g",
    "gx_dps", "gy_dps", "gz_dps", "roll_deg", "pitch_deg", "yaw_deg",
    "inference_label", "exercise_probability", "source_rate_hz", "target_rate_hz",
]
SENSOR_FIELDS = (
    "ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps",
    "roll_deg", "pitch_deg", "yaw_deg",
)
SMOOTH_FIELDS = ("ax_g", "ay_g", "az_g", "gx_dps", "gy_dps", "gz_dps")


@dataclass
class DeviceStatus:
    """Small, serializable health record consumed by the dashboard."""

    name: str
    address: str
    state: str = "starting"
    last_error: str = ""
    connections: int = 0
    notifications: int = 0
    raw_frames: int = 0
    normalized_samples: int = 0
    discarded_bytes: int = 0
    source_rate_hz: float = 0.0
    last_notification_unix_s: float = 0.0
    last_sample_unix_s: float = 0.0


class StatusStore:
    """Atomically publish receiver health without coupling it to the HTTP server."""

    def __init__(self, path: Path, devices: dict[str, str]):
        self.path = path
        self.statuses = {name: DeviceStatus(name, address) for name, address in devices.items()}
        self._last_write = 0.0
        self._lock = threading.Lock()

    def write(self, force: bool = False) -> None:
        with self._lock:
            now = time.monotonic()
            if not force and now - self._last_write < 0.25:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "updated_unix_s": time.time(),
                "devices": {name: asdict(status) for name, status in self.statuses.items()},
            }
            temporary = self.path.with_name(self.path.name + ".tmp")
            try:
                temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                temporary.replace(self.path)
                self._last_write = now
            except OSError:
                # The receiver must keep sampling even if an antivirus scanner or
                # dashboard briefly has the status file open.
                pass


class LiveCsvWriter:
    """Write only normalized samples, keeping the dashboard file bounded in rate."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("w", newline="", encoding="utf-8-sig")
        self.writer = csv.DictWriter(self.file, fieldnames=LIVE_FIELDS)
        self.writer.writeheader()
        self.file.flush()
        self._lock = threading.Lock()

    def write(self, sample: dict) -> None:
        with self._lock:
            self.writer.writerow({field: sample.get(field, "") for field in LIVE_FIELDS})
            self.file.flush()

    def close(self) -> None:
        with self._lock:
            self.file.close()


class FrameDecoder:
    """Incrementally extract valid 20-byte ``55 61`` frames from BLE chunks."""

    def __init__(self):
        self.buffer = bytearray()
        self.frames_decoded = 0
        self.discarded_bytes = 0

    def feed(self, data: bytes) -> Iterable[dict]:
        self.buffer.extend(data)
        while True:
            start = self.buffer.find(FRAME_HEADER)
            if start < 0:
                keep = self.buffer[-1:] if self.buffer.endswith(FRAME_HEADER[:1]) else b""
                self.discarded_bytes += len(self.buffer) - len(keep)
                self.buffer[:] = keep
                return
            if start:
                self.discarded_bytes += start
                del self.buffer[:start]
            if len(self.buffer) < FRAME_SIZE:
                return
            frame = bytes(self.buffer[:FRAME_SIZE])
            del self.buffer[:FRAME_SIZE]
            self.frames_decoded += 1
            yield decode_frame(frame)


def decode_frame(frame: bytes) -> dict:
    """Decode ``55 61`` followed by nine little-endian signed int16 values."""
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


class NotificationClock:
    """Assign times to frames even when Windows delivers a notification batch."""

    def __init__(self, fallback_rate_hz: float = TARGET_RATE_HZ):
        self.period_s = 1.0 / fallback_rate_hz
        self.last_arrival: Optional[float] = None
        self.last_assigned: Optional[float] = None

    @property
    def rate_hz(self) -> float:
        return 1.0 / self.period_s if self.period_s > 0 else 0.0

    def assign(self, count: int, arrival: float) -> list[float]:
        if count <= 0:
            return []
        if self.last_arrival is not None:
            elapsed = arrival - self.last_arrival
            if elapsed > 1e-5:
                observed_period = elapsed / count
                # Ignore debugger pauses and impossible timer noise while still
                # accepting the device's high-rate batched BLE delivery.
                if 1.0 / 5000.0 <= observed_period <= 0.1:
                    # A notification can represent one frame or a complete
                    # batch. Its current span is the best timing source for
                    # that batch; smoothing it would preserve the 100 Hz
                    # fallback long enough to distort an 800 Hz device.
                    self.period_s = observed_period
        if self.last_assigned is None:
            first = arrival
        elif count == 1:
            # A one-frame notification carries its own arrival timestamp. Do
            # not force the old fallback period here: that would turn an 800 Hz
            # raw source into a fictitious multi-second stream.
            first = max(arrival, self.last_assigned + 1e-6)
        else:
            first = max(self.last_assigned + 1e-6, arrival - self.period_s * (count - 1))
        assigned = [first + index * self.period_s for index in range(count)]
        self.last_arrival = arrival
        self.last_assigned = assigned[-1]
        return assigned


class RealtimeNormalizer:
    """Resample a device stream to a stable rate and apply causal smoothing."""

    def __init__(self, target_rate_hz: float = TARGET_RATE_HZ, smooth_seconds: float = SMOOTH_SECONDS):
        self.target_rate_hz = target_rate_hz
        self.period_s = 1.0 / target_rate_hz
        self.smooth_seconds = smooth_seconds
        self.previous: Optional[dict] = None
        self.next_time: Optional[float] = None
        self.smoothing: Deque[dict] = deque()

    def _smoothed(self, sample: dict) -> dict:
        self.smoothing.append(sample)
        cutoff = sample["timestamp_monotonic_s"] - self.smooth_seconds
        while self.smoothing and self.smoothing[0]["timestamp_monotonic_s"] < cutoff:
            self.smoothing.popleft()
        result = sample.copy()
        for field in SMOOTH_FIELDS:
            result[field] = float(np.mean([row[field] for row in self.smoothing]))
        return result

    def _interpolate(self, left: dict, right: dict, timestamp: float) -> dict:
        span = max(right["timestamp_monotonic_s"] - left["timestamp_monotonic_s"], 1e-9)
        ratio = min(1.0, max(0.0, (timestamp - left["timestamp_monotonic_s"]) / span))
        result = {"timestamp_monotonic_s": timestamp}
        for field in SENSOR_FIELDS:
            result[field] = float(left[field] + (right[field] - left[field]) * ratio)
        return result

    def feed(self, sample: dict) -> Iterable[dict]:
        timestamp = float(sample["timestamp_monotonic_s"])
        if self.previous is None:
            self.previous = sample
            self.next_time = timestamp
            yield self._smoothed(sample.copy())
            self.next_time += self.period_s
            return
        if timestamp <= self.previous["timestamp_monotonic_s"]:
            timestamp = self.previous["timestamp_monotonic_s"] + 1e-6
            sample = sample.copy()
            sample["timestamp_monotonic_s"] = timestamp
        if timestamp - self.previous["timestamp_monotonic_s"] > 0.25:
            # A stalled stream must not be filled with synthetic motion.
            self.previous = sample
            self.next_time = timestamp
            self.smoothing.clear()
            yield self._smoothed(sample.copy())
            self.next_time += self.period_s
            return
        while self.next_time is not None and self.next_time <= timestamp:
            yield self._smoothed(self._interpolate(self.previous, sample, self.next_time))
            self.next_time += self.period_s
        self.previous = sample


class LiveExerciseClassifier:
    """Apply the offline two-second logistic model to one normalized stream."""

    def __init__(
        self,
        model_path: Path,
        window_seconds: float = 2.0,
        hop_seconds: float = 1.0,
        sample_rate_hz: float = TARGET_RATE_HZ,
    ):
        with model_path.open("rb") as handle:
            payload = pickle.load(handle)
        self.model = payload["model"]
        self.features = payload["features"]
        self.threshold = float(payload.get("threshold", 0.5))
        self.window_seconds = window_seconds
        self.hop_seconds = hop_seconds
        self.minimum_samples = max(16, round(window_seconds * sample_rate_hz * 0.8))
        self.samples: Deque[dict] = deque()
        self.last_prediction_time = 0.0
        self.stable_label: Optional[str] = None
        self.pending_label: Optional[str] = None
        self.pending_count = 0
        self.last_probability: Optional[float] = None

    @staticmethod
    def _spectral(x: np.ndarray, fs: float) -> tuple[float, float]:
        x = x - np.mean(x)
        power = np.abs(np.fft.rfft(x)) ** 2
        freq = np.fft.rfftfreq(len(x), 1.0 / fs)
        valid = (freq >= 0.2) & (freq <= min(15.0, fs / 2))
        if not valid.any() or power[valid].sum() <= 0:
            return 0.0, 0.0
        p = power[valid] / power[valid].sum()
        dominant = float(freq[valid][np.argmax(power[valid])])
        entropy = float(-(p * np.log2(p + 1e-15)).sum() / np.log2(len(p))) if len(p) > 1 else 0.0
        return dominant, entropy

    @classmethod
    def _stats(cls, prefix: str, values: np.ndarray, fs: float) -> dict:
        x = np.asarray(values, dtype=float)
        dom, entropy = cls._spectral(x, fs)
        return {
            f"{prefix}_mean": float(np.mean(x)),
            f"{prefix}_std": float(np.std(x, ddof=1)),
            f"{prefix}_rms": float(np.sqrt(np.mean(x ** 2))),
            f"{prefix}_dom_freq_hz": dom,
            f"{prefix}_spec_entropy": entropy,
        }

    def _feature_row(self, rows: list[dict]) -> dict:
        elapsed = np.array([row["timestamp_monotonic_s"] for row in rows], dtype=float)
        fs = (len(rows) - 1) / max(elapsed[-1] - elapsed[0], 1e-6)
        acc = np.array([[row["ax_g"], row["ay_g"], row["az_g"]] for row in rows], dtype=float)
        gyro = np.array([[row["gx_dps"], row["gy_dps"], row["gz_dps"]] for row in rows], dtype=float)
        values = {}
        for prefix, column in zip(("acc_x", "acc_y", "acc_z"), acc.T):
            values.update(self._stats(prefix, column, fs))
        for prefix, column in zip(("gyro_x", "gyro_y", "gyro_z"), gyro.T):
            values.update(self._stats(prefix, column, fs))
        acc_mag = np.linalg.norm(acc, axis=1)
        values.update(self._stats("acc_mag", acc_mag, fs))
        values.update(self._stats("dynamic_acc", np.abs(acc_mag - 1.0), fs))
        values["acc_sma"] = float(np.mean(np.sum(np.abs(acc - np.mean(acc, axis=0)), axis=1)))
        values.update(self._stats("gyro_mag", np.linalg.norm(gyro, axis=1), fs))
        return values

    def update(self, sample: dict) -> Optional[tuple[str, float]]:
        now = sample["timestamp_monotonic_s"]
        self.samples.append(sample)
        while self.samples and now - self.samples[0]["timestamp_monotonic_s"] > self.window_seconds:
            self.samples.popleft()
        if now - self.last_prediction_time < self.hop_seconds or len(self.samples) < self.minimum_samples:
            return None
        if now - self.samples[0]["timestamp_monotonic_s"] < self.window_seconds * 0.9:
            return None
        self.last_prediction_time = now
        row = self._feature_row(list(self.samples))
        x = np.array([row.get(name, np.nan) for name in self.features], dtype=float)
        median = np.asarray(self.model["median"], dtype=float)
        x = np.where(np.isfinite(x), x, median)
        z = (x - np.asarray(self.model["mean"])) / np.asarray(self.model["scale"])
        probability = 1.0 / (1.0 + np.exp(-np.clip(
            self.model["intercept"] + z @ np.asarray(self.model["coefficients"]), -35, 35
        )))
        self.last_probability = float(probability)
        raw_label = "运动" if probability >= self.threshold else "非运动"
        if self.stable_label is None:
            self.stable_label = raw_label
            return self.stable_label, self.last_probability
        if raw_label == self.stable_label:
            self.pending_label, self.pending_count = None, 0
            return self.stable_label, self.last_probability
        if raw_label == self.pending_label:
            self.pending_count += 1
        else:
            self.pending_label, self.pending_count = raw_label, 1
        if self.pending_count >= 3:
            self.stable_label = raw_label
            self.pending_label, self.pending_count = None, 0
            return self.stable_label, self.last_probability
        return None


async def stream_device(
    name: str,
    address: str,
    classifier: LiveExerciseClassifier,
    writer: LiveCsvWriter,
    status_store: StatusStore,
    target_rate_hz: float,
    smooth_seconds: float,
) -> None:
    decoder = FrameDecoder()
    clock = NotificationClock(target_rate_hz)
    normalizer = RealtimeNormalizer(target_rate_hz, smooth_seconds)
    status = status_store.statuses[name]
    last_print_time = 0.0
    while True:
        try:
            status.state, status.last_error = "discovering", ""
            status_store.write(force=True)
            device = await BleakScanner.find_device_by_address(address, timeout=5)
            if device is None:
                status.state = "reconnecting"
                status.last_error = "设备未广播或仍被 WitMotion 占用"
                status_store.write(force=True)
                print(f"[{name}] 未发现，3 秒后重试")
                await asyncio.sleep(3)
                continue

            status.state = "connecting"
            status_store.write(force=True)

            def on_notification(_sender: int, data: bytearray) -> None:
                nonlocal last_print_time
                arrival = time.monotonic()
                raw_samples = list(decoder.feed(bytes(data)))
                status.notifications += 1
                status.last_notification_unix_s = time.time()
                status.discarded_bytes = decoder.discarded_bytes
                if not raw_samples:
                    status_store.write()
                    return
                timestamps = clock.assign(len(raw_samples), arrival)
                status.raw_frames += len(raw_samples)
                status.source_rate_hz = clock.rate_hz
                for raw, timestamp in zip(raw_samples, timestamps):
                    raw["timestamp_monotonic_s"] = timestamp
                    for sample in normalizer.feed(raw):
                        sample["device"] = name
                        sample["timestamp_unix_s"] = time.time()
                        prediction = classifier.update(sample)
                        if prediction:
                            label, probability = prediction
                            print(f"[{name}] 实时判定：{label}（运动概率 {probability:.1%}）")
                        sample["inference_label"] = classifier.stable_label or ""
                        sample["exercise_probability"] = (
                            classifier.last_probability if classifier.last_probability is not None else ""
                        )
                        sample["source_rate_hz"] = round(clock.rate_hz, 2)
                        sample["target_rate_hz"] = target_rate_hz
                        writer.write(sample)
                        status.normalized_samples += 1
                        status.last_sample_unix_s = sample["timestamp_unix_s"]
                        if sample["timestamp_monotonic_s"] - last_print_time >= 0.5:
                            print(
                                f"[{name}] a=({sample['ax_g']:.3f}, {sample['ay_g']:.3f}, {sample['az_g']:.3f}) g, "
                                f"w=({sample['gx_dps']:.1f}, {sample['gy_dps']:.1f}, {sample['gz_dps']:.1f}) °/s"
                            )
                            last_print_time = sample["timestamp_monotonic_s"]
                status.state = "live"
                status_store.write()

            async with BleakClient(device, timeout=20) as client:
                status.connections += 1
                status.state = "live"
                status_store.write(force=True)
                print(f"[{name}] 已连接，开始接收")
                await client.start_notify(NOTIFY_UUID, on_notification)
                while client.is_connected:
                    await asyncio.sleep(1)
                status.state = "reconnecting"
                status.last_error = "BLE 连接已断开"
                status_store.write(force=True)
        except Exception as exc:
            status.state = "reconnecting"
            status.last_error = f"{type(exc).__name__}: {exc}"
            status_store.write(force=True)
            print(f"[{name}] 连接/接收异常：{status.last_error}")
        await asyncio.sleep(3)


async def run_receiver(args: argparse.Namespace) -> None:
    model_path = Path(args.model)
    if not model_path.is_file():
        raise SystemExit(f"找不到模型文件：{model_path.resolve()}")
    if args.target_rate_hz <= 0 or args.smooth_seconds < 0:
        raise SystemExit("--target-rate-hz 必须大于 0，--smooth-seconds 不得小于 0")
    writer = LiveCsvWriter(Path(args.csv))
    status_store = StatusStore(Path(args.status), DEVICES)
    classifiers = {
        name: LiveExerciseClassifier(model_path, sample_rate_hz=args.target_rate_hz)
        for name in DEVICES
    }
    print(f"已加载实时模型：{model_path.resolve()}")
    print(f"同时连接 {', '.join(DEVICES)}；输出规范化到 {args.target_rate_hz:g} Hz")
    try:
        await asyncio.gather(*(
            stream_device(
                name, address, classifiers[name], writer, status_store,
                args.target_rate_hz, args.smooth_seconds,
            )
            for name, address in DEVICES.items()
        ))
    finally:
        status_store.write(force=True)
        writer.close()


async def scan_devices(timeout: float) -> None:
    """List nearby BLE devices without changing the fixed two-device receiver set."""
    print(f"扫描附近 BLE 设备（{timeout:.0f} 秒）…")
    devices = await BleakScanner.discover(timeout=timeout)
    if not devices:
        print("未发现任何 BLE 设备。请确认设备已开机、未被其他应用占用且处于可广播状态。")
        return
    for device in devices:
        print(f"{device.name or '<无名称>'}\t{device.address}\tRSSI={getattr(device, 'rssi', 'n/a')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="必填：训练输出的 l2_logistic_model.pkl")
    parser.add_argument("--csv", default="imu_output/live_ble_imu.csv", help="BLE 规范化实时流 CSV 路径")
    parser.add_argument("--status", default="imu_output/live_ble_status.json", help="BLE 设备状态 JSON 路径")
    parser.add_argument("--target-rate-hz", type=float, default=TARGET_RATE_HZ, help="实时规范化采样率")
    parser.add_argument("--smooth-seconds", type=float, default=SMOOTH_SECONDS, help="因果平滑时间窗")
    parser.add_argument("--scan", action="store_true", help="仅扫描附近 BLE 设备")
    parser.add_argument("--scan-seconds", type=float, default=12.0, help="BLE 扫描时长")
    args = parser.parse_args()
    if not args.scan and not args.model:
        parser.error("直接 BLE 推理必须提供 --model")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.scan:
        asyncio.run(scan_devices(arguments.scan_seconds))
    else:
        asyncio.run(run_receiver(arguments))
