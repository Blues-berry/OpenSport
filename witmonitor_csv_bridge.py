"""Tail WitMotion/WitMonitor CSV recordings and feed the live dashboard.

This bridge never opens BLE or a COM port.  Point WitMotion's live recording
folder at ``IMU数据采集`` (or pass ``--source-dir``) and it will consume rows
as the application appends them.  The expected columns are the same as the
previously collected WitMotion CSV exports.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from realtime_ble_imu import LiveExerciseClassifier


RAW_TO_LIVE = {
    "加速度X(g)": "ax_g", "加速度Y(g)": "ay_g", "加速度Z(g)": "az_g",
    "角速度X(°/s)": "gx_dps", "角速度Y(°/s)": "gy_dps", "角速度Z(°/s)": "gz_dps",
    "角度X(°)": "roll_deg", "角度Y(°)": "pitch_deg", "角度Z(°)": "yaw_deg",
}
OUTPUT_FIELDS = [
    "timestamp_monotonic_s", "device", "ax_g", "ay_g", "az_g",
    "gx_dps", "gy_dps", "gz_dps", "roll_deg", "pitch_deg", "yaw_deg",
    "inference_label", "exercise_probability",
]


def numeric(value: str | None) -> float:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return 0.0


class CsvTail:
    def __init__(self, path: Path):
        self.path = path
        self.position = 0
        self.header: list[str] | None = None

    def read_new(self) -> list[dict[str, str]]:
        try:
            size = self.path.stat().st_size
        except OSError:
            return []
        if size < self.position:
            self.position, self.header = 0, None
        with self.path.open("r", encoding="utf-8-sig", newline="") as handle:
            if self.header is None:
                self.header = [item.strip() for item in next(csv.reader([handle.readline()]))]
                self.position = handle.tell()
            handle.seek(self.position)
            text = handle.read()
            # Leave an incomplete final line for the next poll.
            complete = text.rsplit("\n", 1)[0] if "\n" in text else ""
            if not complete:
                return []
            consumed = len(complete.encode("utf-8")) + 1
            self.position += consumed
        return list(csv.DictReader(complete.splitlines(), fieldnames=self.header))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("IMU数据采集"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("imu_output/live_imu.csv"))
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    args = parser.parse_args()

    started = time.time()
    tails: dict[Path, CsvTail] = {}
    classifiers: dict[str, LiveExerciseClassifier] = {}
    sample_time: dict[str, float] = {}
    tracked_files: list[Path] = []
    next_scan = 0.0
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"监视 WitMotion CSV 目录：{args.source_dir.resolve()}")
    print("仅接收本桥接启动后新建或更新的 CSV，避免将历史训练数据当作实时流。")
    with args.csv.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        output.flush()
        while True:
            now_wall = time.time()
            if now_wall >= next_scan:
                tracked_files = [p for p in args.source_dir.rglob("*.csv") if p != args.csv and p.stat().st_mtime >= started]
                next_scan = now_wall + 1.0
            for path in tracked_files:
                tail = tails.setdefault(path, CsvTail(path))
                for raw in tail.read_new():
                    if not set(RAW_TO_LIVE).issubset(raw):
                        continue
                    device = str(raw.get("设备名称") or path.stem).strip()
                    classifier = classifiers.setdefault(device, LiveExerciseClassifier(str(args.model)))
                    now = sample_time.get(device, 0.0) + 1.0 / args.sample_rate
                    sample_time[device] = now
                    sample = {"timestamp_monotonic_s": now, "device": device}
                    sample.update({target: numeric(raw.get(source)) for source, target in RAW_TO_LIVE.items()})
                    prediction = classifier.update(sample)
                    if prediction:
                        label, probability = prediction
                        print(f"[{device}] 实时判定：{label}（运动概率 {probability:.1%}）")
                    sample["inference_label"] = classifier.stable_label or ""
                    sample["exercise_probability"] = classifier.last_probability if classifier.last_probability is not None else ""
                    writer.writerow(sample)
                    output.flush()
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
