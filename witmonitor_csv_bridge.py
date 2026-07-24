"""Tail WitMotion/WitMonitor CSV recordings and feed the live dashboard.

This bridge never opens BLE or a COM port.  Point WitMotion's live recording
folder at a dedicated runtime location (or pass ``--source-dir``) and it will
consume rows as the application appends them.  Archived project recordings
belong under ``data/raw``; generated normalized output belongs under
``imu_output``.
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
    "timestamp_monotonic_s", "timestamp_unix_s", "device", "ax_g", "ay_g", "az_g",
    "gx_dps", "gy_dps", "gz_dps", "roll_deg", "pitch_deg", "yaw_deg",
    "inference_label", "exercise_probability",
]
DEFAULT_WITMOTION_RECORD_DIR = Path(r"D:\download\Witmotion(V2026.6.26.0)\Record")


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
        # Use binary offsets.  The source CSV has Chinese headers, so text-mode
        # ``tell`` cookies plus byte-length arithmetic can skip or duplicate a
        # row when WitMotion appends while we are reading.
        with self.path.open("rb") as handle:
            if self.header is None:
                raw_header = handle.readline()
                if not raw_header:
                    return []
                self.header = [item.strip() for item in next(csv.reader([
                    raw_header.decode("utf-8-sig", errors="replace")
                ]))]
                self.position = handle.tell()
            handle.seek(self.position)
            raw = handle.read()
            # Leave an incomplete final line for the next poll.
            last_newline = raw.rfind(b"\n")
            if last_newline < 0:
                return []
            complete = raw[:last_newline + 1]
            self.position += len(complete)
        text = complete.decode("utf-8", errors="replace")
        return list(csv.DictReader(text.splitlines(), fieldnames=self.header))


def file_identity(path: Path) -> tuple[int, int]:
    """Return an identity that survives a parent-directory rename on Windows."""
    info = path.stat()
    return info.st_dev, info.st_ino


def recently_created_csvs(source_dir: Path, started: float, include_existing: bool) -> list[Path]:
    """List readable source files without failing when WitMotion rotates one."""
    files: list[Path] = []
    for path in source_dir.rglob("*.csv"):
        try:
            if include_existing or path.stat().st_mtime >= started:
                files.append(path)
        except OSError:
            # The app can atomically finish or rename a record between rglob
            # and stat; retry it during the following scan.
            continue
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_WITMOTION_RECORD_DIR)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=Path("imu_output/live_imu.csv"))
    parser.add_argument("--sample-rate", type=float, default=100.0)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--include-existing", action="store_true",
                        help="包含启动前已封存的 CSV；用于只读验证，不影响 WitMotion。")
    parser.add_argument("--once", action="store_true",
                        help="处理当前匹配的文件后退出；必须与 --include-existing 一起使用。")
    args = parser.parse_args()

    if args.once and not args.include_existing:
        parser.error("--once 需要 --include-existing，避免误把历史记录当成实时流。")
    started = time.time()
    # Key by the Windows file identity rather than the path.  A user can rename
    # a recording directory immediately after collection without causing the
    # same data_0.csv to be interpreted as a second, new stream.
    tails: dict[tuple[int, int], CsvTail] = {}
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
                tracked_files = [p for p in recently_created_csvs(
                    args.source_dir, started, args.include_existing
                ) if p != args.csv]
                next_scan = now_wall + 1.0
            for path in tracked_files:
                try:
                    identity = file_identity(path)
                except OSError:
                    continue
                tail = tails.get(identity)
                if tail is None:
                    tail = tails[identity] = CsvTail(path)
                else:
                    # Continue from the previous byte offset after the parent
                    # folder is renamed by the operator.
                    tail.path = path
                for raw in tail.read_new():
                    if not set(RAW_TO_LIVE).issubset(raw):
                        continue
                    device = str(raw.get("设备名称") or path.stem).strip()
                    classifier = classifiers.setdefault(device, LiveExerciseClassifier(args.model))
                    now = sample_time.get(device, 0.0) + 1.0 / args.sample_rate
                    sample_time[device] = now
                    # The vendor CSV records a time-of-day without a date.  Use
                    # the arrival time for dashboard freshness while preserving
                    # the monotonic sample time used by the model.
                    sample = {
                        "timestamp_monotonic_s": now,
                        "timestamp_unix_s": time.time(),
                        "device": device,
                    }
                    sample.update({target: numeric(raw.get(source)) for source, target in RAW_TO_LIVE.items()})
                    prediction = classifier.update(sample)
                    if prediction:
                        label, probability = prediction
                        print(f"[{device}] 实时判定：{label}（运动概率 {probability:.1%}）")
                    sample["inference_label"] = classifier.stable_label or ""
                    sample["exercise_probability"] = classifier.last_probability if classifier.last_probability is not None else ""
                    writer.writerow(sample)
                    output.flush()
            if args.once and tracked_files:
                print("已完成封存 CSV 的只读验证。")
                return
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
