"""Shared utilities for the IMU data quality, cleaning and feature scripts."""

from __future__ import annotations

import re
import csv
from pathlib import Path

import numpy as np
import pandas as pd


ACC_COLS = ["加速度X(g)", "加速度Y(g)", "加速度Z(g)"]
GYRO_COLS = ["角速度X(°/s)", "角速度Y(°/s)", "角速度Z(°/s)"]
ANGLE_COLS = ["角度X(°)", "角度Y(°)", "角度Z(°)"]
QUAT_COLS = ["四元数0()", "四元数1()", "四元数2()", "四元数3()"]
CORE_COLS = ACC_COLS + GYRO_COLS + ANGLE_COLS + QUAT_COLS


def natural_key(text: str) -> list[object]:
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", text)]


def activity_from_folder(name: str) -> str:
    """Normalize logger/file naming variants into an analysis action label."""
    label = re.sub(r"__(csv|txt)$", "", name, flags=re.IGNORECASE)
    # New logger folders use "personA+personB-action".  The people identify
    # the acquisition group, not the action class.
    label = re.sub(r"^[\u4e00-\u9fff]{2,4}\+[\u4e00-\u9fff]{2,4}\s*-?\s*", "", label)
    label = re.sub(r"\s*\(\d+\)\s*", " ", label).strip()
    label = re.sub(r"\s+", " ", label)
    if label.startswith("坐起"):
        return "坐下起立"
    if label.startswith("站姿"):
        return "站立"
    if label.startswith("走路") and "3.0" in label:
        return "慢走"
    if label.startswith("走路") and "5.0" in label:
        return "快走"
    if label in {"自由走路", "自由行走"}:
        return "自由行走"
    return label


def trial_from_folder(name: str) -> int:
    match = re.search(r"\((\d+)\)", name)
    return int(match.group(1)) if match else 1


def find_csv_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("*.csv"), key=lambda p: natural_key(str(p)))


def find_imu_files(data_dir: Path) -> list[Path]:
    files = list(data_dir.rglob("*.csv")) + list(data_dir.rglob("*.txt"))
    return sorted(files, key=lambda p: natural_key(str(p)))


def source_name(path: Path) -> str:
    # In the 0720 logger layout, the action is an ancestor directory and the
    # filename identifies the physical device.  Search ancestors first.
    for part in reversed(path.parts[:-1]):
        if "+" in part:
            return part
    return path.stem if path.suffix.lower() == ".txt" else path.parent.name


def subject_group_from_path(path: Path) -> str:
    for part in reversed(path.parts[:-1]):
        match = re.match(r"^([\u4e00-\u9fff]{2,4}\+[\u4e00-\u9fff]{2,4})", part)
        if match:
            return match.group(1)
    return "unknown"


def session_name_from_path(path: Path) -> str:
    for part in reversed(path.parts[:-1]):
        if re.fullmatch(r"\d{2}-\d{2}-\d{2}-\d{3}", part):
            return part
    return "unknown-session"


def device_id_from_path(path: Path) -> str:
    match = re.search(r"\(([0-9a-fA-F-]{17})\)", path.stem)
    if match:
        return match.group(1).lower()
    return path.stem.lstrip("_").split("(")[0]


def recording_name(path: Path) -> str:
    name = source_name(path)
    # Cleaned files already live in a source-suffixed folder.
    if re.search(r"__(csv|txt)$", name, flags=re.IGNORECASE):
        return name
    if any("+" in part for part in path.parts[:-1]):
        activity = activity_from_folder(source_name(path))
        session = session_name_from_path(path)
        device = device_id_from_path(path)
        return f"{activity}__{session}__{device}__{path.suffix.lower().lstrip('.')}"
    return f"{name}__{path.suffix.lower().lstrip('.')}"


def state_from_activity(activity: str) -> str:
    """Temporary protocol-derived fitness label for the 0720 collection."""
    if activity.startswith(("跑步机", "深蹲", "弓步蹲", "卷腹")):
        return "exercise"
    if activity.startswith(("自由行走", "爬楼", "下楼", "坐下起立")):
        return "ambiguous"
    if activity.startswith(("不对称佩戴", "佩戴取下")):
        return "wear_artifact"
    return "non_exercise"


def read_imu_csv(path: Path) -> pd.DataFrame:
    # Data rows contain one more trailing empty field than the header. Without
    # explicit names/usecols pandas silently promotes the true time column to
    # an index and shifts every sensor channel one position to the left.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader([handle.readline()]))
    header = [str(c).strip() for c in header]
    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        skipinitialspace=True,
        low_memory=False,
        header=0,
        names=header,
        usecols=range(len(header)),
    )
    df.columns = [str(c).strip() for c in df.columns]
    # The logger adds an empty trailing field to every row.
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    for col in df.columns:
        if col not in ("时间", "设备名称", "片上时间()", "版本号()") and not col.startswith("_"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def read_imu_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return read_imu_csv(path)
    if path.suffix.lower() != ".txt":
        raise ValueError(f"Unsupported IMU format: {path}")
    # TXT exports have the same extra trailing empty field as CSV exports.
    # Supplying names/usecols prevents pandas from promoting the real time
    # field into the index and shifting all sensor columns left.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader([handle.readline()], delimiter="\t"))
    header = [str(c).strip() for c in header]
    df = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8-sig",
        low_memory=False,
        header=0,
        names=header,
        usecols=range(len(header)),
    )
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if col not in ("时间", "设备名称", "片上时间()", "版本号()"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_time_column(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df:
        return pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")
    values = df[col].astype(str).str.strip()
    sample = values[df[col].notna()].iloc[0] if df[col].notna().any() else ""
    if col == "时间" and " " not in sample:
        fmt = "%H:%M:%S.%f"
    else:
        fmt = "%Y-%m-%d %H:%M:%S.%f"
    return pd.to_datetime(values, format=fmt, errors="coerce")


def capture_group(path: Path, df: pd.DataFrame) -> str:
    """Identify samples acquired in the same action session."""
    activity = activity_from_folder(source_name(path))
    parsed = parse_time_column(df, "时间")
    if parsed.notna().any():
        stamp = parsed.loc[parsed.notna()].iloc[0]
        date_text = stamp.strftime("%Y-%m-%d")
        if stamp.year == 1900:
            date_text = next((part for part in path.parts if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", part)), "unknown-date")
        # Parallel CSV/TXT samples of one action session can begin 1--7 s apart.
        # Trials in this data set are much farther apart, so a 20-second bucket
        # keeps related samples together without treating them as duplicate rows.
        stamp = stamp.round("20s")
        return f"{date_text}__{activity}__{stamp.strftime('%H%M%S')}"
    return f"unknown-date__{activity}__{recording_name(path)}"


def elapsed_seconds(df: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return a monotonic analysis time axis and its source description."""
    candidates = [("时间", "host"), ("片上时间()", "onboard")]
    for col, source in candidates:
        if col not in df:
            continue
        parsed = parse_time_column(df, col)
        valid = parsed.notna()
        if valid.sum() < 2:
            continue
        first = parsed.loc[valid].iloc[0]
        sec_values = (parsed - first).dt.total_seconds().to_numpy(dtype=float)
        finite = sec_values[np.isfinite(sec_values)]
        span = finite[-1] - finite[0] if len(finite) > 1 else 0
        if span > 0:
            sec = pd.Series(sec_values).interpolate(limit_direction="both").to_numpy()
            # Repeated batched timestamps are spread uniformly inside each run.
            sec = _make_strict_time(sec)
            return sec, source
    return np.arange(len(df), dtype=float), "sample_index"


def _make_strict_time(sec: np.ndarray) -> np.ndarray:
    if len(sec) < 2:
        return sec
    span = float(sec[-1] - sec[0])
    if span <= 0:
        return np.arange(len(sec), dtype=float)
    # The row rate over the complete capture is much more reliable than the
    # logger's millisecond timestamps, which arrive in batches.
    return np.linspace(0.0, span, len(sec), dtype=float)


def sampling_info(df: pd.DataFrame) -> dict[str, float | str]:
    sec, source = elapsed_seconds(df)
    duration = float(sec[-1] - sec[0]) if len(sec) > 1 else 0.0
    fs = (len(sec) - 1) / duration if duration > 0 else np.nan
    return {"duration_s": duration, "sample_rate_hz": float(fs), "time_source": source}


def robust_outlier_mask(values: pd.Series, z_threshold: float = 8.0) -> pd.Series:
    x = pd.to_numeric(values, errors="coerce")
    median = x.median()
    mad = (x - median).abs().median()
    if not np.isfinite(mad) or mad <= 1e-12:
        return pd.Series(False, index=x.index)
    robust_z = 0.67448975 * (x - median).abs() / mad
    return robust_z > z_threshold


def robust_jump_mask(values: pd.Series, z_threshold: float = 10.0) -> pd.Series:
    diff = pd.to_numeric(values, errors="coerce").diff().abs()
    median = diff.median()
    mad = (diff - median).abs().median()
    if not np.isfinite(mad) or mad <= 1e-12:
        return pd.Series(False, index=diff.index)
    return 0.67448975 * (diff - median).abs() / mad > z_threshold


def longest_run(mask: np.ndarray) -> int:
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def longest_constant_run(values: pd.Series) -> int:
    x = pd.to_numeric(values, errors="coerce")
    same = x.eq(x.shift()) & x.notna()
    return longest_run(same.to_numpy()) + (1 if same.any() else 0)


def safe_name(path: Path) -> str:
    return re.sub(r"[^\w\-()\u4e00-\u9fff]+", "_", path.parent.name)


def markdown_table(frame: pd.DataFrame, include_index: bool = False) -> str:
    """Render a small DataFrame as Markdown without the optional tabulate package."""
    table = frame.copy()
    if include_index:
        index_name = str(table.index.name or "index")
        table.insert(0, index_name, table.index.astype(str))
    headers = [str(c).replace("|", "\\|") for c in table.columns]
    rows = []
    for values in table.itertuples(index=False, name=None):
        rows.append([("" if pd.isna(v) else str(v)).replace("|", "\\|") for v in values])
    line1 = "| " + " | ".join(headers) + " |"
    line2 = "| " + " | ".join(["---"] * len(headers)) + " |"
    return "\n".join([line1, line2] + ["| " + " | ".join(row) + " |" for row in rows])
