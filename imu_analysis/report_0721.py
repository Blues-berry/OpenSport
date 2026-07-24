"""Join 0721 split, file-quality and cleaning results into an audit manifest."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()

    split = pd.read_csv(args.result_dir / "split_manifest.csv")
    quality = pd.read_csv(args.result_dir / "quality" / "quality_files.csv")
    cleaning = pd.read_csv(args.result_dir / "cleaned" / "cleaning_log.csv")

    quality["source_id"] = quality["file"].astype(str).str.extract(r"[\\/]([0-9a-f]{10})[\\/]")
    counts = (
        cleaning.groupby("recording", as_index=False)[
            [
                "duplicate_rows_removed", "missing_input", "isolated_zero_replaced",
                "spike_replaced", "jump_candidate", "missing_after_cleaning",
                "physical_limit_candidate",
            ]
        ]
        .sum()
    )
    audit = quality.merge(split, on=["source_id", "device_id"], how="left", suffixes=("", "_split"))
    audit = audit.merge(counts, on="recording", how="left")

    def flags(row: pd.Series) -> str:
        result: list[str] = []
        if str(row.get("missing_core_columns", "")).strip() not in {"", "nan"}:
            result.append("missing_core_columns")
        if not bool(row.get("orientation_valid", False)):
            result.append("orientation_invalid")
        if float(row.get("duration_s", np.nan)) < 20:
            result.append("short_recording")
        rate = float(row.get("sample_rate_hz", np.nan))
        if np.isfinite(rate):
            nominal = max(50.0, round(rate / 50.0) * 50.0)
            if abs(rate - nominal) / nominal > 0.15:
                result.append("sampling_rate_irregular")
        activity = str(row.get("activity", ""))
        if "未标注" in activity:
            result.append("unlabelled_action")
        if "擦耳机" in activity:
            result.append("earphone_contact_artifact")
        if "提前终止" in activity:
            result.append("protocol_truncated")
        if "+" in activity:
            result.append("mixed_action")
        return ";".join(result)

    audit["quality_flags"] = audit.apply(flags, axis=1)
    audit["motion_binary_usable"] = ~audit["quality_flags"].str.contains(
        "missing_core_columns|short_recording|unlabelled_action|earphone_contact_artifact|protocol_truncated|mixed_action",
        regex=True,
    )
    audit["posture_usable"] = audit["motion_binary_usable"] & audit["orientation_valid"].astype(bool)
    audit["sampling_rate_mode_hz"] = (audit["sample_rate_hz"] / 50).round().clip(lower=1) * 50
    audit.to_csv(args.result_dir / "file_quality_audit.csv", index=False, encoding="utf-8-sig")

    flag_counts: dict[str, int] = {}
    for cell in audit["quality_flags"]:
        for flag in str(cell).split(";"):
            if flag and flag != "nan":
                flag_counts[flag] = flag_counts.get(flag, 0) + 1
    summary = {
        "source_csv": int(split["source_id"].nunique()),
        "prepared_device_csv": int(len(split)),
        "multi_device_source_csv": int(split.loc[split["was_split"], "source_id"].nunique()),
        "rows": int(quality["rows"].sum()),
        "duration_hours_sum_by_device": float(quality["duration_s"].sum() / 3600),
        "device_file_counts": quality.groupby("device_id").size().astype(int).to_dict(),
        "sample_rate_hz": {
            "median": float(quality["sample_rate_hz"].median()),
            "min": float(quality["sample_rate_hz"].min()),
            "max": float(quality["sample_rate_hz"].max()),
        },
        "quality_flag_counts": flag_counts,
        "motion_binary_usable_device_files": int(audit["motion_binary_usable"].sum()),
        "posture_usable_device_files": int(audit["posture_usable"].sum()),
        "cleaning_totals": {
            key: int(cleaning[key].sum())
            for key in [
                "duplicate_rows_removed", "missing_input", "isolated_zero_replaced",
                "spike_replaced", "jump_candidate", "missing_after_cleaning",
                "physical_limit_candidate",
            ]
        },
    }
    (args.result_dir / "0721_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
