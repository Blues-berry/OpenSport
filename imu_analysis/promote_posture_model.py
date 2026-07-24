"""Promote a posture candidate only after gold binary acceptance gates pass."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


MODEL_FILES = (
    "head_posture_model.pkl",
    "head_posture_model.txt",
    "feature_importance.csv",
    "metrics.json",
    "MODEL_CARD.md",
    "model_bundle.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_model_dir", type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("imu_output/models/posture"),
    )
    args = parser.parse_args()
    report = json.loads(
        (args.candidate_model_dir / "metrics.json").read_text(
            encoding="utf-8"
        )
    )
    if not report.get("demo_ready") or not report.get("formal_evaluation"):
        raise SystemExit(
            "Candidate lacks a passing gold formal evaluation; champion unchanged"
        )
    metrics = report["metrics"]
    if (
        float(metrics.get("binary_macro_f1", 0.0)) < 0.85
        or float(metrics.get("poor_recall", 0.0)) < 0.85
    ):
        raise SystemExit("Candidate failed posture acceptance gates")
    champion = args.registry / "champion"
    staging = args.registry / "champion.next"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    for filename in MODEL_FILES:
        shutil.copy2(args.candidate_model_dir / filename, staging / filename)
    if champion.exists():
        previous = args.registry / "champion.previous"
        if previous.exists():
            shutil.rmtree(previous)
        champion.replace(previous)
    staging.replace(champion)
    print(f"Promoted {args.candidate_model_dir} -> {champion}")


if __name__ == "__main__":
    main()
