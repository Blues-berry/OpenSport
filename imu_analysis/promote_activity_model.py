"""Promote a versioned candidate only when the fixed acceptance gate passes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = (
    "activity_model.pkl",
    "activity_model.txt",
    "feature_importance.csv",
    "metrics.json",
    "MODEL_CARD.md",
    "model_bundle.json",
)


def score(report: dict) -> tuple[float, float]:
    test = report["metrics"]["test"]
    return float(test["motion"]["macro_f1"]), float(test["macro_f1"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_model_dir", type=Path)
    parser.add_argument(
        "--registry",
        type=Path,
        default=ROOT / "imu_output" / "models" / "activity",
    )
    args = parser.parse_args()
    candidate_report = json.loads(
        (args.candidate_model_dir / "metrics.json").read_text(encoding="utf-8")
    )
    champion = args.registry / "champion"
    champion_report_path = champion / "metrics.json"
    if not candidate_report.get("demo_ready"):
        raise SystemExit("Candidate failed the demo acceptance gate; champion was not changed")
    if not candidate_report.get("formal_evaluation"):
        raise SystemExit("Candidate has no gold formal evaluation; champion was not changed")
    if champion_report_path.exists():
        champion_report = json.loads(champion_report_path.read_text(encoding="utf-8"))
        candidate_score = score(candidate_report)
        champion_score = score(champion_report)
        if candidate_score[0] < champion_score[0] or candidate_score[1] < champion_score[1]:
            raise SystemExit(
                f"Candidate regressed: candidate={candidate_score}, champion={champion_score}"
            )
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
