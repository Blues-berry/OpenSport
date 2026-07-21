"""Run quality inspection, cleaning and feature analysis in sequence."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(script: Path, *args: object) -> None:
    command = [sys.executable, str(script), *map(str, args)]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("imu_output"))
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    quality = args.work_dir / "reports" / "quality"
    cleaned = args.work_dir / "processed" / "cleaned"
    features = args.work_dir / "reports" / "features"
    model = args.work_dir / "model"
    run(here / "check_data.py", args.data_dir, "--output-dir", quality)
    run(here / "clean_data.py", args.data_dir, "--output-dir", cleaned)
    run(here / "extract_features.py", cleaned, "--output-dir", features)
    run(here / "train_logistic.py", features / "window_features.csv", "--output-dir", model)
    print(f"Done. See {args.work_dir.resolve()}")


if __name__ == "__main__":
    main()
