#!/usr/bin/env python3
"""Audit a standards-compliant trial without modifying raw source data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from opensport.data import TrialImporter  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("trial_id")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write validated per-device SI captures and derived labels.",
    )
    args = parser.parse_args()
    importer = TrialImporter(args.manifest, args.trial_id)
    if args.output_dir:
        report, written = importer.write_training_captures(args.output_dir)
    else:
        report, _ = importer.audit()
        written = []
    if args.report:
        importer.write_report(args.report)
    print(
        json.dumps(
            {
                **report.to_dict(),
                "written_files": [str(path) for path in written],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report.status != "recollect_required" else 2


if __name__ == "__main__":
    raise SystemExit(main())
