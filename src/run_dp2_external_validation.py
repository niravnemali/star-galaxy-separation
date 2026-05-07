#!/usr/bin/env python3
"""Run DP2 external-validation diagnostics for a matched field sample."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp2_external_validation import read_table, run_validation_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=["ECDFS", "COSMOS"], required=True)
    parser.add_argument("--dp2-table", type=Path, required=True, help="Prepared DP2 field table; used for documentation.")
    parser.add_argument("--external-matched", type=Path, required=True, help="Matched external-validation table.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ps-version", default="unknown")
    parser.add_argument("--ps-table", type=Path, default=None, help="Optional standardized per-object pS table from a DP2 notebook version.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Reading matched table: {args.external_matched}")
    matched = read_table(args.external_matched)
    print(f"Matched rows: {len(matched):,}; columns: {len(matched.columns):,}")

    if args.ps_table is not None:
        print(f"Reading pS table: {args.ps_table}")
        ps = read_table(args.ps_table)
        if "object_id" not in ps.columns:
            raise ValueError(f"pS table lacks object_id: {args.ps_table}")
        keep = ["object_id"] + [c for c in ps.columns if c.startswith("pS_")]
        dup_count = int(ps["object_id"].duplicated().sum())
        if dup_count:
            print(f"Warning: pS table has {dup_count:,} duplicate object_id rows; keeping the first row per object_id.")
            ps = ps.drop_duplicates(subset=["object_id"], keep="first")
        before = len(matched)
        matched = matched.merge(ps[keep], on="object_id", how="left", validate="one_to_one")
        print(f"Merged pS table onto matched sample: {before:,} rows")
    else:
        print("No --ps-table supplied; pS diagnostics will run only if pS_* columns already exist in matched table.")

    summary = run_validation_suite(matched, args.output_dir, field=args.field, ps_version=args.ps_version)
    if summary.empty:
        print("No pS summary rows were produced. Check pS table availability.")
    else:
        print(summary.to_string(index=False))
    print(f"Wrote validation outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
