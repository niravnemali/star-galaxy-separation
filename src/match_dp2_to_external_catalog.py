#!/usr/bin/env python3
"""Match prepared DP2 field tables to external validation catalogs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp2_external_validation import (
    ensure_dir,
    load_cosmos2020_farmer_validation,
    load_hst_ecdfs_truth,
    match_external_to_dp2,
    read_table,
    save_match_diagnostics,
    write_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True, choices=["ECDFS", "COSMOS"])
    parser.add_argument("--dp2-table", type=Path, required=True, help="Prepared DP2 table.")
    parser.add_argument("--external", type=Path, required=True, help="External catalog path.")
    parser.add_argument("--external-type", choices=["hst-ecdfs", "cosmos2020-farmer"], required=True)
    parser.add_argument("--max-sep-arcsec", type=float, default=0.3)
    parser.add_argument("--test-radii", default="0.3,0.5", help="Comma-separated radii to summarize before final save.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--truth-output",
        type=Path,
        default=None,
        help="Optional matched subset containing only star/galaxy external validation labels.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--max-csv-rows", type=int, default=250_000)
    return parser.parse_args()


def load_external(args: argparse.Namespace) -> pd.DataFrame:
    if args.external_type == "hst-ecdfs":
        return load_hst_ecdfs_truth(args.external)
    if args.external_type == "cosmos2020-farmer":
        return load_cosmos2020_farmer_validation(args.external)
    raise ValueError(args.external_type)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    print(f"Reading DP2 table: {args.dp2_table}")
    dp2 = read_table(args.dp2_table)
    print(f"DP2 rows: {len(dp2):,}; columns: {len(dp2.columns):,}")

    print(f"Reading external catalog: {args.external}")
    external = load_external(args)
    print(f"External rows: {len(external):,}; columns: {len(external.columns):,}")
    if "truth_label" in external.columns:
        print("External label counts:")
        print(external["truth_label"].value_counts(dropna=False).to_string())

    radius_rows = []
    for radius in [float(x) for x in args.test_radii.split(",") if x.strip()]:
        trial = match_external_to_dp2(dp2, external, max_sep_arcsec=radius)
        row = {"radius_arcsec": radius, **trial.summary}
        radius_rows.append(row)
        print(f"Radius {radius:.2f} arcsec: {trial.summary['final_matches']:,} final matches")
    pd.DataFrame(radius_rows).to_csv(args.output_dir / "match_radius_comparison.csv", index=False)

    result = match_external_to_dp2(dp2, external, max_sep_arcsec=args.max_sep_arcsec)
    write_table(result.matched, args.output, write_csv=args.export_csv, max_csv_rows=args.max_csv_rows)
    print(f"Wrote matched table: {args.output}")
    if args.truth_output is not None:
        truth_sample = result.matched[result.matched["truth_label"].isin(["star", "galaxy"])].copy()
        write_table(truth_sample, args.truth_output, write_csv=False)
        print(f"Wrote external validation truth-like subset: {args.truth_output}")
    if args.export_csv and len(result.matched) > args.max_csv_rows:
        print(f"CSV skipped because matched rows exceed --max-csv-rows ({args.max_csv_rows:,}).")

    title = f"DP2 {args.field} vs {args.external_type}"
    save_match_diagnostics(result.matched, result.summary, args.output_dir, title)
    print(f"Wrote diagnostics: {args.output_dir}")
    print("Final summary:")
    for key, val in result.summary.items():
        print(f"  {key}: {val}")


if __name__ == "__main__":
    main()
