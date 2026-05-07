#!/usr/bin/env python3
"""Prepare a Rubin DP2 field object catalog for external validation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp2_external_validation import make_basic_field_plots, read_table, standardize_dp2_table, write_table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="DP2 Object FITS/parquet/csv input.")
    parser.add_argument("--field", required=True, help="Field label, e.g. ECDFS or COSMOS.")
    parser.add_argument("--output", type=Path, required=True, help="Prepared parquet output.")
    parser.add_argument("--plot-dir", type=Path, required=True, help="Directory for sanity plots and clean-mask summaries.")
    parser.add_argument("--export-csv", action="store_true", help="Also write CSV when the table is not too large.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Reading DP2 {args.field}: {args.input}")
    raw = read_table(args.input)
    print(f"Raw rows: {len(raw):,}; columns: {len(raw.columns):,}")

    prepared = standardize_dp2_table(raw, field=args.field)
    print(f"Prepared columns: {len(prepared.columns):,}")
    print(f"RA range: {prepared['ra'].min():.6f} .. {prepared['ra'].max():.6f} deg")
    print(f"Dec range: {prepared['dec'].min():.6f} .. {prepared['dec'].max():.6f} deg")

    write_table(prepared, args.output, write_csv=args.export_csv)
    print(f"Wrote prepared table: {args.output}")

    counts = make_basic_field_plots(prepared, args.field, args.plot_dir)
    print("Clean-mask counts:")
    for key, value in counts.items():
        print(f"  {key}: {value:,}")
    print(f"Wrote plots and summaries: {args.plot_dir}")


if __name__ == "__main__":
    main()
