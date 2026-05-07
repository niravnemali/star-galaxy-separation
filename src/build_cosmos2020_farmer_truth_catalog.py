#!/usr/bin/env python3
"""Build a compact COSMOS2020 FARMER external-validation truth catalog.

The raw COSMOS2020 FARMER FITS file is too large for normal GitHub use.  This
script extracts only the columns needed for star/galaxy external validation and
keeps only ACS_MU_CLASS star/galaxy rows.  The output is analogous to the
ECDFS HST `data/external/hst_truth_catalog.csv` file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp2_external_validation import ensure_dir, read_table

DEFAULT_INPUT = REPO_ROOT / "data" / "external" / "COSMOS2020_FARMER_R1_v2.2_p3.fits"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "external" / "cosmos2020_farmer_truth_catalog.csv"
DEFAULT_SUMMARY = REPO_ROOT / "data" / "external" / "cosmos2020_farmer_truth_catalog_summary.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Raw COSMOS2020 FARMER FITS file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Compact CSV truth catalog output.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY, help="Summary markdown output.")
    parser.add_argument(
        "--include-flagged",
        action="store_true",
        help="Keep rows with FLAG_COMBINED != 0. Default keeps them but records flag counts; this option is reserved for compatibility.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cols = [
        "ID",
        "ALPHA_J2000",
        "DELTA_J2000",
        "FARMER_ID",
        "FLAG_COMBINED",
        "ID_ACS",
        "ACS_F814W_MAG",
        "ACS_MU_CLASS",
        "lp_type",
    ]
    print(f"Reading selected columns from {args.input}")
    raw = read_table(args.input, columns=cols)
    acs = pd.to_numeric(raw["ACS_MU_CLASS"], errors="coerce")
    labels = pd.Series(pd.NA, index=raw.index, dtype="object")
    labels[acs.eq(1)] = "galaxy"
    labels[acs.eq(2)] = "star"
    labels[acs.eq(3)] = "fake"

    finite = np.isfinite(pd.to_numeric(raw["ALPHA_J2000"], errors="coerce")) & np.isfinite(pd.to_numeric(raw["DELTA_J2000"], errors="coerce"))
    secure = finite & labels.isin(["star", "galaxy"])

    lp = pd.to_numeric(raw["lp_type"], errors="coerce") if "lp_type" in raw.columns else pd.Series(np.nan, index=raw.index)
    lp_label = pd.Series(pd.NA, index=raw.index, dtype="object")
    lp_label[lp.eq(0)] = "galaxy"
    lp_label[lp.eq(1)] = "star"
    lp_label[lp.eq(2)] = "xray"
    lp_label[lp.eq(-9)] = "failed"

    out = pd.DataFrame(
        {
            "id": raw["ID"].astype(str),
            "ra": pd.to_numeric(raw["ALPHA_J2000"], errors="coerce"),
            "dec": pd.to_numeric(raw["DELTA_J2000"], errors="coerce"),
            "label": labels,
            "truth_binary": labels.map({"galaxy": 0, "star": 1}),
            "acs_mu_class": acs,
            "flag_combined": pd.to_numeric(raw["FLAG_COMBINED"], errors="coerce") if "FLAG_COMBINED" in raw.columns else pd.NA,
            "lp_type": lp,
            "lp_type_label": lp_label,
            "acs_f814w_mag": pd.to_numeric(raw["ACS_F814W_MAG"], errors="coerce") if "ACS_F814W_MAG" in raw.columns else pd.NA,
        }
    ).loc[secure].reset_index(drop=True)

    ensure_dir(args.output.parent)
    out.to_csv(args.output, index=False)

    label_counts = out["label"].value_counts(dropna=False).to_dict()
    flag_counts = out["flag_combined"].value_counts(dropna=False).head(20).to_dict() if "flag_combined" in out.columns else {}
    summary = [
        "# COSMOS2020 FARMER Compact External-Validation Catalog\n\n",
        f"- Raw input: `{args.input}`\n",
        f"- Output: `{args.output}`\n",
        f"- Raw rows: {len(raw):,}\n",
        f"- Compact star/galaxy rows: {len(out):,}\n",
        f"- Label counts: {label_counts}\n",
        f"- FLAG_COMBINED counts in compact rows: {flag_counts}\n",
        "- Label rule: `ACS_MU_CLASS=1 -> galaxy`, `ACS_MU_CLASS=2 -> star`; fake/spurious and unlabeled rows excluded.\n",
        "- Caveat: labels are external validation labels, not perfect truth.\n",
    ]
    args.summary.write_text("".join(summary))
    print("".join(summary))


if __name__ == "__main__":
    main()
