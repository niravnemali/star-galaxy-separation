#!/usr/bin/env python3
"""Convenience wrapper for matching DP2 ECDFS to the GOODS-S/3D-HST labels."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]

cmd = [
    sys.executable,
    str(REPO_ROOT / "scripts" / "match_dp2_to_external_catalog.py"),
    "--field",
    "ECDFS",
    "--dp2-table",
    str(REPO_ROOT / "outputs" / "dp2_ecdfs_analysis_table.parquet"),
    "--external",
    str(REPO_ROOT / "data" / "external" / "hst_truth_catalog.csv"),
    "--external-type",
    "hst-ecdfs",
    "--max-sep-arcsec",
    "0.3",
    "--output",
    str(REPO_ROOT / "outputs" / "dp2_ecdfs_hst_matched.parquet"),
    "--output-dir",
    str(REPO_ROOT / "results" / "dp2_ecdfs_hst_validation"),
    "--export-csv",
]

raise SystemExit(subprocess.call(cmd))
