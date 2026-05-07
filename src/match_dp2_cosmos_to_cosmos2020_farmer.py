#!/usr/bin/env python3
"""Convenience wrapper for matching DP2 COSMOS to COSMOS2020 FARMER."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FARMER = REPO_ROOT / "data" / "external" / "cosmos2020_farmer_truth_catalog.csv"

cmd = [
    sys.executable,
    str(REPO_ROOT / "scripts" / "match_dp2_to_external_catalog.py"),
    "--field",
    "COSMOS",
    "--dp2-table",
    str(REPO_ROOT / "outputs" / "dp2_cosmos_analysis_table.parquet"),
    "--external",
    str(DEFAULT_FARMER),
    "--external-type",
    "cosmos2020-farmer",
    "--max-sep-arcsec",
    "0.3",
    "--output",
    str(REPO_ROOT / "outputs" / "dp2_cosmos_cosmos2020_farmer_matched.parquet"),
    "--truth-output",
    str(REPO_ROOT / "outputs" / "dp2_cosmos_external_truth_sample.parquet"),
    "--output-dir",
    str(REPO_ROOT / "results" / "dp2_cosmos_external_validation"),
]

raise SystemExit(subprocess.call(cmd))
