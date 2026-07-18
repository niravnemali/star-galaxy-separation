#!/usr/bin/env python3
"""Sample 100k objects from DP2_COSMOS_objects.fits and export a CSV
ready for the end-to-end morphology pS pipeline.

Reads the FITS table, converts nJy fluxes to AB magnitudes, propagates
flux errors to magnitude errors, randomly samples 100,000 rows, and
writes the result with the column names expected by ps_end_to_end.py.

Usage
-----
    python sample_dp2_cosmos_for_pipeline.py                 # defaults
    python sample_dp2_cosmos_for_pipeline.py --n 50000       # fewer rows
    python sample_dp2_cosmos_for_pipeline.py --seed 12345    # reproducibility
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

BANDS = ("u", "g", "r", "i", "z", "y")
NANOMAGGY_ZEROPOINT = 31.4
COORD_JITTER_ARCMIN = 10.0

DEFAULT_INPUT = Path(__file__).resolve().parents[1].parent / "star-galaxy-DP2-data" / "DP2_COSMOS_objects.fits"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1].parent / "End-to-end-morphology-star-probability" / "examples" / "dp2_cosmos_sample.csv"


def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    out = np.full_like(flux, np.nan, dtype="float64")
    good = np.isfinite(flux) & (flux > 0)
    out[good] = NANOMAGGY_ZEROPOINT - 2.5 * np.log10(flux[good])
    return out


def flux_err_to_mag_err(flux: np.ndarray, flux_err: np.ndarray) -> np.ndarray:
    out = np.full_like(flux, np.nan, dtype="float64")
    good = np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0) & (flux_err >= 0)
    out[good] = (2.5 / np.log(10.0)) * flux_err[good] / flux[good]
    return out


def read_fits_to_dataframe(path: Path) -> pd.DataFrame:
    with fits.open(path) as hdul:
        table = hdul[1].data
        return pd.DataFrame({col: table[col] for col in table.columns.names})


def build_pipeline_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["object_id"] = df["objectId"]
    out["coord_ra"] = np.asarray(df["coord_ra"], dtype="float64")
    out["coord_dec"] = np.asarray(df["coord_dec"], dtype="float64")

    for band in BANDS:
        psf_flux = np.asarray(df[f"{band}_psfFlux"], dtype="float64")
        psf_flux_err = np.asarray(df[f"{band}_psfFluxErr"], dtype="float64")
        cmodel_flux = np.asarray(df[f"{band}_cModelFlux"], dtype="float64")
        cmodel_flux_err = np.asarray(df[f"{band}_cModelFluxErr"], dtype="float64")

        out[f"cmodel_mag_{band}"] = flux_to_mag(cmodel_flux)
        out[f"cmodel_mag_err_{band}"] = flux_err_to_mag_err(cmodel_flux, cmodel_flux_err)
        out[f"psf_mag_{band}"] = flux_to_mag(psf_flux)
        out[f"psf_mag_err_{band}"] = flux_err_to_mag_err(psf_flux, psf_flux_err)

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to DP2_COSMOS_objects.fits")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--n", type=int, default=100_000, help="Number of objects to sample (default: 100000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading {args.input} ...")
    raw = read_fits_to_dataframe(args.input)
    print(f"  Total rows: {len(raw):,}")

    pipeline_df = build_pipeline_columns(raw)

    n_sample = min(args.n, len(pipeline_df))
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(len(pipeline_df), size=n_sample, replace=False)
    sample_idx.sort()
    sampled = pipeline_df.iloc[sample_idx].reset_index(drop=True)

    jitter_deg = COORD_JITTER_ARCMIN / 60.0
    sampled["coord_ra"] += rng.normal(0.0, jitter_deg, len(sampled))
    sampled["coord_dec"] += rng.normal(0.0, jitter_deg, len(sampled))

    n_finite = sampled[[f"cmodel_mag_{b}" for b in BANDS]].notna().all(axis=1).sum()
    print(f"  Sampled {len(sampled):,} objects ({n_finite:,} with finite CModel mags in all bands)")
    print(f"  Coordinates jittered by Gaussian sigma = {COORD_JITTER_ARCMIN} arcmin")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output, index=False)
    print(f"  Wrote {args.output}")
    print(f"  File size: {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
