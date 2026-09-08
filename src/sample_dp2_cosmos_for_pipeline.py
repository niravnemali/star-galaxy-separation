#!/usr/bin/env python3
"""Sample objects from DP2_COSMOS_objects.fits and export a CSV
ready for the end-to-end morphology pS pipeline.

Reads the FITS table, cross-matches against the COSMOS2020 FARMER truth
catalog (HST ACS labels) to obtain star/galaxy labels, converts nJy
fluxes to AB magnitudes, propagates flux errors to magnitude errors,
randomly samples from the full DP2 catalog, and writes the result with
the column names expected by ps_end_to_end.py.

Objects without a COSMOS2020 FARMER match receive hst_label = -1.

Usage
-----
    python sample_dp2_cosmos_for_pipeline.py                 # full catalog
    python sample_dp2_cosmos_for_pipeline.py --n 50000       # subsample
    python sample_dp2_cosmos_for_pipeline.py --seed 12345    # reproducible subsample
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

BANDS = ("u", "g", "r", "i", "z", "y")
NANOMAGGY_ZEROPOINT = 31.4

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT.parent / "star-galaxy-DP2-data" / "DP2_COSMOS_objects.fits"
DEFAULT_TRUTH = REPO_ROOT / "data" / "cosmos2020_farmer_truth_catalog_github.csv"
DEFAULT_OUTPUT = REPO_ROOT.parent / "End-to-end-morphology-star-probability" / "examples" / "dp2_cosmos_sample.csv"

MAX_SEP_ARCSEC = 0.3


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


def add_hst_labels(dp2: pd.DataFrame, truth_path: Path) -> pd.DataFrame:
    """Cross-match DP2 objects to COSMOS2020 FARMER truth catalog.

    Returns all dp2 rows with an added ``hst_label`` column:
    0 = star, 1 = galaxy, -1 = no match.
    """
    truth = pd.read_csv(truth_path)

    dp2_ra = np.asarray(dp2["coord_ra"], dtype="float64")
    dp2_dec = np.asarray(dp2["coord_dec"], dtype="float64")
    truth_ra = np.asarray(truth["ra"], dtype="float64")
    truth_dec = np.asarray(truth["dec"], dtype="float64")

    dp2_coord = SkyCoord(dp2_ra * u.deg, dp2_dec * u.deg)
    truth_coord = SkyCoord(truth_ra * u.deg, truth_dec * u.deg)

    idx, sep2d, _ = dp2_coord.match_to_catalog_sky(truth_coord)
    within = sep2d.arcsec <= MAX_SEP_ARCSEC

    result = dp2.copy()
    result["hst_label"] = -1

    matched_truth_labels = truth.iloc[idx[within]]["label"].values
    label_map = {"star": 0, "galaxy": 1}
    mapped = pd.Series(matched_truth_labels).map(label_map)

    sep_arcsec = sep2d.arcsec[within]
    truth_ids = truth.iloc[idx[within]]["id"].values

    dedup_df = pd.DataFrame({
        "_dp2_idx": np.where(within)[0],
        "_truth_id": truth_ids,
        "_sep_arcsec": sep_arcsec,
        "_label": mapped.values,
    })
    dedup_df = dedup_df.dropna(subset=["_label"])
    dedup_df = dedup_df.sort_values("_sep_arcsec").drop_duplicates(subset=["_truth_id"], keep="first")
    dedup_df["_label"] = dedup_df["_label"].astype(int)

    result.iloc[dedup_df["_dp2_idx"].values, result.columns.get_loc("hst_label")] = dedup_df["_label"].values

    return result.reset_index(drop=True)


def build_pipeline_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["object_id"] = df["objectId"].values
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

    out["hst_label"] = df["hst_label"].values
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to DP2_COSMOS_objects.fits")
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH, help="Path to COSMOS2020 FARMER truth catalog CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--n", type=int, default=None, help="Max number of objects to sample (default: all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (only used when --n is set)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading {args.input} ...")
    raw = read_fits_to_dataframe(args.input)
    print(f"  Total DP2 rows: {len(raw):,}")

    print(f"Cross-matching to {args.truth} (max sep = {MAX_SEP_ARCSEC} arcsec) ...")
    labeled = add_hst_labels(raw, args.truth)
    n_stars = (labeled["hst_label"] == 0).sum()
    n_galaxies = (labeled["hst_label"] == 1).sum()
    n_unmatched = (labeled["hst_label"] == -1).sum()
    print(f"  HST labels: {n_stars:,} stars, {n_galaxies:,} galaxies, {n_unmatched:,} unmatched (-1)")

    pipeline_df = build_pipeline_columns(labeled)

    if args.n is not None and args.n < len(pipeline_df):
        rng = np.random.default_rng(args.seed)
        sample_idx = rng.choice(len(pipeline_df), size=args.n, replace=False)
        sample_idx.sort()
        sampled = pipeline_df.iloc[sample_idx].reset_index(drop=True)
    else:
        sampled = pipeline_df

    n_finite = sampled[[f"cmodel_mag_{b}" for b in BANDS]].notna().all(axis=1).sum()
    n_labeled = (sampled["hst_label"] >= 0).sum()
    print(f"  Output: {len(sampled):,} objects ({n_labeled:,} with HST labels, {n_finite:,} with finite CModel mags in all bands)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(args.output, index=False)
    print(f"  Wrote {args.output}")
    print(f"  File size: {args.output.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
