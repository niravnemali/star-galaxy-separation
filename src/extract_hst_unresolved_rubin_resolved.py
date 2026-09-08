#!/usr/bin/env python3
"""Extract COSMOS sources where HST says point source but Rubin says extended.

Selects DP2 COSMOS objects that are

  * bright:            r_cModelMag < 23
  * HST-unresolved:    matched to a COSMOS2020 FARMER row with ACS_MU_CLASS = 2
                       (``label == "star"`` in the compact truth catalog)
  * Rubin-resolved:    refExtendedness == 1

and pass standard measurement-quality cuts, then writes them to CSV with the
RA/DEC columns the cutout notebook expects plus diagnostic columns.

These are the objects where the two classifiers disagree, so the point of the
output is visual inspection: real HST/Rubin disagreements, deblending failures,
and marginally resolved galaxies all land in here and only cutouts separate them.

Usage
-----
    python extract_hst_unresolved_rubin_resolved.py
    python extract_hst_unresolved_rubin_resolved.py --r-max 22.5 --no-quality-cuts
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
NANOJANSKY_ZEROPOINT = 31.4

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT.parent / "star-galaxy-DP2-data" / "DP2_COSMOS_objects.fits"
DEFAULT_TRUTH = REPO_ROOT / "data" / "cosmos2020_farmer_truth_catalog_github.csv"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "cosmos_hst_unresolved_rubin_resolved.csv"

MAX_SEP_ARCSEC = 0.3
MAX_BLENDEDNESS = 0.1

# Columns carried through to the output for inspection alongside the cutouts.
DIAGNOSTIC_COLUMNS = [
    "objectId",
    "refBand",
    "refExtendedness",
    "detect_fromBlend",
    "detect_isIsolated",
    "parentObjectId",
]


def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    out = np.full(flux.shape, np.nan, dtype="float64")
    good = np.isfinite(flux) & (flux > 0)
    out[good] = NANOJANSKY_ZEROPOINT - 2.5 * np.log10(flux[good])
    return out


def read_fits_to_dataframe(path: Path) -> pd.DataFrame:
    with fits.open(path) as hdul:
        table = hdul[1].data
        # Byte-swap big-endian FITS columns into native order for pandas.
        data = {}
        for col in table.columns.names:
            arr = np.asarray(table[col])
            if arr.dtype.byteorder == ">":
                arr = arr.astype(arr.dtype.newbyteorder("="))
            data[col] = arr
        return pd.DataFrame(data)


def match_to_truth(dp2: pd.DataFrame, truth_path: Path) -> pd.DataFrame:
    """Nearest-neighbour match DP2 to the COSMOS2020 FARMER truth catalog.

    Adds ``hst_label`` (0 = star/unresolved, 1 = galaxy, -1 = no match),
    ``hst_id`` and ``hst_sep_arcsec``. Each truth source is used at most once,
    keeping the closest DP2 counterpart, mirroring the convention in
    sample_dp2_cosmos_for_pipeline.py.
    """
    truth = pd.read_csv(truth_path)

    dp2_coord = SkyCoord(
        np.asarray(dp2["coord_ra"], dtype="float64") * u.deg,
        np.asarray(dp2["coord_dec"], dtype="float64") * u.deg,
    )
    truth_coord = SkyCoord(
        np.asarray(truth["ra"], dtype="float64") * u.deg,
        np.asarray(truth["dec"], dtype="float64") * u.deg,
    )

    idx, sep2d, _ = dp2_coord.match_to_catalog_sky(truth_coord)
    within = sep2d.arcsec <= MAX_SEP_ARCSEC

    result = dp2.copy()
    result["hst_label"] = -1
    result["hst_id"] = -1
    result["hst_sep_arcsec"] = np.nan
    result["hst_acs_mu_class"] = -1
    result["hst_flag_combined"] = -1

    matched = truth.iloc[idx[within]]
    label_map = {"star": 0, "galaxy": 1}

    dedup = pd.DataFrame({
        "_dp2_idx": np.where(within)[0],
        "_truth_id": matched["id"].to_numpy(),
        "_sep": sep2d.arcsec[within],
        "_label": pd.Series(matched["label"].to_numpy()).map(label_map).to_numpy(),
        "_mu_class": matched["acs_mu_class"].to_numpy(),
        "_flag": matched["flag_combined"].to_numpy(),
    })
    dedup = dedup.dropna(subset=["_label"])
    dedup = dedup.sort_values("_sep").drop_duplicates(subset=["_truth_id"], keep="first")

    rows = dedup["_dp2_idx"].to_numpy()
    result.iloc[rows, result.columns.get_loc("hst_label")] = dedup["_label"].astype(int).to_numpy()
    result.iloc[rows, result.columns.get_loc("hst_id")] = dedup["_truth_id"].to_numpy()
    result.iloc[rows, result.columns.get_loc("hst_sep_arcsec")] = dedup["_sep"].to_numpy()
    result.iloc[rows, result.columns.get_loc("hst_acs_mu_class")] = dedup["_mu_class"].to_numpy()
    result.iloc[rows, result.columns.get_loc("hst_flag_combined")] = dedup["_flag"].to_numpy()

    return result


def select(df: pd.DataFrame, r_max: float, quality_cuts: bool) -> pd.DataFrame:
    """Apply the selection, printing how many objects survive each stage."""
    r_mag = np.asarray(df["r_cModelMag"], dtype="float64")
    ref_ext = np.asarray(df["refExtendedness"], dtype="float64")

    stages = [
        ("all DP2 COSMOS objects", np.ones(len(df), dtype=bool)),
        (f"r_cModelMag < {r_max}", np.isfinite(r_mag) & (r_mag < r_max)),
        ("HST unresolved (ACS_MU_CLASS = 2)", np.asarray(df["hst_label"]) == 0),
        ("Rubin resolved (refExtendedness = 1)", ref_ext == 1),
    ]

    if quality_cuts:
        blendedness = np.asarray(df["r_blendedness"], dtype="float64")
        stages += [
            ("r cModel flag clear", ~np.asarray(df["r_cModel_flag"], dtype=bool)),
            ("r extendedness flag clear", ~np.asarray(df["r_extendedness_flag"], dtype=bool)),
            (f"r blendedness < {MAX_BLENDEDNESS}", np.isfinite(blendedness) & (blendedness < MAX_BLENDEDNESS)),
        ]

    mask = np.ones(len(df), dtype=bool)
    for name, stage_mask in stages:
        mask &= stage_mask
        print(f"  {name:<40s} {int(mask.sum()):>8,d}")

    return df[mask].reset_index(drop=True)


def build_output(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble the output table: RA/DEC first, then mags and diagnostics."""
    out = pd.DataFrame()
    # The cutout notebook reads 'RA' and 'DEC'; keep coord_* too for traceability.
    out["RA"] = np.asarray(df["coord_ra"], dtype="float64")
    out["DEC"] = np.asarray(df["coord_dec"], dtype="float64")

    for col in DIAGNOSTIC_COLUMNS:
        out[col] = df[col].to_numpy()

    for band in BANDS:
        out[f"{band}_cModelMag"] = flux_to_mag(np.asarray(df[f"{band}_cModelFlux"], dtype="float64"))
        out[f"{band}_psfMag"] = flux_to_mag(np.asarray(df[f"{band}_psfFlux"], dtype="float64"))
        # psf - cModel is the classic point-source discriminant; near 0 means unresolved.
        out[f"{band}_psf_minus_cModel"] = out[f"{band}_psfMag"] - out[f"{band}_cModelMag"]
        out[f"{band}_extendedness"] = np.asarray(df[f"{band}_extendedness"], dtype="float64")
        out[f"{band}_blendedness"] = np.asarray(df[f"{band}_blendedness"], dtype="float64")

    for col in ["hst_id", "hst_label", "hst_sep_arcsec", "hst_acs_mu_class", "hst_flag_combined"]:
        out[col] = df[col].to_numpy()

    # Brightest first, so a truncated cutout run still inspects the best-measured objects.
    return out.sort_values("r_cModelMag").reset_index(drop=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to DP2_COSMOS_objects.fits")
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH, help="COSMOS2020 FARMER truth catalog CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV path")
    parser.add_argument("--r-max", type=float, default=23.0, help="Faint limit in r cModel mag (default: 23)")
    parser.add_argument(
        "--no-quality-cuts",
        action="store_true",
        help="Skip the flag/blendedness cuts and select on r-mag and classification only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Reading {args.input} ...")
    raw = read_fits_to_dataframe(args.input)
    print(f"  {len(raw):,} rows")

    print(f"Matching to {args.truth} (max sep {MAX_SEP_ARCSEC} arcsec) ...")
    labeled = match_to_truth(raw, args.truth)
    print(f"  {int((labeled['hst_label'] == 0).sum()):,} HST stars, "
          f"{int((labeled['hst_label'] == 1).sum()):,} HST galaxies, "
          f"{int((labeled['hst_label'] == -1).sum()):,} unmatched")

    print("Selection:")
    selected = select(labeled, args.r_max, quality_cuts=not args.no_quality_cuts)

    out = build_output(selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"\nWrote {len(out):,} objects to {args.output}")
    if len(out):
        print(f"  r_cModelMag range: {out['r_cModelMag'].min():.2f} - {out['r_cModelMag'].max():.2f}")
        print(f"  median r psf-cModel: {out['r_psf_minus_cModel'].median():.3f} mag")
        print(f"  mosaics at 25 per page: {int(np.ceil(len(out) / 25))}")


if __name__ == "__main__":
    main()
