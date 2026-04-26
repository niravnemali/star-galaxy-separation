"""Build a first-pass EDFS external label catalog from Gaia DR3 and NSC DR2.

Gaia DR3 is used for high-purity stars.  NSC DR2 low ``class_star`` objects are
used as first-pass extended/galaxy-like labels.  These galaxy labels are not as
strong as HST labels; they are intended to bootstrap EDFS validation where no
local HST truth table is currently available.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
EXTERNAL_DIR = DATA_DIR / "external"
DEFAULT_GAIA = EXTERNAL_DIR / "gaia_edfs_region.fits"
DEFAULT_NSC = EXTERNAL_DIR / "nsc_edfs_region.fits"


def read_table(path: Path) -> pd.DataFrame:
    from astropy.table import Table

    return Table.read(path, format="fits").to_pandas()


def as_float(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def build_gaia_star_labels(gaia: pd.DataFrame, parallax_snr_min: float = 5.0, pm_snr_min: float = 5.0) -> pd.DataFrame:
    """Select high-purity Gaia stars using astrometric significance and RUWE."""
    g = gaia.copy()
    for col in ["ra", "dec", "parallax", "parallax_error", "pmra", "pmra_error", "pmdec", "pmdec_error", "phot_g_mean_mag", "ruwe"]:
        g[col] = as_float(g, col)
    parallax_snr = np.abs(g["parallax"] / g["parallax_error"])
    pm_snr = np.sqrt((g["pmra"] / g["pmra_error"]) ** 2 + (g["pmdec"] / g["pmdec_error"]) ** 2)
    secure = (
        np.isfinite(g["ra"])
        & np.isfinite(g["dec"])
        & np.isfinite(g["ruwe"])
        & (g["ruwe"] < 1.4)
        & ((parallax_snr > parallax_snr_min) | (pm_snr > pm_snr_min))
    )
    out = pd.DataFrame(
        {
            "id": "gaia_" + g.loc[secure, "source_id"].astype(str),
            "ra": g.loc[secure, "ra"].to_numpy(),
            "dec": g.loc[secure, "dec"].to_numpy(),
            "label": "star",
            "label_source": "Gaia DR3",
            "quality_flag": "secure_gaia_star",
            "notes": f"RUWE<1.4 and parallax_snr>{parallax_snr_min} or pm_snr>{pm_snr_min}",
            "gaia_source_id": g.loc[secure, "source_id"].to_numpy(),
            "gaia_g_mag": g.loc[secure, "phot_g_mean_mag"].to_numpy(),
            "gaia_ruwe": g.loc[secure, "ruwe"].to_numpy(),
            "gaia_parallax_snr": parallax_snr.loc[secure].to_numpy(),
            "gaia_pm_snr": pm_snr.loc[secure].to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def build_nsc_galaxy_labels(nsc: pd.DataFrame, gaia_stars: pd.DataFrame, class_star_max: float = 0.1) -> pd.DataFrame:
    """Select first-pass galaxy-like NSC objects and remove Gaia-star overlaps."""
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    n = nsc.copy()
    for col in ["ra", "dec", "class_star", "imag", "ierr"]:
        n[col] = as_float(n, col)
    candidate = (
        np.isfinite(n["ra"])
        & np.isfinite(n["dec"])
        & np.isfinite(n["class_star"])
        & (n["class_star"] <= class_star_max)
        & np.isfinite(n["imag"])
        & (n["imag"] > 15)
        & (n["imag"] < 25)
        & np.isfinite(n["ierr"])
        & (n["ierr"] < 0.2)
    )
    cand = n.loc[candidate].copy().reset_index(drop=True)

    if len(cand) and len(gaia_stars):
        nsc_coord = SkyCoord(cand["ra"].to_numpy() * u.deg, cand["dec"].to_numpy() * u.deg)
        gaia_coord = SkyCoord(gaia_stars["ra"].to_numpy() * u.deg, gaia_stars["dec"].to_numpy() * u.deg)
        _, sep2d, _ = nsc_coord.match_to_catalog_sky(gaia_coord)
        cand = cand.loc[sep2d.arcsec > 1.0].copy().reset_index(drop=True)

    out = pd.DataFrame(
        {
            "id": "nsc_" + cand["id"].astype(str),
            "ra": cand["ra"].to_numpy(),
            "dec": cand["dec"].to_numpy(),
            "label": "galaxy",
            "label_source": "NSC DR2",
            "quality_flag": "first_pass_nsc_extended",
            "notes": f"class_star<={class_star_max}, 15<imag<25, ierr<0.2, >1 arcsec from secure Gaia star",
            "nsc_id": cand["id"].to_numpy(),
            "nsc_class_star": cand["class_star"].to_numpy(),
            "nsc_imag": cand["imag"].to_numpy(),
            "nsc_ierr": cand["ierr"].to_numpy(),
        }
    )
    return out.reset_index(drop=True)


def build_truth_catalog(
    gaia_path: Path = DEFAULT_GAIA,
    nsc_path: Path = DEFAULT_NSC,
    output_dir: Path = OUTPUT_DIR,
    parallax_snr_min: float = 5.0,
    pm_snr_min: float = 5.0,
    nsc_class_star_max: float = 0.1,
) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not gaia_path.exists():
        raise FileNotFoundError(f"Missing Gaia regional file: {gaia_path}")
    if not nsc_path.exists():
        raise FileNotFoundError(f"Missing NSC regional file: {nsc_path}")

    gaia = read_table(gaia_path)
    nsc = read_table(nsc_path)
    stars = build_gaia_star_labels(gaia, parallax_snr_min=parallax_snr_min, pm_snr_min=pm_snr_min)
    galaxies = build_nsc_galaxy_labels(nsc, stars, class_star_max=nsc_class_star_max)
    truth = pd.concat([stars, galaxies], ignore_index=True, sort=False)
    truth_path = output_dir / "edfs_external_truth_catalog.csv"
    truth.to_csv(truth_path, index=False)

    summary = (
        "EDFS external truth-label catalog summary\n"
        "=========================================\n\n"
        f"Gaia input: {gaia_path}\n"
        f"NSC input: {nsc_path}\n"
        f"Gaia input rows: {len(gaia)}\n"
        f"NSC input rows: {len(nsc)}\n"
        f"Secure Gaia stars: {len(stars)}\n"
        f"First-pass NSC galaxy-like labels: {len(galaxies)}\n"
        f"Combined labels: {len(truth)}\n\n"
        "Label rules:\n"
        f"  star: Gaia DR3 source with RUWE < 1.4 and parallax_snr > {parallax_snr_min} or pm_snr > {pm_snr_min}.\n"
        f"  galaxy: NSC DR2 source with class_star <= {nsc_class_star_max}, 15 < imag < 25, ierr < 0.2, and >1 arcsec from secure Gaia stars.\n\n"
        "Caution:\n"
        "  Gaia stars are high-purity astrometric labels. NSC galaxy labels are first-pass morphology-based labels, not HST-quality truth.\n"
        f"\nWrote: {truth_path}\n"
    )
    (output_dir / "edfs_external_truth_catalog_summary.txt").write_text(summary)
    print(summary)
    return truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaia", type=Path, default=DEFAULT_GAIA)
    parser.add_argument("--nsc", type=Path, default=DEFAULT_NSC)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--parallax-snr-min", type=float, default=5.0)
    parser.add_argument("--pm-snr-min", type=float, default=5.0)
    parser.add_argument("--nsc-class-star-max", type=float, default=0.1)
    args = parser.parse_args()
    build_truth_catalog(
        args.gaia,
        args.nsc,
        args.output_dir,
        parallax_snr_min=args.parallax_snr_min,
        pm_snr_min=args.pm_snr_min,
        nsc_class_star_max=args.nsc_class_star_max,
    )


if __name__ == "__main__":
    main()
