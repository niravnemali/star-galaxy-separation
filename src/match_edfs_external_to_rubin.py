"""Match EDFS Gaia/NSC external labels to the local Rubin EDFS FITS catalog."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs" / "edfs_validation"
DEFAULT_TRUTH = REPO_ROOT / "outputs" / "edfs_external_truth_catalog.csv"
DEFAULT_RUBIN = DATA_DIR / "EDFS.fits"

RA_CANDIDATES = ["coord_ra", "ra", "RA", "ALPHA_J2000", "alpha_j2000"]
DEC_CANDIDATES = ["coord_dec", "dec", "DEC", "DELTA_J2000", "delta_j2000"]
ID_CANDIDATES = ["objectId", "sourceId", "id", "ID", "object_id", "source_id"]


def first_existing_col(columns, candidates) -> str | None:
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)))


def read_rubin_table(path: Path) -> pd.DataFrame:
    from astropy.table import Table

    return Table.read(path, format="fits").to_pandas()


def match_external_to_rubin(
    truth_path: Path = DEFAULT_TRUTH,
    rubin_path: Path = DEFAULT_RUBIN,
    output_dir: Path = OUTPUT_DIR,
    max_sep_arcsec: float = 1.0,
) -> pd.DataFrame | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not truth_path.exists():
        msg = f"External EDFS truth catalog not found: {truth_path}\nRun src/build_edfs_external_truth_catalog.py first.\n"
        (output_dir / "edfs_external_match_summary.txt").write_text(msg)
        print(msg)
        return None

    from astropy import units as u
    from astropy.coordinates import SkyCoord
    import matplotlib.pyplot as plt

    truth = pd.read_csv(truth_path)
    rubin = read_rubin_table(rubin_path)
    ra_col = first_existing_col(rubin.columns, RA_CANDIDATES)
    dec_col = first_existing_col(rubin.columns, DEC_CANDIDATES)
    id_col = first_existing_col(rubin.columns, ID_CANDIDATES)
    if ra_col is None or dec_col is None:
        raise ValueError(f"Could not identify Rubin RA/Dec columns. Columns include: {list(rubin.columns)[:80]}")

    truth_coord = SkyCoord(truth["ra"].to_numpy() * u.deg, truth["dec"].to_numpy() * u.deg)
    rubin_coord = SkyCoord(rubin[ra_col].to_numpy() * u.deg, rubin[dec_col].to_numpy() * u.deg)
    idx, sep2d, _ = truth_coord.match_to_catalog_sky(rubin_coord)
    sep_arcsec = sep2d.arcsec
    within = sep_arcsec <= max_sep_arcsec

    nearest = rubin.iloc[idx].reset_index(drop=True)
    truth_reset = truth.reset_index(drop=True)
    dec_ref = np.deg2rad(truth_reset["dec"].to_numpy())
    dx = (nearest[ra_col].to_numpy() - truth_reset["ra"].to_numpy()) * np.cos(dec_ref) * 3600.0
    dy = (nearest[dec_col].to_numpy() - truth_reset["dec"].to_numpy()) * 3600.0

    # Prefix truth columns as hst_* for compatibility with existing validation code.
    truth_part = truth_reset.add_prefix("hst_")
    rubin_part = nearest.add_prefix("dp1_")
    all_matches = pd.concat([truth_part, rubin_part], axis=1)
    all_matches["match_sep_arcsec"] = sep_arcsec
    all_matches["delta_ra_cosdec_arcsec"] = dx
    all_matches["delta_dec_arcsec"] = dy
    all_matches["matched_within_arcsec"] = within
    matched = all_matches.loc[within].reset_index(drop=True)

    offset_x = float(np.nanmedian(matched["delta_ra_cosdec_arcsec"])) if len(matched) else np.nan
    offset_y = float(np.nanmedian(matched["delta_dec_arcsec"])) if len(matched) else np.nan
    matched["delta_ra_cosdec_recentered_arcsec"] = matched["delta_ra_cosdec_arcsec"] - offset_x
    matched["delta_dec_recentered_arcsec"] = matched["delta_dec_arcsec"] - offset_y
    sigma_x = robust_sigma(matched["delta_ra_cosdec_recentered_arcsec"].to_numpy())
    sigma_y = robust_sigma(matched["delta_dec_recentered_arcsec"].to_numpy())
    sx = sigma_x if np.isfinite(sigma_x) and sigma_x > 0 else 1.0
    sy = sigma_y if np.isfinite(sigma_y) and sigma_y > 0 else 1.0
    matched["clean_radial_0p5arcsec"] = matched["match_sep_arcsec"] < 0.5
    matched["match_ellipse_radius2"] = (
        (matched["delta_ra_cosdec_recentered_arcsec"] / sx) ** 2
        + (matched["delta_dec_recentered_arcsec"] / sy) ** 2
    )
    matched["clean_ellipse_3sigma"] = matched["match_ellipse_radius2"] < 9.0

    one_arcsec = output_dir / "edfs_external_dp1_matched_1arcsec.csv"
    radial_path = output_dir / "edfs_external_dp1_matched_clean_radial.csv"
    ellipse_path = output_dir / "edfs_external_dp1_matched_clean_ellipse.csv"
    matched.to_csv(one_arcsec, index=False)
    matched.loc[matched["clean_radial_0p5arcsec"]].reset_index(drop=True).to_csv(radial_path, index=False)
    matched.loc[matched["clean_ellipse_3sigma"]].reset_index(drop=True).to_csv(ellipse_path, index=False)

    plt.figure(figsize=(5, 5))
    plt.scatter(matched["delta_ra_cosdec_arcsec"], matched["delta_dec_arcsec"], s=8, alpha=0.45)
    plt.axhline(0, color="k", lw=0.8)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel(r"$\Delta$RA cos(Dec) [arcsec]")
    plt.ylabel(r"$\Delta$Dec [arcsec]")
    plt.title("EDFS: Gaia DR3 stars + NSC DR2 galaxy-like labels - Rubin EDFS residuals")
    plt.tight_layout()
    plt.savefig(output_dir / "edfs_match_delta_ra_dec.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(matched["delta_ra_cosdec_recentered_arcsec"], matched["delta_dec_recentered_arcsec"], s=8, alpha=0.45)
    theta = np.linspace(0, 2 * np.pi, 400)
    plt.plot(3 * sx * np.cos(theta), 3 * sy * np.sin(theta), color="red", lw=1.2)
    plt.axhline(0, color="k", lw=0.8)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel(r"Recentered $\Delta$RA cos(Dec) [arcsec]")
    plt.ylabel(r"Recentered $\Delta$Dec [arcsec]")
    plt.title("EDFS: recentered Gaia/NSC labels - Rubin EDFS residuals")
    plt.tight_layout()
    plt.savefig(output_dir / "edfs_match_delta_ra_dec_recentered.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(matched["match_sep_arcsec"], bins=60, range=(0, max_sep_arcsec), histtype="stepfilled", alpha=0.45, label="1 arcsec")
    plt.hist(matched.loc[matched["clean_radial_0p5arcsec"], "match_sep_arcsec"], bins=60, range=(0, max_sep_arcsec), histtype="step", lw=1.8, label="<0.5 arcsec")
    plt.hist(matched.loc[matched["clean_ellipse_3sigma"], "match_sep_arcsec"], bins=60, range=(0, max_sep_arcsec), histtype="step", lw=1.8, label="3-sigma ellipse")
    plt.axvline(0.5, color="red", ls="--")
    plt.xlabel("Nearest-neighbor separation [arcsec]")
    plt.ylabel("Matched EDFS Gaia/NSC labels")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "edfs_match_separation_hist_clean.png", dpi=180)
    plt.close()

    clean_ellipse = int(matched["clean_ellipse_3sigma"].sum())
    summary = (
        "EDFS external-label to Rubin positional match summary\n"
        "=====================================================\n\n"
        f"External truth labels: {len(truth)}\n"
        f"Rubin EDFS sources: {len(rubin)}\n"
        f"Rubin catalog: {rubin_path}\n"
        f"Rubin RA column: {ra_col}\n"
        f"Rubin Dec column: {dec_col}\n"
        f"Rubin ID column: {id_col or 'not identified'}\n"
        f"Matches within {max_sep_arcsec:.1f} arcsec: {len(matched)}\n"
        f"Median separation within cut: {np.nanmedian(matched['match_sep_arcsec']):.4f} arcsec\n"
        f"Median offset dx: {offset_x:.4f} arcsec\n"
        f"Median offset dy: {offset_y:.4f} arcsec\n"
        f"Robust sigma_x after recentering: {sigma_x:.4f} arcsec\n"
        f"Robust sigma_y after recentering: {sigma_y:.4f} arcsec\n"
        f"Surviving radial separation < 0.5 arcsec: {int(matched['clean_radial_0p5arcsec'].sum())}\n"
        f"Surviving recentered 3-sigma elliptical cut: {clean_ellipse}\n"
        f"Labels within 1 arcsec by class:\n{matched['hst_label'].value_counts().to_string()}\n\n"
        "Recommended default clean matched sample: recentered 3-sigma elliptical cut.\n"
        "Caution: hst_label column is used only for compatibility; labels come from Gaia DR3 and NSC DR2, not HST.\n\n"
        f"Wrote: {one_arcsec}\n"
        f"Wrote: {radial_path}\n"
        f"Wrote: {ellipse_path}\n"
    )
    (output_dir / "edfs_external_match_summary.txt").write_text(summary)
    print(summary)
    return matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--rubin", type=Path, default=DEFAULT_RUBIN)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-sep-arcsec", type=float, default=1.0)
    args = parser.parse_args()
    match_external_to_rubin(args.truth, args.rubin, args.output_dir, args.max_sep_arcsec)


if __name__ == "__main__":
    main()
