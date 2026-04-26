"""Positionally match HST truth labels to the repo DP1 FITS catalog."""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_HST_TRUTH = OUTPUT_DIR / "hst_truth_catalog.csv"
DEFAULT_DP1_CATALOG = DATA_DIR / "ECDFS.fits"

RA_CANDIDATES = ["coord_ra", "ra", "RA", "ALPHA_J2000", "alpha_j2000"]
DEC_CANDIDATES = ["coord_dec", "dec", "DEC", "DELTA_J2000", "delta_j2000"]
ID_CANDIDATES = ["objectId", "sourceId", "id", "ID", "object_id", "source_id"]


def is_git_lfs_pointer(path: Path) -> bool:
    return path.exists() and path.read_bytes()[:64].startswith(b"version https://git-lfs.github.com")


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
    return 1.4826 * np.median(np.abs(x - med))


def _safe_sigma(value: float, fallback: float = 1.0) -> float:
    """Avoid division by zero in normalized match-residual cuts."""
    if np.isfinite(value) and value > 0:
        return float(value)
    return fallback


def read_dp1_table(path: Path):
    if is_git_lfs_pointer(path):
        raise RuntimeError(
            f"{path} is currently a Git LFS pointer, not a materialized FITS table. "
            "Install/use git-lfs and pull the LFS object before matching."
        )
    try:
        from astropy.table import Table
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("astropy is required to read the DP1 FITS table.") from exc
    return Table.read(path, format="fits").to_pandas()


def match_hst_to_dp1(
    hst_truth_path: Path = DEFAULT_HST_TRUTH,
    dp1_path: Path = DEFAULT_DP1_CATALOG,
    output_dir: Path = OUTPUT_DIR,
    max_sep_arcsec: float = 1.0,
) -> pd.DataFrame | None:
    """Run nearest-neighbor sky matching and save the matched table plus diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if not hst_truth_path.exists():
        msg = (
            f"HST truth catalog not found: {hst_truth_path}\n"
            "Run src/build_hst_truth_catalog.py first with a local HST/3D-HST GOODS-S catalog.\n"
        )
        (output_dir / "match_summary.txt").write_text(msg)
        print(msg)
        return None

    try:
        dp1 = read_dp1_table(dp1_path)
    except Exception as exc:
        msg = (
            "Could not read the DP1 matching catalog.\n\n"
            f"DP1 catalog: {dp1_path}\n"
            f"Error: {exc}\n"
        )
        (output_dir / "match_summary.txt").write_text(msg)
        print(msg)
        return None

    hst = pd.read_csv(hst_truth_path)
    dp1_ra_col = first_existing_col(dp1.columns, RA_CANDIDATES)
    dp1_dec_col = first_existing_col(dp1.columns, DEC_CANDIDATES)
    dp1_id_col = first_existing_col(dp1.columns, ID_CANDIDATES)
    if dp1_ra_col is None or dp1_dec_col is None:
        raise ValueError(f"Could not identify DP1 RA/Dec columns. Columns include: {list(dp1.columns)[:80]}")

    from astropy.coordinates import SkyCoord
    from astropy import units as u
    import matplotlib.pyplot as plt

    hst_coord = SkyCoord(hst["ra"].to_numpy() * u.deg, hst["dec"].to_numpy() * u.deg, frame="icrs")
    dp1_coord = SkyCoord(dp1[dp1_ra_col].to_numpy() * u.deg, dp1[dp1_dec_col].to_numpy() * u.deg, frame="icrs")

    idx, sep2d, _ = hst_coord.match_to_catalog_sky(dp1_coord)
    sep_arcsec = sep2d.arcsec
    within = sep_arcsec <= max_sep_arcsec

    nearest = dp1.iloc[idx].reset_index(drop=True)
    matched_hst = hst.reset_index(drop=True)
    dec_ref = np.deg2rad(matched_hst["dec"].to_numpy())
    delta_ra_arcsec = (nearest[dp1_ra_col].to_numpy() - matched_hst["ra"].to_numpy()) * np.cos(dec_ref) * 3600.0
    delta_dec_arcsec = (nearest[dp1_dec_col].to_numpy() - matched_hst["dec"].to_numpy()) * 3600.0

    hst_part = matched_hst.add_prefix("hst_")
    dp1_part = nearest.add_prefix("dp1_")
    all_matches = pd.concat([hst_part, dp1_part], axis=1)
    all_matches["match_sep_arcsec"] = sep_arcsec
    all_matches["delta_ra_cosdec_arcsec"] = delta_ra_arcsec
    all_matches["delta_dec_arcsec"] = delta_dec_arcsec
    all_matches["matched_within_arcsec"] = within

    matched = all_matches.loc[within].reset_index(drop=True)
    out_path = output_dir / "hst_dp1_matched_1arcsec.csv"

    # Matching-quality cuts are defined only from the compact 1 arcsec match core.
    # First measure and subtract any systematic HST-DP1 astrometric offset.
    offset_x = float(np.nanmedian(matched["delta_ra_cosdec_arcsec"])) if len(matched) else np.nan
    offset_y = float(np.nanmedian(matched["delta_dec_arcsec"])) if len(matched) else np.nan
    matched["delta_ra_cosdec_recentered_arcsec"] = matched["delta_ra_cosdec_arcsec"] - offset_x
    matched["delta_dec_recentered_arcsec"] = matched["delta_dec_arcsec"] - offset_y

    sigma_x = robust_sigma(matched["delta_ra_cosdec_recentered_arcsec"].to_numpy())
    sigma_y = robust_sigma(matched["delta_dec_recentered_arcsec"].to_numpy())
    sigma_x_safe = _safe_sigma(sigma_x)
    sigma_y_safe = _safe_sigma(sigma_y)

    matched["clean_radial_0p5arcsec"] = matched["match_sep_arcsec"] < 0.5
    ellipse_radius2 = (
        (matched["delta_ra_cosdec_recentered_arcsec"] / sigma_x_safe) ** 2
        + (matched["delta_dec_recentered_arcsec"] / sigma_y_safe) ** 2
    )
    matched["match_ellipse_radius2"] = ellipse_radius2
    matched["clean_ellipse_3sigma"] = ellipse_radius2 < 9.0

    matched.to_csv(out_path, index=False)
    radial_clean = matched.loc[matched["clean_radial_0p5arcsec"]].reset_index(drop=True)
    ellipse_clean = matched.loc[matched["clean_ellipse_3sigma"]].reset_index(drop=True)
    radial_path = output_dir / "hst_dp1_matched_clean_radial.csv"
    ellipse_path = output_dir / "hst_dp1_matched_clean_ellipse.csv"
    radial_clean.to_csv(radial_path, index=False)
    ellipse_clean.to_csv(ellipse_path, index=False)

    plt.figure(figsize=(6, 4))
    plt.hist(sep_arcsec[np.isfinite(sep_arcsec)], bins=60, range=(0, min(5, np.nanmax(sep_arcsec))), histtype="stepfilled", alpha=0.75)
    plt.axvline(max_sep_arcsec, color="red", ls="--", label=f"{max_sep_arcsec:.1f} arcsec")
    plt.xlabel("Nearest-neighbor separation [arcsec]")
    plt.ylabel("Number of ECDFS HST/3D-HST sources")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "match_separation_hist.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(delta_ra_arcsec[within], delta_dec_arcsec[within], s=8, alpha=0.45)
    plt.axhline(0, color="k", lw=0.8)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel(r"$\Delta$RA cos(Dec) [arcsec]")
    plt.ylabel(r"$\Delta$Dec [arcsec]")
    plt.title("ECDFS: HST/3D-HST GOODS-S labels - Rubin ECDFS residuals")
    plt.tight_layout()
    plt.savefig(output_dir / "match_delta_ra_dec.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(
        matched["delta_ra_cosdec_recentered_arcsec"],
        matched["delta_dec_recentered_arcsec"],
        s=8,
        alpha=0.45,
        label="1 arcsec matches",
    )
    theta = np.linspace(0, 2 * np.pi, 400)
    plt.plot(3 * sigma_x_safe * np.cos(theta), 3 * sigma_y_safe * np.sin(theta),
             color="red", lw=1.2, label="3-sigma ellipse")
    plt.axhline(0, color="k", lw=0.8)
    plt.axvline(0, color="k", lw=0.8)
    plt.xlabel(r"Recentered $\Delta$RA cos(Dec) [arcsec]")
    plt.ylabel(r"Recentered $\Delta$Dec [arcsec]")
    plt.title("ECDFS: recentered HST/3D-HST - Rubin ECDFS residuals")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "match_delta_ra_dec_recentered.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(matched["match_sep_arcsec"], bins=60, range=(0, max_sep_arcsec),
             histtype="stepfilled", alpha=0.45, label="1 arcsec matches")
    plt.hist(radial_clean["match_sep_arcsec"], bins=60, range=(0, max_sep_arcsec),
             histtype="step", lw=1.8, label="radial < 0.5 arcsec")
    plt.hist(ellipse_clean["match_sep_arcsec"], bins=60, range=(0, max_sep_arcsec),
             histtype="step", lw=1.8, label="3-sigma ellipse")
    plt.axvline(0.5, color="red", ls="--", label="0.5 arcsec")
    plt.xlabel("Nearest-neighbor separation [arcsec]")
    plt.ylabel("Number of ECDFS HST/3D-HST sources")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "match_separation_hist_clean.png", dpi=180)
    plt.close()

    sig_ra = robust_sigma(delta_ra_arcsec[within])
    sig_dec = robust_sigma(delta_dec_arcsec[within])
    sig_2d = np.sqrt(sig_ra**2 + sig_dec**2)
    median_sep = float(np.nanmedian(sep_arcsec[within])) if np.any(within) else np.nan
    rec_05 = "yes" if np.isfinite(sig_2d) and 3 * sig_2d <= 0.5 else "inspect residual plots first"
    recommended_default = "3-sigma elliptical cut on recentered residuals"
    summary = (
        "ECDFS HST/3D-HST-to-Rubin positional match summary\n"
        "===================================\n\n"
        f"HST labeled sources: {len(hst)}\n"
        f"DP1 sources in chosen catalog: {len(dp1)}\n"
        f"DP1 catalog: {dp1_path}\n"
        f"DP1 RA column: {dp1_ra_col}\n"
        f"DP1 Dec column: {dp1_dec_col}\n"
        f"DP1 ID column: {dp1_id_col or 'not identified'}\n"
        f"Nearest-neighbor matches attempted: {len(hst)}\n"
        f"Matches within {max_sep_arcsec:.1f} arcsec: {int(within.sum())}\n"
        f"Median separation within cut: {median_sep:.4f} arcsec\n"
        f"Median astrometric offset Delta RA cos(Dec): {offset_x:.4f} arcsec\n"
        f"Median astrometric offset Delta Dec: {offset_y:.4f} arcsec\n"
        f"Robust sigma Delta RA cos(Dec), before recentering: {sig_ra:.4f} arcsec\n"
        f"Robust sigma Delta Dec, before recentering: {sig_dec:.4f} arcsec\n"
        f"Robust sigma_x after recentering: {sigma_x:.4f} arcsec\n"
        f"Robust sigma_y after recentering: {sigma_y:.4f} arcsec\n"
        f"Quadrature robust residual scale: {sig_2d:.4f} arcsec\n"
        f"Surviving radial separation < 0.5 arcsec: {len(radial_clean)}\n"
        f"Surviving recentered 3-sigma elliptical cut: {len(ellipse_clean)}\n"
        f"Recommended default clean matched sample: {recommended_default}\n"
        f"Is 0.5 arcsec likely reasonable? {rec_05}\n"
        "Suggested 3-sigma cut: use the recentered residuals and require "
        "((dx'/sigma_x)^2 + (dy'/sigma_y)^2) < 9 after inspecting the residual plots.\n"
        f"\nWrote radial clean sample: {radial_path}\n"
        f"Wrote elliptical clean sample: {ellipse_path}\n"
    )
    (output_dir / "match_summary.txt").write_text(summary)
    print(summary)
    print(f"Wrote {out_path}")
    return matched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hst-truth", type=Path, default=DEFAULT_HST_TRUTH)
    parser.add_argument("--dp1", type=Path, default=DEFAULT_DP1_CATALOG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--max-sep-arcsec", type=float, default=1.0)
    args = parser.parse_args()
    match_hst_to_dp1(args.hst_truth, args.dp1, args.output_dir, args.max_sep_arcsec)


if __name__ == "__main__":
    main()
