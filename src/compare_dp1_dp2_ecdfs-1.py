"""Direct DP1 vs DP2 ECDFS diagnostics for PSF/CModel quantities.

This script is intentionally diagnostic rather than a replacement for the
existing pS pipelines.  It compares the DP1 ECDFS FITS export against the DP2
ECDFS FITS export using the same derived quantities and the same sign
convention:

    delta = psf_mag - cmodel_mag

Positive delta means the PSF magnitude is fainter than the CModel magnitude.
The same convention is used by the prepared DP1/DP2 analysis tables.

For the unresolved/resolved follow-up diagnostics, two magnitude-difference
sign conventions are reported explicitly:

    psfDP1 - psfDP2
    CModelDP1 - CModelDP2

and the morphology change is reported in both directions:

    delta_change_DP2_minus_DP1 = delta_DP2 - delta_DP1
    delta_change_DP1_minus_DP2 = delta_DP1 - delta_DP2
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dp2_external_validation import read_table

BANDS = ("g", "r", "i", "z")
MAG_BINS = [
    ("mag < 22", None, 22.0),
    ("22 <= mag < 23", 22.0, 23.0),
    ("23 <= mag < 24", 23.0, 24.0),
    ("24 <= mag < 25", 24.0, 25.0),
]
MAG_RANGE = (15.0, 28.0)
DELTA_RANGE = (-3.0, 3.0)
SLICE_DELTA_RANGE = (-1.5, 1.5)
HST_MATCH_RADIUS_ARCSEC = 0.3
COMMON_MATCH_RADIUS_ARCSEC = 0.3


@dataclass
class StandardizedCatalog:
    name: str
    path: Path
    df: pd.DataFrame


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def flux_to_mag(flux: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(flux, dtype="float64")
    out = np.full(arr.shape, np.nan, dtype="float64")
    good = np.isfinite(arr) & (arr > 0)
    out[good] = 31.4 - 2.5 * np.log10(arr[good])
    return out


def robust_sigma(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan")
    med = np.nanmedian(arr)
    return float(1.4826 * np.nanmedian(np.abs(arr - med)))


def finite_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def flag_clean(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(True, index=index)
    if series.dtype == bool:
        return ~series.fillna(True)
    vals = pd.to_numeric(series, errors="coerce")
    return vals.fillna(1).eq(0)


def selected_columns_for_dp1(bands: Iterable[str]) -> list[str]:
    cols = ["objectId", "coord_ra", "coord_dec"]
    for band in bands:
        cols.extend(
            [
                f"{band}_free_psfFlux",
                f"{band}_free_psfFluxErr",
                f"{band}_free_psfFlux_flag",
                f"{band}_free_cModelFlux",
                f"{band}_free_cModelFluxErr",
                f"{band}_free_cModelFlux_flag",
                f"{band}_extendedness",
                f"{band}_extendedness_flag",
            ]
        )
    return cols


def selected_columns_for_dp2(bands: Iterable[str]) -> list[str]:
    cols = ["objectId", "coord_ra", "coord_dec"]
    for band in bands:
        cols.extend(
            [
                f"{band}_psfFlux",
                f"{band}_psfFluxErr",
                f"{band}_psfFlux_flag",
                f"{band}_cModelFlux",
                f"{band}_cModelFluxErr",
                f"{band}_cModel_flag",
                f"{band}_extendedness",
                f"{band}_extendedness_flag",
            ]
        )
    return cols


def standardize_dp1(path: Path, bands: Iterable[str]) -> StandardizedCatalog:
    raw = read_table(path, columns=selected_columns_for_dp1(bands))
    out = pd.DataFrame(
        {
            "object_id": pd.to_numeric(raw["objectId"], errors="coerce"),
            "ra": pd.to_numeric(raw["coord_ra"], errors="coerce"),
            "dec": pd.to_numeric(raw["coord_dec"], errors="coerce"),
        }
    )
    for band in bands:
        pflux = finite_numeric(raw[f"{band}_free_psfFlux"])
        cflux = finite_numeric(raw[f"{band}_free_cModelFlux"])
        out[f"psf_flux_{band}"] = pflux
        out[f"cmodel_flux_{band}"] = cflux
        out[f"psf_flux_err_{band}"] = finite_numeric(raw[f"{band}_free_psfFluxErr"])
        out[f"cmodel_flux_err_{band}"] = finite_numeric(raw[f"{band}_free_cModelFluxErr"])
        out[f"psf_mag_{band}"] = flux_to_mag(pflux)
        out[f"cmodel_mag_{band}"] = flux_to_mag(cflux)
        out[f"psf_minus_cmodel_{band}"] = out[f"psf_mag_{band}"] - out[f"cmodel_mag_{band}"]
        out[f"extendedness_{band}"] = finite_numeric(raw[f"{band}_extendedness"])
        pclean = flag_clean(raw.get(f"{band}_free_psfFlux_flag"), out.index)
        cclean = flag_clean(raw.get(f"{band}_free_cModelFlux_flag"), out.index)
        eclean = flag_clean(raw.get(f"{band}_extendedness_flag"), out.index)
        out[f"clean_{band}"] = (
            np.isfinite(out[f"psf_mag_{band}"])
            & np.isfinite(out[f"cmodel_mag_{band}"])
            & np.isfinite(out[f"psf_minus_cmodel_{band}"])
            & (pflux > 0)
            & (cflux > 0)
            & pclean
            & cclean
        )
        out[f"extendedness_clean_{band}"] = np.isfinite(out[f"extendedness_{band}"]) & eclean
    return StandardizedCatalog("DP1", path, out)


def standardize_dp2(path: Path, bands: Iterable[str], ps_path: Path | None = None) -> StandardizedCatalog:
    raw = read_table(path, columns=selected_columns_for_dp2(bands))
    out = pd.DataFrame(
        {
            "object_id": pd.to_numeric(raw["objectId"], errors="coerce"),
            "ra": pd.to_numeric(raw["coord_ra"], errors="coerce"),
            "dec": pd.to_numeric(raw["coord_dec"], errors="coerce"),
        }
    )
    for band in bands:
        pflux = finite_numeric(raw[f"{band}_psfFlux"])
        cflux = finite_numeric(raw[f"{band}_cModelFlux"])
        out[f"psf_flux_{band}"] = pflux
        out[f"cmodel_flux_{band}"] = cflux
        out[f"psf_flux_err_{band}"] = finite_numeric(raw[f"{band}_psfFluxErr"])
        out[f"cmodel_flux_err_{band}"] = finite_numeric(raw[f"{band}_cModelFluxErr"])
        out[f"psf_mag_{band}"] = flux_to_mag(pflux)
        out[f"cmodel_mag_{band}"] = flux_to_mag(cflux)
        out[f"psf_minus_cmodel_{band}"] = out[f"psf_mag_{band}"] - out[f"cmodel_mag_{band}"]
        out[f"extendedness_{band}"] = finite_numeric(raw[f"{band}_extendedness"])
        pclean = flag_clean(raw.get(f"{band}_psfFlux_flag"), out.index)
        cclean = flag_clean(raw.get(f"{band}_cModel_flag"), out.index)
        eclean = flag_clean(raw.get(f"{band}_extendedness_flag"), out.index)
        out[f"clean_{band}"] = (
            np.isfinite(out[f"psf_mag_{band}"])
            & np.isfinite(out[f"cmodel_mag_{band}"])
            & np.isfinite(out[f"psf_minus_cmodel_{band}"])
            & (pflux > 0)
            & (cflux > 0)
            & pclean
            & cclean
        )
        out[f"extendedness_clean_{band}"] = np.isfinite(out[f"extendedness_{band}"]) & eclean

    if ps_path and ps_path.exists():
        ps = pd.read_parquet(ps_path)
        ps = ps.drop_duplicates("object_id", keep="first")
        keep = ["object_id"] + [f"pS_{band}" for band in bands if f"pS_{band}" in ps.columns]
        out = out.merge(ps[keep], on="object_id", how="left")
    return StandardizedCatalog("DP2", path, out)


def save_fig(fig: plt.Figure, path: Path) -> Path:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def valid_band(cat: StandardizedCatalog, band: str) -> pd.Series:
    return cat.df[f"clean_{band}"].astype(bool)


def plot_mag_hist(dp1: StandardizedCatalog, dp2: StandardizedCatalog, band: str, mag_kind: str, plot_dir: Path) -> Path:
    col = f"{mag_kind}_mag_{band}"
    label = "PSF" if mag_kind == "psf" else "CModel"
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    bins = np.linspace(*MAG_RANGE, 90)
    for cat, color in [(dp1, "tab:blue"), (dp2, "tab:orange")]:
        m = valid_band(cat, band) & cat.df[col].between(*MAG_RANGE)
        ax.hist(cat.df.loc[m, col], bins=bins, histtype="step", density=True, lw=2.1, label=f"{cat.name} N={int(m.sum()):,}", color=color)
    ax.set_xlabel(f"{band} {label} magnitude (AB)")
    ax.set_ylabel("Normalized density")
    ax.set_xlim(*MAG_RANGE)
    ax.set_title(f"ECDFS {band}-band {label} magnitude distribution: DP1 vs DP2")
    ax.grid(True, alpha=0.2)
    ax.legend()
    return save_fig(fig, plot_dir / f"dp1_dp2_compare_{band}_{mag_kind}_mag_hist_ECDFS.png")


def plot_delta_hist(dp1: StandardizedCatalog, dp2: StandardizedCatalog, band: str, plot_dir: Path) -> Path:
    col = f"psf_minus_cmodel_{band}"
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    bins = np.linspace(*DELTA_RANGE, 120)
    for cat, color in [(dp1, "tab:blue"), (dp2, "tab:orange")]:
        m = valid_band(cat, band) & cat.df[col].between(*DELTA_RANGE)
        ax.hist(cat.df.loc[m, col], bins=bins, histtype="step", density=True, lw=2.1, label=f"{cat.name} N={int(m.sum()):,}", color=color)
    ax.axvline(0.0, color="0.35", lw=1.2, ls=":")
    ax.set_xlabel(f"{band} psf_mag - cmodel_mag (AB)")
    ax.set_ylabel("Normalized density")
    ax.set_xlim(*DELTA_RANGE)
    ax.set_title(f"ECDFS {band}-band PSF-CModel delta distribution: DP1 vs DP2")
    ax.grid(True, alpha=0.2)
    ax.legend()
    return save_fig(fig, plot_dir / f"dp1_dp2_compare_{band}_psf_cmodel_delta_hist_ECDFS.png")


def plot_psf_vs_cmodel(dp1: StandardizedCatalog, dp2: StandardizedCatalog, band: str, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), sharex=True, sharey=True)
    for ax, cat in zip(axes, [dp1, dp2]):
        m = valid_band(cat, band) & cat.df[f"psf_mag_{band}"].between(*MAG_RANGE) & cat.df[f"cmodel_mag_{band}"].between(*MAG_RANGE)
        hb = ax.hexbin(
            cat.df.loc[m, f"cmodel_mag_{band}"],
            cat.df.loc[m, f"psf_mag_{band}"],
            gridsize=130,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        ax.plot(MAG_RANGE, MAG_RANGE, color="white", lw=1.0, ls="--")
        ax.set_title(f"{cat.name} N={int(m.sum()):,}")
        ax.set_xlabel(f"{band} CModel magnitude (AB)")
        ax.grid(True, alpha=0.18)
        fig.colorbar(hb, ax=ax, label="log10(N)")
    axes[0].set_ylabel(f"{band} PSF magnitude (AB)")
    axes[0].set_xlim(*MAG_RANGE)
    axes[0].set_ylim(*MAG_RANGE)
    fig.suptitle(f"ECDFS {band}-band PSF vs CModel magnitude: DP1 vs DP2")
    return save_fig(fig, plot_dir / f"dp1_dp2_compare_{band}_psf_vs_cmodel_ECDFS.png")


def plot_delta_vs_mag(dp1: StandardizedCatalog, dp2: StandardizedCatalog, band: str, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), sharex=True, sharey=True)
    for ax, cat in zip(axes, [dp1, dp2]):
        m = (
            valid_band(cat, band)
            & cat.df[f"cmodel_mag_{band}"].between(*MAG_RANGE)
            & cat.df[f"psf_minus_cmodel_{band}"].between(*DELTA_RANGE)
        )
        hb = ax.hexbin(
            cat.df.loc[m, f"cmodel_mag_{band}"],
            cat.df.loc[m, f"psf_minus_cmodel_{band}"],
            gridsize=130,
            bins="log",
            mincnt=1,
            cmap="magma",
        )
        ax.axhline(0.0, color="white", lw=1.0, ls="--")
        ax.set_title(f"{cat.name} N={int(m.sum()):,}")
        ax.set_xlabel(f"{band} CModel magnitude (AB)")
        ax.grid(True, alpha=0.18)
        fig.colorbar(hb, ax=ax, label="log10(N)")
    axes[0].set_ylabel(f"{band} psf_mag - cmodel_mag (AB)")
    axes[0].set_xlim(*MAG_RANGE)
    axes[0].set_ylim(*DELTA_RANGE)
    fig.suptitle(f"ECDFS {band}-band PSF-CModel delta vs magnitude: DP1 vs DP2")
    return save_fig(fig, plot_dir / f"dp1_dp2_compare_{band}_delta_vs_mag_ECDFS.png")


def plot_ps_vs_mag(dp1: StandardizedCatalog, dp2: StandardizedCatalog, band: str, plot_dir: Path) -> Path | None:
    dp2_col = f"pS_{band}"
    dp1_col = f"pS_{band}"
    if dp2_col not in dp2.df.columns and dp1_col not in dp1.df.columns:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), sharex=True, sharey=True)
    for ax, cat, col in zip(axes, [dp1, dp2], [dp1_col, dp2_col]):
        if col not in cat.df.columns:
            ax.text(0.5, 0.5, f"{cat.name} per-object pS not found", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(cat.name)
            ax.set_xlabel(f"{band} CModel magnitude (AB)")
            ax.grid(True, alpha=0.18)
            continue
        m = valid_band(cat, band) & cat.df[f"cmodel_mag_{band}"].between(*MAG_RANGE) & np.isfinite(cat.df[col])
        ax.scatter(cat.df.loc[m, f"cmodel_mag_{band}"], cat.df.loc[m, col], s=2, alpha=0.18)
        ax.axhline(0.5, color="red", lw=1.4, ls="--")
        ax.set_title(f"{cat.name} N={int(m.sum()):,}")
        ax.set_xlabel(f"{band} CModel magnitude (AB)")
        ax.grid(True, alpha=0.18)
    axes[0].set_ylabel(f"pS_{band} (P star)")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_xlim(*MAG_RANGE)
    fig.suptitle(f"ECDFS {band}-band pS vs magnitude: DP1 vs DP2")
    return save_fig(fig, plot_dir / f"dp1_dp2_compare_{band}_pS_vs_mag_ECDFS.png")


def plot_slice_distributions(dp1: StandardizedCatalog, dp2: StandardizedCatalog, band: str, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2), sharey=True)
    bins = np.linspace(*SLICE_DELTA_RANGE, 90)
    for ax, (label, lo, hi) in zip(axes, MAG_BINS):
        for cat, color in [(dp1, "tab:blue"), (dp2, "tab:orange")]:
            mag = cat.df[f"cmodel_mag_{band}"]
            delta = cat.df[f"psf_minus_cmodel_{band}"]
            m = valid_band(cat, band) & np.isfinite(mag) & np.isfinite(delta)
            if lo is not None:
                m &= mag >= lo
            if hi is not None:
                m &= mag < hi
            m &= delta.between(*SLICE_DELTA_RANGE)
            ax.hist(delta[m], bins=bins, density=True, histtype="step", lw=2.0, color=color, label=f"{cat.name} N={int(m.sum()):,}")
        ax.axvline(0.0, color="0.35", lw=1.1, ls=":")
        ax.set_title(label)
        ax.set_xlabel(f"{band} psf_mag - cmodel_mag")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("Normalized density")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"ECDFS {band}-band slice distributions: DP1 vs DP2")
    return save_fig(fig, plot_dir / f"dp1_dp2_compare_{band}_slice_distributions_ECDFS.png")


def mag_bin_mask(mag: pd.Series, lo: float | None, hi: float | None) -> pd.Series:
    mask = pd.Series(np.isfinite(mag), index=mag.index)
    if lo is not None:
        mask &= mag >= lo
    if hi is not None:
        mask &= mag < hi
    return mask


def bright_end_table(cats: list[StandardizedCatalog], bands: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for band in bands:
        for cat in cats:
            mag = cat.df[f"cmodel_mag_{band}"]
            delta = cat.df[f"psf_minus_cmodel_{band}"]
            for label, lo, hi in MAG_BINS:
                m = valid_band(cat, band) & mag_bin_mask(mag, lo, hi) & np.isfinite(delta)
                vals = delta[m]
                rows.append(
                    {
                        "band": band,
                        "dataset": cat.name,
                        "mag_bin": label,
                        "n": int(m.sum()),
                        "delta_mean": float(np.nanmean(vals)) if len(vals) else np.nan,
                        "delta_median": float(np.nanmedian(vals)) if len(vals) else np.nan,
                        "delta_mad_sigma": robust_sigma(vals),
                        "delta_p16": float(np.nanpercentile(vals, 16)) if len(vals) else np.nan,
                        "delta_p84": float(np.nanpercentile(vals, 84)) if len(vals) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def plot_bright_end_offsets(summary: pd.DataFrame, band: str, plot_dir: Path) -> Path:
    use = summary[summary["band"].eq(band)].copy()
    centers = {"mag < 22": 21.5, "22 <= mag < 23": 22.5, "23 <= mag < 24": 23.5, "24 <= mag < 25": 24.5}
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for dataset, color in [("DP1", "tab:blue"), ("DP2", "tab:orange")]:
        d = use[use["dataset"].eq(dataset)].copy()
        x = [centers.get(v, np.nan) for v in d["mag_bin"]]
        ax.errorbar(x, d["delta_median"], yerr=d["delta_mad_sigma"], marker="o", lw=1.8, capsize=3, label=dataset, color=color)
    ax.axhline(0.0, color="0.35", ls=":", lw=1.2)
    ax.set_xlabel(f"{band} CModel magnitude bin center (AB)")
    ax.set_ylabel(f"median {band} psf_mag - cmodel_mag (AB)")
    ax.set_title(f"ECDFS {band}-band bright-end offset check")
    ax.grid(True, alpha=0.2)
    ax.legend()
    return save_fig(fig, plot_dir / f"dp1_dp2_{band}_bright_end_offset_vs_mag_ECDFS.png")


def sample_counts(cats: list[StandardizedCatalog], bands: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cat in cats:
        coord_valid = np.isfinite(cat.df["ra"]) & np.isfinite(cat.df["dec"])
        for band in bands:
            mag = cat.df[f"cmodel_mag_{band}"]
            row = {
                "dataset": cat.name,
                "band": band,
                "total_rows": int(len(cat.df)),
                "coord_valid": int(coord_valid.sum()),
                "positive_finite_psf_mag": int(np.isfinite(cat.df[f"psf_mag_{band}"]).sum()),
                "positive_finite_cmodel_mag": int(np.isfinite(cat.df[f"cmodel_mag_{band}"]).sum()),
                "clean_band": int(valid_band(cat, band).sum()),
                "extendedness_available_clean": int(cat.df[f"extendedness_clean_{band}"].sum()),
            }
            for label, lo, hi in MAG_BINS:
                m = valid_band(cat, band) & mag_bin_mask(mag, lo, hi)
                row[f"clean_{label}"] = int(m.sum())
            rows.append(row)
    return pd.DataFrame(rows)


def plot_footprint(dp1: StandardizedCatalog, dp2: StandardizedCatalog, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharex=True, sharey=True)
    for ax, cat in zip(axes, [dp1, dp2]):
        m = np.isfinite(cat.df["ra"]) & np.isfinite(cat.df["dec"])
        hb = ax.hexbin(cat.df.loc[m, "ra"], cat.df.loc[m, "dec"], gridsize=150, bins="log", mincnt=1, cmap="viridis")
        ax.set_title(f"{cat.name} ECDFS N={int(m.sum()):,}")
        ax.set_xlabel("RA (deg)")
        ax.grid(True, alpha=0.16)
        fig.colorbar(hb, ax=ax, label="log10(N)")
    axes[0].set_ylabel("Dec (deg)")
    fig.suptitle("ECDFS footprint: DP1 vs DP2")
    return save_fig(fig, plot_dir / "dp1_dp2_footprint_ECDFS.png")


def load_hst_truth(path: Path) -> pd.DataFrame:
    truth = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "external_id": truth.get("id", pd.RangeIndex(len(truth))).astype(str),
            "external_ra": pd.to_numeric(truth["ra"], errors="coerce"),
            "external_dec": pd.to_numeric(truth["dec"], errors="coerce"),
            "truth_label": truth["label"].astype(str).str.lower(),
        }
    )
    out = out[out["truth_label"].isin(["star", "galaxy"])].dropna(subset=["external_ra", "external_dec"])
    return out.reset_index(drop=True)


def match_to_hst(cat: StandardizedCatalog, truth: pd.DataFrame, max_sep_arcsec: float) -> pd.DataFrame:
    from astropy import units as u
    from astropy.coordinates import SkyCoord

    base = cat.df.dropna(subset=["ra", "dec", "object_id"]).copy()
    coord_cat = SkyCoord(base["ra"].to_numpy() * u.deg, base["dec"].to_numpy() * u.deg)
    coord_truth = SkyCoord(truth["external_ra"].to_numpy() * u.deg, truth["external_dec"].to_numpy() * u.deg)
    idx, sep2d, _ = coord_truth.match_to_catalog_sky(coord_cat)
    sep_arcsec = sep2d.arcsec
    within = sep_arcsec <= max_sep_arcsec
    ext = truth.loc[within].reset_index(drop=True)
    nearest = base.iloc[idx[within]].reset_index(drop=True)
    matched = pd.concat([ext, nearest.add_prefix(f"{cat.name.lower()}_")], axis=1)
    matched["dataset"] = cat.name
    matched["object_id"] = matched[f"{cat.name.lower()}_object_id"]
    matched["match_sep_arcsec"] = sep_arcsec[within]
    matched = matched.sort_values(["object_id", "match_sep_arcsec"], kind="mergesort").drop_duplicates("object_id", keep="first")
    matched = matched.sort_values(["external_id", "match_sep_arcsec"], kind="mergesort").drop_duplicates("external_id", keep="first")
    return matched.reset_index(drop=True)


def plot_hst_delta_hist(dp1_hst: pd.DataFrame, dp2_hst: pd.DataFrame, band: str, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True, sharey=True)
    bins = np.linspace(*SLICE_DELTA_RANGE, 90)
    panels = [(dp1_hst, "DP1", "star"), (dp1_hst, "DP1", "galaxy"), (dp2_hst, "DP2", "star"), (dp2_hst, "DP2", "galaxy")]
    for ax, (df, dataset, label) in zip(axes.ravel(), panels):
        col = f"{dataset.lower()}_psf_minus_cmodel_{band}"
        m = df["truth_label"].eq(label) & np.isfinite(df[col]) & df[col].between(*SLICE_DELTA_RANGE)
        color = "red" if label == "star" else "tab:blue"
        ax.hist(df.loc[m, col], bins=bins, density=True, histtype="step", lw=2.2, color=color)
        ax.axvline(0.0, color="0.35", lw=1.1, ls=":")
        ax.set_title(f"{dataset} HST {label}s\nN={int(m.sum()):,}")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel(f"{band} psf_mag - cmodel_mag")
    axes[0, 0].set_ylabel("Normalized density")
    axes[1, 0].set_ylabel("Normalized density")
    fig.suptitle(f"ECDFS {band}-band HST-labeled PSF-CModel delta distributions")
    return save_fig(fig, plot_dir / f"dp1_dp2_{band}_delta_hist_HST_stars_galaxies_ECDFS.png")


def plot_hst_delta_vs_mag(dp1_hst: pd.DataFrame, dp2_hst: pd.DataFrame, band: str, plot_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharex=True, sharey=True)
    for ax, df, dataset in zip(axes, [dp1_hst, dp2_hst], ["DP1", "DP2"]):
        prefix = dataset.lower()
        mag_col = f"{prefix}_cmodel_mag_{band}"
        delta_col = f"{prefix}_psf_minus_cmodel_{band}"
        for label, color, size, alpha in [("galaxy", "tab:blue", 5, 0.22), ("star", "red", 16, 0.8)]:
            m = (
                df["truth_label"].eq(label)
                & df[mag_col].between(*MAG_RANGE)
                & df[delta_col].between(*DELTA_RANGE)
                & np.isfinite(df[mag_col])
                & np.isfinite(df[delta_col])
            )
            ax.scatter(df.loc[m, mag_col], df.loc[m, delta_col], s=size, alpha=alpha, color=color, label=f"HST {label} N={int(m.sum()):,}")
        ax.axhline(0.0, color="0.35", lw=1.1, ls=":")
        ax.set_title(dataset)
        ax.set_xlabel(f"{band} CModel magnitude (AB)")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(f"{band} psf_mag - cmodel_mag (AB)")
    axes[0].set_xlim(*MAG_RANGE)
    axes[0].set_ylim(*DELTA_RANGE)
    fig.suptitle(f"ECDFS {band}-band HST-labeled delta vs magnitude: DP1 vs DP2")
    return save_fig(fig, plot_dir / f"dp1_dp2_{band}_delta_vs_mag_HST_ECDFS.png")


def hst_offset_rows(dp1_hst: pd.DataFrame, dp2_hst: pd.DataFrame, band: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for df, dataset in [(dp1_hst, "DP1"), (dp2_hst, "DP2")]:
        prefix = dataset.lower()
        mag = df[f"{prefix}_cmodel_mag_{band}"]
        delta = df[f"{prefix}_psf_minus_cmodel_{band}"]
        for truth_label in ["star", "galaxy"]:
            for label, lo, hi in MAG_BINS:
                m = df["truth_label"].eq(truth_label) & mag_bin_mask(mag, lo, hi) & np.isfinite(delta)
                vals = delta[m]
                rows.append(
                    {
                        "band": band,
                        "dataset": dataset,
                        "truth_label": truth_label,
                        "mag_bin": label,
                        "n": int(m.sum()),
                        "delta_mean": float(np.nanmean(vals)) if len(vals) else np.nan,
                        "delta_median": float(np.nanmedian(vals)) if len(vals) else np.nan,
                        "delta_mad_sigma": robust_sigma(vals),
                    }
                )
    return pd.DataFrame(rows)


def match_dp1_dp2_common(
    dp1: StandardizedCatalog,
    dp2: StandardizedCatalog,
    bands: Iterable[str],
    max_sep_arcsec: float = COMMON_MATCH_RADIUS_ARCSEC,
) -> pd.DataFrame:
    """Build a one-to-one spatial DP1-DP2 common-object catalog."""

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    dp1_use = dp1.df.dropna(subset=["ra", "dec", "object_id"]).copy()
    dp2_use = dp2.df.dropna(subset=["ra", "dec", "object_id"]).copy()
    dp1_coord = SkyCoord(dp1_use["ra"].to_numpy() * u.deg, dp1_use["dec"].to_numpy() * u.deg)
    dp2_coord = SkyCoord(dp2_use["ra"].to_numpy() * u.deg, dp2_use["dec"].to_numpy() * u.deg)
    idx, sep2d, _ = dp1_coord.match_to_catalog_sky(dp2_coord)
    sep_arcsec = sep2d.arcsec
    within = sep_arcsec <= max_sep_arcsec

    dp1_part = dp1_use.loc[within].reset_index(drop=True)
    dp2_part = dp2_use.iloc[idx[within]].reset_index(drop=True)
    common = pd.concat([dp1_part.add_prefix("dp1_"), dp2_part.add_prefix("dp2_")], axis=1)
    common["match_sep_arcsec"] = sep_arcsec[within]
    common = common.sort_values(["dp2_object_id", "match_sep_arcsec"], kind="mergesort")
    common = common.drop_duplicates("dp2_object_id", keep="first")
    common = common.sort_values(["dp1_object_id", "match_sep_arcsec"], kind="mergesort")
    common = common.drop_duplicates("dp1_object_id", keep="first").reset_index(drop=True)

    for band in bands:
        common[f"mag_ref_{band}"] = 0.5 * (common[f"dp1_cmodel_mag_{band}"] + common[f"dp2_cmodel_mag_{band}"])
        common[f"psf_change_{band}"] = common[f"dp2_psf_mag_{band}"] - common[f"dp1_psf_mag_{band}"]
        common[f"cmodel_change_{band}"] = common[f"dp2_cmodel_mag_{band}"] - common[f"dp1_cmodel_mag_{band}"]
        common[f"delta_dp1_{band}"] = common[f"dp1_psf_minus_cmodel_{band}"]
        common[f"delta_dp2_{band}"] = common[f"dp2_psf_minus_cmodel_{band}"]
        common[f"delta_change_{band}"] = common[f"delta_dp2_{band}"] - common[f"delta_dp1_{band}"]
        common[f"common_clean_{band}"] = (
            common[f"dp1_clean_{band}"].astype(bool)
            & common[f"dp2_clean_{band}"].astype(bool)
            & np.isfinite(common[f"mag_ref_{band}"])
            & np.isfinite(common[f"psf_change_{band}"])
            & np.isfinite(common[f"cmodel_change_{band}"])
            & np.isfinite(common[f"delta_change_{band}"])
        )
    return common


def add_hst_labels_to_common(common: pd.DataFrame, truth: pd.DataFrame, max_sep_arcsec: float = HST_MATCH_RADIUS_ARCSEC) -> pd.DataFrame:
    """Attach HST labels to common DP1-DP2 objects using DP2 positions."""

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    out = common.copy()
    out["truth_label"] = pd.NA
    out["truth_external_id"] = pd.NA
    out["truth_match_sep_arcsec"] = np.nan
    base = out.dropna(subset=["dp2_ra", "dp2_dec"]).copy()
    if len(base) == 0 or len(truth) == 0:
        return out

    common_coord = SkyCoord(base["dp2_ra"].to_numpy() * u.deg, base["dp2_dec"].to_numpy() * u.deg)
    truth_coord = SkyCoord(truth["external_ra"].to_numpy() * u.deg, truth["external_dec"].to_numpy() * u.deg)
    idx, sep2d, _ = truth_coord.match_to_catalog_sky(common_coord)
    sep_arcsec = sep2d.arcsec
    within = sep_arcsec <= max_sep_arcsec

    matched_truth = truth.loc[within].reset_index(drop=True)
    common_index = base.index.to_numpy()[idx[within]]
    tmp = pd.DataFrame(
        {
            "common_index": common_index,
            "external_id": matched_truth["external_id"].to_numpy(),
            "truth_label": matched_truth["truth_label"].to_numpy(),
            "truth_match_sep_arcsec": sep_arcsec[within],
        }
    )
    tmp = tmp.sort_values(["common_index", "truth_match_sep_arcsec"], kind="mergesort").drop_duplicates("common_index", keep="first")
    tmp = tmp.sort_values(["external_id", "truth_match_sep_arcsec"], kind="mergesort").drop_duplicates("external_id", keep="first")
    for _, row in tmp.iterrows():
        ix = int(row["common_index"])
        out.loc[ix, "truth_label"] = row["truth_label"]
        out.loc[ix, "truth_external_id"] = row["external_id"]
        out.loc[ix, "truth_match_sep_arcsec"] = row["truth_match_sep_arcsec"]
    return out


def common_delta_decomposition(common: pd.DataFrame, bands: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    samples = [("all_common", None), ("hst_star", "star"), ("hst_galaxy", "galaxy")]
    for band in bands:
        mag = common[f"mag_ref_{band}"]
        clean = common[f"common_clean_{band}"].astype(bool)
        for sample_name, truth_label in samples:
            sample_mask = clean.copy()
            if truth_label is not None:
                sample_mask &= common["truth_label"].eq(truth_label)
            for bin_label, lo, hi in MAG_BINS:
                m = sample_mask & mag_bin_mask(mag, lo, hi)
                row: dict[str, object] = {
                    "band": band,
                    "mag_bin": bin_label,
                    "sample": sample_name,
                    "n": int(m.sum()),
                    "mag_reference": "0.5 * (DP1 CModel mag + DP2 CModel mag)",
                }
                for col in ["psf_change", "cmodel_change", "delta_change", "delta_dp1", "delta_dp2"]:
                    values = common.loc[m, f"{col}_{band}"]
                    row[f"median_{col}"] = float(np.nanmedian(values)) if len(values) else np.nan
                    row[f"mad_sigma_{col}"] = robust_sigma(values)
                rows.append(row)
    return pd.DataFrame(rows)


def common_bin_value(common_summary: pd.DataFrame, band: str, sample: str, mag_bin: str, col: str) -> float:
    row = common_summary[
        common_summary["band"].eq(band)
        & common_summary["sample"].eq(sample)
        & common_summary["mag_bin"].eq(mag_bin)
    ]
    return float(row[col].iloc[0]) if len(row) else np.nan


def common_bin_n(common_summary: pd.DataFrame, band: str, sample: str, mag_bin: str) -> int:
    row = common_summary[
        common_summary["band"].eq(band)
        & common_summary["sample"].eq(sample)
        & common_summary["mag_bin"].eq(mag_bin)
    ]
    return int(row["n"].iloc[0]) if len(row) else 0


def bin_centers() -> dict[str, float]:
    return {"mag < 22": 21.5, "22 <= mag < 23": 22.5, "23 <= mag < 24": 23.5, "24 <= mag < 25": 24.5}


def plot_common_mag_hist(common: pd.DataFrame, band: str, mag_kind: str, plot_dir: Path) -> Path:
    """Compare DP1 and DP2 magnitudes for common objects only."""

    if mag_kind not in {"psf", "cmodel"}:
        raise ValueError(f"Unsupported mag_kind: {mag_kind}")
    label = "PSF" if mag_kind == "psf" else "CModel"
    dp1_col = f"dp1_{mag_kind}_mag_{band}"
    dp2_col = f"dp2_{mag_kind}_mag_{band}"
    clean = (
        common[f"common_clean_{band}"].astype(bool)
        & common[dp1_col].between(*MAG_RANGE)
        & common[dp2_col].between(*MAG_RANGE)
        & np.isfinite(common[dp1_col])
        & np.isfinite(common[dp2_col])
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    bins = np.linspace(*MAG_RANGE, 90)
    ax.hist(
        common.loc[clean, dp1_col],
        bins=bins,
        histtype="step",
        density=True,
        lw=2.2,
        color="tab:blue",
        label=f"DP1 N={int(clean.sum()):,}",
    )
    ax.hist(
        common.loc[clean, dp2_col],
        bins=bins,
        histtype="step",
        density=True,
        lw=2.2,
        color="tab:orange",
        label=f"DP2 N={int(clean.sum()):,}",
    )
    ax.set_xlabel(f"{band} {label} magnitude (AB)")
    ax.set_ylabel("Normalized density")
    ax.set_xlim(*MAG_RANGE)
    ax.set_title(f"ECDFS common objects: {band}-band {label} magnitude")
    ax.grid(True, alpha=0.2)
    ax.legend()
    return save_fig(fig, plot_dir / f"dp1_dp2_common_{band}_{mag_kind}_mag_hist_ECDFS.png")


def save_common_g_psf_cmodel_separate_summary(common_summary: pd.DataFrame, result_dir: Path, band: str = "g") -> Path:
    """Save compact g-band PSF/CModel decomposition table for common objects."""

    use = common_summary[
        common_summary["band"].eq(band)
        & common_summary["sample"].eq("all_common")
        & common_summary["mag_bin"].isin([label for label, _, _ in MAG_BINS])
    ].copy()
    order = [label for label, _, _ in MAG_BINS]
    use["mag_bin"] = pd.Categorical(use["mag_bin"], categories=order, ordered=True)
    use = use.sort_values("mag_bin")
    cols = [
        "band",
        "mag_bin",
        "n",
        "median_psf_change",
        "mad_sigma_psf_change",
        "median_cmodel_change",
        "mad_sigma_cmodel_change",
        "median_delta_change",
    ]
    path = result_dir / "dp1_dp2_common_g_psf_cmodel_separate_summary_ECDFS.csv"
    use[cols].to_csv(path, index=False)
    return path


def common_unresolved_mask(common: pd.DataFrame, band: str, threshold: float = 0.03) -> pd.Series:
    """Select common objects that are unresolved-like in both DP1 and DP2."""

    return (
        common[f"common_clean_{band}"].astype(bool)
        & np.isfinite(common[f"delta_dp1_{band}"])
        & np.isfinite(common[f"delta_dp2_{band}"])
        & (common[f"delta_dp1_{band}"] < threshold)
        & (common[f"delta_dp2_{band}"] < threshold)
    )


def save_common_g_unresolved_summary(common: pd.DataFrame, result_dir: Path, band: str = "g", threshold: float = 0.03) -> Path:
    """Save compact g-band PSF/CModel change table for unresolved common objects."""

    rows: list[dict[str, object]] = []
    base = common_unresolved_mask(common, band, threshold=threshold)
    for bin_label, lo, hi in MAG_BINS:
        m = base & common[f"mag_ref_{band}"].between(*MAG_RANGE) & mag_bin_mask(common[f"mag_ref_{band}"], lo, hi)
        row: dict[str, object] = {
            "band": band,
            "mag_bin": bin_label,
            "selection": f"DP1 and DP2 psf_mag-cmodel_mag < {threshold:.2f}",
            "n": int(m.sum()),
        }
        for col in ["psf_change", "cmodel_change", "delta_change", "delta_dp1", "delta_dp2"]:
            values = common.loc[m, f"{col}_{band}"]
            row[f"median_{col}"] = float(np.nanmedian(values)) if len(values) else np.nan
            row[f"mad_sigma_{col}"] = robust_sigma(values)
        rows.append(row)
    path = result_dir / "dp1_dp2_common_g_unresolved_psf_cmodel_change_summary_ECDFS.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def cut_diagnostic_masks(common: pd.DataFrame, band: str = "g", threshold: float = 0.03) -> dict[str, list[tuple[str, str, pd.Series]]]:
    """Return unresolved/resolved diagnostic masks.

    The `unresolved_both` cut is intentionally tracked separately because
    requiring both releases to satisfy `psf_mag - cmodel_mag < threshold`
    restricts both endpoints of the morphology delta and can suppress the
    allowed range of `delta_change = delta_DP2 - delta_DP1` by construction.
    """

    clean = common[f"common_clean_{band}"].astype(bool) & common[f"mag_ref_{band}"].between(*MAG_RANGE)
    finite_delta = np.isfinite(common[f"delta_dp1_{band}"]) & np.isfinite(common[f"delta_dp2_{band}"])
    clean &= finite_delta
    dp1_unresolved = clean & (common[f"delta_dp1_{band}"] < threshold)
    dp1_resolved = clean & (common[f"delta_dp1_{band}"] > threshold)
    dp2_unresolved = clean & (common[f"delta_dp2_{band}"] < threshold)
    dp2_resolved = clean & (common[f"delta_dp2_{band}"] > threshold)
    both_unresolved = dp1_unresolved & (common[f"delta_dp2_{band}"] < threshold)
    both_resolved = dp1_resolved & (common[f"delta_dp2_{band}"] > threshold)
    return {
        "all_common": [("all_common", "all common objects", clean)],
        "cut_on_DP1": [
            ("unresolved", f"unresolved by DP1: delta_DP1 < {threshold:.2f}", dp1_unresolved),
            ("resolved", f"resolved by DP1: delta_DP1 > {threshold:.2f}", dp1_resolved),
        ],
        "cut_on_DP2": [
            ("unresolved", f"unresolved by DP2: delta_DP2 < {threshold:.2f}", dp2_unresolved),
            ("resolved", f"resolved by DP2: delta_DP2 > {threshold:.2f}", dp2_resolved),
        ],
        "unresolved_both": [("unresolved_both", f"unresolved in both: delta_DP1 and delta_DP2 < {threshold:.2f}", both_unresolved)],
        "resolved_both": [("resolved_both", f"resolved in both: delta_DP1 and delta_DP2 > {threshold:.2f}", both_resolved)],
    }


def finite_values(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    return arr[np.isfinite(arr)]


def robust_plot_limits(values: list[np.ndarray], default: tuple[float, float] = (-0.5, 0.5), symmetric: bool = False) -> tuple[float, float]:
    good = [arr[np.isfinite(arr)] for arr in values if len(arr)]
    good = [arr for arr in good if len(arr)]
    if not good:
        return default
    combined = np.concatenate(good)
    if len(combined) == 0:
        return default
    lo, hi = np.nanpercentile(combined, [1, 99])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return default
    pad = max(0.03, 0.12 * (hi - lo))
    if symmetric:
        lim = max(abs(lo), abs(hi)) + pad
        lim = min(max(lim, 0.12), 1.5)
        return -lim, lim
    return max(default[0], lo - pad), min(default[1], hi + pad)


def mag_bin_selection(common: pd.DataFrame, band: str, lo: float | None, hi: float | None) -> pd.Series:
    return mag_bin_mask(common[f"mag_ref_{band}"], lo, hi)


def plot_g_histogram_grid_by_mag(
    common: pd.DataFrame,
    plot_dir: Path,
    cut_key: str,
    hist_kind: str,
    band: str = "g",
    threshold: float = 0.03,
) -> Path:
    """Shared g-band histogram grid for morphology and DP1-DP2 differences.

    Supported histogram kinds:
    - `delta_overlay`: overlaid DP1/DP2 distributions of delta = psf_mag - cmodel_mag.
    - `psf_diff`: single distribution of psfDP1 - psfDP2.
    - `cmodel_diff`: single distribution of CModelDP1 - CModelDP2.

    For the difference histograms, positive `psfDP1 - psfDP2` means DP2 PSF
    magnitude is smaller/brighter; negative `CModelDP1 - CModelDP2` means DP2
    CModel magnitude is larger/fainter.
    """

    if hist_kind not in {"delta_overlay", "psf_diff", "cmodel_diff"}:
        raise ValueError(f"Unsupported hist_kind: {hist_kind}")
    masks = cut_diagnostic_masks(common, band, threshold)[cut_key]
    nrows = len(masks)
    fig, axes = plt.subplots(nrows, len(MAG_BINS), figsize=(4.1 * len(MAG_BINS), 3.4 * nrows), squeeze=False)

    all_vals: list[np.ndarray] = []
    if hist_kind == "delta_overlay":
        for _, _, mask in masks:
            for _, lo, hi in MAG_BINS:
                m = mask & mag_bin_selection(common, band, lo, hi)
                all_vals.append(common.loc[m, f"delta_dp1_{band}"].to_numpy(dtype="float64"))
                all_vals.append(common.loc[m, f"delta_dp2_{band}"].to_numpy(dtype="float64"))
        xlim = robust_plot_limits(all_vals, default=(-0.2, 1.5), symmetric=False)
        xlim = (min(xlim[0], -0.05), max(xlim[1], 0.12))
        bins = np.linspace(xlim[0], xlim[1], 60)
        file_label = "psf_model"
        suptitle = (
            f"ECDFS common objects, g band: morphology delta histograms by magnitude bin\n"
            f"cut definition: {cut_key}; delta = psf_mag - cmodel_mag"
        )
    else:
        if hist_kind == "psf_diff":
            values = -common[f"psf_change_{band}"]
            label = "PSF"
            file_label = "psfDP1_minus_psfDP2"
            xlabel = "g psfDP1 - psfDP2"
            note = "Positive DP1-DP2 means DP2 PSF magnitude is smaller/brighter."
        else:
            values = -common[f"cmodel_change_{band}"]
            label = "CModel"
            file_label = "cModelDP1_minus_cModelDP2"
            xlabel = "g CModelDP1 - CModelDP2"
            note = "Negative DP1-DP2 means DP2 CModel magnitude is larger/fainter."
        for _, _, mask in masks:
            for _, lo, hi in MAG_BINS:
                m = mask & mag_bin_selection(common, band, lo, hi) & np.isfinite(values)
                all_vals.append(values[m].to_numpy(dtype="float64"))
        xlim = robust_plot_limits(all_vals, default=(-0.5, 0.5), symmetric=True)
        bins = np.linspace(xlim[0], xlim[1], 58)
        suptitle = (
            f"ECDFS common objects, g band: {label} magnitude difference histograms\n"
            f"cut definition: {cut_key}; sign convention: DP1 - DP2"
        )

    for row, (_, subset_label, subset_mask) in enumerate(masks):
        for col, (bin_label, lo, hi) in enumerate(MAG_BINS):
            ax = axes[row, col]
            m = subset_mask & mag_bin_selection(common, band, lo, hi)
            if hist_kind == "delta_overlay":
                dp1_vals = finite_values(common.loc[m, f"delta_dp1_{band}"])
                dp2_vals = finite_values(common.loc[m, f"delta_dp2_{band}"])
                if len(dp1_vals):
                    dp1_med = float(np.nanmedian(dp1_vals))
                    ax.hist(dp1_vals, bins=bins, density=True, histtype="step", lw=2.0, color="tab:blue", label=f"DP1 N={len(dp1_vals):,}")
                    ax.axvline(dp1_med, color="tab:blue", lw=1.8, ls="-", alpha=0.95, label=f"DP1 median={dp1_med:.3f}")
                if len(dp2_vals):
                    dp2_med = float(np.nanmedian(dp2_vals))
                    ax.hist(dp2_vals, bins=bins, density=True, histtype="step", lw=2.0, color="tab:orange", label=f"DP2 N={len(dp2_vals):,}")
                    ax.axvline(dp2_med, color="tab:orange", lw=1.8, ls="-", alpha=0.95, label=f"DP2 median={dp2_med:.3f}")
                ax.axvline(threshold, color="red", lw=1.4, ls="--", label="0.03 threshold" if row == 0 and col == 0 else None)
                ax.set_title(
                    f"{bin_label}\n{subset_label}\n"
                    f"med DP1={np.nanmedian(dp1_vals):.3f}, DP2={np.nanmedian(dp2_vals):.3f}" if len(dp1_vals) and len(dp2_vals) else f"{bin_label}\n{subset_label}",
                    fontsize=9,
                )
                if row == nrows - 1:
                    ax.set_xlabel("g psf_mag - cmodel_mag")
            else:
                panel_mask = m & np.isfinite(values)
                vals = finite_values(values[panel_mask])
                if len(vals):
                    med = float(np.nanmedian(vals))
                    mean = float(np.nanmean(vals))
                    ax.hist(vals, bins=bins, density=True, histtype="stepfilled", color="0.55", alpha=0.42, edgecolor="0.2", lw=1.2)
                    ax.axvline(med, color="red", lw=1.6, label=f"median={med:.3f}")
                    ax.axvline(mean, color="black", lw=1.4, ls="--", label=f"mean={mean:.3f}")
                    ax.text(
                        0.03,
                        0.96,
                        f"N={len(vals):,}\nmed={med:.3f}\nmean={mean:.3f}",
                        transform=ax.transAxes,
                        ha="left",
                        va="top",
                        fontsize=8,
                        bbox=dict(facecolor="white", alpha=0.72, edgecolor="none"),
                    )
                else:
                    ax.text(0.5, 0.5, "No objects", transform=ax.transAxes, ha="center", va="center")
                ax.axvline(0.0, color="0.35", lw=1.0, ls=":")
                ax.set_title(f"{bin_label}\n{subset_label}", fontsize=9)
                if row == nrows - 1:
                    ax.set_xlabel(xlabel)
            ax.set_xlim(*xlim)
            ax.grid(True, alpha=0.18)
            if col == 0:
                ax.set_ylabel("Normalized density")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)

    fig.suptitle(suptitle, y=1.02, fontsize=13)
    if hist_kind == "delta_overlay":
        fig.tight_layout()
    else:
        fig.text(0.5, 0.005, note, ha="center", fontsize=9)
        fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
    return save_fig(fig, plot_dir / f"dp1_dp2_common_g_{file_label}_hist_by_mag_{cut_key}_ECDFS.png")


def plot_g_psf_diff_vs_cmodel_diff_scatter_by_cut(
    common: pd.DataFrame,
    plot_dir: Path,
    cut_key: str,
    band: str = "g",
    threshold: float = 0.03,
) -> Path:
    """Plot psfDP1-psfDP2 against cModelDP1-cModelDP2 by mag bin and cut.

    The dashed diagonal is y=x.  Points below that line have
    (psfDP1-psfDP2) > (CModelDP1-CModelDP2), which is equivalent to
    delta_DP1 > delta_DP2 for delta = psf_mag - cmodel_mag.
    """

    masks = cut_diagnostic_masks(common, band, threshold)[cut_key]
    nrows = len(masks)
    fig, axes = plt.subplots(nrows, len(MAG_BINS), figsize=(4.1 * len(MAG_BINS), 3.7 * nrows), squeeze=False)
    x_col = f"psf_dp1_minus_dp2_{band}"
    y_col = f"cmodel_dp1_minus_dp2_{band}"
    x = -common[f"psf_change_{band}"]
    y = -common[f"cmodel_change_{band}"]
    common[x_col] = x
    common[y_col] = y

    all_x: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for _, _, mask in masks:
        for _, lo, hi in MAG_BINS:
            m = mask & mag_bin_selection(common, band, lo, hi) & np.isfinite(x) & np.isfinite(y)
            all_x.append(x[m].to_numpy(dtype="float64"))
            all_y.append(y[m].to_numpy(dtype="float64"))
    xlim = robust_plot_limits(all_x, default=(-0.5, 0.5), symmetric=True)
    ylim = robust_plot_limits(all_y, default=(-0.5, 0.5), symmetric=True)

    for row, (subset_key, subset_label, subset_mask) in enumerate(masks):
        for col, (bin_label, lo, hi) in enumerate(MAG_BINS):
            ax = axes[row, col]
            m = subset_mask & mag_bin_selection(common, band, lo, hi) & np.isfinite(x) & np.isfinite(y)
            n = int(m.sum())
            if n >= 80:
                hb = ax.hexbin(x[m], y[m], gridsize=45, bins="log", mincnt=1, cmap="viridis", extent=(*xlim, *ylim))
                if row == 0 and col == len(MAG_BINS) - 1:
                    fig.colorbar(hb, ax=ax, label="log10(N)", fraction=0.046, pad=0.04)
            elif n:
                ax.scatter(x[m], y[m], s=7, alpha=0.45, color="tab:blue")
            else:
                ax.text(0.5, 0.5, "No objects", transform=ax.transAxes, ha="center", va="center")
            if n:
                x_med = float(np.nanmedian(x[m]))
                y_med = float(np.nanmedian(y[m]))
                x_mean = float(np.nanmean(x[m]))
                y_mean = float(np.nanmean(y[m]))
                ax.scatter([x_med], [y_med], marker="o", s=55, color="red", edgecolor="white", linewidth=0.7, label="median")
                ax.scatter([x_mean], [y_mean], marker="x", s=55, color="black", linewidth=1.4, label="mean")
                ax.text(
                    0.03,
                    0.97,
                    f"N={n:,}\nmed=({x_med:.3f},{y_med:.3f})\nmean=({x_mean:.3f},{y_mean:.3f})",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.72, edgecolor="none"),
                )
            ax.axhline(0, color="0.45", lw=0.9, ls=":")
            ax.axvline(0, color="0.45", lw=0.9, ls=":")
            diag_lo = max(xlim[0], ylim[0])
            diag_hi = min(xlim[1], ylim[1])
            if diag_lo < diag_hi:
                ax.plot([diag_lo, diag_hi], [diag_lo, diag_hi], color="0.15", lw=1.0, ls="--")
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.grid(True, alpha=0.18)
            ax.set_title(f"{bin_label}\n{subset_label}", fontsize=9)
            if row == nrows - 1:
                ax.set_xlabel("g PSF mag: DP1 - DP2")
            if col == 0:
                ax.set_ylabel("g CModel mag: DP1 - DP2")
            if row == 0 and col == 0 and n:
                ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        f"ECDFS common objects, g band: PSF vs CModel magnitude differences\n"
        f"cut definition: {cut_key}; matched within 0.3 arcsec; sign convention: DP1 - DP2",
        y=1.02,
        fontsize=13,
    )
    fig.text(
        0.5,
        0.005,
        "Dashed diagonal is y=x; below it means delta_DP1 > delta_DP2, so DP2 has smaller psf_mag - cmodel_mag.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.0, 0.03, 1.0, 0.97])
    return save_fig(fig, plot_dir / f"dp1_dp2_common_g_psf_diff_vs_cmodel_diff_scatter_{cut_key}_ECDFS.png")


def plot_g_psf_model_hist_by_mag(
    common: pd.DataFrame,
    plot_dir: Path,
    cut_key: str,
    band: str = "g",
    threshold: float = 0.03,
) -> Path:
    """Overlay DP1 and DP2 psf_mag-cmodel_mag histograms by cut and mag bin."""
    return plot_g_histogram_grid_by_mag(common, plot_dir, cut_key, "delta_overlay", band, threshold)


def plot_g_mag_diff_hist_by_mag(
    common: pd.DataFrame,
    plot_dir: Path,
    cut_key: str,
    mag_kind: str,
    band: str = "g",
    threshold: float = 0.03,
) -> Path:
    """Plot DP1-DP2 magnitude-difference histograms by cut and magnitude bin."""

    if mag_kind not in {"psf", "cmodel"}:
        raise ValueError(f"Unsupported mag_kind: {mag_kind}")
    hist_kind = "psf_diff" if mag_kind == "psf" else "cmodel_diff"
    return plot_g_histogram_grid_by_mag(common, plot_dir, cut_key, hist_kind, band, threshold)


def save_g_unresolved_resolved_cut_diagnostics(
    common: pd.DataFrame,
    result_dir: Path,
    band: str = "g",
    threshold: float = 0.03,
) -> Path:
    """Save summary statistics for all unresolved/resolved cut definitions."""

    rows: list[dict[str, object]] = []
    masks = cut_diagnostic_masks(common, band, threshold)
    for cut_key, subsets in masks.items():
        for subset_key, subset_label, subset_mask in subsets:
            for bin_label, lo, hi in MAG_BINS:
                m = subset_mask & mag_bin_selection(common, band, lo, hi)
                psf_dp1_minus_dp2 = -common.loc[m, f"psf_change_{band}"]
                cmodel_dp1_minus_dp2 = -common.loc[m, f"cmodel_change_{band}"]
                delta_change = common.loc[m, f"delta_change_{band}"]
                delta_change_dp1_minus_dp2 = -delta_change
                delta_dp1 = common.loc[m, f"delta_dp1_{band}"]
                delta_dp2 = common.loc[m, f"delta_dp2_{band}"]
                n = int(m.sum())
                subset_definition = cut_key if cut_key == "all_common" else f"{cut_key}_{subset_key}"
                row: dict[str, object] = {
                    "band": band,
                    "cut_definition": cut_key,
                    "subset": subset_key,
                    "subset_definition": subset_definition,
                    "selection": subset_label,
                    "mag_bin": bin_label,
                    "n": n,
                    "median_delta_DP1": float(np.nanmedian(delta_dp1)) if n else np.nan,
                    "mean_delta_DP1": float(np.nanmean(delta_dp1)) if n else np.nan,
                    "mad_sigma_delta_DP1": robust_sigma(delta_dp1),
                    "median_delta_DP2": float(np.nanmedian(delta_dp2)) if n else np.nan,
                    "mean_delta_DP2": float(np.nanmean(delta_dp2)) if n else np.nan,
                    "mad_sigma_delta_DP2": robust_sigma(delta_dp2),
                    "median_delta_change_DP2_minus_DP1": float(np.nanmedian(delta_change)) if n else np.nan,
                    "mean_delta_change_DP2_minus_DP1": float(np.nanmean(delta_change)) if n else np.nan,
                    "mad_sigma_delta_change_DP2_minus_DP1": robust_sigma(delta_change),
                    "median_delta_change_DP1_minus_DP2": float(np.nanmedian(delta_change_dp1_minus_dp2)) if n else np.nan,
                    "mean_delta_change_DP1_minus_DP2": float(np.nanmean(delta_change_dp1_minus_dp2)) if n else np.nan,
                    "mad_sigma_delta_change_DP1_minus_DP2": robust_sigma(delta_change_dp1_minus_dp2),
                    "median_delta_dp1_psf_minus_cmodel": float(np.nanmedian(delta_dp1)) if n else np.nan,
                    "mean_delta_dp1_psf_minus_cmodel": float(np.nanmean(delta_dp1)) if n else np.nan,
                    "mad_sigma_delta_dp1_psf_minus_cmodel": robust_sigma(delta_dp1),
                    "median_delta_dp2_psf_minus_cmodel": float(np.nanmedian(delta_dp2)) if n else np.nan,
                    "mean_delta_dp2_psf_minus_cmodel": float(np.nanmean(delta_dp2)) if n else np.nan,
                    "mad_sigma_delta_dp2_psf_minus_cmodel": robust_sigma(delta_dp2),
                    "median_psfDP1_minus_psfDP2": float(np.nanmedian(psf_dp1_minus_dp2)) if n else np.nan,
                    "mean_psfDP1_minus_psfDP2": float(np.nanmean(psf_dp1_minus_dp2)) if n else np.nan,
                    "mad_sigma_psfDP1_minus_psfDP2": robust_sigma(psf_dp1_minus_dp2),
                    "median_cModelDP1_minus_cModelDP2": float(np.nanmedian(cmodel_dp1_minus_dp2)) if n else np.nan,
                    "mean_cModelDP1_minus_cModelDP2": float(np.nanmean(cmodel_dp1_minus_dp2)) if n else np.nan,
                    "mad_sigma_cModelDP1_minus_cModelDP2": robust_sigma(cmodel_dp1_minus_dp2),
                    "median_delta_change_dp2_minus_dp1": float(np.nanmedian(delta_change)) if n else np.nan,
                    "mean_delta_change_dp2_minus_dp1": float(np.nanmean(delta_change)) if n else np.nan,
                    "mad_sigma_delta_change_dp2_minus_dp1": robust_sigma(delta_change),
                    "median_delta_change_dp1_minus_dp2": float(np.nanmedian(delta_change_dp1_minus_dp2)) if n else np.nan,
                    "mean_delta_change_dp1_minus_dp2": float(np.nanmean(delta_change_dp1_minus_dp2)) if n else np.nan,
                    "mad_sigma_delta_change_dp1_minus_dp2": robust_sigma(delta_change_dp1_minus_dp2),
                }
                rows.append(row)
    path = result_dir / "dp1_dp2_g_unresolved_resolved_cut_diagnostics_ECDFS.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_g_histogram_median_mean_summary(cut_diagnostics: pd.DataFrame, result_dir: Path) -> Path:
    """Save the median/mean values annotated in the g-band histogram figures."""

    columns = {
        "band": "band",
        "sample_definition": "sample_definition",
        "cut_definition": "cut_definition",
        "subset": "subset",
        "selection": "selection",
        "mag_bin": "mag_bin",
        "n": "n",
        "median_delta_DP1": "median_delta_DP1",
        "median_delta_DP2": "median_delta_DP2",
        "mean_delta_DP1": "mean_delta_DP1",
        "mean_delta_DP2": "mean_delta_DP2",
        "median_psfDP1_minus_psfDP2": "median_psfDP1_minus_psfDP2",
        "mean_psfDP1_minus_psfDP2": "mean_psfDP1_minus_psfDP2",
        "median_cModelDP1_minus_cModelDP2": "median_CModelDP1_minus_CModelDP2",
        "mean_cModelDP1_minus_cModelDP2": "mean_CModelDP1_minus_CModelDP2",
    }
    available = {source: target for source, target in columns.items() if source in cut_diagnostics.columns}
    summary = cut_diagnostics.loc[:, list(available)].rename(columns=available)
    path = result_dir / "dp1_dp2_g_histogram_median_mean_summary_ECDFS.csv"
    summary.to_csv(path, index=False)
    return path


def save_g_unresolved_resolved_transition_counts(
    common: pd.DataFrame,
    result_dir: Path,
    band: str = "g",
    threshold: float = 0.03,
) -> Path:
    """Count movement across the unresolved/resolved threshold between DP1 and DP2."""

    rows: list[dict[str, object]] = []
    clean = common[f"common_clean_{band}"].astype(bool) & common[f"mag_ref_{band}"].between(*MAG_RANGE)
    finite_delta = np.isfinite(common[f"delta_dp1_{band}"]) & np.isfinite(common[f"delta_dp2_{band}"])
    clean &= finite_delta
    dp1_unresolved = common[f"delta_dp1_{band}"] < threshold
    dp1_resolved = common[f"delta_dp1_{band}"] > threshold
    dp2_unresolved = common[f"delta_dp2_{band}"] < threshold
    dp2_resolved = common[f"delta_dp2_{band}"] > threshold
    transitions = [
        ("DP1_unresolved_to_DP2_unresolved", "unresolved", "unresolved", dp1_unresolved & dp2_unresolved, dp1_unresolved),
        ("DP1_unresolved_to_DP2_resolved", "unresolved", "resolved", dp1_unresolved & dp2_resolved, dp1_unresolved),
        ("DP1_resolved_to_DP2_unresolved", "resolved", "unresolved", dp1_resolved & dp2_unresolved, dp1_resolved),
        ("DP1_resolved_to_DP2_resolved", "resolved", "resolved", dp1_resolved & dp2_resolved, dp1_resolved),
    ]
    for bin_label, lo, hi in MAG_BINS:
        bin_mask = clean & mag_bin_selection(common, band, lo, hi)
        total_bin_n = int(bin_mask.sum())
        for transition, dp1_class, dp2_class, transition_mask, dp1_class_mask in transitions:
            dp1_class_n = int((bin_mask & dp1_class_mask).sum())
            n = int((bin_mask & transition_mask).sum())
            rows.append(
                {
                    "band": band,
                    "threshold_delta_psf_minus_cmodel": threshold,
                    "mag_bin": bin_label,
                    "transition": transition,
                    "dp1_class": dp1_class,
                    "dp2_class": dp2_class,
                    "n": n,
                    "total_bin_n": total_bin_n,
                    "fraction_of_total_bin": n / total_bin_n if total_bin_n else np.nan,
                    "dp1_class_n": dp1_class_n,
                    "fraction_within_dp1_class": n / dp1_class_n if dp1_class_n else np.nan,
                }
            )
    path = result_dir / "dp1_dp2_g_unresolved_resolved_transition_counts_ECDFS.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def plot_g_psf_cmodel_offset_medians_simple(common: pd.DataFrame, plot_dir: Path, band: str = "g") -> Path:
    """Presentation-style median offset plot for PSF and CModel separately."""

    labels = ["<22", "22-23", "23-24", "24-25"]
    x = np.arange(len(MAG_BINS))
    width = 0.34
    clean = common[f"common_clean_{band}"].astype(bool) & common[f"mag_ref_{band}"].between(*MAG_RANGE)
    psf_vals = -common[f"psf_change_{band}"]
    cmodel_vals = -common[f"cmodel_change_{band}"]

    rows = []
    for bin_label, lo, hi in MAG_BINS:
        m = clean & mag_bin_selection(common, band, lo, hi) & np.isfinite(psf_vals) & np.isfinite(cmodel_vals)
        rows.append(
            {
                "n": int(m.sum()),
                "psf_median": float(np.nanmedian(psf_vals[m])) if int(m.sum()) else np.nan,
                "psf_sigma": robust_sigma(psf_vals[m]),
                "cmodel_median": float(np.nanmedian(cmodel_vals[m])) if int(m.sum()) else np.nan,
                "cmodel_sigma": robust_sigma(cmodel_vals[m]),
            }
        )
    stats = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    ax.bar(
        x - width / 2,
        stats["psf_median"],
        width,
        yerr=stats["psf_sigma"],
        label="PSF mag: DP1 - DP2",
        color="tab:blue",
        alpha=0.86,
        capsize=4,
    )
    ax.bar(
        x + width / 2,
        stats["cmodel_median"],
        width,
        yerr=stats["cmodel_sigma"],
        label="CModel mag: DP1 - DP2",
        color="tab:orange",
        alpha=0.86,
        capsize=4,
    )
    ax.axhline(0.0, color="0.25", lw=1.1, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("g reference CModel magnitude bin")
    ax.set_ylabel("Median magnitude difference, DP1 - DP2")
    ax.set_title("ECDFS common objects, g band: PSF and CModel offsets separately")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(loc="upper right")
    for i, row in stats.iterrows():
        ymax = max(row["psf_median"], row["cmodel_median"], 0.0)
        ax.text(i, ymax + 0.02, f"N={int(row['n']):,}", ha="center", va="bottom", fontsize=9)
        ax.text(i - width / 2, row["psf_median"], f"{row['psf_median']:+.3f}", ha="center", va="bottom" if row["psf_median"] >= 0 else "top", fontsize=8)
        ax.text(i + width / 2, row["cmodel_median"], f"{row['cmodel_median']:+.3f}", ha="center", va="bottom" if row["cmodel_median"] >= 0 else "top", fontsize=8)
    fig.text(
        0.5,
        0.01,
        "Positive DP1-DP2 means the DP2 magnitude is smaller/brighter. Error bars show robust MAD scatter.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])
    return save_fig(fig, plot_dir / "dp1_dp2_common_g_psf_cmodel_offset_medians_simple_ECDFS.png")


def plot_g_delta_cut_effect_simple(cut_diagnostics: pd.DataFrame, plot_dir: Path) -> Path:
    """Compact plot showing how the unresolved/resolved definition changes delta offsets."""

    bright = cut_diagnostics[cut_diagnostics["mag_bin"].eq("mag < 22")].copy()
    order = [
        ("all_common", "all_common", "All\ncommon"),
        ("unresolved_both", "unresolved_both", "Unresolved\nin both"),
        ("cut_on_DP1", "unresolved", "DP1 cut\nunresolved"),
        ("cut_on_DP1", "resolved", "DP1 cut\nresolved"),
        ("cut_on_DP2", "unresolved", "DP2 cut\nunresolved"),
        ("cut_on_DP2", "resolved", "DP2 cut\nresolved"),
    ]
    rows = []
    for cut_key, subset, label in order:
        row = bright[bright["cut_definition"].eq(cut_key) & bright["subset"].eq(subset)]
        if len(row):
            rows.append(
                {
                    "label": label,
                    "n": int(row["n"].iloc[0]),
                    "value": float(row["median_delta_change_dp2_minus_dp1"].iloc[0]),
                    "sigma": float(row["mad_sigma_delta_change_dp2_minus_dp1"].iloc[0]),
                    "kind": "resolved" if "resolved" in subset and subset != "unresolved_both" else "unresolved",
                }
            )
    stats = pd.DataFrame(rows)
    colors = [
        "0.55" if label.startswith("All") else "tab:orange" if kind == "resolved" else "tab:blue"
        for label, kind in zip(stats["label"], stats["kind"])
    ]

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    x = np.arange(len(stats))
    ax.bar(x, stats["value"], color=colors, alpha=0.88)
    ax.axhline(0.0, color="0.25", lw=1.1, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{row['label']}\nN={int(row['n']):,}" for _, row in stats.iterrows()])
    ax.set_ylabel("Median delta_change = delta_DP2 - delta_DP1")
    ax.set_title("ECDFS common objects, g band, mag < 22: effect of unresolved/resolved cuts")
    ax.grid(True, axis="y", alpha=0.22)
    for i, row in stats.iterrows():
        ax.text(i, row["value"], f"{row['value']:+.3f}", ha="center", va="top" if row["value"] < 0 else "bottom", fontsize=9)
    fig.text(
        0.5,
        0.01,
        "Bars show medians only. Negative values mean DP2 has smaller psf_mag - cmodel_mag. The strict unresolved-in-both cut compresses the offset by construction.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])
    return save_fig(fig, plot_dir / "dp1_dp2_common_g_unresolved_cut_effect_simple_ECDFS.png")


def plot_g_transition_matrix_simple(transition_counts: pd.DataFrame, plot_dir: Path) -> Path:
    """Show unresolved/resolved boundary movement as a simple 2x2 matrix."""

    bright = transition_counts[transition_counts["mag_bin"].eq("mag < 22")].copy()
    classes = ["unresolved", "resolved"]
    matrix = np.zeros((2, 2), dtype=float)
    row_frac = np.zeros((2, 2), dtype=float)
    for i, dp1_class in enumerate(classes):
        for j, dp2_class in enumerate(classes):
            row = bright[bright["dp1_class"].eq(dp1_class) & bright["dp2_class"].eq(dp2_class)]
            if len(row):
                matrix[i, j] = float(row["n"].iloc[0])
                row_frac[i, j] = float(row["fraction_within_dp1_class"].iloc[0])

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["DP2 unresolved", "DP2 resolved"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["DP1 unresolved", "DP1 resolved"])
    ax.set_title("ECDFS common objects, g band, mag < 22\nmovement across delta = 0.03 boundary")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{int(matrix[i, j]):,}\n{100.0 * row_frac[i, j]:.1f}%", ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax, label="N objects")
    fig.tight_layout()
    return save_fig(fig, plot_dir / "dp1_dp2_common_g_transition_matrix_simple_ECDFS.png")


def _plot_common_change_panel(ax: plt.Axes, common: pd.DataFrame, band: str, value: str, cmap: str = "viridis") -> None:
    y_col = f"{value}_{band}"
    labels = {
        "psf_change": "PSF mag change, DP2 - DP1",
        "cmodel_change": "CModel mag change, DP2 - DP1",
    }
    m = (
        common[f"common_clean_{band}"].astype(bool)
        & common[f"mag_ref_{band}"].between(*MAG_RANGE)
        & np.isfinite(common[y_col])
    )
    vals = common.loc[m, y_col]
    if len(vals):
        lo, hi = np.nanpercentile(vals, [1, 99])
        pad = max(0.04, 0.1 * (hi - lo))
        y_min, y_max = max(-1.5, lo - pad), min(1.5, hi + pad)
    else:
        y_min, y_max = -0.5, 0.5
    hb = ax.hexbin(
        common.loc[m, f"mag_ref_{band}"],
        common.loc[m, y_col],
        gridsize=115,
        bins="log",
        mincnt=1,
        cmap=cmap,
    )
    for edge in [22, 23, 24, 25]:
        ax.axvline(edge, color="0.82", lw=0.8, ls=":")
    centers = bin_centers()
    for bin_label, lo, hi in MAG_BINS:
        bm = m & mag_bin_mask(common[f"mag_ref_{band}"], lo, hi)
        if int(bm.sum()):
            ax.plot(centers[bin_label], np.nanmedian(common.loc[bm, y_col]), marker="o", ms=5, color="red")
    ax.axhline(0.0, color="white", lw=1.1, ls="--")
    ax.set_xlabel(f"{band} reference CModel mag = mean(DP1, DP2)")
    ax.set_ylabel(labels[value])
    ax.set_xlim(*MAG_RANGE)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.16)
    return hb


def plot_common_g_psf_cmodel_separate_diagnostic(common: pd.DataFrame, plot_dir: Path, band: str = "g") -> Path:
    """Four-panel PSF/CModel diagnostic for common objects."""

    fig, axes = plt.subplots(2, 2, figsize=(13.2, 10.0))
    bins = np.linspace(*MAG_RANGE, 80)
    clean = common[f"common_clean_{band}"].astype(bool)
    panels = [
        (axes[0, 0], "psf", "PSF"),
        (axes[0, 1], "cmodel", "CModel"),
    ]
    for ax, mag_kind, label in panels:
        dp1_col = f"dp1_{mag_kind}_mag_{band}"
        dp2_col = f"dp2_{mag_kind}_mag_{band}"
        m = (
            clean
            & common[dp1_col].between(*MAG_RANGE)
            & common[dp2_col].between(*MAG_RANGE)
            & np.isfinite(common[dp1_col])
            & np.isfinite(common[dp2_col])
        )
        ax.hist(common.loc[m, dp1_col], bins=bins, density=True, histtype="step", lw=2.2, color="tab:blue", label=f"DP1 N={int(m.sum()):,}")
        ax.hist(common.loc[m, dp2_col], bins=bins, density=True, histtype="step", lw=2.2, color="tab:orange", label=f"DP2 N={int(m.sum()):,}")
        ax.set_xlabel(f"{band} {label} magnitude (AB)")
        ax.set_ylabel("Normalized density")
        ax.set_xlim(*MAG_RANGE)
        ax.set_title(f"{label} magnitude histogram")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)

    hb1 = _plot_common_change_panel(axes[1, 0], common, band, "psf_change", cmap="viridis")
    axes[1, 0].set_title("PSF magnitude change vs magnitude")
    hb2 = _plot_common_change_panel(axes[1, 1], common, band, "cmodel_change", cmap="magma")
    axes[1, 1].set_title("CModel magnitude change vs magnitude")
    fig.colorbar(hb1, ax=axes[1, 0], label="log10(N)")
    fig.colorbar(hb2, ax=axes[1, 1], label="log10(N)")
    fig.suptitle(
        "DP1 vs DP2 ECDFS common-object g-band diagnostic\n"
        "common objects matched within 0.3 arcsec; change = DP2 - DP1",
        y=1.02,
        fontsize=14,
    )
    fig.text(
        0.5,
        0.005,
        "Red points are magnitude-bin medians for mag < 22, 22-23, 23-24, and 24-25.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.96])
    return save_fig(fig, plot_dir / "dp1_dp2_common_g_psf_cmodel_separate_diagnostic_ECDFS.png")


def plot_common_change_violin_by_mag(common: pd.DataFrame, band: str, plot_dir: Path, unresolved_only: bool = False) -> Path:
    """Show DP2-DP1 change distributions by magnitude bin for common objects."""

    metrics = [
        ("psf_change", "PSF mag change\nDP2 - DP1", "tab:blue"),
        ("cmodel_change", "CModel mag change\nDP2 - DP1", "tab:orange"),
        ("delta_change", "(PSF-CModel) change\nDP2 - DP1", "tab:green"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), sharex=True)
    base = common[f"common_clean_{band}"].astype(bool) & common[f"mag_ref_{band}"].between(*MAG_RANGE)
    title_selection = "all common objects"
    file_selection = ""
    if unresolved_only:
        base &= common_unresolved_mask(common, band)
        title_selection = "unresolved common objects: DP1 and DP2 psf_mag - cmodel_mag < 0.03"
        file_selection = "_unresolved"
    positions = np.arange(1, len(MAG_BINS) + 1)
    xticklabels = ["<22", "22-23", "23-24", "24-25"]

    for ax, (value, ylabel, color) in zip(axes, metrics):
        y_col = f"{value}_{band}"
        data = []
        data_positions = []
        medians = []
        sigmas = []
        counts = []
        for pos, (_, lo, hi) in zip(positions, MAG_BINS):
            bm = base & mag_bin_mask(common[f"mag_ref_{band}"], lo, hi) & np.isfinite(common[y_col])
            vals = common.loc[bm, y_col].to_numpy(dtype="float64")
            vals = vals[np.isfinite(vals)]
            counts.append(len(vals))
            if len(vals):
                data.append(vals)
                data_positions.append(pos)
                medians.append(float(np.nanmedian(vals)))
                sigmas.append(robust_sigma(vals))
            else:
                medians.append(np.nan)
                sigmas.append(np.nan)

        if data:
            parts = ax.violinplot(data, positions=data_positions, widths=0.75, showextrema=False, showmedians=False)
            for body in parts["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("0.25")
                body.set_alpha(0.35)
                body.set_linewidth(0.8)
            valid = np.isfinite(medians)
            ax.errorbar(
                positions[valid],
                np.asarray(medians)[valid],
                yerr=np.asarray(sigmas)[valid],
                fmt="o",
                ms=5,
                color="red",
                ecolor="red",
                elinewidth=1.1,
                capsize=3,
                label="median +/- MAD sigma",
            )
            finite_medians = np.asarray(medians, dtype="float64")[valid]
            finite_sigmas = np.asarray(sigmas, dtype="float64")[valid]
            lo = np.nanmin(finite_medians - 4.0 * finite_sigmas)
            hi = np.nanmax(finite_medians + 4.0 * finite_sigmas)
            pad = max(0.04, 0.12 * (hi - lo))
            ax.set_ylim(max(-1.0, lo - pad), min(1.0, hi + pad))
        else:
            ax.text(0.5, 0.5, "No valid common objects", transform=ax.transAxes, ha="center", va="center")

        ax.axhline(0.0, color="0.35", lw=1.0, ls="--")
        ax.set_xticks(positions)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(f"{band} reference CModel mag bin")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_title(ylabel.replace("\n", " "))
        for pos, n in zip(positions, counts):
            ax.text(pos, 0.98, f"N={n:,}", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8)
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle(
        f"DP1 vs DP2 ECDFS common-object {band}-band change distributions by magnitude bin\n"
        f"{title_selection}; matched within 0.3 arcsec; sign convention: change = DP2 - DP1",
        y=1.03,
        fontsize=13,
    )
    fig.text(
        0.5,
        -0.01,
        "Y limits use median +/- 4 MAD sigma so the central distribution is visible; red points use full-bin medians.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout()
    return save_fig(fig, plot_dir / f"dp1_dp2_common_{band}{file_selection}_psf_cmodel_change_violin_by_mag_ECDFS.png")


def plot_common_change_vs_mag(common: pd.DataFrame, band: str, value: str, plot_dir: Path) -> Path:
    y_col = f"{value}_{band}"
    y_labels = {
        "psf_change": f"{band} PSF mag change: DP2 - DP1 (AB)",
        "cmodel_change": f"{band} CModel mag change: DP2 - DP1 (AB)",
        "delta_change": f"{band} delta change: DP2 - DP1 (AB)",
    }
    titles = {
        "psf_change": "PSF magnitude change",
        "cmodel_change": "CModel magnitude change",
        "delta_change": "PSF-CModel delta change",
    }
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    m = (
        common[f"common_clean_{band}"].astype(bool)
        & common[f"mag_ref_{band}"].between(*MAG_RANGE)
        & np.isfinite(common[y_col])
    )
    vals = common.loc[m, y_col]
    if len(vals):
        lo, hi = np.nanpercentile(vals, [1, 99])
        pad = max(0.05, 0.1 * (hi - lo))
        y_min, y_max = max(-3.0, lo - pad), min(3.0, hi + pad)
    else:
        y_min, y_max = -1.0, 1.0
    hb = ax.hexbin(
        common.loc[m, f"mag_ref_{band}"],
        common.loc[m, y_col],
        gridsize=130,
        bins="log",
        mincnt=1,
        cmap="viridis",
    )
    centers = {"mag < 22": 21.5, "22 <= mag < 23": 22.5, "23 <= mag < 24": 23.5, "24 <= mag < 25": 24.5}
    for bin_label, lo, hi in MAG_BINS:
        bm = m & mag_bin_mask(common[f"mag_ref_{band}"], lo, hi)
        if int(bm.sum()):
            ax.plot(centers[bin_label], np.nanmedian(common.loc[bm, y_col]), marker="o", color="red")
    ax.axhline(0.0, color="white", lw=1.2, ls="--")
    ax.set_xlabel(f"{band} reference CModel mag = mean(DP1, DP2)")
    ax.set_ylabel(y_labels[value])
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(*MAG_RANGE)
    ax.set_title(f"ECDFS common objects: {titles[value]} in {band} band")
    ax.grid(True, alpha=0.18)
    fig.colorbar(hb, ax=ax, label="log10(N)")
    return save_fig(fig, plot_dir / f"dp1_dp2_common_{band}_{value}_vs_mag_ECDFS.png")


def plot_common_delta_dp1_vs_dp2(common: pd.DataFrame, band: str, plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.6, 5.8))
    m = (
        common[f"common_clean_{band}"].astype(bool)
        & common[f"delta_dp1_{band}"].between(*DELTA_RANGE)
        & common[f"delta_dp2_{band}"].between(*DELTA_RANGE)
    )
    hb = ax.hexbin(
        common.loc[m, f"delta_dp1_{band}"],
        common.loc[m, f"delta_dp2_{band}"],
        gridsize=130,
        bins="log",
        mincnt=1,
        cmap="magma",
    )
    ax.plot(DELTA_RANGE, DELTA_RANGE, color="white", lw=1.2, ls="--")
    ax.axhline(0.0, color="0.75", lw=0.9, ls=":")
    ax.axvline(0.0, color="0.75", lw=0.9, ls=":")
    ax.set_xlabel(f"DP1 {band} psf_mag - cmodel_mag (AB)")
    ax.set_ylabel(f"DP2 {band} psf_mag - cmodel_mag (AB)")
    ax.set_xlim(*DELTA_RANGE)
    ax.set_ylim(*DELTA_RANGE)
    ax.set_title(f"ECDFS common objects: DP1 vs DP2 {band}-band morphology delta")
    ax.grid(True, alpha=0.18)
    fig.colorbar(hb, ax=ax, label="log10(N)")
    return save_fig(fig, plot_dir / f"dp1_dp2_common_{band}_delta_DP1_vs_delta_DP2_ECDFS.png")


def plot_common_hst_delta_change(common: pd.DataFrame, band: str, plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for label, color, size, alpha in [("galaxy", "tab:blue", 8, 0.28), ("star", "red", 22, 0.82)]:
        m = (
            common[f"common_clean_{band}"].astype(bool)
            & common["truth_label"].eq(label)
            & common[f"mag_ref_{band}"].between(*MAG_RANGE)
            & np.isfinite(common[f"delta_change_{band}"])
        )
        ax.scatter(
            common.loc[m, f"mag_ref_{band}"],
            common.loc[m, f"delta_change_{band}"],
            s=size,
            alpha=alpha,
            color=color,
            label=f"HST {label} N={int(m.sum()):,}",
        )
    ax.axhline(0.0, color="0.35", lw=1.2, ls=":")
    ax.set_xlabel(f"{band} reference CModel mag = mean(DP1, DP2)")
    ax.set_ylabel(f"{band} delta change: DP2 - DP1 (AB)")
    ax.set_xlim(*MAG_RANGE)
    ax.set_ylim(-2.0, 2.0)
    ax.set_title(f"ECDFS common HST-labeled objects: {band} delta change")
    ax.grid(True, alpha=0.2)
    ax.legend()
    return save_fig(fig, plot_dir / f"dp1_dp2_common_{band}_delta_change_HST_stars_galaxies_ECDFS.png")


def plot_common_hst_psf_cmodel_change(common: pd.DataFrame, band: str, plot_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    for label, color, size, alpha in [("galaxy", "tab:blue", 8, 0.28), ("star", "red", 24, 0.84)]:
        m = (
            common[f"common_clean_{band}"].astype(bool)
            & common["truth_label"].eq(label)
            & np.isfinite(common[f"psf_change_{band}"])
            & np.isfinite(common[f"cmodel_change_{band}"])
        )
        ax.scatter(
            common.loc[m, f"psf_change_{band}"],
            common.loc[m, f"cmodel_change_{band}"],
            s=size,
            alpha=alpha,
            color=color,
            label=f"HST {label} N={int(m.sum()):,}",
        )
    ax.axhline(0.0, color="0.55", lw=1.0, ls=":")
    ax.axvline(0.0, color="0.55", lw=1.0, ls=":")
    ax.plot((-2.0, 2.0), (-2.0, 2.0), color="0.35", lw=1.1, ls="--", label="equal PSF/CModel change")
    ax.set_xlabel(f"{band} PSF mag change: DP2 - DP1 (AB)")
    ax.set_ylabel(f"{band} CModel mag change: DP2 - DP1 (AB)")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_title(f"ECDFS common HST-labeled objects: {band} PSF vs CModel change")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    return save_fig(fig, plot_dir / f"dp1_dp2_common_{band}_psf_cmodel_change_HST_stars_galaxies_ECDFS.png")


def save_common_bright_delta_change_by_band(common_summary: pd.DataFrame, plot_dir: Path, result_dir: Path) -> tuple[Path, Path]:
    """Save compact bright-bin by-band common-object delta-change summary."""

    bands = ["g", "r", "i", "z"]
    use = common_summary[
        common_summary["sample"].eq("all_common")
        & common_summary["mag_bin"].eq("mag < 22")
        & common_summary["band"].isin(bands)
    ].copy()
    use["band"] = pd.Categorical(use["band"], categories=bands, ordered=True)
    use = use.sort_values("band")
    cols = [
        "band",
        "mag_bin",
        "sample",
        "n",
        "median_delta_change",
        "mad_sigma_delta_change",
        "median_psf_change",
        "mad_sigma_psf_change",
        "median_cmodel_change",
        "mad_sigma_cmodel_change",
    ]
    csv_path = result_dir / "dp1_dp2_common_bright_delta_change_by_band_ECDFS.csv"
    use[cols].to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    colors = ["tab:red" if band == "g" else "0.55" for band in use["band"].astype(str)]
    ax.bar(use["band"].astype(str), use["median_delta_change"], color=colors, alpha=0.88)
    ax.errorbar(
        use["band"].astype(str),
        use["median_delta_change"],
        yerr=use["mad_sigma_delta_change"],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
    )
    ax.axhline(0.0, color="0.25", lw=1.1, ls=":")
    ax.set_xlabel("Band")
    ax.set_ylabel("median Δ(psf_mag - cmodel_mag), DP2 - DP1")
    ax.set_title("ECDFS common objects, bright bin (mag < 22)")
    ax.grid(True, axis="y", alpha=0.22)
    for x, (_, row) in enumerate(use.iterrows()):
        ax.text(x, row["median_delta_change"], f"N={int(row['n']):,}", ha="center", va="bottom" if row["median_delta_change"] >= 0 else "top", fontsize=8)
    png_path = plot_dir / "dp1_dp2_common_bright_delta_change_by_band_ECDFS.png"
    save_fig(fig, png_path)
    return csv_path, png_path


def write_cut_notes(path: Path, dp1: StandardizedCatalog, dp2: StandardizedCatalog, bands: Iterable[str]) -> Path:
    lines = [
        "# DP1 vs DP2 Cut and Column Differences\n\n",
        "This diagnostic compares DP1 and DP2 ECDFS using matched derived quantities rather than changing the science pipeline.\n\n",
        "## Data Products\n\n",
        f"- DP1 input: `{dp1.path}`\n",
        f"- DP2 input: `{dp2.path}`\n",
        "- HST matching radius used for labeled checks: 0.3 arcsec.\n\n",
        "## Sign Convention\n\n",
        "- Classifier delta is defined consistently as `psf_mag - cmodel_mag`.\n",
        "- Positive values mean PSF magnitude is fainter than CModel magnitude.\n\n",
        "## Column Mapping\n\n",
        "- DP1 PSF flux: `[band]_free_psfFlux`; DP2 PSF flux: `[band]_psfFlux`.\n",
        "- DP1 CModel flux: `[band]_free_cModelFlux`; DP2 CModel flux: `[band]_cModelFlux`.\n",
        "- DP1 CModel flag: `[band]_free_cModelFlux_flag`; DP2 CModel flag: `[band]_cModel_flag`.\n",
        "- Extendedness uses `[band]_extendedness` in both DP1 and DP2, with `[band]_extendedness_flag` where available.\n\n",
        "## First-Pass Clean Mask\n\n",
        "For each band, the comparison clean mask requires finite positive PSF and CModel fluxes, finite derived magnitudes, finite delta, and clean PSF/CModel flux flags. This is a diagnostic cut, not a final science cut.\n\n",
        "## Clean Counts\n\n",
    ]
    for band in bands:
        lines.append(f"- `{band}` clean rows: DP1 {int(dp1.df[f'clean_{band}'].sum()):,}; DP2 {int(dp2.df[f'clean_{band}'].sum()):,}\n")
    path.write_text("".join(lines))
    return path


def summarize_findings(
    path: Path,
    bright: pd.DataFrame,
    hst_offsets: pd.DataFrame | None,
    sample: pd.DataFrame,
    generated: list[Path],
    missing: list[str],
    common_summary: pd.DataFrame | None = None,
    common_counts: dict[str, int] | None = None,
    cut_diagnostics: pd.DataFrame | None = None,
    transition_counts: pd.DataFrame | None = None,
) -> Path:
    def med(dataset: str, band: str, bin_label: str = "mag < 22") -> float:
        row = bright[(bright["dataset"].eq(dataset)) & (bright["band"].eq(band)) & (bright["mag_bin"].eq(bin_label))]
        return float(row["delta_median"].iloc[0]) if len(row) else np.nan

    g_dp1 = med("DP1", "g")
    g_dp2 = med("DP2", "g")
    rows = [
        "# DP1 vs DP2 ECDFS g-band Diagnostic Summary\n\n",
        "This is a first-pass diagnostic intended to explain why some DP2 g-band plots differ from analogous DP1 plots.\n\n",
        "## Inputs\n\n",
        "- DP1: `data/ECDFS.fits`\n",
        "- DP2: `data/private/DP2_ECDFS_objects.fits`\n",
        "- DP2 pS: `outputs/dp2_ecdfs_ps_v6.parquet` when available.\n",
        "- HST labels: `data/hst_truth_catalog.csv`, positionally matched with a 0.3 arcsec radius.\n\n",
        "## Reproducibility\n\n",
        "- Run command: `python3 src/compare_dp1_dp2_ecdfs.py --bands g r i z`\n",
        "- DP1-DP2 common objects are spatially matched with a 0.3 arcsec radius.\n",
        "- Unresolved/resolved threshold: `delta = psf_mag - cmodel_mag`; unresolved means `delta < 0.03`, resolved means `delta > 0.03`.\n\n",
        "## Main g-band Result\n\n",
        f"- Bright-bin (`mag < 22`) median `psf_mag - cmodel_mag`: DP1 = {g_dp1:.4f}, DP2 = {g_dp2:.4f}, DP2-DP1 = {g_dp2 - g_dp1:.4f} mag.\n",
        "- The generated histograms and slice plots show whether the g-band delta distribution is shifted, broadened, or shaped differently between releases.\n",
        "- The PSF-vs-CModel and delta-vs-magnitude plots separate whether the difference is mainly in PSF, CModel, or their difference.\n\n",
        "## Band Check\n\n",
    ]
    for band in sorted(bright["band"].unique()):
        d1 = med("DP1", band)
        d2 = med("DP2", band)
        rows.append(f"- `{band}` bright-bin median delta: DP1 {d1:.4f}; DP2 {d2:.4f}; DP2-DP1 {d2 - d1:.4f} mag.\n")

    rows.extend(
        [
            "\n## HST-Labeled Check\n\n",
            "- The HST-labeled comparison uses independent position matching to both DP1 and DP2 with a 0.3 arcsec radius.\n",
        ]
    )
    if hst_offsets is not None and len(hst_offsets):
        for label in ["star", "galaxy"]:
            for dataset in ["DP1", "DP2"]:
                row = hst_offsets[
                    hst_offsets["dataset"].eq(dataset)
                    & hst_offsets["truth_label"].eq(label)
                    & hst_offsets["mag_bin"].eq("mag < 22")
                ]
                if len(row):
                    plural = "galaxies" if label == "galaxy" else "stars"
                    rows.append(f"- `{dataset}` HST {plural}, `mag < 22`: N={int(row['n'].iloc[0]):,}, median delta={float(row['delta_median'].iloc[0]):.4f}.\n")
    else:
        rows.append("- HST-labeled offsets were not available.\n")

    rows.extend(
        [
            "\n## Sample Selection\n\n",
            "- See `results/dp1_dp2_sample_counts_ECDFS.csv` for total rows, coordinate-valid rows, band-clean rows, and magnitude-bin counts.\n",
            "- The comparison uses matched sign conventions and first-pass band-specific clean masks; it does not impose a global all-band science cut.\n\n",
        ]
    )

    if common_summary is not None and common_counts is not None and len(common_summary):
        g_bin = "mag < 22"
        g_psf = common_bin_value(common_summary, "g", "all_common", g_bin, "median_psf_change")
        g_cmodel = common_bin_value(common_summary, "g", "all_common", g_bin, "median_cmodel_change")
        g_delta = common_bin_value(common_summary, "g", "all_common", g_bin, "median_delta_change")
        g_star_dp1 = common_bin_value(common_summary, "g", "hst_star", g_bin, "median_delta_dp1")
        g_star_dp2 = common_bin_value(common_summary, "g", "hst_star", g_bin, "median_delta_dp2")
        g_gal_dp1 = common_bin_value(common_summary, "g", "hst_galaxy", g_bin, "median_delta_dp1")
        g_gal_dp2 = common_bin_value(common_summary, "g", "hst_galaxy", g_bin, "median_delta_dp2")
        if abs(g_cmodel) > 1.5 * abs(g_psf):
            driver = "mostly by the CModel magnitude change"
        elif abs(g_psf) > 1.5 * abs(g_cmodel):
            driver = "mostly by the PSF magnitude change"
        else:
            driver = "by both PSF and CModel magnitude changes"
        rows.extend(
            [
                "## Common-object DP1-DP2 comparison\n\n",
                f"- DP1-DP2 common objects matched within 0.3 arcsec: {common_counts.get('common_objects', 0):,}.\n",
                f"- Common objects with HST labels: {common_counts.get('hst_labeled_common', 0):,} "
                f"({common_counts.get('hst_star_common', 0):,} stars; {common_counts.get('hst_galaxy_common', 0):,} galaxies).\n",
                f"- In g band, `{g_bin}`, median PSF change `DP2-DP1` = {g_psf:.4f} mag; median CModel change `DP2-DP1` = {g_cmodel:.4f} mag; median delta change = {g_delta:.4f} mag.\n",
                f"- At the median level, the g-band delta change is driven {driver}.\n",
                "- Note: medians of `psf_change` and `cmodel_change` do not have to add exactly to the median `delta_change`. "
                "The decomposition is therefore a robust diagnostic of the typical shifts, not an exact additive equality for the median statistics.\n",
                f"- HST stars remain near zero in the common sample: DP1 median delta = {g_star_dp1:.4f}, DP2 median delta = {g_star_dp2:.4f}.\n",
                f"- HST galaxies still show smaller DP2 morphology delta: DP1 median delta = {g_gal_dp1:.4f}, DP2 median delta = {g_gal_dp2:.4f}.\n",
                "- Bright-bin median delta changes by band in the common sample:\n",
            ]
        )
        for band in sorted(common_summary["band"].unique()):
            val = common_bin_value(common_summary, band, "all_common", g_bin, "median_delta_change")
            rows.append(f"  - `{band}`: {val:.4f} mag\n")
        rows.extend(
            [
                "- Clearest common-object evidence: `results/dp1_dp2_common_object_delta_decomposition_ECDFS.csv`, "
                "`results/dp1_dp2_common_bright_delta_change_by_band_ECDFS.csv`, "
                "`results/dp1_dp2_common_g_psf_cmodel_separate_summary_ECDFS.csv`, "
                "`plots/dp1_dp2_common_bright_delta_change_by_band_ECDFS.png`, "
                "`plots/dp1_dp2_common_g_psf_cmodel_separate_diagnostic_ECDFS.png`, "
                "`plots/dp1_dp2_common_g_delta_change_vs_mag_ECDFS.png`, "
                "`plots/dp1_dp2_common_g_delta_DP1_vs_delta_DP2_ECDFS.png`, and "
                "`plots/dp1_dp2_common_g_psf_cmodel_change_HST_stars_galaxies_ECDFS.png`.\n\n",
            ]
        )

    if cut_diagnostics is not None and len(cut_diagnostics):
        def cut_row(cut_definition: str, subset: str, mag_bin: str = "mag < 22") -> pd.Series | None:
            match = cut_diagnostics[
                cut_diagnostics["cut_definition"].eq(cut_definition)
                & cut_diagnostics["subset"].eq(subset)
                & cut_diagnostics["mag_bin"].eq(mag_bin)
            ]
            return match.iloc[0] if len(match) else None

        all_bright = cut_row("all_common", "all_common")
        both_bright = cut_row("unresolved_both", "unresolved_both")
        dp1_unres = cut_row("cut_on_DP1", "unresolved")
        dp1_res = cut_row("cut_on_DP1", "resolved")
        dp2_unres = cut_row("cut_on_DP2", "unresolved")
        dp2_res = cut_row("cut_on_DP2", "resolved")

        rows.extend(
            [
                "## Unresolved/resolved cut diagnostic\n\n",
                "- This check uses `delta = psf_mag - cmodel_mag` and compares multiple ways of defining unresolved/resolved objects.\n",
                "- For the requested magnitude-difference histograms, the sign convention is `DP1 - DP2`; for morphology change, the convention remains `delta_change = delta_DP2 - delta_DP1`.\n",
                "- In the scatter plots, the dashed `y=x` line marks equal PSF and CModel magnitude differences. Points below that line have `delta_DP1 > delta_DP2`, meaning DP2 has smaller `psf_mag - cmodel_mag`.\n",
                "- A strict `unresolved_both` selection can make the offset look smaller because it requires both `delta_DP1 < 0.03` and `delta_DP2 < 0.03`, which directly limits the range of the two endpoint morphology deltas.\n",
            ]
        )
        if all_bright is not None:
            rows.append(
                f"- All common objects, `mag < 22`: N={int(all_bright['n']):,}, "
                f"median `delta_change`={float(all_bright['median_delta_change_dp2_minus_dp1']):.4f}, "
                f"median `psfDP1-psfDP2`={float(all_bright['median_psfDP1_minus_psfDP2']):.4f}, "
                f"median `cModelDP1-cModelDP2`={float(all_bright['median_cModelDP1_minus_cModelDP2']):.4f}.\n"
            )
        if both_bright is not None:
            rows.append(
                f"- `unresolved_both`, `mag < 22`: N={int(both_bright['n']):,}, "
                f"median `delta_change`={float(both_bright['median_delta_change_dp2_minus_dp1']):.4f}; "
                "this reproduces the smaller offset in the strict unresolved-only sample.\n"
            )
        if dp1_unres is not None and dp1_res is not None:
            rows.append(
                f"- Cut on DP1: unresolved `mag < 22` has median `delta_change`={float(dp1_unres['median_delta_change_dp2_minus_dp1']):.4f} "
                f"(N={int(dp1_unres['n']):,}); resolved has median `delta_change`={float(dp1_res['median_delta_change_dp2_minus_dp1']):.4f} "
                f"(N={int(dp1_res['n']):,}).\n"
            )
        if dp2_unres is not None and dp2_res is not None:
            rows.append(
                f"- Cut on DP2: unresolved `mag < 22` has median `delta_change`={float(dp2_unres['median_delta_change_dp2_minus_dp1']):.4f} "
                f"(N={int(dp2_unres['n']):,}); resolved has median `delta_change`={float(dp2_res['median_delta_change_dp2_minus_dp1']):.4f} "
                f"(N={int(dp2_res['n']):,}).\n"
            )
        if transition_counts is not None and len(transition_counts):
            def transition_row(transition: str, mag_bin: str = "mag < 22") -> pd.Series | None:
                match = transition_counts[
                    transition_counts["transition"].eq(transition)
                    & transition_counts["mag_bin"].eq(mag_bin)
                ]
                return match.iloc[0] if len(match) else None

            res_to_unres = transition_row("DP1_resolved_to_DP2_unresolved")
            unres_to_res = transition_row("DP1_unresolved_to_DP2_resolved")
            if res_to_unres is not None and unres_to_res is not None:
                rows.append(
                    f"- Boundary migration, `mag < 22`: DP1 resolved -> DP2 unresolved is "
                    f"{int(res_to_unres['n']):,}/{int(res_to_unres['dp1_class_n']):,} "
                    f"({100.0 * float(res_to_unres['fraction_within_dp1_class']):.2f}% of DP1-resolved objects); "
                    f"DP1 unresolved -> DP2 resolved is {int(unres_to_res['n']):,}/{int(unres_to_res['dp1_class_n']):,} "
                    f"({100.0 * float(unres_to_res['fraction_within_dp1_class']):.2f}% of DP1-unresolved objects).\n"
                )
        rows.extend(
            [
                "- The clearest plots for this check are "
                "`plots/dp1_dp2_common_g_psf_diff_vs_cmodel_diff_scatter_cut_on_DP1_ECDFS.png`, "
                "`plots/dp1_dp2_common_g_psf_diff_vs_cmodel_diff_scatter_cut_on_DP2_ECDFS.png`, "
                "`plots/dp1_dp2_common_g_psf_model_hist_by_mag_cut_on_DP1_ECDFS.png`, and "
                "`plots/dp1_dp2_common_g_psfDP1_minus_psfDP2_hist_by_mag_cut_on_DP1_ECDFS.png`.\n",
                "- The clearest table for the boundary test is `results/dp1_dp2_g_unresolved_resolved_transition_counts_ECDFS.csv`.\n\n",
            ]
        )

        rows.extend(
            [
                "## Histogram median/mean annotations\n\n",
                "- The overlaid `psf_mag - cmodel_mag` histograms now explicitly label the DP1 and DP2 median lines in the same color family as the corresponding histogram.\n",
                "- The `psfDP1 - psfDP2` histograms isolate the PSF magnitude offset. Positive values mean the DP2 PSF magnitude is smaller/brighter.\n",
                "- The `CModelDP1 - CModelDP2` histograms isolate the CModel magnitude offset. Negative values mean the DP2 CModel magnitude is larger/fainter.\n",
                "- The PSF and CModel difference histograms show both the median and mean in each magnitude bin and sample definition.\n",
                "- The annotated values are also saved in `results/dp1_dp2_g_histogram_median_mean_summary_ECDFS.csv`.\n\n",
            ]
        )

    rows.extend(
        [
            "## Missing or Limited Pieces\n\n",
        ]
    )
    if missing:
        for item in missing:
            rows.append(f"- {item}\n")
    else:
        rows.append("- No blocking missing files/columns were encountered for the requested core comparisons.\n")

    rows.extend(
        [
            "\n## Clearest Plots to Show\n\n",
            "- `plots/dp1_dp2_compare_g_psf_cmodel_delta_hist_ECDFS.png`\n",
            "- `plots/dp1_dp2_compare_g_delta_vs_mag_ECDFS.png`\n",
            "- `plots/dp1_dp2_compare_g_slice_distributions_ECDFS.png`\n",
            "- `plots/dp1_dp2_g_bright_end_offset_vs_mag_ECDFS.png`\n",
            "- `plots/dp1_dp2_g_delta_hist_HST_stars_galaxies_ECDFS.png`\n",
            "- `plots/dp1_dp2_footprint_ECDFS.png`\n\n",
            f"Generated {len(generated)} diagnostic plots/tables in this run.\n",
        ]
    )
    path.write_text("".join(rows))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dp1", default="data/ECDFS.fits", type=Path)
    parser.add_argument("--dp2", default="data/private/DP2_ECDFS_objects.fits", type=Path)
    parser.add_argument("--dp2-ps", default="outputs/dp2_ecdfs_ps_v6.parquet", type=Path)
    parser.add_argument("--hst-truth", default="data/hst_truth_catalog.csv", type=Path)
    parser.add_argument("--bands", nargs="+", default=list(BANDS))
    parser.add_argument("--plot-dir", default="plots", type=Path)
    parser.add_argument("--result-dir", default="results", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dir(args.plot_dir)
    ensure_dir(args.result_dir)
    generated: list[Path] = []
    missing: list[str] = []

    bands = tuple(args.bands)
    dp1 = standardize_dp1(args.dp1, bands)
    dp2 = standardize_dp2(args.dp2, bands, args.dp2_ps if args.dp2_ps.exists() else None)
    if not args.dp2_ps.exists():
        missing.append(f"DP2 pS file not found: `{args.dp2_ps}`; pS-vs-magnitude plots skipped.")

    for band in bands:
        generated.append(plot_mag_hist(dp1, dp2, band, "psf", args.plot_dir))
        generated.append(plot_mag_hist(dp1, dp2, band, "cmodel", args.plot_dir))
        generated.append(plot_delta_hist(dp1, dp2, band, args.plot_dir))
        generated.append(plot_psf_vs_cmodel(dp1, dp2, band, args.plot_dir))
        generated.append(plot_delta_vs_mag(dp1, dp2, band, args.plot_dir))
        ps_plot = plot_ps_vs_mag(dp1, dp2, band, args.plot_dir)
        if ps_plot is not None:
            generated.append(ps_plot)
            if f"pS_{band}" not in dp1.df.columns:
                missing.append(f"DP1 per-object `pS_{band}` is not present in this fresh clone; `{ps_plot}` includes DP2 pS and an annotated missing-DP1 panel.")
        else:
            missing.append(f"No per-object pS values found for either DP1 or DP2 in `{band}`.")
        generated.append(plot_slice_distributions(dp1, dp2, band, args.plot_dir))

    bright = bright_end_table([dp1, dp2], bands)
    bright_all_path = args.result_dir / "dp1_dp2_bright_end_offset_summary_ECDFS.csv"
    bright.to_csv(bright_all_path, index=False)
    generated.append(bright_all_path)
    g_bright_path = args.result_dir / "dp1_dp2_g_bright_end_offset_summary_ECDFS.csv"
    bright[bright["band"].eq("g")].to_csv(g_bright_path, index=False)
    generated.append(g_bright_path)
    for band in bands:
        generated.append(plot_bright_end_offsets(bright, band, args.plot_dir))

    counts = sample_counts([dp1, dp2], bands)
    sample_path = args.result_dir / "dp1_dp2_sample_counts_ECDFS.csv"
    counts.to_csv(sample_path, index=False)
    generated.append(sample_path)
    generated.append(plot_footprint(dp1, dp2, args.plot_dir))

    hst_offsets = None
    truth = None
    if args.hst_truth.exists():
        truth = load_hst_truth(args.hst_truth)
        dp1_hst = match_to_hst(dp1, truth, HST_MATCH_RADIUS_ARCSEC)
        dp2_hst = match_to_hst(dp2, truth, HST_MATCH_RADIUS_ARCSEC)
        hst_counts = pd.DataFrame(
            [
                {"dataset": "DP1", "matched_hst_rows_0p3arcsec": len(dp1_hst)},
                {"dataset": "DP2", "matched_hst_rows_0p3arcsec": len(dp2_hst)},
            ]
        )
        hst_counts.to_csv(args.result_dir / "dp1_dp2_hst_match_counts_ECDFS.csv", index=False)
        generated.append(args.result_dir / "dp1_dp2_hst_match_counts_ECDFS.csv")
        generated.append(plot_hst_delta_hist(dp1_hst, dp2_hst, "g", args.plot_dir))
        generated.append(plot_hst_delta_vs_mag(dp1_hst, dp2_hst, "g", args.plot_dir))
        hst_offsets = hst_offset_rows(dp1_hst, dp2_hst, "g")
        hst_offset_path = args.result_dir / "dp1_dp2_g_hst_labeled_offset_summary_ECDFS.csv"
        hst_offsets.to_csv(hst_offset_path, index=False)
        generated.append(hst_offset_path)
    else:
        missing.append(f"HST truth file not found: `{args.hst_truth}`; HST-labeled plots skipped.")

    common = match_dp1_dp2_common(dp1, dp2, bands, COMMON_MATCH_RADIUS_ARCSEC)
    if truth is not None:
        common = add_hst_labels_to_common(common, truth, HST_MATCH_RADIUS_ARCSEC)
    else:
        common["truth_label"] = pd.NA
    common_summary = common_delta_decomposition(common, bands)
    common_summary_path = args.result_dir / "dp1_dp2_common_object_delta_decomposition_ECDFS.csv"
    common_summary.to_csv(common_summary_path, index=False)
    generated.append(common_summary_path)
    compact_csv, compact_plot = save_common_bright_delta_change_by_band(common_summary, args.plot_dir, args.result_dir)
    generated.extend([compact_csv, compact_plot])
    generated.append(save_common_g_psf_cmodel_separate_summary(common_summary, args.result_dir, "g"))
    generated.append(save_common_g_unresolved_summary(common, args.result_dir, "g"))
    cut_diagnostics_path = save_g_unresolved_resolved_cut_diagnostics(common, args.result_dir, "g")
    generated.append(cut_diagnostics_path)
    cut_diagnostics = pd.read_csv(cut_diagnostics_path)
    generated.append(save_g_histogram_median_mean_summary(cut_diagnostics, args.result_dir))
    transition_counts_path = save_g_unresolved_resolved_transition_counts(common, args.result_dir, "g")
    generated.append(transition_counts_path)
    transition_counts = pd.read_csv(transition_counts_path)
    generated.append(plot_common_g_psf_cmodel_separate_diagnostic(common, args.plot_dir, "g"))
    generated.append(plot_g_psf_cmodel_offset_medians_simple(common, args.plot_dir, "g"))
    generated.append(plot_g_delta_cut_effect_simple(cut_diagnostics, args.plot_dir))
    generated.append(plot_g_transition_matrix_simple(transition_counts, args.plot_dir))
    common_counts = {
        "common_objects": int(len(common)),
        "hst_labeled_common": int(common["truth_label"].isin(["star", "galaxy"]).sum()),
        "hst_star_common": int(common["truth_label"].eq("star").sum()),
        "hst_galaxy_common": int(common["truth_label"].eq("galaxy").sum()),
    }
    common_counts_path = args.result_dir / "dp1_dp2_common_object_match_counts_ECDFS.csv"
    pd.DataFrame([common_counts]).to_csv(common_counts_path, index=False)
    generated.append(common_counts_path)

    for band in bands:
        generated.append(plot_common_mag_hist(common, band, "psf", args.plot_dir))
        generated.append(plot_common_mag_hist(common, band, "cmodel", args.plot_dir))
        generated.append(plot_common_change_violin_by_mag(common, band, args.plot_dir))
        generated.append(plot_common_change_vs_mag(common, band, "delta_change", args.plot_dir))
        if band == "g":
            generated.append(plot_common_change_violin_by_mag(common, band, args.plot_dir, unresolved_only=True))
            for cut_key in ["cut_on_DP1", "cut_on_DP2", "unresolved_both", "resolved_both"]:
                generated.append(plot_g_psf_diff_vs_cmodel_diff_scatter_by_cut(common, args.plot_dir, cut_key, band))
            for cut_key in ["all_common", "cut_on_DP1", "cut_on_DP2", "unresolved_both", "resolved_both"]:
                generated.append(plot_g_psf_model_hist_by_mag(common, args.plot_dir, cut_key, band))
                generated.append(plot_g_mag_diff_hist_by_mag(common, args.plot_dir, cut_key, "psf", band))
                generated.append(plot_g_mag_diff_hist_by_mag(common, args.plot_dir, cut_key, "cmodel", band))
            generated.append(plot_common_change_vs_mag(common, band, "psf_change", args.plot_dir))
            generated.append(plot_common_change_vs_mag(common, band, "cmodel_change", args.plot_dir))
            generated.append(plot_common_delta_dp1_vs_dp2(common, band, args.plot_dir))
            if common["truth_label"].isin(["star", "galaxy"]).any():
                generated.append(plot_common_hst_delta_change(common, band, args.plot_dir))
                generated.append(plot_common_hst_psf_cmodel_change(common, band, args.plot_dir))
            else:
                missing.append("No HST labels were attached to the DP1-DP2 common sample; common HST plots skipped.")

    generated.append(write_cut_notes(args.result_dir / "dp1_dp2_cut_differences_notes.md", dp1, dp2, bands))
    generated.append(
        summarize_findings(
            args.result_dir / "dp1_dp2_gband_diagnostic_summary.md",
            bright,
            hst_offsets,
            counts,
            generated,
            missing,
            common_summary,
            common_counts,
            cut_diagnostics,
            transition_counts,
        )
    )

    print("DP1 input:", dp1.path)
    print("DP2 input:", dp2.path)
    print("Compared bands:", ", ".join(bands))
    print("Generated files:")
    for path in generated:
        print(" -", path)
    if missing:
        print("Warnings:")
        for item in missing:
            print(" -", item)


if __name__ == "__main__":
    main()
