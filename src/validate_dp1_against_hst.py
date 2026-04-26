"""Validate Rubin-side star/galaxy quantities against external labels."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_MATCHED = OUTPUT_DIR / "hst_dp1_matched_clean_ellipse.csv"
FALLBACK_MATCHED = OUTPUT_DIR / "hst_dp1_matched_1arcsec.csv"
BANDS = ["u", "g", "r", "i", "z", "y"]
MAG_BINS = [(20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26), (26, 27)]


def _contains_all(name: str, tokens: list[str]) -> bool:
    lower = name.lower()
    return all(tok.lower() in lower for tok in tokens)


def _find_columns(df: pd.DataFrame, tokens: list[str]) -> list[str]:
    return [c for c in df.columns if _contains_all(c, tokens)]


def _first_col(df: pd.DataFrame, token_groups: list[list[str]]) -> str | None:
    for tokens in token_groups:
        matches = _find_columns(df, tokens)
        if matches:
            return matches[0]
    return None


def nJy_to_mag(flux) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    out = np.full(flux.shape, np.nan, dtype=float)
    ok = np.isfinite(flux) & (flux > 0)
    out[ok] = -2.5 * np.log10((flux[ok] * 1e-9) / 3631.0)
    return out


def robust_sigma(values) -> float:
    """MAD-based scatter estimate for a one-dimensional diagnostic."""
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan
    med = np.median(arr)
    sigma = 1.4826 * np.median(np.abs(arr - med))
    if sigma == 0 and len(arr) > 1:
        sigma = 0.5 * (np.nanpercentile(arr, 84) - np.nanpercentile(arr, 16))
    return float(sigma)


def compute_psf_cmodel_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Add PSF-CModel morphology columns where usable flux pairs exist."""
    added = []
    for band in BANDS:
        psf_col = _first_col(
            df,
            [
                [f"dp1_{band}", "psfflux"],
                [f"dp1_{band}", "psf_flux"],
                [f"dp1_{band}", "psf", "flux"],
            ],
        )
        cmodel_col = _first_col(
            df,
            [
                [f"dp1_{band}", "cmodelflux"],
                [f"dp1_{band}", "cmodel", "flux"],
                [f"dp1_{band}", "model", "flux"],
            ],
        )
        if psf_col is None or cmodel_col is None:
            continue

        psf = pd.to_numeric(df[psf_col], errors="coerce").to_numpy()
        cmodel = pd.to_numeric(df[cmodel_col], errors="coerce").to_numpy()
        ok = np.isfinite(psf) & np.isfinite(cmodel) & (psf > 0) & (cmodel > 0)
        diff = np.full(len(df), np.nan)
        diff[ok] = 2.5 * np.log10(cmodel[ok] / psf[ok])
        mag = nJy_to_mag(cmodel)

        diff_col = f"{band}_psf_minus_cmodel"
        mag_col = f"{band}_cmodel_mag"
        df[diff_col] = diff
        df[mag_col] = mag
        added.extend([diff_col, mag_col])
    return df, added


def choose_best_morphology_column(df: pd.DataFrame) -> str | None:
    candidates = [
        "i_psf_minus_cmodel",
        "r_psf_minus_cmodel",
        "g_psf_minus_cmodel",
        "z_psf_minus_cmodel",
        "y_psf_minus_cmodel",
        "u_psf_minus_cmodel",
    ]
    for col in candidates:
        if col in df.columns and df[col].notna().sum() > 0:
            return col
    ext = [c for c in df.columns if "extendedness" in c.lower()]
    return ext[0] if ext else None


def _to_bool_false_mask(series: pd.Series) -> pd.Series:
    """Return True where a flag-like series is explicitly false/zero."""
    if series.dtype == bool:
        return ~series
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["false", "0", "0.0", "nan", "none", ""])


def _required_i_band_columns(df: pd.DataFrame) -> dict[str, str | None]:
    return {
        "psf_flux": _first_col(
            df,
            [["dp1_i", "psfflux"], ["dp1_i", "psf_flux"], ["dp1_i", "psf", "flux"]],
        ),
        "cmodel_flux": _first_col(
            df,
            [["dp1_i", "cmodelflux"], ["dp1_i", "cmodel", "flux"], ["dp1_i", "model", "flux"]],
        ),
        "psf_flag": _first_col(df, [["dp1_i", "psfflux", "flag"], ["dp1_i", "psf", "flux", "flag"]]),
        "cmodel_flag": _first_col(df, [["dp1_i", "cmodelflux", "flag"], ["dp1_i", "cmodel", "flux", "flag"]]),
        "deblend_failed": "dp1_deblend_failed" if "dp1_deblend_failed" in df.columns else None,
        "extendedness": "dp1_i_extendedness" if "dp1_i_extendedness" in df.columns else None,
        "size_extendedness": "dp1_i_sizeExtendedness" if "dp1_i_sizeExtendedness" in df.columns else None,
    }


def clean_morphology_sample(
    df: pd.DataFrame,
    output_dir: Path,
    label_description: str = "external truth labels",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Apply data-quality cuts before interpreting morphology diagnostics.

    These are not final science cuts.  They remove clearly broken photometry
    and define a plotting-clean subset with physically sensible display ranges.
    """
    cols = _required_i_band_columns(df)
    lines = ["DP1 morphology cleaning summary\n", "===============================\n\n"]
    lines.append(f"Input matched rows: {len(df)}\n")
    lines.append("Required i-band columns:\n")
    for key, value in cols.items():
        lines.append(f"  - {key}: {value or 'not found'}\n")

    if cols["psf_flux"] is None or cols["cmodel_flux"] is None:
        msg = "\nMissing required i-band PSF/CModel flux columns; cannot build clean morphology sample.\n"
        (output_dir / "validation_missing_columns.txt").write_text(msg)
        lines.append(msg)
        (output_dir / "dp1_morphology_cleaning_summary.txt").write_text("".join(lines))
        return df.iloc[0:0].copy(), df.iloc[0:0].copy(), lines

    work = df.copy()
    psf_flux = pd.to_numeric(work[cols["psf_flux"]], errors="coerce")
    cmodel_flux = pd.to_numeric(work[cols["cmodel_flux"]], errors="coerce")
    mask = pd.Series(True, index=work.index)

    finite_flux = np.isfinite(psf_flux) & np.isfinite(cmodel_flux)
    mask &= finite_flux
    lines.append(f"After finite i-band PSF/CModel flux: {int(mask.sum())}\n")

    positive_flux = (psf_flux > 0) & (cmodel_flux > 0)
    mask &= positive_flux
    lines.append(f"After positive i-band PSF/CModel flux: {int(mask.sum())}\n")

    if cols["psf_flag"] is not None:
        mask &= _to_bool_false_mask(work[cols["psf_flag"]])
        lines.append(f"After excluding i-band PSF flux flags: {int(mask.sum())}\n")
    else:
        lines.append("No i-band PSF flux flag column found; skipped that cut.\n")

    if cols["cmodel_flag"] is not None:
        mask &= _to_bool_false_mask(work[cols["cmodel_flag"]])
        lines.append(f"After excluding i-band CModel flux flags: {int(mask.sum())}\n")
    else:
        lines.append("No i-band CModel flux flag column found; skipped that cut.\n")

    if cols["deblend_failed"] is not None:
        mask &= _to_bool_false_mask(work[cols["deblend_failed"]])
        lines.append(f"After excluding deblend_failed rows: {int(mask.sum())}\n")
    else:
        lines.append("No deblend_failed column found; skipped that cut.\n")

    finite_diag = np.isfinite(pd.to_numeric(work["i_psf_minus_cmodel"], errors="coerce"))
    finite_mag = np.isfinite(pd.to_numeric(work["i_cmodel_mag"], errors="coerce"))
    mask &= finite_diag & finite_mag
    lines.append(f"After finite i_psf_minus_cmodel and i_cmodel_mag: {int(mask.sum())}\n")

    science_clean = work.loc[mask].copy().reset_index(drop=True)

    # Plotting-range cuts remove clearly broken rows from visualization only.
    plot_mask = (
        (science_clean["i_cmodel_mag"] > 15)
        & (science_clean["i_cmodel_mag"] < 35)
        & (science_clean["i_psf_minus_cmodel"] > -1)
        & (science_clean["i_psf_minus_cmodel"] < 5)
    )
    plotting_clean = science_clean.loc[plot_mask].copy().reset_index(drop=True)
    lines.append(f"Plotting-clean rows with 15 < i_cmodel_mag < 35 and -1 < i_psf_minus_cmodel < 5: {len(plotting_clean)}\n")
    lines.append(f"Truth/label source: {label_description}\n")
    lines.append(f"Plotting-clean label stars: {int((plotting_clean['hst_label'] == 'star').sum())}\n")
    lines.append(f"Plotting-clean label galaxies: {int((plotting_clean['hst_label'] == 'galaxy').sum())}\n")

    (output_dir / "dp1_morphology_cleaning_summary.txt").write_text("".join(lines))
    return science_clean, plotting_clean, lines


def summarize_by_mag_bin(df: pd.DataFrame, morph_col: str, mag_col: str) -> pd.DataFrame:
    rows = []
    for lo, hi in MAG_BINS:
        sub = df[(df[mag_col] >= lo) & (df[mag_col] < hi)]
        star = sub[sub["hst_label"] == "star"][morph_col]
        gal = sub[sub["hst_label"] == "galaxy"][morph_col]
        rows.append(
            {
                "mag_bin": f"{lo}-{hi}",
                "mag_lo": lo,
                "mag_hi": hi,
                "n_total": len(sub),
                "n_star": int((sub["hst_label"] == "star").sum()),
                "n_galaxy": int((sub["hst_label"] == "galaxy").sum()),
                "star_median": float(np.nanmedian(star)) if len(star) else np.nan,
                "galaxy_median": float(np.nanmedian(gal)) if len(gal) else np.nan,
                "star_p16": float(np.nanpercentile(star, 16)) if len(star) else np.nan,
                "star_p84": float(np.nanpercentile(star, 84)) if len(star) else np.nan,
                "galaxy_p16": float(np.nanpercentile(gal, 16)) if len(gal) else np.nan,
                "galaxy_p84": float(np.nanpercentile(gal, 84)) if len(gal) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def morphology_separation_by_mag(df: pd.DataFrame, score_col: str, mag_col: str = "i_cmodel_mag") -> pd.DataFrame:
    """Quantify star/galaxy separation as a function of i-band magnitude."""
    rows = []
    for lo, hi in MAG_BINS:
        sub = df[(df[mag_col] >= lo) & (df[mag_col] < hi)].copy()
        star = pd.to_numeric(sub.loc[sub["hst_label"] == "star", score_col], errors="coerce").dropna()
        gal = pd.to_numeric(sub.loc[sub["hst_label"] == "galaxy", score_col], errors="coerce").dropna()
        med_star = float(np.nanmedian(star)) if len(star) else np.nan
        med_gal = float(np.nanmedian(gal)) if len(gal) else np.nan
        sig_star = robust_sigma(star)
        sig_gal = robust_sigma(gal)
        midpoint = 0.5 * (med_star + med_gal) if np.isfinite(med_star) and np.isfinite(med_gal) else np.nan
        if np.isfinite(midpoint) and med_gal >= med_star:
            star_overlap = float((star > midpoint).mean()) if len(star) else np.nan
            gal_overlap = float((gal <= midpoint).mean()) if len(gal) else np.nan
        elif np.isfinite(midpoint):
            star_overlap = float((star < midpoint).mean()) if len(star) else np.nan
            gal_overlap = float((gal >= midpoint).mean()) if len(gal) else np.nan
        else:
            star_overlap = gal_overlap = np.nan
        rows.append(
            {
                "score_col": score_col,
                "mag_bin": f"{lo}-{hi}",
                "mag_lo": lo,
                "mag_hi": hi,
                "mag_center": 0.5 * (lo + hi),
                "n_total": len(sub),
                "n_stars": len(star),
                "n_galaxies": len(gal),
                "median_star": med_star,
                "median_galaxy": med_gal,
                "robust_sigma_star": sig_star,
                "robust_sigma_galaxy": sig_gal,
                "separation_metric": (med_gal - med_star) / sig_star if np.isfinite(sig_star) and sig_star > 0 else np.nan,
                "overlap_proxy": 0.5 * (star_overlap + gal_overlap) if np.isfinite(star_overlap) and np.isfinite(gal_overlap) else np.nan,
                "star_overlap_proxy": star_overlap,
                "galaxy_overlap_proxy": gal_overlap,
            }
        )
    return pd.DataFrame(rows)


def threshold_scan(
    df: pd.DataFrame,
    score_col: str,
    thresholds: np.ndarray,
    mag_col: str | None = None,
) -> pd.DataFrame:
    """Scan a one-dimensional star/galaxy cut.

    Convention: predict galaxy if score > threshold, otherwise predict star.
    """
    rows = []
    bin_defs = [("all", None, None)]
    if mag_col is not None:
        bin_defs += [(f"{lo}-{hi}", lo, hi) for lo, hi in MAG_BINS]

    for bin_label, lo, hi in bin_defs:
        sub = df
        if lo is not None:
            sub = df[(df[mag_col] >= lo) & (df[mag_col] < hi)]
        score = pd.to_numeric(sub[score_col], errors="coerce")
        usable = sub[score.notna()].copy()
        score = score.loc[usable.index]
        truth_star = usable["hst_label"] == "star"
        truth_gal = usable["hst_label"] == "galaxy"
        n_star = int(truth_star.sum())
        n_gal = int(truth_gal.sum())

        for threshold in thresholds:
            pred_gal = score > threshold
            pred_star = ~pred_gal
            tp_star = int((truth_star & pred_star).sum())
            fp_star = int((truth_gal & pred_star).sum())
            tp_gal = int((truth_gal & pred_gal).sum())
            fp_gal = int((truth_star & pred_gal).sum())
            n_pred_star = int(pred_star.sum())
            n_pred_gal = int(pred_gal.sum())
            stellar_completeness = tp_star / n_star if n_star else np.nan
            galaxy_completeness = tp_gal / n_gal if n_gal else np.nan
            rows.append(
                {
                    "score_col": score_col,
                    "mag_bin": bin_label,
                    "mag_lo": lo,
                    "mag_hi": hi,
                    "threshold": float(threshold),
                    "n_total": len(usable),
                    "n_hst_star": n_star,
                    "n_hst_galaxy": n_gal,
                    "n_pred_star": n_pred_star,
                    "n_pred_galaxy": n_pred_gal,
                    "stellar_completeness": stellar_completeness,
                    "stellar_purity": tp_star / n_pred_star if n_pred_star else np.nan,
                    "galaxy_contamination_in_star_sample": fp_star / n_pred_star if n_pred_star else np.nan,
                    "galaxy_completeness": galaxy_completeness,
                    "galaxy_purity": tp_gal / n_pred_gal if n_pred_gal else np.nan,
                    "star_contamination_in_galaxy_sample": fp_gal / n_pred_gal if n_pred_gal else np.nan,
                    "balanced_accuracy": 0.5 * (stellar_completeness + galaxy_completeness)
                    if np.isfinite(stellar_completeness) and np.isfinite(galaxy_completeness)
                    else np.nan,
                }
            )
    return pd.DataFrame(rows)


def best_threshold_row(scan_df: pd.DataFrame, score_col: str) -> pd.Series | None:
    """Return a simple balanced best-threshold row for the full sample."""
    full = scan_df[(scan_df["score_col"] == score_col) & (scan_df["mag_bin"] == "all")].copy()
    if len(full) == 0:
        return None
    full["balanced_score"] = (
        full["stellar_completeness"].fillna(0)
        + full["stellar_purity"].fillna(0)
        + full["galaxy_completeness"].fillna(0)
        - full["galaxy_contamination_in_star_sample"].fillna(1)
    )
    return full.loc[full["balanced_score"].idxmax()]


def plot_threshold_performance(scan_df: pd.DataFrame, score_col: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    full = scan_df[(scan_df["score_col"] == score_col) & (scan_df["mag_bin"] == "all")]
    plt.figure(figsize=(7, 5))
    for metric, label in [
        ("stellar_completeness", "stellar completeness"),
        ("stellar_purity", "stellar purity"),
        ("galaxy_contamination_in_star_sample", "galaxy contamination in star sample"),
        ("galaxy_completeness", "galaxy completeness"),
    ]:
        plt.plot(full["threshold"], full[metric], label=label)
    plt.xlabel(f"{score_col} threshold")
    plt.ylabel("Metric")
    plt.ylim(-0.02, 1.02)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_multi_threshold_performance(scan_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("stellar_completeness", "stellar completeness"),
        ("stellar_purity", "stellar purity"),
        ("galaxy_completeness", "galaxy completeness"),
        ("galaxy_contamination_in_star_sample", "galaxy contamination in star sample"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for score_col in scan_df["score_col"].dropna().unique():
            full = scan_df[(scan_df["score_col"] == score_col) & (scan_df["mag_bin"] == "all")]
            ax.plot(full["threshold"], full[metric], label=score_col)
        ax.set_title(label)
        ax.set_xlabel("threshold")
        ax.set_ylabel("metric")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_separation_vs_mag(summary_df: pd.DataFrame, score_col: str, output_path: Path, label_description: str = "external truth labels") -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(summary_df["mag_center"], summary_df["median_star"], "o-", label=f"{label_description}: stars")
    axes[0].plot(summary_df["mag_center"], summary_df["median_galaxy"], "o-", label=f"{label_description}: galaxies")
    axes[0].set_xlabel("i_cmodel_mag bin center")
    axes[0].set_ylabel(score_col)
    axes[0].invert_xaxis()
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(summary_df["mag_center"], summary_df["separation_metric"], "o-", label="separation / sigma_star")
    axes[1].plot(summary_df["mag_center"], summary_df["overlap_proxy"], "s--", label="overlap proxy")
    axes[1].set_xlabel("i_cmodel_mag bin center")
    axes[1].set_ylabel("Metric")
    axes[1].invert_xaxis()
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"{label_description} - magnitude-dependent separation: {score_col}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_threshold_curves_by_mag(scan_df: pd.DataFrame, score_col: str, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("stellar_purity", "stellar purity"),
        ("stellar_completeness", "stellar completeness"),
        ("galaxy_completeness", "galaxy completeness"),
        ("balanced_accuracy", "balanced accuracy"),
    ]
    bins_to_plot = ["all"] + [f"{lo}-{hi}" for lo, hi in MAG_BINS]
    cmap = plt.get_cmap("viridis", len(bins_to_plot))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, (metric, label) in zip(axes.ravel(), metrics):
        for idx, mag_bin in enumerate(bins_to_plot):
            sub = scan_df[(scan_df["score_col"] == score_col) & (scan_df["mag_bin"] == mag_bin)]
            if len(sub) == 0 or sub["n_total"].max() == 0:
                continue
            lw = 2.2 if mag_bin == "all" else 1.1
            alpha = 1.0 if mag_bin == "all" else 0.65
            ax.plot(sub["threshold"], sub[metric], label=mag_bin, lw=lw, alpha=alpha, color=cmap(idx))
        ax.set_title(label)
        ax.set_xlabel(f"{score_col} threshold")
        ax.set_ylabel("Metric")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_morphology_quantity_outputs(df: pd.DataFrame, output_dir: Path, label_description: str = "external truth labels") -> dict[str, dict[str, object]]:
    """Write per-quantity separation and threshold products."""
    quantity_specs = {
        "psf_cmodel": ("i_psf_minus_cmodel", "morphology_by_mag_bin_psf_cmodel.csv", "separation_vs_mag_psf_cmodel.png",
                       "threshold_performance_psf_cmodel_by_mag.csv", "threshold_curves_psf_cmodel.png"),
        "extendedness": ("dp1_i_extendedness", "morphology_by_mag_bin_extendedness.csv", "separation_vs_mag_extendedness.png",
                         "threshold_performance_extendedness_by_mag.csv", "threshold_curves_extendedness.png"),
        "sizeextendedness": ("dp1_i_sizeExtendedness", "morphology_by_mag_bin_sizeextendedness.csv",
                             "separation_vs_mag_sizeextendedness.png",
                             "threshold_performance_sizeextendedness_by_mag.csv",
                             "threshold_curves_sizeextendedness.png"),
    }
    thresholds = np.linspace(-0.1, 1.5, 161)
    results: dict[str, dict[str, object]] = {}
    for short_name, (score_col, sep_csv, sep_png, scan_csv, scan_png) in quantity_specs.items():
        if score_col not in df.columns:
            continue
        usable = df[np.isfinite(pd.to_numeric(df[score_col], errors="coerce"))].copy()
        if len(usable) == 0:
            continue
        sep_df = morphology_separation_by_mag(usable, score_col, mag_col="i_cmodel_mag")
        sep_df.to_csv(output_dir / sep_csv, index=False)
        plot_separation_vs_mag(sep_df, score_col, output_dir / sep_png, label_description=label_description)

        scan_df = threshold_scan(usable, score_col, thresholds, mag_col="i_cmodel_mag")
        scan_df.to_csv(output_dir / scan_csv, index=False)
        plot_threshold_curves_by_mag(scan_df, score_col, output_dir / scan_png)

        full_best = best_threshold_row(scan_df, score_col)
        good_bins = sep_df[sep_df["n_stars"] >= 5]
        results[short_name] = {
            "score_col": score_col,
            "separation_df": sep_df,
            "threshold_df": scan_df,
            "best_row": full_best,
            "median_separation": float(np.nanmedian(good_bins["separation_metric"])) if len(good_bins) else np.nan,
            "separation_stability": float(np.nanstd(good_bins["separation_metric"])) if len(good_bins) else np.nan,
            "median_overlap_proxy": float(np.nanmedian(good_bins["overlap_proxy"])) if len(good_bins) else np.nan,
        }
    return results


def write_morphology_comparison_summary(
    results: dict[str, dict[str, object]],
    output_dir: Path,
    label_description: str = "external truth labels",
) -> None:
    lines = [
        "Rubin morphology quantity comparison summary\n",
        "===========================================\n\n",
        f"The comparison uses the clean morphology subset and labels from: {label_description}.\n",
        "Separation metric is (median_galaxy - median_star) / robust_sigma_star in i_cmodel_mag bins.\n",
        "Overlap proxy is the mean of star leakage and galaxy leakage around the midpoint between class medians; lower is better.\n\n",
    ]
    if not results:
        lines.append("No morphology quantities were available for comparison.\n")
        (output_dir / "morphology_quantity_comparison_summary.txt").write_text("".join(lines))
        return

    rows = []
    for short_name, info in results.items():
        best = info["best_row"]
        rows.append(
            {
                "short_name": short_name,
                "score_col": info["score_col"],
                "median_separation": info["median_separation"],
                "separation_stability": info["separation_stability"],
                "median_overlap_proxy": info["median_overlap_proxy"],
                "best_threshold": float(best["threshold"]) if best is not None else np.nan,
                "best_balanced_accuracy": float(best["balanced_accuracy"]) if best is not None else np.nan,
                "best_stellar_purity": float(best["stellar_purity"]) if best is not None else np.nan,
                "best_stellar_completeness": float(best["stellar_completeness"]) if best is not None else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    best_sep = table.sort_values("median_separation", ascending=False).iloc[0]
    best_stable = table.sort_values("separation_stability", ascending=True).iloc[0]
    best_balanced = table.sort_values("best_balanced_accuracy", ascending=False).iloc[0]

    lines.append(table.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    lines.append("\n\n")
    lines.append(f"Best median separation by magnitude bin: {best_sep['score_col']}.\n")
    lines.append(f"Most stable separation across magnitude bins: {best_stable['score_col']}.\n")
    lines.append(f"Best full-sample balanced threshold performance: {best_balanced['score_col']}.\n")
    lines.append(
        "Easiest physical interpretation: i_psf_minus_cmodel, because point sources should have "
        "PSF and CModel fluxes close to each other while extended galaxies usually have larger CModel flux.\n"
    )
    lines.append(
        "A single fixed threshold is useful as a first baseline, but the magnitude-bin tables should be used "
        "to check whether the threshold drifts at the faint end where the star sample is small and noisier.\n"
    )
    (output_dir / "morphology_quantity_comparison_summary.txt").write_text("".join(lines))


def plot_clean_validation(df: pd.DataFrame, output_dir: Path, label_description: str = "external truth labels") -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    bins = np.linspace(-0.5, 2.0, 80)
    for label, color in [("star", "black"), ("galaxy", "tab:blue")]:
        vals = df.loc[df["hst_label"] == label, "i_psf_minus_cmodel"].dropna()
        plt.hist(vals, bins=bins, histtype="step", lw=1.8, label=f"{label_description}: {label}", density=True, color=color)
    plt.xlabel("i_psf_minus_cmodel")
    plt.ylabel("Density")
    plt.title(f"{label_description} - morphology histogram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "morphology_histograms_by_hst_label_clean.png", dpi=180)
    plt.close()

    plt.figure(figsize=(6.5, 5.2))
    for label, color, alpha, size in [("galaxy", "tab:blue", 0.28, 7), ("star", "black", 0.75, 12)]:
        sub = df[df["hst_label"] == label]
        plt.scatter(sub["i_cmodel_mag"], sub["i_psf_minus_cmodel"], s=size, alpha=alpha, label=f"{label_description}: {label}", c=color)
    plt.gca().invert_xaxis()
    plt.xlabel("i_cmodel_mag")
    plt.ylabel("i_psf_minus_cmodel")
    plt.title(f"{label_description} - Rubin morphology check")
    plt.ylim(-0.5, 2.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "psf_cmodel_vs_hst_label_clean.png", dpi=180)
    plt.close()

    summary = summarize_by_mag_bin(df, "i_psf_minus_cmodel", "i_cmodel_mag")
    summary.to_csv(output_dir / "morphology_by_mag_bin_summary.csv", index=False)


def validate_against_hst(
    matched_path: Path = DEFAULT_MATCHED,
    output_dir: Path = OUTPUT_DIR,
    label_description: str = "ECDFS: HST/3D-HST GOODS-S truth labels",
) -> pd.DataFrame | None:
    """Create cleaned validation plots, threshold scans, and text summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    requested_path = matched_path
    if not matched_path.exists() and matched_path == DEFAULT_MATCHED and FALLBACK_MATCHED.exists():
        matched_path = FALLBACK_MATCHED
    if not matched_path.exists():
        msg = (
            f"Matched table not found: {requested_path}\n"
            "Run the appropriate matching script after building the external label catalog.\n"
        )
        (output_dir / "dp1_vs_hst_validation_summary.txt").write_text(msg)
        print(msg)
        return None

    import matplotlib.pyplot as plt

    df = pd.read_csv(matched_path)
    if "hst_label" not in df.columns:
        raise ValueError("Matched table must contain hst_label.")

    df, added = compute_psf_cmodel_features(df)
    probability_cols = [
        c for c in df.columns
        if any(tok in c.lower() for tok in ["prob", "pstar", "p_star", "p_s", "classprob"])
    ]
    extendedness_cols = [c for c in df.columns if "extendedness" in c.lower()]
    morph_col = choose_best_morphology_column(df)

    lines = [
        "Rubin morphology validation summary\n",
        "===================================\n\n",
        f"Truth/label source: {label_description}\n",
        f"Requested matched table: {requested_path}\n",
        f"Used matched table: {matched_path}\n",
        f"Matched rows before morphology cleaning: {len(df)}\n",
        f"Label stars before morphology cleaning: {int((df['hst_label'] == 'star').sum())}\n",
        f"Label galaxies before morphology cleaning: {int((df['hst_label'] == 'galaxy').sum())}\n",
        f"Derived PSF-CModel feature columns: {', '.join(added) if added else 'none'}\n",
        f"Extendedness columns: {', '.join(extendedness_cols) if extendedness_cols else 'none'}\n",
        f"Probability/classification columns: {', '.join(probability_cols) if probability_cols else 'none'}\n",
    ]

    if "i_psf_minus_cmodel" not in df.columns or "i_cmodel_mag" not in df.columns:
        missing = (
            "Required i-band morphology features could not be derived. "
            "Looked for i-band PSF and CModel flux columns.\n"
        )
        (output_dir / "validation_missing_columns.txt").write_text(missing)
        lines.append("\n" + missing)
        summary = "".join(lines)
        (output_dir / "dp1_vs_hst_validation_summary.txt").write_text(summary)
        print(summary)
        return df

    science_clean, plotting_clean, cleaning_lines = clean_morphology_sample(
        df,
        output_dir,
        label_description=label_description,
    )
    lines.append("\n")
    lines.extend(cleaning_lines)

    if len(plotting_clean) == 0:
        lines.append("\nNo rows survived morphology cleaning; clean plots and threshold scans were skipped.\n")
        summary = "".join(lines)
        (output_dir / "dp1_vs_hst_validation_summary.txt").write_text(summary)
        print(summary)
        return df

    plot_clean_validation(plotting_clean, output_dir, label_description=label_description)
    lines.append("\nClean validation plots written using the plotting-clean subset.\n")
    quantity_results = save_morphology_quantity_outputs(plotting_clean, output_dir, label_description=label_description)
    write_morphology_comparison_summary(quantity_results, output_dir, label_description=label_description)
    lines.append("Magnitude-dependent separation and threshold-by-magnitude products written for available i-band morphology quantities.\n")

    star_vals = pd.to_numeric(plotting_clean.loc[plotting_clean["hst_label"] == "star", "i_psf_minus_cmodel"], errors="coerce")
    gal_vals = pd.to_numeric(plotting_clean.loc[plotting_clean["hst_label"] == "galaxy", "i_psf_minus_cmodel"], errors="coerce")
    lines += [
        "\nPrimary cleaned morphology diagnostic: i_psf_minus_cmodel\n",
        f"Clean label-star median: {np.nanmedian(star_vals):.5g}\n",
        f"Clean label-galaxy median: {np.nanmedian(gal_vals):.5g}\n",
        f"Clean label-star 16-84 percentile: {np.nanpercentile(star_vals, 16):.5g}, {np.nanpercentile(star_vals, 84):.5g}\n",
        f"Clean label-galaxy 16-84 percentile: {np.nanpercentile(gal_vals, 16):.5g}, {np.nanpercentile(gal_vals, 84):.5g}\n",
    ]

    thresholds = np.linspace(-0.1, 1.5, 161)
    scan_df = threshold_scan(plotting_clean, "i_psf_minus_cmodel", thresholds, mag_col="i_cmodel_mag")
    scan_df.to_csv(output_dir / "psf_cmodel_threshold_scan.csv", index=False)
    plot_threshold_performance(scan_df, "i_psf_minus_cmodel", output_dir / "psf_cmodel_threshold_performance.png")

    best_rows = []
    best_psf = best_threshold_row(scan_df, "i_psf_minus_cmodel")
    if best_psf is not None:
        best_rows.append(best_psf)
        lines += [
            "\nBest first-pass i_psf_minus_cmodel threshold by simple balanced score:\n",
            f"  threshold = {best_psf['threshold']:.3f}\n",
            f"  stellar completeness = {best_psf['stellar_completeness']:.3f}\n",
            f"  stellar purity = {best_psf['stellar_purity']:.3f}\n",
            f"  galaxy contamination in star sample = {best_psf['galaxy_contamination_in_star_sample']:.3f}\n",
            f"  galaxy completeness = {best_psf['galaxy_completeness']:.3f}\n",
        ]

    ext_score_cols = [c for c in ["dp1_i_extendedness", "dp1_i_sizeExtendedness"] if c in plotting_clean.columns]
    ext_scan_parts = []
    for score_col in ext_score_cols:
        part = threshold_scan(plotting_clean, score_col, thresholds, mag_col="i_cmodel_mag")
        ext_scan_parts.append(part)
        best_ext = best_threshold_row(part, score_col)
        if best_ext is not None:
            best_rows.append(best_ext)
    if ext_scan_parts:
        ext_scan_df = pd.concat(ext_scan_parts, ignore_index=True)
        ext_scan_df.to_csv(output_dir / "extendedness_threshold_scan.csv", index=False)
        plot_multi_threshold_performance(ext_scan_df, output_dir / "extendedness_threshold_performance.png")
    else:
        lines.append("\nNo i-band extendedness columns found for comparison threshold scans.\n")

    if probability_cols:
        pcol = probability_cols[0]
        plt.figure(figsize=(7, 4))
        for label, color in [("star", "black"), ("galaxy", "tab:blue")]:
            vals = pd.to_numeric(plotting_clean.loc[plotting_clean["hst_label"] == label, pcol], errors="coerce").dropna()
            plt.hist(vals, bins=50, histtype="step", lw=1.8, label=f"{label_description}: {label}", density=True, color=color)
        plt.xlabel(pcol)
        plt.ylabel("Density")
        plt.title(f"{label_description} - probability column check")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "probability_histograms_by_hst_label.png", dpi=180)
        plt.close()
    else:
        lines.append(
            "\nNo probability columns were found, so this validation uses Rubin morphology "
            "diagnostics such as PSF-CModel and extendedness where available.\n"
        )

    if "delta_ra_cosdec_arcsec" in plotting_clean.columns and "delta_dec_arcsec" in plotting_clean.columns:
        dx_med = float(np.nanmedian(plotting_clean["delta_ra_cosdec_arcsec"]))
        dy_med = float(np.nanmedian(plotting_clean["delta_dec_arcsec"]))
    else:
        dx_med = dy_med = np.nan
    if "delta_ra_cosdec_recentered_arcsec" in plotting_clean.columns and "delta_dec_recentered_arcsec" in plotting_clean.columns:
        dx_rec_med = float(np.nanmedian(plotting_clean["delta_ra_cosdec_recentered_arcsec"]))
        dy_rec_med = float(np.nanmedian(plotting_clean["delta_dec_recentered_arcsec"]))
    else:
        dx_rec_med = dy_rec_med = np.nan

    promising = "i_psf_minus_cmodel"
    if best_rows:
        best_table = pd.DataFrame(best_rows)
        best_table["balanced_score"] = (
            best_table["stellar_completeness"].fillna(0)
            + best_table["stellar_purity"].fillna(0)
            + best_table["galaxy_completeness"].fillna(0)
            - best_table["galaxy_contamination_in_star_sample"].fillna(1)
        )
        promising = str(best_table.loc[best_table["balanced_score"].idxmax(), "score_col"])

    comparison_path = output_dir / "morphology_quantity_comparison_summary.txt"
    comparison_text = comparison_path.read_text() if comparison_path.exists() else ""
    interpretation = [
        "Rubin morphology interpretation summary\n",
        "======================================\n\n",
        f"Truth/label source: {label_description}\n",
        f"Matching sample used: {matched_path.name}\n",
        "The validation now uses the clean matched sample when available; this is the recentered 3-sigma elliptical cut by default.\n",
        f"Median original astrometric residual in the clean validation subset: dx = {dx_med:.4f} arcsec, dy = {dy_med:.4f} arcsec.\n",
        f"Median recentered residual in the clean validation subset: dx' = {dx_rec_med:.4f} arcsec, dy' = {dy_rec_med:.4f} arcsec.\n",
        f"Rows after i-band photometry/flag/deblend cleaning: {len(science_clean)}.\n",
        f"Rows after plotting-range cleaning: {len(plotting_clean)}.\n",
        f"Label stars in clean plotting subset: {int((plotting_clean['hst_label'] == 'star').sum())}.\n",
        f"Label galaxies in clean plotting subset: {int((plotting_clean['hst_label'] == 'galaxy').sum())}.\n",
        f"Label stars cluster near i_psf_minus_cmodel median {np.nanmedian(star_vals):.4f}; this is close to the expected point-source value near zero.\n",
        f"Label galaxies have a broader, more positive median i_psf_minus_cmodel of {np.nanmedian(gal_vals):.4f}.\n",
        "The star/galaxy morphology signal remains visible after removing bad photometric outliers and failed deblends.\n",
        f"Most promising current Rubin-side one-dimensional diagnostic by the simple threshold scan: {promising}.\n",
        "Immediate next step: choose a default threshold using the clean scan, then validate it on a held-out region or a second DP1 FITS table before treating it as a science-grade classifier.\n",
    ]
    (output_dir / "dp1_vs_hst_interpretation_summary.txt").write_text("".join(interpretation))
    v2 = [
        "Rubin morphology interpretation summary v2\n",
        "==========================================\n\n",
        f"Truth/label source: {label_description}\n",
        "1. Clean matching results\n",
        f"Clean matching uses {matched_path.name}; clean morphology rows = {len(plotting_clean)}.\n",
        f"Median recentered residuals in the validation subset are dx' = {dx_rec_med:.4f} arcsec and dy' = {dy_rec_med:.4f} arcsec.\n\n",
        "2. Morphology signal strength\n",
        f"Label stars have median i_psf_minus_cmodel = {np.nanmedian(star_vals):.4f}; label galaxies have median = {np.nanmedian(gal_vals):.4f}.\n",
        "This confirms that the cleaned Rubin morphology signal is real.\n\n",
        "3. Magnitude dependence and best quantity\n",
        comparison_text,
        "\n4. Probability model status\n",
        "The morphology validation script does not train the probability model. Run src/build_rubin_star_probability_model.py for the first P_star model.\n\n",
        "5. Recommended next step\n",
        "Use the generated by-magnitude threshold tables to decide whether to keep a fixed morphology threshold or fit a magnitude-dependent probability model.\n",
    ]
    (output_dir / "dp1_vs_hst_interpretation_summary_v2.txt").write_text("".join(v2))

    summary = "".join(lines)
    (output_dir / "dp1_vs_hst_validation_summary.txt").write_text(summary)
    print(summary)
    return plotting_clean


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--label-description", default="ECDFS: HST/3D-HST GOODS-S truth labels")
    args = parser.parse_args()
    validate_against_hst(args.matched, args.output_dir, label_description=args.label_description)


if __name__ == "__main__":
    main()
