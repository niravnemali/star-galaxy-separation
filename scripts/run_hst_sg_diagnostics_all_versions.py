#!/usr/bin/env python3
"""Generate external-label DP2 star/galaxy diagnostics for all pS versions.

This script extends the existing external-labeled pS-vs-magnitude, pS histogram,
and extendedness histogram diagnostics into magnitude-binned four-panel plots.
It also adds CMD/CCD confusion plots, magnitude-binned ROC curves, and a
Slater-style within-version probabilistic pS combination prototype.

Conventions used here match the current project diagnostics:
- pS is treated as P(star); predicted star if pS >= 0.5.
- Rubin extendedness is treated as 0=star-like, 1=galaxy-like; predicted star
  if extendedness < 0.5, and ROC uses 1 - extendedness as the star-like score.
- External labels are validation labels, not perfect truth.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp2_external_validation import BANDS, DEFAULT_COLOR_RANGE, read_table, write_table


MAG_BINS = [
    ("mag < 22", -np.inf, 22.0),
    ("22 <= mag < 23", 22.0, 23.0),
    ("23 <= mag < 24", 23.0, 24.0),
    ("24 <= mag < 25", 24.0, 25.0),
]
PS_THRESHOLD = 0.5
EXT_THRESHOLD = 0.5
STAR_LABEL = 1
GALAXY_LABEL = 0


@dataclass
class PlotRecord:
    version: str
    filter: str
    classifier: str
    plot_type: str
    output_path: str
    n_objects: int
    n_external_stars: int
    n_external_galaxies: int
    auc: float | None = None
    skipped_reason: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched", type=Path, default=REPO_ROOT / "outputs" / "dp2_ecdfs_hst_matched.parquet")
    parser.add_argument("--ps-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--plot-dir", type=Path, default=REPO_ROOT / "plots")
    parser.add_argument("--summary", type=Path, default=REPO_ROOT / "plots" / "hst_sg_diagnostics_summary_all_versions_filters.csv")
    parser.add_argument("--combined-output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--field", default="ECDFS", help="Field label for plot titles and filenames, e.g. ECDFS or COSMOS.")
    parser.add_argument("--external-label-source", default="HST", help="External label source for plot titles and legends.")
    parser.add_argument("--ps-prefix", default=None, help="pS table prefix. Default: outputs/dp2_<field_lower>_ps_<version>.parquet")
    parser.add_argument("--versions", default="v1,v2,v3,v4,v5,v6")
    parser.add_argument("--filters", default=None, help="Comma-separated filters. Default: infer from pS tables.")
    parser.add_argument("--prior", choices=["empirical", "flat"], default="empirical")
    parser.add_argument("--min-class-count", type=int, default=5)
    parser.add_argument("--hist-bins", type=int, default=40)
    parser.add_argument("--skip-combined", action="store_true", help="Skip the Slater-style combined_pS prototype outputs.")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def truth_binary(df: pd.DataFrame) -> pd.Series:
    if "truth_binary" in df.columns:
        return pd.to_numeric(df["truth_binary"], errors="coerce")
    if "truth_label" in df.columns:
        return df["truth_label"].astype(str).str.lower().map({"star": 1, "galaxy": 0})
    raise ValueError("Matched table needs truth_binary or truth_label.")


def finite_series(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df.columns else pd.Series(np.nan, index=df.index)


def mag_col_for_filter(band: str) -> str:
    return f"dp2_cmodel_mag_{band}"


def ext_col_for_filter(band: str) -> str:
    return f"dp2_extendedness_{band}"


def ps_col_for_filter(band: str) -> str:
    return f"pS_{band}"


def classifier_label(band: str, classifier: str) -> str:
    if band == "combined":
        return classifier
    return f"{band}-band {classifier}"


def safe_tag(text: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in text.upper()).strip("_")


def plot_tag(args: argparse.Namespace) -> str:
    return f"{safe_tag(args.field)}_{safe_tag(args.external_label_source)}"


def ps_table_path(args: argparse.Namespace, version: str) -> Path:
    if args.ps_prefix:
        return args.ps_dir / f"{args.ps_prefix}{version}.parquet"
    return args.ps_dir / f"dp2_{args.field.lower()}_ps_{version}.parquet"


def available_filters(ps: pd.DataFrame, matched: pd.DataFrame, requested: str | None = None) -> list[str]:
    if requested:
        candidates = [x.strip() for x in requested.split(",") if x.strip()]
    else:
        candidates = [b for b in BANDS if ps_col_for_filter(b) in ps.columns]
    return [b for b in candidates if mag_col_for_filter(b) in matched.columns]


def load_version_sample(matched: pd.DataFrame, ps_path: Path) -> tuple[pd.DataFrame | None, str]:
    if not ps_path.exists():
        return None, f"missing pS table: {ps_path}"
    ps = read_table(ps_path)
    if "object_id" not in ps.columns:
        return None, f"pS table lacks object_id: {ps_path}"
    dup = int(ps["object_id"].duplicated().sum())
    if dup:
        print(f"Warning: {ps_path.name} has {dup:,} duplicate object_id rows; keeping first.")
        ps = ps.drop_duplicates(subset=["object_id"], keep="first")
    keep = ["object_id"] + [c for c in ps.columns if c.startswith("pS_") and c[-1:] in BANDS]
    sample = matched.merge(ps[keep], on="object_id", how="left", validate="one_to_one")
    return sample, ""


def valid_labeled_mask(df: pd.DataFrame) -> pd.Series:
    return truth_binary(df).isin([GALAXY_LABEL, STAR_LABEL])


def count_labels(df: pd.DataFrame, mask: pd.Series | np.ndarray | None = None) -> tuple[int, int, int]:
    if mask is None:
        mask = pd.Series(True, index=df.index)
    truth = truth_binary(df)
    use = mask & truth.isin([GALAXY_LABEL, STAR_LABEL])
    n_star = int((truth[use] == STAR_LABEL).sum())
    n_gal = int((truth[use] == GALAXY_LABEL).sum())
    return int(use.sum()), n_star, n_gal


def save_fig(fig: plt.Figure, path: Path) -> str:
    ensure_dir(path.parent)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def add_record(records: list[PlotRecord], version: str, band: str, classifier: str, plot_type: str, path: Path | str, df: pd.DataFrame, mask=None, auc=None, skipped="") -> None:
    n, ns, ng = count_labels(df, mask)
    records.append(
        PlotRecord(
            version=version,
            filter=band,
            classifier=classifier,
            plot_type=plot_type,
            output_path=str(path),
            n_objects=n,
            n_external_stars=ns,
            n_external_galaxies=ng,
            auc=auc,
            skipped_reason=skipped,
        )
    )


def panel_masks_by_mag(df: pd.DataFrame, mag_col: str, base_mask: pd.Series | None = None) -> list[tuple[str, pd.Series]]:
    mag = finite_series(df, mag_col)
    if base_mask is None:
        base_mask = pd.Series(True, index=df.index)
    panels = []
    for label, lo, hi in MAG_BINS:
        m = base_mask & np.isfinite(mag) & (mag < hi)
        if np.isfinite(lo):
            m &= mag >= lo
        panels.append((label, m))
    return panels


def mag_bin_file_tag(label: str) -> str:
    return (
        label.replace(" <= mag < ", "_")
        .replace("mag < ", "mag_lt")
        .replace(" ", "")
        .replace("<", "lt")
        .replace("=", "")
    )


def plot_ps_vs_mag_4panel(df: pd.DataFrame, version: str, band: str, out: Path, field: str, source: str) -> Path | None:
    score_col = ps_col_for_filter(band)
    mag_col = mag_col_for_filter(band)
    if score_col not in df.columns or mag_col not in df.columns:
        return None
    truth = truth_binary(df)
    score = finite_series(df, score_col)
    mag = finite_series(df, mag_col)
    base = valid_labeled_mask(df) & np.isfinite(score) & np.isfinite(mag)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), sharey=True)
    for ax, (label, mask) in zip(axes, panel_masks_by_mag(df, mag_col, base)):
        star = mask & (truth == STAR_LABEL)
        gal = mask & (truth == GALAXY_LABEL)
        ax.scatter(mag[gal], score[gal], s=6, alpha=0.28, color="tab:blue", label=f"{source} galaxy")
        ax.scatter(mag[star], score[star], s=10, alpha=0.75, color="black", label=f"{source} star")
        ax.axhline(PS_THRESHOLD, color="red", lw=2.8, ls="--", label="pS=0.5")
        ax.set_title(f"{label}\nN={int(mask.sum()):,}")
        ax.set_xlabel(f"{band} CModel AB")
        ax.grid(True, alpha=0.2)
        if label == "mag < 22":
            ax.set_xlim(15, 22)
        else:
            ax.set_xlim(max(15, mag[mask].min()) if mask.any() else 22, min(25, mag[mask].max()) if mask.any() else 25)
    axes[0].set_ylabel(f"pS_{band} (P star)")
    axes[0].set_ylim(-0.03, 1.03)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{version} {field} pS vs magnitude, {band}-band classifier. External labels: {source}", y=1.06)
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94])
    return Path(save_fig(fig, out))


def plot_score_hist_4panel(df: pd.DataFrame, version: str, band: str, out: Path, kind: str, field: str, source: str) -> Path | None:
    if kind == "pS":
        score_col = ps_col_for_filter(band)
        xlabel = f"pS_{band} (P star)"
        threshold = PS_THRESHOLD
        title_name = "pS"
    else:
        score_col = ext_col_for_filter(band)
        xlabel = f"{band} extendedness (0=star-like, 1=galaxy-like)"
        threshold = EXT_THRESHOLD
        title_name = "extendedness"
    mag_col = mag_col_for_filter(band)
    if score_col not in df.columns or mag_col not in df.columns:
        return None
    truth = truth_binary(df)
    score = finite_series(df, score_col)
    mag = finite_series(df, mag_col)
    base = valid_labeled_mask(df) & np.isfinite(score) & np.isfinite(mag)
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.0), sharey=True)
    bins = np.linspace(0, 1, 41)
    for ax, (label, mask) in zip(axes, panel_masks_by_mag(df, mag_col, base)):
        star = mask & (truth == STAR_LABEL)
        gal = mask & (truth == GALAXY_LABEL)
        ax.hist(
            score[gal],
            bins=bins,
            histtype="step",
            lw=3.0,
            ls="--",
            color="tab:blue",
            density=True,
            label=f"{source} galaxy N={int(gal.sum()):,}",
        )
        ax.hist(
            score[star],
            bins=bins,
            histtype="step",
            lw=3.0,
            ls="--",
            color="red",
            density=True,
            label=f"{source} star N={int(star.sum()):,}",
        )
        ax.axvline(threshold, color="0.45", lw=1.2, ls=":")
        ax.set_title(f"{label}\nN_star={int(star.sum()):,}, N_gal={int(gal.sum()):,}")
        ax.set_xlabel(xlabel)
        ax.set_xlim(-0.03, 1.03)
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Normalized density")
    fig.suptitle(f"{version} {field} {title_name} histograms, {band}-band. External labels: {source}", y=1.05)
    return Path(save_fig(fig, out))


def confusion_masks(df: pd.DataFrame, predicted_star: pd.Series) -> list[tuple[str, pd.Series, str]]:
    truth = truth_binary(df)
    pred = predicted_star.astype("boolean")
    valid = truth.isin([GALAXY_LABEL, STAR_LABEL]) & pred.notna()
    return [
        ("Correct stars", valid & (truth == STAR_LABEL) & pred, "black"),
        ("Misclassified stars", valid & (truth == STAR_LABEL) & ~pred, "tab:orange"),
        ("Correct galaxies", valid & (truth == GALAXY_LABEL) & ~pred, "tab:blue"),
        ("Misclassified galaxies", valid & (truth == GALAXY_LABEL) & pred, "tab:red"),
    ]


def plot_confusion_4panel(
    df: pd.DataFrame,
    version: str,
    band: str,
    classifier: str,
    predicted_star: pd.Series,
    out: Path,
    diagram: str,
    field: str,
    source: str,
    selection_mask: pd.Series | None = None,
    selection_label: str = "",
) -> Path | None:
    if diagram == "cmd":
        x_col, y_col = "dp2_cmd_g_minus_i", "dp2_cmd_i_mag"
        xlabel, ylabel = "g - i (CModel AB)", "i (CModel AB)"
        xlim, ylim = DEFAULT_COLOR_RANGE, (28, 15)
    else:
        x_col, y_col = "dp2_gr", "dp2_ri"
        xlabel, ylabel = "g - r (CModel AB)", "r - i (CModel AB)"
        xlim, ylim = DEFAULT_COLOR_RANGE, DEFAULT_COLOR_RANGE
    if x_col not in df.columns or y_col not in df.columns:
        return None
    x = finite_series(df, x_col)
    y = finite_series(df, y_col)
    base = valid_labeled_mask(df) & predicted_star.notna() & np.isfinite(x) & np.isfinite(y) & x.between(*DEFAULT_COLOR_RANGE)
    if selection_mask is not None:
        base &= selection_mask.reindex(df.index, fill_value=False)
    if diagram == "cmd":
        base &= y.between(15, 28)
    else:
        base &= y.between(*DEFAULT_COLOR_RANGE)
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.5), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (name, mask, color) in zip(axes, confusion_masks(df, predicted_star)):
        bg = base
        if int(bg.sum()) > 20:
            ax.hexbin(x[bg], y[bg], gridsize=60, bins="log", mincnt=1, cmap="Greys", alpha=0.22)
        m = mask & base
        if int(m.sum()) > 600:
            ax.hexbin(x[m], y[m], gridsize=55, bins="log", mincnt=1, cmap="Reds" if color == "tab:red" else "Blues")
        else:
            ax.scatter(x[m], y[m], s=9, alpha=0.7, color=color)
        ax.set_title(f"{name}\nN={int(m.sum()):,}")
        ax.grid(True, alpha=0.18)
    for ax in axes[2:]:
        ax.set_xlabel(xlabel)
    for ax in axes[::2]:
        ax.set_ylabel(ylabel)
    axes[0].set_xlim(*xlim)
    axes[0].set_ylim(*ylim)
    threshold_text = "pS >= 0.5 means star" if classifier == "pS" else "extendedness < 0.5 means star"
    title = f"{version} {field} {diagram.upper()} confusion, {band}-band {classifier}. External labels: {source}; {threshold_text}"
    if selection_label:
        title += f"; {selection_label}"
    fig.suptitle(title)
    return Path(save_fig(fig, out))


def plot_ccd_mag_bins_4panel(df: pd.DataFrame, version: str, out: Path, field: str, source: str) -> Path | None:
    """Color-color diagram split by i-band magnitude bins and external labels."""

    x_col, y_col, mag_col = "dp2_gr", "dp2_ri", "dp2_cmd_i_mag"
    if x_col not in df.columns or y_col not in df.columns or mag_col not in df.columns:
        return None
    truth = truth_binary(df)
    x = finite_series(df, x_col)
    y = finite_series(df, y_col)
    mag = finite_series(df, mag_col)
    base = (
        valid_labeled_mask(df)
        & np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(mag)
        & x.between(*DEFAULT_COLOR_RANGE)
        & y.between(*DEFAULT_COLOR_RANGE)
        & mag.between(15, 28)
    )
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.3), sharex=True, sharey=True)
    for ax, (label, mask) in zip(axes, panel_masks_by_mag(df, mag_col, base)):
        star = mask & (truth == STAR_LABEL)
        gal = mask & (truth == GALAXY_LABEL)
        if int(gal.sum()) > 300:
            ax.hexbin(x[gal], y[gal], gridsize=55, bins="log", mincnt=1, cmap="Blues", alpha=0.72)
        else:
            ax.scatter(x[gal], y[gal], s=8, alpha=0.35, color="tab:blue", label=f"{source} galaxy")
        if int(star.sum()) > 300:
            ax.hexbin(x[star], y[star], gridsize=45, bins="log", mincnt=1, cmap="Reds", alpha=0.72)
        else:
            ax.scatter(x[star], y[star], s=12, alpha=0.75, color="red", label=f"{source} star")
        ax.set_title(f"{label}\nN_star={int(star.sum()):,}, N_gal={int(gal.sum()):,}")
        ax.set_xlabel("g - r (CModel AB)")
        ax.grid(True, alpha=0.18)
    axes[0].set_ylabel("r - i (CModel AB)")
    axes[0].set_xlim(*DEFAULT_COLOR_RANGE)
    axes[0].set_ylim(*DEFAULT_COLOR_RANGE)
    fig.suptitle(f"{version} {field} color-color diagram by i magnitude bin. External labels: {source}", y=1.08)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    return Path(save_fig(fig, out))


def plot_cmd_mag_bins_4panel(df: pd.DataFrame, version: str, out: Path, field: str, source: str) -> Path | None:
    """Color-magnitude diagram split by i-band magnitude bins and external labels."""

    x_col, mag_col = "dp2_cmd_g_minus_i", "dp2_cmd_i_mag"
    if x_col not in df.columns or mag_col not in df.columns:
        return None
    truth = truth_binary(df)
    x = finite_series(df, x_col)
    mag = finite_series(df, mag_col)
    base = (
        valid_labeled_mask(df)
        & np.isfinite(x)
        & np.isfinite(mag)
        & x.between(*DEFAULT_COLOR_RANGE)
        & mag.between(15, 28)
    )
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.3), sharex=True)
    for ax, (label, mask) in zip(axes, panel_masks_by_mag(df, mag_col, base)):
        star = mask & (truth == STAR_LABEL)
        gal = mask & (truth == GALAXY_LABEL)
        if int(gal.sum()) > 300:
            ax.hexbin(x[gal], mag[gal], gridsize=55, bins="log", mincnt=1, cmap="Blues", alpha=0.72)
        else:
            ax.scatter(x[gal], mag[gal], s=8, alpha=0.35, color="tab:blue", label=f"{source} galaxy")
        if int(star.sum()) > 300:
            ax.hexbin(x[star], mag[star], gridsize=45, bins="log", mincnt=1, cmap="Reds", alpha=0.72)
        else:
            ax.scatter(x[star], mag[star], s=12, alpha=0.75, color="red", label=f"{source} star")
        ax.set_title(f"{label}\nN_star={int(star.sum()):,}, N_gal={int(gal.sum()):,}")
        ax.set_xlabel("g - i (CModel AB)")
        ax.set_xlim(*DEFAULT_COLOR_RANGE)
        if label == "mag < 22":
            ax.set_ylim(22, 15)
        else:
            _, lo, hi = next(item for item in MAG_BINS if item[0] == label)
            ax.set_ylim(hi, lo)
        ax.grid(True, alpha=0.18)
    axes[0].set_ylabel("i (CModel AB)")
    fig.suptitle(f"{version} {field} color-magnitude diagram by i magnitude bin. External labels: {source}", y=1.08)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.92])
    return Path(save_fig(fig, out))


def roc_curve_and_auc(y_true: pd.Series, score: pd.Series) -> tuple[np.ndarray, np.ndarray, float] | None:
    y = pd.to_numeric(y_true, errors="coerce")
    s = pd.to_numeric(score, errors="coerce")
    valid = y.isin([0, 1]) & np.isfinite(s)
    y = y[valid].astype(int).to_numpy()
    s = s[valid].astype(float).to_numpy()
    if len(y) == 0 or np.sum(y == 1) == 0 or np.sum(y == 0) == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    s = s[order]
    pos = np.sum(y == 1)
    neg = np.sum(y == 0)
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y == 1)[distinct]
    fps = np.cumsum(y == 0)[distinct]
    tpr = np.r_[0.0, tps / pos, 1.0]
    fpr = np.r_[0.0, fps / neg, 1.0]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def plot_roc_4panel(df: pd.DataFrame, version: str, band: str, classifier: str, score: pd.Series, mag_col: str, out: Path, min_class_count: int, field: str, source: str) -> tuple[Path | None, dict[str, float]]:
    truth = truth_binary(df)
    mag = finite_series(df, mag_col)
    base = valid_labeled_mask(df) & np.isfinite(score) & np.isfinite(mag)
    aucs = {}
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.4), sharex=True, sharey=True)
    for ax, (label, mask) in zip(axes, panel_masks_by_mag(df, mag_col, base)):
        ns = int(((truth == STAR_LABEL) & mask).sum())
        ng = int(((truth == GALAXY_LABEL) & mask).sum())
        if ns < min_class_count or ng < min_class_count:
            ax.text(0.5, 0.5, f"insufficient data\nN_star={ns}, N_gal={ng}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            aucs[label] = np.nan
        else:
            roc = roc_curve_and_auc(truth[mask], score[mask])
            if roc is None:
                ax.text(0.5, 0.5, "insufficient data", ha="center", va="center", transform=ax.transAxes)
                aucs[label] = np.nan
            else:
                fpr, tpr, auc = roc
                aucs[label] = auc
                ax.plot(fpr, tpr, lw=2.0, label=f"AUC={auc:.3f}")
                ax.legend(fontsize=8)
            ax.set_title(f"{label}\nN={int(mask.sum()):,}, stars={ns:,}, gal={ng:,}")
        ax.plot([0, 1], [0, 1], color="0.55", ls="--", lw=1.0)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("False positive rate (contamination)")
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("True positive rate (completeness)")
    fig.suptitle(
        f"{version} {field} ROC by magnitude, {classifier_label(band, classifier)}. "
        f"Positive class: {source} star; External labels: {source}",
        y=1.08,
        fontsize=12,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.9])
    return Path(save_fig(fig, out)), aucs


def plot_roc_singlepanel(df: pd.DataFrame, version: str, band: str, classifier: str, score: pd.Series, mag_col: str, out: Path, min_class_count: int, field: str, source: str) -> tuple[Path | None, dict[str, float]]:
    truth = truth_binary(df)
    mag = finite_series(df, mag_col)
    base = valid_labeled_mask(df) & np.isfinite(score) & np.isfinite(mag)
    aucs = {}
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for label, mask in panel_masks_by_mag(df, mag_col, base):
        ns = int(((truth == STAR_LABEL) & mask).sum())
        ng = int(((truth == GALAXY_LABEL) & mask).sum())
        if ns < min_class_count or ng < min_class_count:
            aucs[label] = np.nan
            continue
        roc = roc_curve_and_auc(truth[mask], score[mask])
        if roc is None:
            aucs[label] = np.nan
            continue
        fpr, tpr, auc = roc
        aucs[label] = auc
        ax.plot(fpr, tpr, lw=2.0, label=f"{label}: AUC={auc:.3f}, N={int(mask.sum()):,}")
    ax.plot([0, 1], [0, 1], color="0.55", ls="--", lw=1.0)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel("False positive rate (contamination)")
    ax.set_ylabel("True positive rate (completeness)")
    ax.set_title(f"{version} {field} ROC, {classifier_label(band, classifier)}. Positive class: {source} star; External labels: {source}")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    return Path(save_fig(fig, out)), aucs


def histogram_lr(star_scores: np.ndarray, gal_scores: np.ndarray, bins: np.ndarray, floor: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    star_hist, _ = np.histogram(star_scores, bins=bins)
    gal_hist, _ = np.histogram(gal_scores, bins=bins)
    star_prob = (star_hist.astype(float) + floor) / (star_hist.sum() + floor * len(star_hist))
    gal_prob = (gal_hist.astype(float) + floor) / (gal_hist.sum() + floor * len(gal_hist))
    return np.log(star_prob) - np.log(gal_prob), bins


def assign_score_bins(scores: pd.Series, bins: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(bins, scores.to_numpy(dtype=float), side="right") - 1
    return np.clip(idx, 0, len(bins) - 2)


def combine_ps_slater_style(
    df: pd.DataFrame,
    filters: list[str],
    version: str,
    prior: str = "empirical",
    hist_bins: int = 40,
    min_class_count: int = 5,
) -> pd.DataFrame:
    """Prototype within-version pS combination using summed log likelihood ratios."""

    out = df[["object_id", "truth_label", "truth_binary"]].copy()
    truth = truth_binary(df)
    score_bins = np.linspace(0, 1, hist_bins + 1)
    total_loglr = pd.Series(0.0, index=df.index)
    n_used = pd.Series(0, index=df.index, dtype=int)
    base_labeled = truth.isin([GALAXY_LABEL, STAR_LABEL])
    prior_log_odds = pd.Series(0.0, index=df.index)

    common_mag = finite_series(df, "dp2_cmd_i_mag") if "dp2_cmd_i_mag" in df.columns else finite_series(df, mag_col_for_filter("i"))
    if prior == "empirical":
        for _, lo, hi in MAG_BINS:
            m = base_labeled & np.isfinite(common_mag) & (common_mag < hi)
            if np.isfinite(lo):
                m &= common_mag >= lo
            ns = int(((truth == STAR_LABEL) & m).sum())
            ng = int(((truth == GALAXY_LABEL) & m).sum())
            logodds = np.log((ns + 1.0) / (ng + 1.0))
            prior_log_odds.loc[np.isfinite(common_mag) & (common_mag < hi) & ((common_mag >= lo) if np.isfinite(lo) else True)] = logodds

    for band in filters:
        score_col = ps_col_for_filter(band)
        mag_col = mag_col_for_filter(band)
        if score_col not in df.columns or mag_col not in df.columns:
            continue
        score = finite_series(df, score_col).clip(0, 1)
        mag = finite_series(df, mag_col)
        for _, lo, hi in MAG_BINS:
            in_bin = np.isfinite(score) & np.isfinite(mag) & (mag < hi)
            if np.isfinite(lo):
                in_bin &= mag >= lo
            train = in_bin & base_labeled
            star_train = train & (truth == STAR_LABEL)
            gal_train = train & (truth == GALAXY_LABEL)
            if int(star_train.sum()) < min_class_count or int(gal_train.sum()) < min_class_count:
                continue
            log_lr_bins, bins = histogram_lr(score[star_train].to_numpy(), score[gal_train].to_numpy(), score_bins)
            target = in_bin & np.isfinite(score)
            idx = assign_score_bins(score[target], bins)
            total_loglr.loc[target] += log_lr_bins[idx]
            n_used.loc[target] += 1

    log_odds = prior_log_odds + total_loglr
    log_odds = log_odds.clip(-50, 50)
    out["combined_pS"] = 1.0 / (1.0 + np.exp(-log_odds))
    out["combined_logLR"] = total_loglr
    out["combined_n_filters"] = n_used
    out["combined_prior"] = prior
    out["ps_version"] = version
    return out


def run_for_version(args: argparse.Namespace, matched: pd.DataFrame, version: str, records: list[PlotRecord]) -> None:
    version_tag = f"DP2{version}"
    tag = plot_tag(args)
    ps_path = ps_table_path(args, version)
    sample, reason = load_version_sample(matched, ps_path)
    if sample is None:
        add_record(records, version_tag, "all", "pS", "version", ps_path, matched, skipped=reason)
        return
    filters = available_filters(read_table(ps_path), sample, args.filters)
    if not filters:
        add_record(records, version_tag, "all", "pS", "version", ps_path, sample, skipped="no filters with both pS and magnitude columns")
        return

    cmd_base = (
        valid_labeled_mask(sample)
        & np.isfinite(finite_series(sample, "dp2_cmd_g_minus_i"))
        & np.isfinite(finite_series(sample, "dp2_cmd_i_mag"))
    )
    out = args.plot_dir / f"cmd_mag_bins_4panel_{version_tag}_{tag}.png"
    path = plot_cmd_mag_bins_4panel(sample, version_tag, out, field=args.field, source=args.external_label_source)
    add_record(
        records,
        version_tag,
        "gi_i",
        "external_label",
        "cmd_mag_bins_4panel",
        path or out,
        sample,
        cmd_base,
        skipped="" if path else "missing CMD color or i magnitude columns",
    )

    ccd_base = (
        valid_labeled_mask(sample)
        & np.isfinite(finite_series(sample, "dp2_gr"))
        & np.isfinite(finite_series(sample, "dp2_ri"))
        & np.isfinite(finite_series(sample, "dp2_cmd_i_mag"))
    )
    out = args.plot_dir / f"ccd_mag_bins_4panel_{version_tag}_{tag}.png"
    path = plot_ccd_mag_bins_4panel(sample, version_tag, out, field=args.field, source=args.external_label_source)
    add_record(
        records,
        version_tag,
        "gri",
        "external_label",
        "ccd_mag_bins_4panel",
        path or out,
        sample,
        ccd_base,
        skipped="" if path else "missing CCD color or i magnitude columns",
    )

    for band in filters:
        score_col = ps_col_for_filter(band)
        mag_col = mag_col_for_filter(band)
        ext_col = ext_col_for_filter(band)
        base_ps = valid_labeled_mask(sample) & np.isfinite(finite_series(sample, score_col)) & np.isfinite(finite_series(sample, mag_col))
        if int(base_ps.sum()) == 0:
            add_record(records, version_tag, band, "pS", "all_plots", "", sample, skipped=f"missing finite {score_col}/{mag_col}")
            continue

        out = args.plot_dir / f"pS_vs_mag_4panel_{version_tag}_{tag}_{band}.png"
        path = plot_ps_vs_mag_4panel(sample, version_tag, band, out, field=args.field, source=args.external_label_source)
        add_record(records, version_tag, band, "pS", "pS_vs_mag_4panel", path or out, sample, base_ps, skipped="" if path else "plot failed")

        out = args.plot_dir / f"pS_hist_4panel_{version_tag}_{tag}_{band}.png"
        path = plot_score_hist_4panel(sample, version_tag, band, out, kind="pS", field=args.field, source=args.external_label_source)
        add_record(records, version_tag, band, "pS", "pS_hist_4panel", path or out, sample, base_ps, skipped="" if path else "plot failed")

        pred_ps = finite_series(sample, score_col) >= PS_THRESHOLD
        for diagram in ["cmd", "ccd"]:
            out = args.plot_dir / f"{diagram}_confusion_4panel_{version_tag}_{tag}_{band}_pS.png"
            path = plot_confusion_4panel(sample, version_tag, band, "pS", pred_ps, out, diagram=diagram, field=args.field, source=args.external_label_source)
            add_record(records, version_tag, band, "pS", f"{diagram}_confusion_4panel", path or out, sample, base_ps, skipped="" if path else "missing CMD/CCD columns")
            for mag_label, mag_mask in panel_masks_by_mag(sample, mag_col, base_ps):
                mag_tag = mag_bin_file_tag(mag_label)
                out = args.plot_dir / f"{diagram}_confusion_4panel_{version_tag}_{tag}_{band}_pS_{mag_tag}.png"
                path = plot_confusion_4panel(
                    sample,
                    version_tag,
                    band,
                    "pS",
                    pred_ps,
                    out,
                    diagram=diagram,
                    field=args.field,
                    source=args.external_label_source,
                    selection_mask=mag_mask,
                    selection_label=mag_label,
                )
                add_record(
                    records,
                    version_tag,
                    band,
                    "pS",
                    f"{diagram}_confusion_4panel_magbin",
                    path or out,
                    sample,
                    mag_mask,
                    skipped="" if path else "missing CMD/CCD columns",
                )

        ps_score = finite_series(sample, score_col)
        out = args.plot_dir / f"roc_4panel_{version_tag}_{tag}_{band}_pS.png"
        path, aucs = plot_roc_4panel(sample, version_tag, band, "pS", ps_score, mag_col, out, args.min_class_count, field=args.field, source=args.external_label_source)
        add_record(records, version_tag, band, "pS", "roc_4panel", path or out, sample, base_ps, auc=np.nanmean(list(aucs.values())), skipped="" if path else "ROC failed")

        out = args.plot_dir / f"roc_singlepanel_{version_tag}_{tag}_{band}_pS.png"
        path, aucs = plot_roc_singlepanel(sample, version_tag, band, "pS", ps_score, mag_col, out, args.min_class_count, field=args.field, source=args.external_label_source)
        add_record(records, version_tag, band, "pS", "roc_singlepanel", path or out, sample, base_ps, auc=np.nanmean(list(aucs.values())), skipped="" if path else "ROC failed")

        if ext_col in sample.columns:
            base_ext = valid_labeled_mask(sample) & np.isfinite(finite_series(sample, ext_col)) & np.isfinite(finite_series(sample, mag_col))
            out = args.plot_dir / f"extendedness_hist_4panel_{version_tag}_{tag}_{band}.png"
            path = plot_score_hist_4panel(sample, version_tag, band, out, kind="extendedness", field=args.field, source=args.external_label_source)
            add_record(records, version_tag, band, "extendedness", "extendedness_hist_4panel", path or out, sample, base_ext, skipped="" if path else "plot failed")

            pred_ext = finite_series(sample, ext_col) < EXT_THRESHOLD
            for diagram in ["cmd", "ccd"]:
                out = args.plot_dir / f"{diagram}_confusion_4panel_{version_tag}_{tag}_{band}_extendedness.png"
                path = plot_confusion_4panel(sample, version_tag, band, "extendedness", pred_ext, out, diagram=diagram, field=args.field, source=args.external_label_source)
                add_record(records, version_tag, band, "extendedness", f"{diagram}_confusion_4panel", path or out, sample, base_ext, skipped="" if path else "missing CMD/CCD columns")
                for mag_label, mag_mask in panel_masks_by_mag(sample, mag_col, base_ext):
                    mag_tag = mag_bin_file_tag(mag_label)
                    out = args.plot_dir / f"{diagram}_confusion_4panel_{version_tag}_{tag}_{band}_extendedness_{mag_tag}.png"
                    path = plot_confusion_4panel(
                        sample,
                        version_tag,
                        band,
                        "extendedness",
                        pred_ext,
                        out,
                        diagram=diagram,
                        field=args.field,
                        source=args.external_label_source,
                        selection_mask=mag_mask,
                        selection_label=mag_label,
                    )
                    add_record(
                        records,
                        version_tag,
                        band,
                        "extendedness",
                        f"{diagram}_confusion_4panel_magbin",
                        path or out,
                        sample,
                        mag_mask,
                        skipped="" if path else "missing CMD/CCD columns",
                    )

            ext_star_score = 1.0 - finite_series(sample, ext_col)
            out = args.plot_dir / f"roc_4panel_{version_tag}_{tag}_{band}_extendedness.png"
            path, aucs = plot_roc_4panel(sample, version_tag, band, "extendedness", ext_star_score, mag_col, out, args.min_class_count, field=args.field, source=args.external_label_source)
            add_record(records, version_tag, band, "extendedness", "roc_4panel", path or out, sample, base_ext, auc=np.nanmean(list(aucs.values())), skipped="" if path else "ROC failed")

            out = args.plot_dir / f"roc_singlepanel_{version_tag}_{tag}_{band}_extendedness.png"
            path, aucs = plot_roc_singlepanel(sample, version_tag, band, "extendedness", ext_star_score, mag_col, out, args.min_class_count, field=args.field, source=args.external_label_source)
            add_record(records, version_tag, band, "extendedness", "roc_singlepanel", path or out, sample, base_ext, auc=np.nanmean(list(aucs.values())), skipped="" if path else "ROC failed")
        else:
            add_record(records, version_tag, band, "extendedness", "extendedness_plots", "", sample, skipped=f"missing {ext_col}")

    if args.skip_combined:
        return

    combined = combine_ps_slater_style(sample, filters, version_tag, prior=args.prior, hist_bins=args.hist_bins, min_class_count=args.min_class_count)
    combined_path = args.combined_output_dir / f"dp2_{args.field.lower()}_{safe_tag(args.external_label_source).lower()}_combined_ps_{version}.parquet"
    write_table(combined, combined_path)
    merged = sample.merge(combined[["object_id", "combined_pS", "combined_logLR", "combined_n_filters"]], on="object_id", how="left", validate="one_to_one")
    base_combined = valid_labeled_mask(merged) & np.isfinite(merged["combined_pS"]) & (merged["combined_n_filters"] > 0)
    add_record(records, version_tag, "combined", "combined_pS", "combined_table", combined_path, merged, base_combined)

    combined_ready = merged["combined_n_filters"] > 0
    combined_score = finite_series(merged, "combined_pS").where(combined_ready)
    pred_combined = pd.Series(pd.NA, index=merged.index, dtype="boolean")
    pred_combined.loc[combined_ready] = combined_score.loc[combined_ready] >= PS_THRESHOLD
    for diagram in ["cmd", "ccd"]:
        out = args.plot_dir / f"{diagram}_confusion_4panel_{version_tag}_{tag}_combined_pS.png"
        path = plot_confusion_4panel(merged, version_tag, "combined", "pS", pred_combined, out, diagram=diagram, field=args.field, source=args.external_label_source)
        add_record(records, version_tag, "combined", "combined_pS", f"{diagram}_confusion_4panel", path or out, merged, base_combined, skipped="" if path else "missing CMD/CCD columns")

    mag_col = "dp2_cmd_i_mag" if "dp2_cmd_i_mag" in merged.columns else mag_col_for_filter("i")
    out = args.plot_dir / f"roc_4panel_{version_tag}_{tag}_combined_pS.png"
    path, aucs = plot_roc_4panel(merged, version_tag, "combined", "combined_pS", combined_score, mag_col, out, args.min_class_count, field=args.field, source=args.external_label_source)
    add_record(records, version_tag, "combined", "combined_pS", "roc_4panel", path or out, merged, base_combined, auc=np.nanmean(list(aucs.values())), skipped="" if path else "ROC failed")

    out = args.plot_dir / f"roc_singlepanel_{version_tag}_{tag}_combined_pS.png"
    path, aucs = plot_roc_singlepanel(merged, version_tag, "combined", "combined_pS", combined_score, mag_col, out, args.min_class_count, field=args.field, source=args.external_label_source)
    add_record(records, version_tag, "combined", "combined_pS", "roc_singlepanel", path or out, merged, base_combined, auc=np.nanmean(list(aucs.values())), skipped="" if path else "ROC failed")


def main() -> None:
    args = parse_args()
    ensure_dir(args.plot_dir)
    ensure_dir(args.combined_output_dir)
    matched = read_table(args.matched)
    if "object_id" not in matched.columns:
        raise ValueError(f"Matched table lacks object_id: {args.matched}")
    if "truth_label" not in matched.columns and "truth_binary" not in matched.columns:
        raise ValueError(f"Matched table lacks external truth label columns: {args.matched}")
    matched = matched.drop_duplicates(subset=["object_id"], keep="first").reset_index(drop=True)

    records: list[PlotRecord] = []
    versions = [v.strip() for v in args.versions.split(",") if v.strip()]
    print(f"Matched {args.external_label_source} sample for {args.field}: {len(matched):,} rows")
    print(f"Versions: {versions}")
    for version in versions:
        print(f"Running {version}...")
        run_for_version(args, matched, version, records)

    summary = pd.DataFrame([r.__dict__ for r in records])
    ensure_dir(args.summary.parent)
    summary.to_csv(args.summary, index=False)
    generated = summary[(summary["skipped_reason"].fillna("") == "") & (summary["output_path"].fillna("") != "")]
    skipped = summary[summary["skipped_reason"].fillna("") != ""]
    print(f"Wrote summary: {args.summary}")
    print(f"Generated records: {len(generated):,}")
    print(f"Skipped records: {len(skipped):,}")
    if len(generated):
        print("Generated files:")
        for path in generated["output_path"].dropna().unique():
            print(f"  {path}")
    if len(skipped):
        print("Skipped cases:")
        print(skipped[["version", "filter", "classifier", "plot_type", "skipped_reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
