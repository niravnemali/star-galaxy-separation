"""Extendedness baseline figures for COSMOS paper convergence."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper_plot_style import COLOR_COLOR_LIMITS, COLORS, FIG_SIZES, downsample_frame, save_figure, set_paper_style
from paper_sample_selection import truth_masks

MAG_SPLITS = ((16.0, 25.0), (25.0, 26.0))
PERFORMANCE_BINS = ((16.0, 24.0), (24.0, 25.0), (25.0, 26.0))


def class_masks(df: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return star-like, galaxy-like, and valid masks for extendedness-like columns."""

    values = pd.to_numeric(df[col], errors="coerce")
    valid = values.isin([0, 1])
    star_like = values.eq(0) & valid
    galaxy_like = values.eq(1) & valid
    return star_like, galaxy_like, valid


def verify_extendedness_convention(matched: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    stars, galaxies = truth_masks(matched)
    rows = []
    for col in columns:
        star_like, galaxy_like, valid = class_masks(matched, col)
        rows.extend(
            [
                {
                    "column": col,
                    "group": "external_star",
                    "N_total": int(stars.sum()),
                    "N_value_0_unresolved": int((stars & star_like).sum()),
                    "N_value_1_resolved": int((stars & galaxy_like).sum()),
                    "N_nan_or_other": int((stars & ~valid).sum()),
                    "convention": "0=unresolved/star-like, 1=resolved/galaxy-like",
                },
                {
                    "column": col,
                    "group": "external_galaxy",
                    "N_total": int(galaxies.sum()),
                    "N_value_0_unresolved": int((galaxies & star_like).sum()),
                    "N_value_1_resolved": int((galaxies & galaxy_like).sum()),
                    "N_nan_or_other": int((galaxies & ~valid).sum()),
                    "convention": "0=unresolved/star-like, 1=resolved/galaxy-like",
                },
            ]
        )
    return pd.DataFrame(rows)


def _limit_string(limits) -> str:
    return str(tuple(float(x) for x in limits))


def _scatter_class(ax, df: pd.DataFrame, class_col: str, x_col: str, y_col: str, max_gal: int = 90000) -> dict:
    star_like, galaxy_like, valid = class_masks(df, class_col)
    finite = valid & np.isfinite(df[x_col]) & np.isfinite(df[y_col])
    gal = df.loc[galaxy_like & finite]
    star = df.loc[star_like & finite]
    gal_plot = downsample_frame(gal, max_gal)
    star_plot = downsample_frame(star, 30000)
    ax.scatter(gal_plot[x_col], gal_plot[y_col], s=2.0, c=COLORS["galaxy"], alpha=0.10, linewidths=0, label=f"resolved N={len(gal):,}")
    ax.scatter(star_plot[x_col], star_plot[y_col], s=3.0, c=COLORS["star"], alpha=0.45, linewidths=0, label=f"unresolved N={len(star):,}")
    return {
        "N_unresolved_finite": int(len(star)),
        "N_resolved_finite": int(len(gal)),
        "N_nan_or_other": int((~valid).sum()),
    }


def plot_extendedness_color_color(
    df: pd.DataFrame,
    class_col: str,
    title_label: str,
    output_png: Path,
) -> tuple[list[Path], pd.DataFrame]:
    """Create Fig 1.4-style color-color diagram using extendedness labels."""

    set_paper_style()
    fig, axes = plt.subplots(2, 4, figsize=FIG_SIZES["2x4"])
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.13, top=0.84, hspace=0.50, wspace=0.30)
    specs = [
        ("color_ug", "color_gr", "dust-corrected u-g", "dust-corrected g-r", (16, 25), ("ug", "gr")),
        ("color_ug", "color_gr", "dust-corrected u-g", "dust-corrected g-r", (25, 26), ("ug", "gr")),
        ("color_gr", "color_ri", "dust-corrected g-r", "dust-corrected r-i", (16, 25), ("gr", "ri")),
        ("color_gr", "color_ri", "dust-corrected g-r", "dust-corrected r-i", (25, 26), ("gr", "ri")),
        ("color_ri", "color_iz", "dust-corrected r-i", "dust-corrected i-z", (16, 25), ("ri", "iz")),
        ("color_ri", "color_iz", "dust-corrected r-i", "dust-corrected i-z", (25, 26), ("ri", "iz")),
        ("color_iz", "color_zy", "dust-corrected i-z", "dust-corrected z-y", (16, 25), ("iz", "zy")),
        ("color_iz", "color_zy", "dust-corrected i-z", "dust-corrected z-y", (25, 26), ("iz", "zy")),
    ]
    rows = []
    rmag = pd.to_numeric(df["cmodel_mag_r"], errors="coerce")
    for ax, (x_col, y_col, x_label, y_label, mag_range, limit_key) in zip(axes.flat, specs):
        lo, hi = mag_range
        use = df.loc[rmag.gt(lo) & rmag.lt(hi)]
        counts = _scatter_class(ax, use, class_col, x_col, y_col)
        xlim, ylim = COLOR_COLOR_LIMITS[limit_key]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"{lo:g} < rmag < {hi:g}\nN_unres={counts['N_unresolved_finite']:,}, N_res={counts['N_resolved_finite']:,}")
        rows.append(
            {
                "classification_column": class_col,
                "x_column": x_col,
                "y_column": y_col,
                "mag_low": lo,
                "mag_high": hi,
                **counts,
                "xlim": _limit_string(xlim),
                "ylim": _limit_string(ylim),
                "convention": "0=unresolved/star-like, 1=resolved/galaxy-like",
            }
        )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["galaxy"], markeredgewidth=0, markersize=5, alpha=0.55, label="resolved"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["star"], markeredgewidth=0, markersize=5, alpha=0.85, label="unresolved"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.02))
    for ax in axes.flat:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.suptitle(f"Fig 1.4 COSMOS {title_label} color-color diagrams", y=0.965, fontsize=13)
    return save_figure(fig, output_png), pd.DataFrame(rows)


def plot_confusion_cmd(
    matched: pd.DataFrame,
    class_col: str,
    title_label: str,
    output_png: Path,
) -> tuple[list[Path], pd.DataFrame]:
    """Create Fig 1.5-style CMD confusion panels against COSMOS2020 truth labels."""

    set_paper_style()
    truth_star, truth_gal = truth_masks(matched)
    pred_star, pred_gal, valid = class_masks(matched, class_col)
    finite = valid & np.isfinite(matched["color_gi"]) & np.isfinite(matched["dp2_cmodel_mag_r"])
    categories = [
        ("SS", truth_star & pred_star & finite, COLORS["star"], "truth star, class star"),
        ("SG", truth_star & pred_gal & finite, COLORS["star"], "truth star, class galaxy"),
        ("GS", truth_gal & pred_star & finite, COLORS["galaxy"], "truth galaxy, class star"),
        ("GG", truth_gal & pred_gal & finite, COLORS["galaxy"], "truth galaxy, class galaxy"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES["2x2"], sharex=True, sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.10, top=0.86, hspace=0.35, wspace=0.12)
    rows = []
    for idx, (ax, (label, mask, color, subtitle)) in enumerate(zip(axes.flat, categories)):
        data = matched.loc[mask]
        plot_data = downsample_frame(data, 90000)
        ax.scatter(plot_data["color_gi"], plot_data["dp2_cmodel_mag_r"], s=2.5, c=color, alpha=0.18, linewidths=0)
        ax.set_xlim(-0.8, 4.0)
        ax.set_ylim(26, 16)
        ax.set_title(f"{label}: {subtitle}\nN={len(data):,}")
        if idx // 2 == 1:
            ax.set_xlabel("dust-corrected g-i CModel color")
        else:
            ax.set_xlabel("")
        if idx % 2 == 0:
            ax.set_ylabel("uncorrected r CModel magnitude")
        else:
            ax.set_ylabel("")
        rows.append(
            {
                "classification_column": class_col,
                "panel": label,
                "description": subtitle,
                "N": int(len(data)),
                "x_column": "color_gi",
                "y_column": "dp2_cmodel_mag_r",
                "xlim": "(-0.8, 4.0)",
                "ylim": "(26.0, 16.0)",
                "convention": "first letter=COSMOS2020 truth, second letter=extendedness classification",
            }
        )
    fig.suptitle(f"Fig 1.5 COSMOS {title_label} confusion CMD", y=0.975, fontsize=14)
    return save_figure(fig, output_png), pd.DataFrame(rows)


def _binary_metrics(truth_positive: pd.Series, pred_positive: pd.Series, valid: pd.Series) -> dict:
    truth_positive = truth_positive & valid
    truth_negative = (~truth_positive) & valid
    pred_positive = pred_positive & valid
    pred_negative = (~pred_positive) & valid
    tp = int((truth_positive & pred_positive).sum())
    fn = int((truth_positive & pred_negative).sum())
    fp = int((truth_negative & pred_positive).sum())
    tn = int((truth_negative & pred_negative).sum())
    tpr = tp / (tp + fn) if (tp + fn) else np.nan
    fpr = fp / (fp + tn) if (fp + tn) else np.nan
    precision = tp / (tp + fp) if (tp + fp) else np.nan
    contamination = fp / (tp + fp) if (tp + fp) else np.nan
    auc_step = 0.5 * (1 + tpr - fpr) if np.isfinite(tpr) and np.isfinite(fpr) else np.nan
    return {
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "TN": tn,
        "completeness_TPR": tpr,
        "false_positive_rate": fpr,
        "purity_precision": precision,
        "contamination": contamination,
        "step_auc": auc_step,
    }


def extendedness_metrics_by_mag(matched: pd.DataFrame, class_col: str) -> pd.DataFrame:
    truth_star, truth_gal = truth_masks(matched)
    pred_star, pred_gal, valid_class = class_masks(matched, class_col)
    rmag = pd.to_numeric(matched["dp2_cmodel_mag_r"], errors="coerce")
    rows = []
    for lo, hi in PERFORMANCE_BINS:
        in_bin = rmag.gt(lo) & rmag.lt(hi)
        valid = in_bin & valid_class & (truth_star | truth_gal)
        for positive_class, truth_positive, pred_positive in [
            ("star", truth_star, pred_star),
            ("galaxy", truth_gal, pred_gal),
        ]:
            metrics = _binary_metrics(truth_positive, pred_positive, valid)
            rows.append(
                {
                    "classification_column": class_col,
                    "mag_low": lo,
                    "mag_high": hi,
                    "positive_class": positive_class,
                    "N_valid": int(valid.sum()),
                    "N_star": int((truth_star & valid).sum()),
                    "N_galaxy": int((truth_gal & valid).sum()),
                    "N_nan_or_other_classification": int((in_bin & ~valid_class).sum()),
                    **metrics,
                    "convention": "0=unresolved/star-like, 1=resolved/galaxy-like",
                }
            )
    return pd.DataFrame(rows)


def plot_extendedness_performance(
    matched: pd.DataFrame,
    class_col: str,
    title_label: str,
    output_png: Path,
) -> tuple[list[Path], pd.DataFrame]:
    """Create Fig 1.6 binary operating-point ROC panels."""

    set_paper_style()
    metrics = extendedness_metrics_by_mag(matched, class_col)
    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZES["1x3"])
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.18, top=0.84, wspace=0.24)
    for ax, (lo, hi) in zip(axes, PERFORMANCE_BINS):
        ax.plot([0, 1], [0, 1], color="#999999", ls="--", lw=1.0, label="random")
        panel = metrics[(metrics["mag_low"] == lo) & (metrics["mag_high"] == hi)]
        for _, row in panel.iterrows():
            color = COLORS["star"] if row["positive_class"] == "star" else COLORS["galaxy"]
            label = (
                f"{row['positive_class']} positive "
                f"AUC={row['step_auc']:.3f}, N_S={int(row['N_star']):,}, N_G={int(row['N_galaxy']):,}"
            )
            fpr = row["false_positive_rate"]
            tpr = row["completeness_TPR"]
            ax.step([0, fpr, 1], [0, tpr, 1], where="post", color=color, lw=1.4, alpha=0.85)
            ax.scatter([fpr], [tpr], color=color, s=35, label=label, zorder=5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("false positive rate")
        ax.set_ylabel("true positive rate")
        ax.set_title(f"{lo:g} < rmag < {hi:g}")
        ax.legend(loc="lower right", frameon=True, fontsize=7)
    if title_label == "r-band extendedness":
        fig.suptitle("Fig 1.6 COSMOS r-band extendedness performance", y=0.965, fontsize=13)
    elif title_label == "refExtendedness":
        fig.suptitle("Fig 1.6 COSMOS refExtendedness performance", y=0.965, fontsize=13)
    else:
        fig.suptitle(f"Fig 1.6 COSMOS {title_label} extendedness performance", y=0.965, fontsize=13)
    return save_figure(fig, output_png), metrics


def compare_extendedness_metrics(r_metrics: pd.DataFrame, ref_metrics: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    merged = r_metrics.merge(
        ref_metrics,
        on=["mag_low", "mag_high", "positive_class"],
        suffixes=("_r_extendedness", "_refExtendedness"),
    )
    keep = [
        "mag_low",
        "mag_high",
        "positive_class",
        "N_valid_r_extendedness",
        "N_valid_refExtendedness",
        "step_auc_r_extendedness",
        "step_auc_refExtendedness",
        "completeness_TPR_r_extendedness",
        "completeness_TPR_refExtendedness",
        "false_positive_rate_r_extendedness",
        "false_positive_rate_refExtendedness",
        "purity_precision_r_extendedness",
        "purity_precision_refExtendedness",
    ]
    out = merged[keep].copy()
    out["delta_step_auc_ref_minus_r"] = out["step_auc_refExtendedness"] - out["step_auc_r_extendedness"]
    md = ["# Extendedness vs refExtendedness Comparison", ""]
    md.append("Conventions: `0=unresolved/star-like`, `1=resolved/galaxy-like`; NaN/other values are excluded from operating-point metrics and counted in the CSV summaries.")
    md.append("")
    for _, row in out.iterrows():
        md.append(
            f"- {row['mag_low']:g}-{row['mag_high']:g}, {row['positive_class']} positive: "
            f"AUC r={row['step_auc_r_extendedness']:.3f}, "
            f"ref={row['step_auc_refExtendedness']:.3f}, "
            f"delta={row['delta_step_auc_ref_minus_r']:.3f}."
        )
    return out, "\n".join(md) + "\n"
