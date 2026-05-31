"""Section 1 COSMOS paper figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from paper_plot_style import COLOR_COLOR_LIMITS, COLORS, FIG_SIZES, downsample_frame, save_figure, set_paper_style
from paper_sample_selection import truth_masks


def _scatter_truth(ax, df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, *, max_gal=90000):
    stars, galaxies = truth_masks(df)
    gal = df.loc[galaxies & np.isfinite(df[x_col]) & np.isfinite(df[y_col])]
    star = df.loc[stars & np.isfinite(df[x_col]) & np.isfinite(df[y_col])]
    gal_plot = downsample_frame(gal, max_gal)
    star_plot = downsample_frame(star, 25000)
    ax.scatter(gal_plot[x_col], gal_plot[y_col], s=2.0, c=COLORS["galaxy"], alpha=0.10, linewidths=0, label=f"galaxy N={len(gal):,}")
    ax.scatter(star_plot[x_col], star_plot[y_col], s=3.0, c=COLORS["star"], alpha=0.45, linewidths=0, label=f"star N={len(star):,}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)


def _limit_string(limits) -> str:
    return str(tuple(float(x) for x in limits))


def _hist_log_counts(ax, rows: list[dict], bins: np.ndarray) -> None:
    for row in rows:
        data = pd.to_numeric(row["data"], errors="coerce").dropna()
        counts, edges = np.histogram(data, bins=bins)
        mids = 0.5 * (edges[:-1] + edges[1:])
        y = np.full(counts.shape, np.nan, dtype=float)
        positive = counts > 0
        y[positive] = np.log10(counts[positive])
        ax.plot(mids, y, drawstyle="steps-mid", color=row["color"], lw=row.get("lw", 1.8), label=row["label"])
    ax.set_xlabel("uncorrected r CModel magnitude")
    ax.set_ylabel("log10(counts per 0.5 mag)")
    ax.set_xlim(16, 26)
    ax.legend(loc="best", frameon=True)


def _binned_ratio(df: pd.DataFrame, category: str, bins: np.ndarray, mag_col: str, flux_col: str, err_col: str) -> pd.DataFrame:
    mag = pd.to_numeric(df[mag_col], errors="coerce")
    flux = pd.to_numeric(df[flux_col], errors="coerce")
    err = pd.to_numeric(df[err_col], errors="coerce")
    ratio = err / flux
    use = pd.DataFrame({"mag": mag, "ratio": ratio})
    use = use[np.isfinite(use["mag"]) & np.isfinite(use["ratio"]) & (flux > 0) & (err > 0)]
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        vals = use.loc[use["mag"].ge(lo) & use["mag"].lt(hi), "ratio"]
        rows.append(
            {
                "panel": "flux_uncertainty_ratio",
                "category": category,
                "mag_low": lo,
                "mag_high": hi,
                "count": int(vals.size),
                "median_flux_err_over_flux": vals.median() if vals.size else np.nan,
                "p16_flux_err_over_flux": vals.quantile(0.16) if vals.size else np.nan,
                "p84_flux_err_over_flux": vals.quantile(0.84) if vals.size else np.nan,
                "notes": "",
            }
        )
    return pd.DataFrame(rows)


def plot_fig1_1(samples: dict[str, pd.DataFrame], output_png: Path) -> tuple[list[Path], pd.DataFrame]:
    set_paper_style()
    matched = samples["matched_paper"]
    dp2_only = samples["dp2_only"]
    external_only = samples.get("external_only", pd.DataFrame())
    stars, galaxies = truth_masks(matched)

    fig, axes = plt.subplots(1, 3, figsize=FIG_SIZES["1x3"])
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.18, top=0.84, wspace=0.26)

    ax = axes[0]
    gal = matched.loc[galaxies]
    star = matched.loc[stars]
    gal_plot = downsample_frame(gal, 120000)
    ax.scatter(gal_plot["dp2_ra"], gal_plot["dp2_dec"], s=1.5, c=COLORS["galaxy"], alpha=0.12, linewidths=0, label=f"COSMOS2020 galaxy N={len(gal):,}")
    ax.scatter(star["dp2_ra"], star["dp2_dec"], s=3.0, c=COLORS["star"], alpha=0.50, linewidths=0, label=f"COSMOS2020 star N={len(star):,}")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("COSMOS/COSMOS2020 matched DP2 sample")
    ax.invert_xaxis()
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), frameon=True)

    bins = np.arange(16, 26.0001, 0.5)
    ax = axes[1]
    hist_rows = [
        {"data": dp2_only["cmodel_mag_r"], "color": COLORS["dp2_only"], "label": f"DP2-only N={len(dp2_only):,}"},
        {"data": matched.loc[stars, "dp2_cmodel_mag_r"], "color": COLORS["star"], "label": f"matched stars N={int(stars.sum()):,}"},
        {"data": matched.loc[galaxies, "dp2_cmodel_mag_r"], "color": COLORS["galaxy"], "label": f"matched galaxies N={int(galaxies.sum()):,}"},
    ]
    external_rmag_available = "external_hsc_r_mag" in external_only.columns
    if external_rmag_available:
        ext_r = pd.to_numeric(external_only["external_hsc_r_mag"], errors="coerce")
        if "external_hsc_r_valid" in external_only.columns:
            valid = external_only["external_hsc_r_valid"].fillna(False).astype(bool)
            ext_r = ext_r.loc[valid]
        ext_r = ext_r[np.isfinite(ext_r)]
        hist_rows.append(
            {
                "data": ext_r,
                "color": COLORS["external_only"],
                "label": f"COSMOS2020-only HSC r N={len(ext_r):,}",
                "lw": 1.6,
            }
        )
    else:
        ext_r = pd.Series(dtype=float)
    _hist_log_counts(
        ax,
        hist_rows,
        bins,
    )
    if external_rmag_available:
        ax.set_xlabel("r magnitude (DP2 CModel; COSMOS2020 HSC r)")
        ax.text(
            0.98,
            0.05,
            "COSMOS2020-only uses HSC_r_MAG",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=2.0),
        )
    else:
        ax.text(
            0.98,
            0.05,
            "COSMOS2020-only r magnitude\nunavailable",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.80, pad=2.0),
        )
    ax.set_title("Counts vs r magnitude")

    ax = axes[2]
    ratio_rows = []
    categories = [
        ("DP2-only", dp2_only, "cmodel_mag_r", "cmodel_flux_r", "cmodel_flux_err_r", COLORS["dp2_only"]),
        ("matched stars", matched.loc[stars], "dp2_cmodel_mag_r", "dp2_cmodel_flux_r", "dp2_cmodel_flux_err_r", COLORS["star"]),
        ("matched galaxies", matched.loc[galaxies], "dp2_cmodel_mag_r", "dp2_cmodel_flux_r", "dp2_cmodel_flux_err_r", COLORS["galaxy"]),
    ]
    for label, df, mag_col, flux_col, err_col, color in categories:
        stat = _binned_ratio(df, label, bins, mag_col, flux_col, err_col)
        ratio_rows.append(stat)
        x = 0.5 * (stat["mag_low"] + stat["mag_high"])
        ax.plot(x, stat["median_flux_err_over_flux"], marker="o", ms=3, lw=1.5, color=color, label=label)
        ax.fill_between(x, stat["p16_flux_err_over_flux"], stat["p84_flux_err_over_flux"], color=color, alpha=0.12, linewidth=0)
    ax.set_yscale("log")
    ax.set_xlim(16, 26)
    ax.set_xlabel("uncorrected r CModel magnitude")
    ax.set_ylabel("r cModelFluxErr / cModelFlux")
    ax.set_title("Flux uncertainty ratio")
    ax.legend(loc="best", frameon=True)

    fig.suptitle("Fig 1.1 COSMOS DP2 / COSMOS2020 dataset overview", y=0.96, fontsize=13)
    saved = save_figure(fig, output_png)

    count_rows = []
    for label, data in [
        ("DP2-only", dp2_only["cmodel_mag_r"]),
        ("matched stars", matched.loc[stars, "dp2_cmodel_mag_r"]),
        ("matched galaxies", matched.loc[galaxies, "dp2_cmodel_mag_r"]),
        ("COSMOS2020-only HSC r", ext_r),
    ]:
        if label.startswith("COSMOS2020-only") and not external_rmag_available:
            continue
        counts, edges = np.histogram(pd.to_numeric(data, errors="coerce").dropna(), bins=bins)
        for lo, hi, count in zip(edges[:-1], edges[1:], counts):
            count_rows.append({"panel": "counts_vs_rmag", "category": label, "mag_low": lo, "mag_high": hi, "count": int(count), "notes": ""})
    if not external_rmag_available:
        count_rows.append(
            {
                "panel": "counts_vs_rmag",
                "category": "COSMOS2020-only",
                "mag_low": np.nan,
                "mag_high": np.nan,
                "count": np.nan,
                "notes": "Unavailable: external truth CSV has no r magnitude directly comparable to DP2 r CModelMag.",
            }
        )
    else:
        count_rows.append(
            {
                "panel": "counts_vs_rmag",
                "category": "COSMOS2020-only HSC r",
                "mag_low": np.nan,
                "mag_high": np.nan,
                "count": int(ext_r.size),
                "notes": "External-only curve uses COSMOS2020 FARMER HSC_r_MAG from the local full FARMER FITS; it is not DP2 r CModelMag.",
            }
        )
    summary = pd.concat([pd.DataFrame(count_rows), *ratio_rows], ignore_index=True)
    return saved, summary


def plot_fig1_2(samples: dict[str, pd.DataFrame], output_png: Path) -> tuple[list[Path], pd.DataFrame]:
    set_paper_style()
    matched = samples["matched_paper"]
    fig, axes = plt.subplots(2, 2, figsize=FIG_SIZES["2x2"])
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.88, hspace=0.34, wspace=0.24)

    panels = [
        (axes[0, 0], "dp2_cmodel_mag_r", "dp2_psf_minus_cmodel_r", "uncorrected r CModel magnitude", "r psfMag - CModelMag", (16, 26), None),
        (axes[0, 1], "dp2_cmodel_mag_r", "color_gi", "uncorrected r CModel magnitude", "dust-corrected g-i CModel color", (16, 26), None),
        (axes[1, 0], "color_ug", "color_gr", "dust-corrected u-g", "dust-corrected g-r", *COLOR_COLOR_LIMITS[("ug", "gr")]),
        (axes[1, 1], "color_gr", "color_ri", "dust-corrected g-r", "dust-corrected r-i", *COLOR_COLOR_LIMITS[("gr", "ri")]),
    ]

    rows = []
    for ax, x_col, y_col, x_label, y_label, xlim, ylim in panels:
        _scatter_truth(ax, matched, x_col, y_col, x_label, y_label)
        ax.set_xlim(xlim)
        if ylim is None:
            vals = pd.to_numeric(matched[y_col], errors="coerce")
            lo, hi = np.nanpercentile(vals, [1, 99])
            pad = 0.08 * (hi - lo) if np.isfinite(hi - lo) and hi > lo else 0.1
            ylim = (lo - pad, hi + pad)
        ax.set_ylim(ylim)
        stars, galaxies = truth_masks(matched)
        finite = np.isfinite(matched[x_col]) & np.isfinite(matched[y_col])
        rows.append(
            {
                "panel": y_label,
                "x_column": x_col,
                "y_column": y_col,
                "N_star_finite": int((stars & finite).sum()),
                "N_galaxy_finite": int((galaxies & finite).sum()),
                "xlim": _limit_string(xlim),
                "ylim": _limit_string(ylim),
            }
        )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.94))
    fig.suptitle("Fig 1.2 COSMOS2020 truth-label morphology and color overview", y=0.995, fontsize=13)
    return save_figure(fig, output_png), pd.DataFrame(rows)


def plot_fig1_3(samples: dict[str, pd.DataFrame], output_png: Path) -> tuple[list[Path], pd.DataFrame]:
    set_paper_style()
    matched = samples["matched_paper"]
    fig, axes = plt.subplots(2, 4, figsize=FIG_SIZES["2x4"])
    fig.subplots_adjust(left=0.055, right=0.99, bottom=0.13, top=0.84, hspace=0.48, wspace=0.30)
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
    for ax, (x_col, y_col, x_label, y_label, mag_range, limit_key) in zip(axes.flat, specs):
        lo, hi = mag_range
        use = matched.loc[pd.to_numeric(matched["dp2_cmodel_mag_r"], errors="coerce").gt(lo) & pd.to_numeric(matched["dp2_cmodel_mag_r"], errors="coerce").lt(hi)]
        _scatter_truth(ax, use, x_col, y_col, x_label, y_label, max_gal=65000)
        xlim, ylim = COLOR_COLOR_LIMITS[limit_key]
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        stars, galaxies = truth_masks(use)
        finite = np.isfinite(use[x_col]) & np.isfinite(use[y_col])
        ax.set_title(f"{lo:g} < rmag < {hi:g}\nN_star={int((stars & finite).sum()):,}, N_gal={int((galaxies & finite).sum()):,}")
        rows.append(
            {
                "x_column": x_col,
                "y_column": y_col,
                "mag_low": lo,
                "mag_high": hi,
                "N_star_finite": int((stars & finite).sum()),
                "N_galaxy_finite": int((galaxies & finite).sum()),
                "xlim": _limit_string(xlim),
                "ylim": _limit_string(ylim),
            }
        )
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["galaxy"], markeredgewidth=0, markersize=5, alpha=0.55, label="galaxy"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["star"], markeredgewidth=0, markersize=5, alpha=0.85, label="star"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.02))
    for ax in axes.flat:
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()
    fig.suptitle("Fig 1.3 COSMOS2020 truth color-color diagrams by r magnitude", y=0.965, fontsize=13)
    return save_figure(fig, output_png), pd.DataFrame(rows)
