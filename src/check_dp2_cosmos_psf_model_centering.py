#!/usr/bin/env python3
"""Check DP2 COSMOS PSF-CModel magnitude-difference centering.

This diagnostic is intentionally DP2-only. It does not compare against DP1.

The quantity checked throughout is:

    delta = psf_mag - cmodel_mag

If delta is centered near zero for point-like sources, PSF and CModel
magnitudes agree for those sources. Positive delta means PSF magnitude is
fainter than CModel magnitude; negative delta means PSF magnitude is brighter
than CModel magnitude.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
BANDS = ("u", "g", "r", "i", "z", "y")
MAG_BINS = (
    ("mag < 22", -np.inf, 22.0),
    ("22 <= mag < 23", 22.0, 23.0),
    ("23 <= mag < 24", 23.0, 24.0),
    ("24 <= mag < 25", 24.0, 25.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table",
        type=Path,
        default=REPO_ROOT / "outputs" / "dp2_cosmos_analysis_table.parquet",
        help="Prepared DP2 COSMOS analysis table.",
    )
    parser.add_argument(
        "--matched",
        type=Path,
        default=REPO_ROOT / "outputs" / "dp2_cosmos_cosmos2020_farmer_matched.parquet",
        help="Optional DP2 COSMOS table matched to COSMOS2020 external labels.",
    )
    parser.add_argument("--plot-dir", type=Path, default=REPO_ROOT / "plots")
    parser.add_argument("--result-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--mag-min", type=float, default=15.0)
    parser.add_argument("--mag-max", type=float, default=28.0)
    parser.add_argument("--x-min", type=float, default=-0.2)
    parser.add_argument("--x-max", type=float, default=0.8)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in df.columns), None)


def delta_col(df: pd.DataFrame, band: str) -> str | None:
    return first_existing(df, [f"psf_minus_cmodel_{band}", f"dp2_psf_minus_cmodel_{band}"])


def mag_col(df: pd.DataFrame, band: str) -> str | None:
    return first_existing(df, [f"cmodel_mag_{band}", f"dp2_cmodel_mag_{band}", f"{band}_cModelMag"])


def finite_values(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce")


def mad_sigma(values: pd.Series | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    med = np.nanmedian(arr)
    return float(1.4826 * np.nanmedian(np.abs(arr - med)))


def mag_mask(df: pd.DataFrame, band: str, lo: float, hi: float) -> pd.Series:
    mcol = mag_col(df, band)
    dcol = delta_col(df, band)
    if mcol is None or dcol is None:
        return pd.Series(False, index=df.index)
    mag = finite_values(df, mcol)
    delta = finite_values(df, dcol)
    mask = np.isfinite(mag) & np.isfinite(delta) & (mag < hi)
    if np.isfinite(lo):
        mask &= mag >= lo
    return pd.Series(mask, index=df.index)


def base_mask(df: pd.DataFrame, band: str, mag_min: float, mag_max: float) -> pd.Series:
    return mag_mask(df, band, mag_min, mag_max)


def summarize_sample(
    rows: list[dict[str, object]],
    df: pd.DataFrame,
    sample_name: str,
    band: str,
    mag_label: str,
    mask: pd.Series,
) -> None:
    dcol = delta_col(df, band)
    if dcol is None:
        return
    delta = finite_values(df, dcol)[mask]
    finite = delta[np.isfinite(delta)]
    n = int(finite.size)
    if n == 0:
        rows.append(
            {
                "sample": sample_name,
                "band": band,
                "mag_bin": mag_label,
                "n": 0,
                "median_delta": np.nan,
                "mean_delta": np.nan,
                "mad_sigma_delta": np.nan,
                "fraction_abs_delta_lt_0p03": np.nan,
                "fraction_delta_gt_0p03": np.nan,
            }
        )
        return
    rows.append(
        {
            "sample": sample_name,
            "band": band,
            "mag_bin": mag_label,
            "n": n,
            "median_delta": float(np.nanmedian(finite)),
            "mean_delta": float(np.nanmean(finite)),
            "mad_sigma_delta": mad_sigma(finite),
            "fraction_abs_delta_lt_0p03": float((np.abs(finite) < 0.03).mean()),
            "fraction_delta_gt_0p03": float((finite > 0.03).mean()),
        }
    )


def hist_delta(ax, values, label: str, color: str, x_range: tuple[float, float], lw: float = 2.0) -> None:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    bins = np.linspace(x_range[0], x_range[1], 80)
    ax.hist(arr, bins=bins, histtype="step", density=True, lw=lw, color=color, label=f"{label} N={arr.size:,}")
    med = float(np.nanmedian(arr))
    ax.axvline(med, color=color, lw=2.2, ls="-", alpha=0.9, label=f"{label} med={med:.4f}")


def plot_all_band_hist(df: pd.DataFrame, args: argparse.Namespace) -> Path:
    out = args.plot_dir / "dp2_cosmos_psf_minus_cmodel_delta_hist_all_bands.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, band in zip(axes.ravel(), BANDS):
        dcol = delta_col(df, band)
        if dcol is None:
            ax.set_title(f"{band}: missing delta column")
            continue
        mask = base_mask(df, band, args.mag_min, args.mag_max)
        values = finite_values(df, dcol)[mask]
        hist_delta(ax, values, "all DP2 COSMOS", "tab:blue", (args.x_min, args.x_max))
        ax.axvline(0.0, color="black", lw=1.8, ls=":", label="0")
        ax.axvline(0.03, color="gray", lw=1.4, ls="--", label="0.03")
        ax.set_title(f"{band}-band")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel(f"{band} psf_mag - cmodel_mag")
        ax.set_ylabel("Normalized density")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(
        f"DP2 COSMOS PSF-CModel magnitude difference by band; {args.mag_min:g} < CModel mag < {args.mag_max:g}\n"
        "delta = psf_mag - cmodel_mag"
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_mag_bin_hist(df: pd.DataFrame, args: argparse.Namespace) -> Path:
    out = args.plot_dir / "dp2_cosmos_psf_minus_cmodel_delta_hist_by_mag_bins.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(BANDS), len(MAG_BINS), figsize=(16, 18), sharex=True)
    for row, band in enumerate(BANDS):
        dcol = delta_col(df, band)
        for col, (label, lo, hi) in enumerate(MAG_BINS):
            ax = axes[row, col]
            if dcol is None:
                ax.set_title(f"{band}: missing")
                continue
            mask = mag_mask(df, band, lo, hi)
            values = finite_values(df, dcol)[mask]
            hist_delta(ax, values, "all", "tab:blue", (args.x_min, args.x_max))
            ax.axvline(0.0, color="black", lw=1.5, ls=":")
            ax.axvline(0.03, color="gray", lw=1.2, ls="--")
            med = np.nanmedian(values) if np.isfinite(values).any() else np.nan
            ax.set_title(f"{band}, {label}\nN={int(mask.sum()):,}, med={med:.4f}")
            ax.grid(True, alpha=0.2)
            if row == len(BANDS) - 1:
                ax.set_xlabel("psf_mag - cmodel_mag")
            if col == 0:
                ax.set_ylabel("Normalized density")
    fig.suptitle("DP2 COSMOS delta = psf_mag - cmodel_mag by band and magnitude bin")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_external_label_hist(df: pd.DataFrame, args: argparse.Namespace) -> Path | None:
    if "truth_binary" not in df.columns:
        return None
    out = args.plot_dir / "dp2_cosmos_psf_minus_cmodel_delta_hist_cosmos2020_labels_all_bands.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    truth = pd.to_numeric(df["truth_binary"], errors="coerce")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, band in zip(axes.ravel(), BANDS):
        dcol = delta_col(df, band)
        if dcol is None:
            ax.set_title(f"{band}: missing delta column")
            continue
        mask = base_mask(df, band, args.mag_min, args.mag_max)
        delta = finite_values(df, dcol)
        star = mask & truth.eq(1)
        galaxy = mask & truth.eq(0)
        hist_delta(ax, delta[galaxy], "COSMOS2020 galaxy", "tab:blue", (args.x_min, args.x_max), lw=2.0)
        hist_delta(ax, delta[star], "COSMOS2020 star", "tab:red", (args.x_min, args.x_max), lw=2.4)
        ax.axvline(0.0, color="black", lw=1.8, ls=":", label="0")
        ax.axvline(0.03, color="gray", lw=1.4, ls="--", label="0.03")
        ax.set_title(f"{band}-band")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel(f"{band} psf_mag - cmodel_mag")
        ax.set_ylabel("Normalized density")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)
    fig.suptitle(
        f"DP2 COSMOS PSF-CModel difference with COSMOS2020 external labels; "
        f"{args.mag_min:g} < CModel mag < {args.mag_max:g}\n"
        "delta = psf_mag - cmodel_mag; external labels are validation labels"
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.92])
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_median_by_band(summary: pd.DataFrame, args: argparse.Namespace) -> Path:
    out = args.plot_dir / "dp2_cosmos_psf_minus_cmodel_delta_median_by_band.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    main = summary[(summary["mag_bin"] == f"{args.mag_min:g}<mag<{args.mag_max:g}")].copy()
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(BANDS))
    width = 0.24
    sample_order = [
        ("all_dp2", "All DP2 COSMOS", "tab:blue"),
        ("cosmos2020_star", "COSMOS2020 star", "tab:red"),
        ("cosmos2020_galaxy", "COSMOS2020 galaxy", "tab:cyan"),
    ]
    plotted = False
    for i, (sample, label, color) in enumerate(sample_order):
        sub = main[main["sample"] == sample].set_index("band").reindex(BANDS)
        if sub["median_delta"].notna().any():
            ax.bar(x + (i - 1) * width, sub["median_delta"].to_numpy(), width=width, label=label, color=color)
            plotted = True
    ax.axhline(0.0, color="black", lw=1.4, ls=":")
    ax.axhline(0.03, color="gray", lw=1.2, ls="--", label="0.03")
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_ylabel("median(psf_mag - cmodel_mag)")
    ax.set_xlabel("Band")
    ax.set_title(f"DP2 COSMOS PSF-CModel median by band; {args.mag_min:g} < CModel mag < {args.mag_max:g}")
    ax.grid(True, axis="y", alpha=0.25)
    if plotted:
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def write_markdown(summary: pd.DataFrame, plots: list[Path], args: argparse.Namespace) -> Path:
    out = args.result_dir / "dp2_cosmos_psf_model_centering_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    main = summary[(summary["sample"] == "all_dp2") & (summary["mag_bin"] == f"{args.mag_min:g}<mag<{args.mag_max:g}")]
    lines = [
        "# DP2 COSMOS PSF-CModel Centering Check",
        "",
        "Quantity: `delta = psf_mag - cmodel_mag`.",
        "",
        f"Main plotted range: `{args.mag_min:g} < CModel mag < {args.mag_max:g}`.",
        "",
        "A median close to zero means PSF and CModel magnitudes agree for the selected sample. "
        "Positive delta means PSF magnitude is fainter than CModel magnitude; negative delta means PSF magnitude is brighter.",
        "",
        "## All DP2 COSMOS Median Delta",
        "",
        "| band | N | median delta | mean delta | MAD sigma | frac |delta| < 0.03 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in main.sort_values("band").iterrows():
        lines.append(
            f"| {row['band']} | {int(row['n']):,} | {row['median_delta']:.5f} | "
            f"{row['mean_delta']:.5f} | {row['mad_sigma_delta']:.5f} | "
            f"{row['fraction_abs_delta_lt_0p03']:.3f} |"
        )
    lines += ["", "## Generated Plots", ""]
    for plot in plots:
        lines.append(f"- `{plot.relative_to(REPO_ROOT)}`")
    lines += [
        "",
        "Notes:",
        "- COSMOS2020 labels, where used, are external validation labels.",
        "- These plots do not compare DP1 and DP2; they only check DP2 COSMOS centering.",
    ]
    out.write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    args = parse_args()
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.result_dir.mkdir(parents=True, exist_ok=True)

    table = read_table(args.table)
    rows: list[dict[str, object]] = []
    mag_all_label = f"{args.mag_min:g}<mag<{args.mag_max:g}"

    for band in BANDS:
        summarize_sample(rows, table, "all_dp2", band, mag_all_label, base_mask(table, band, args.mag_min, args.mag_max))
        for label, lo, hi in MAG_BINS:
            summarize_sample(rows, table, "all_dp2", band, label, mag_mask(table, band, lo, hi))

    matched = None
    if args.matched.exists():
        matched = read_table(args.matched)
        if "truth_binary" in matched.columns:
            truth = pd.to_numeric(matched["truth_binary"], errors="coerce")
            for band in BANDS:
                base = base_mask(matched, band, args.mag_min, args.mag_max)
                summarize_sample(rows, matched, "cosmos2020_star", band, mag_all_label, base & truth.eq(1))
                summarize_sample(rows, matched, "cosmos2020_galaxy", band, mag_all_label, base & truth.eq(0))
                for label, lo, hi in MAG_BINS:
                    mb = mag_mask(matched, band, lo, hi)
                    summarize_sample(rows, matched, "cosmos2020_star", band, label, mb & truth.eq(1))
                    summarize_sample(rows, matched, "cosmos2020_galaxy", band, label, mb & truth.eq(0))

    summary = pd.DataFrame(rows)
    summary_path = args.result_dir / "dp2_cosmos_psf_model_centering_summary.csv"
    summary.to_csv(summary_path, index=False)

    plots: list[Path] = []
    plots.append(plot_all_band_hist(table, args))
    plots.append(plot_mag_bin_hist(table, args))
    if matched is not None:
        label_plot = plot_external_label_hist(matched, args)
        if label_plot is not None:
            plots.append(label_plot)
    plots.append(plot_median_by_band(summary, args))
    md_path = write_markdown(summary, plots, args)

    print(f"Wrote summary CSV: {summary_path}")
    print(f"Wrote summary markdown: {md_path}")
    print("Generated plots:")
    for plot in plots:
        print(f"  {plot}")


if __name__ == "__main__":
    main()
