#!/usr/bin/env python3
"""Compare standardized DP2 pS output files against external labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dp2_external_validation import (
    BANDS,
    dataframe_to_markdown,
    ensure_dir,
    infer_truth_binary,
    metrics_vs_mag,
    metrics_from_counts,
    confusion_counts,
    plot_purity_completeness,
    read_table,
    threshold_scan,
)


def savefig(fig, path: Path) -> None:
    ensure_dir(path.parent)
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def plot_overlay_curves(scans: pd.DataFrame, outdir: Path, field: str) -> None:
    import matplotlib.pyplot as plt

    if scans.empty:
        return
    for band in BANDS:
        part_band = scans[scans["band"] == band]
        if part_band.empty:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True)
        for version, part in part_band.groupby("version"):
            axes[0].plot(part["star_completeness"], part["star_purity"], lw=1.4, label=version)
            axes[1].plot(part["galaxy_completeness"], part["galaxy_purity"], lw=1.4, label=version)
        axes[0].set_title("star")
        axes[1].set_title("galaxy")
        for ax in axes:
            ax.set_xlabel("completeness")
            ax.set_ylabel("purity")
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"DP2 {field}: pS version purity/completeness, {band}-band")
        fig.tight_layout()
        savefig(fig, outdir / f"purity_completeness_{band}")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), sharex=True, sharey=True)
        for version, part in part_band.groupby("version"):
            axes[0].plot(part["star_completeness"], part["star_contamination"], lw=1.4, label=version)
            axes[1].plot(part["galaxy_completeness"], part["galaxy_contamination"], lw=1.4, label=version)
        axes[0].set_title("star")
        axes[1].set_title("galaxy")
        for ax in axes:
            ax.set_xlabel("completeness")
            ax.set_ylabel("contamination")
            ax.set_xlim(-0.03, 1.03)
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=8)
        fig.suptitle(f"DP2 {field}: pS version contamination/completeness, {band}-band")
        fig.tight_layout()
        savefig(fig, outdir / f"contamination_completeness_{band}")
        plt.close(fig)


def plot_metrics_vs_mag_overlay(mag_metrics: pd.DataFrame, outdir: Path, field: str) -> None:
    import matplotlib.pyplot as plt

    if mag_metrics.empty:
        return
    for band in BANDS:
        part_band = mag_metrics[mag_metrics["band"] == band]
        if part_band.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), sharex=True, sharey=True)
        for version, part in part_band.groupby("version"):
            axes[0].plot(part["mag_center"], part["star_completeness"], marker="o", ms=2.5, lw=1.2, label=f"{version} comp")
            axes[0].plot(part["mag_center"], part["star_contamination"], ls="--", lw=1.2, label=f"{version} cont")
            axes[1].plot(part["mag_center"], part["galaxy_completeness"], marker="o", ms=2.5, lw=1.2, label=f"{version} comp")
            axes[1].plot(part["mag_center"], part["galaxy_contamination"], ls="--", lw=1.2, label=f"{version} cont")
        axes[0].set_title("star")
        axes[1].set_title("galaxy")
        for ax in axes:
            ax.set_xlabel("i magnitude bin center")
            ax.set_ylabel("metric")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.2)
            ax.legend(fontsize=6, ncol=2)
        fig.suptitle(f"DP2 {field}: pS version metrics vs i magnitude, {band}-band")
        fig.tight_layout()
        savefig(fig, outdir / f"metrics_vs_mag_{band}")
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", choices=["ECDFS", "COSMOS"], default="ECDFS")
    parser.add_argument("--matched", type=Path, required=True)
    parser.add_argument("--versions", default="v1,v2,v3,v4,v5,v6")
    parser.add_argument("--ps-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results" / "dp2_ps_version_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = ensure_dir(args.output_dir)
    matched = read_table(args.matched)
    truth = infer_truth_binary(matched)
    rows = []
    scan_rows = []
    mag_rows = []
    missing = []

    prefix = f"dp2_{args.field.lower()}_ps_"
    for version in [v.strip() for v in args.versions.split(",") if v.strip()]:
        ps_path = args.ps_dir / f"{prefix}{version}.parquet"
        if not ps_path.exists():
            missing.append(str(ps_path))
            continue
        ps = read_table(ps_path)
        if "object_id" not in ps.columns:
            missing.append(f"{ps_path} (missing object_id)")
            continue
        dup_count = int(ps["object_id"].duplicated().sum())
        if dup_count:
            print(f"Warning: {ps_path} has {dup_count:,} duplicate object_id rows; keeping the first row per object_id.")
            ps = ps.drop_duplicates(subset=["object_id"], keep="first")
        use = matched.merge(ps, on="object_id", how="inner", validate="one_to_one")
        truth_use = infer_truth_binary(use)
        for band in BANDS:
            col = f"pS_{band}"
            if col not in use.columns:
                continue
            score = pd.to_numeric(use[col], errors="coerce")
            valid = truth_use.isin([0, 1]) & np.isfinite(score)
            if not valid.any():
                continue
            counts = confusion_counts(truth_use[valid], score[valid] >= 0.5)
            rows.append({"version": version, "band": band, "score_col": col, **metrics_from_counts(counts)})
            scan = threshold_scan(use.loc[valid], col)
            scan["version"] = version
            scan["band"] = band
            scan_rows.append(scan)
            mag = metrics_vs_mag(use.loc[valid], score[valid] >= 0.5)
            if not mag.empty:
                mag["version"] = version
                mag["band"] = band
                mag_rows.append(mag)

    summary = pd.DataFrame(rows)
    scans = pd.concat(scan_rows, ignore_index=True) if scan_rows else pd.DataFrame()
    mag_metrics = pd.concat(mag_rows, ignore_index=True) if mag_rows else pd.DataFrame()
    summary.to_csv(outdir / "version_band_summary.csv", index=False)
    scans.to_csv(outdir / "threshold_metrics.csv", index=False)
    mag_metrics.to_csv(outdir / "metrics_vs_mag.csv", index=False)

    if not scans.empty:
        for band in BANDS:
            for version, part in scans[scans["band"] == band].groupby("version"):
                plot_purity_completeness(part, outdir / f"purity_completeness_{band}_{version}", f"{args.field} {version} {band}")
        plot_overlay_curves(scans, outdir, args.field)
    plot_metrics_vs_mag_overlay(mag_metrics, outdir, args.field)

    lines = [f"# DP2 pS version comparison: {args.field}\n\n"]
    if missing:
        lines.append("## Missing standardized pS files\n\n")
        for path in missing:
            lines.append(f"- `{path}`\n")
        lines.append("\nRun the standardized notebook pS-output cells before treating this comparison as complete.\n\n")
    if summary.empty:
        lines.append("No version metrics were produced because no usable standardized pS files were found.\n")
    else:
        best = summary.sort_values(["star_purity", "star_completeness"], ascending=False).head(10)
        lines.append("## Top rows by star purity/completeness\n\n")
        lines.append(dataframe_to_markdown(best))
        lines.append("\n")
        balanced = summary.assign(
            star_balanced=2 * summary["star_purity"] * summary["star_completeness"] / (summary["star_purity"] + summary["star_completeness"])
        ).sort_values("star_balanced", ascending=False).head(10)
        lines.append("\n## Top rows by star F1-like balance\n\n")
        lines.append(dataframe_to_markdown(balanced))
        lines.append("\n")
    (outdir / "version_recommendation.md").write_text("".join(lines))
    print(f"Wrote comparison outputs to {outdir}")


if __name__ == "__main__":
    main()
