"""Small helpers to attach external labels to the ECDFS DP1 notebook workflow.

The intent is minimal notebook disruption:

1. keep the existing Rubin-side feature construction and pS model,
2. load the matched external-label table already produced elsewhere,
3. attach labels to the Rubin ECDFS dataframe by objectId,
4. make simple comparison plots against those labels.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_external_label_table(
    matched_path: str | Path,
    matched_object_col: str = "dp1_objectId",
    label_col: str = "hst_label",
    label_source_col: str = "hst_label_source",
    sep_col: str = "match_sep_arcsec",
) -> pd.DataFrame:
    """Load a matched external-label table and keep the best row per Rubin object.

    The clean ellipse matched sample can still contain duplicate Rubin object IDs if
    more than one external source lands on the same DP1 source. For notebook use we
    keep the smallest-separation match per DP1 object.
    """

    matched_path = Path(matched_path)
    df = pd.read_csv(matched_path)

    required = [matched_object_col, label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Matched label table is missing required columns: {missing}")

    keep = [matched_object_col, label_col]
    for col in [label_source_col, sep_col]:
        if col in df.columns:
            keep.append(col)

    label_df = df[keep].copy()

    if sep_col in label_df.columns:
        label_df = label_df.sort_values([matched_object_col, sep_col], kind="mergesort")
    else:
        label_df = label_df.sort_values([matched_object_col], kind="mergesort")

    label_df = label_df.drop_duplicates(subset=[matched_object_col], keep="first").reset_index(drop=True)
    label_df = label_df.rename(
        columns={
            matched_object_col: "objectId",
            label_col: "external_label",
            label_source_col: "external_label_source",
        }
    )
    return label_df


def attach_external_labels(
    rubin_df: pd.DataFrame,
    matched_path: str | Path,
    object_col: str = "objectId",
) -> pd.DataFrame:
    """Attach external labels to a Rubin dataframe by objectId."""

    if object_col not in rubin_df.columns:
        raise ValueError(f"Rubin dataframe is missing object id column: {object_col}")

    labels = load_external_label_table(matched_path)
    merged = rubin_df.merge(labels, on=object_col, how="left", validate="one_to_one")
    return merged


def summarize_external_labels(df: pd.DataFrame, label_col: str = "external_label") -> pd.DataFrame:
    """Simple label-count summary for notebook display."""

    if label_col not in df.columns:
        raise ValueError(f"Missing label column: {label_col}")

    counts = (
        df[label_col]
        .fillna("unmatched")
        .value_counts(dropna=False)
        .rename_axis(label_col)
        .reset_index(name="count")
    )
    return counts


def _label_subsets(df: pd.DataFrame, label_col: str = "external_label") -> dict[str, pd.DataFrame]:
    subsets = {}
    for label in ["star", "galaxy"]:
        subsets[label] = df[df[label_col] == label].copy()
    return subsets


def plot_diff_vs_mag_with_labels(
    df: pd.DataFrame,
    band: str,
    label_col: str = "external_label",
    file_path: str | Path = "../plots/",
    name: str | None = None,
    xlim: tuple[float, float] = (-0.1, 1.5),
    ylim: tuple[float, float] = (15, 25),
) -> None:
    """Scatter of Rubin PSF-CModel morphology against Rubin CModel magnitude."""

    diff_col = f"{band}_diff"
    mag_col = f"{band}_modelFlux_mag"
    required = [diff_col, mag_col, label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for plot_diff_vs_mag_with_labels: {missing}")

    use = df[np.isfinite(df[diff_col]) & np.isfinite(df[mag_col]) & df[label_col].isin(["star", "galaxy"])].copy()
    subsets = _label_subsets(use, label_col=label_col)

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    if len(use):
        ax.hexbin(
            use[diff_col],
            use[mag_col],
            gridsize=160,
            bins="log",
            mincnt=1,
            cmap="Greys",
            alpha=0.35,
        )

    styles = {
        "galaxy": dict(c="tab:blue", s=10, alpha=0.35),
        "star": dict(c="black", s=18, alpha=0.8),
    }
    for label in ["galaxy", "star"]:
        sub = subsets[label]
        if len(sub) == 0:
            continue
        ax.scatter(sub[diff_col], sub[mag_col], label=label, **styles[label])

    ax.set_xlabel(f"{band}_diff (Rubin PSF-CModel)")
    ax.set_ylabel(f"{band}_modelFlux_mag (Rubin CModel mag)")
    ax.set_title(f"ECDFS Rubin {band}-band morphology with external labels")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()

    if name is not None:
        out = Path(file_path) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
    plt.show()


def plot_score_hist_by_label(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "external_label",
    bins: np.ndarray | None = None,
    file_path: str | Path = "../plots/",
    name: str | None = None,
    source: str | None = None,
) -> None:
    """Histogram of a Rubin-side score split by external labels."""

    required = [score_col, label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for plot_score_hist_by_label: {missing}")

    use = df[np.isfinite(df[score_col]) & df[label_col].isin(["star", "galaxy"])].copy()
    if bins is None:
        bins = np.linspace(0, 1, 51)

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    for label, color, ls in [("star", "red", "--"), ("galaxy", "tab:blue", "-")]:
        sub = use.loc[use[label_col] == label, score_col]
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=bins, histtype="step", density=True, lw=3.0, ls=ls, label=label, color=color)

    ax.set_xlabel(score_col)
    ax.set_ylabel("Density")
    source_suffix = f" ({source})" if source else ""
    ax.set_title(f"{score_col} split by external labels{source_suffix}")
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()

    if name is not None:
        out = Path(file_path) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
    plt.show()


def plot_score_vs_mag_by_label(
    df: pd.DataFrame,
    score_col: str,
    mag_col: str,
    label_col: str = "external_label",
    file_path: str | Path = "../plots/",
    name: str | None = None,
) -> None:
    """Scatter of Rubin-side score against Rubin magnitude, split by labels."""

    required = [score_col, mag_col, label_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for plot_score_vs_mag_by_label: {missing}")

    use = df[np.isfinite(df[score_col]) & np.isfinite(df[mag_col]) & df[label_col].isin(["star", "galaxy"])].copy()

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for label, color, alpha, size in [("galaxy", "tab:blue", 0.30, 10), ("star", "black", 0.80, 18)]:
        sub = use[use[label_col] == label]
        if len(sub) == 0:
            continue
        ax.scatter(sub[mag_col], sub[score_col], c=color, alpha=alpha, s=size, label=label)

    ax.set_xlabel(mag_col)
    ax.set_ylabel(score_col)
    ax.set_title(f"{score_col} vs {mag_col} by external label")
    ax.invert_xaxis()
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()

    if name is not None:
        out = Path(file_path) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
    plt.show()


def _score_col_for_band(band: str, score_prefix: str = "pS_", score_suffix: str | None = None) -> str:
    if score_suffix is not None:
        return f"{band}{score_suffix}"
    return f"{score_prefix}{band}"


def build_multiband_score_long_table(
    df: pd.DataFrame,
    bands: list[str] | tuple[str, ...],
    score_prefix: str = "pS_",
    score_suffix: str | None = None,
    label_col: str = "external_label",
) -> pd.DataFrame:
    """Return a long-form table of score values across multiple bands."""

    records: list[pd.DataFrame] = []
    for band in bands:
        score_col = _score_col_for_band(band, score_prefix=score_prefix, score_suffix=score_suffix)
        if score_col not in df.columns:
            continue
        use = df[
            np.isfinite(df[score_col]) & df[label_col].isin(["star", "galaxy"])
        ][[label_col, score_col]].copy()
        if len(use) == 0:
            continue
        use = use.rename(columns={score_col: "score", label_col: "label"})
        use["band"] = band
        use["score_col"] = score_col
        records.append(use)

    if not records:
        return pd.DataFrame(columns=["label", "score", "band", "score_col"])
    return pd.concat(records, ignore_index=True)


def build_multiband_score_mag_long_table(
    df: pd.DataFrame,
    bands: list[str] | tuple[str, ...],
    score_prefix: str = "pS_",
    score_suffix: str | None = None,
    label_col: str = "external_label",
) -> pd.DataFrame:
    """Return a long-form table of score and magnitude values across multiple bands."""

    records: list[pd.DataFrame] = []
    for band in bands:
        score_col = _score_col_for_band(band, score_prefix=score_prefix, score_suffix=score_suffix)
        mag_col = f"{band}_modelFlux_mag"
        if score_col not in df.columns or mag_col not in df.columns:
            continue
        use = df[
            np.isfinite(df[score_col]) & np.isfinite(df[mag_col]) & df[label_col].isin(["star", "galaxy"])
        ][[label_col, score_col, mag_col]].copy()
        if len(use) == 0:
            continue
        use = use.rename(columns={score_col: "score", mag_col: "mag", label_col: "label"})
        use["band"] = band
        use["score_col"] = score_col
        use["mag_col"] = mag_col
        records.append(use)

    if not records:
        return pd.DataFrame(columns=["label", "score", "mag", "band", "score_col", "mag_col"])
    return pd.concat(records, ignore_index=True)


def summarize_multiband_scores(
    df: pd.DataFrame,
    bands: list[str] | tuple[str, ...],
    score_prefix: str = "pS_",
    score_suffix: str | None = None,
    label_col: str = "external_label",
) -> pd.DataFrame:
    """Summarize score distributions by band and label."""

    long_df = build_multiband_score_long_table(
        df,
        bands,
        score_prefix=score_prefix,
        score_suffix=score_suffix,
        label_col=label_col,
    )
    if len(long_df) == 0:
        return pd.DataFrame(
            columns=["band", "label", "count", "mean", "median", "std", "p16", "p84"]
        )

    summary = (
        long_df.groupby(["band", "label"])["score"]
        .agg(
            count="size",
            mean="mean",
            median="median",
            std="std",
            p16=lambda x: np.nanpercentile(x, 16),
            p84=lambda x: np.nanpercentile(x, 84),
        )
        .reset_index()
    )
    return summary


def plot_multiband_score_hist_grid(
    df: pd.DataFrame,
    bands: list[str] | tuple[str, ...],
    score_prefix: str = "pS_",
    score_suffix: str | None = None,
    label_col: str = "external_label",
    bins: np.ndarray | None = None,
    file_path: str | Path = "../plots/",
    name: str | None = None,
    score_label: str = "pS",
) -> None:
    """Plot a 2x3 histogram grid for a score evaluated in multiple bands."""

    if bins is None:
        bins = np.linspace(0, 1, 51)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, band in zip(axes, bands):
        score_col = _score_col_for_band(band, score_prefix=score_prefix, score_suffix=score_suffix)
        if score_col not in df.columns:
            ax.set_visible(False)
            continue

        use = df[np.isfinite(df[score_col]) & df[label_col].isin(["star", "galaxy"])].copy()
        for label, color in [("star", "black"), ("galaxy", "tab:blue")]:
            sub = use.loc[use[label_col] == label, score_col]
            if len(sub) == 0:
                continue
            ax.hist(sub, bins=bins, histtype="step", density=True, lw=1.8, label=label, color=color)

        ax.set_title(f"{band}-band")
        ax.grid(True, alpha=0.2)
        ax.set_xlabel(score_col)
        ax.set_ylabel("Density")

    for ax in axes[len(bands):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"ECDFS Rubin {score_label} split by external labels across ugrizy", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if name is not None:
        out = Path(file_path) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
    plt.show()


def plot_multiband_score_summary(
    df: pd.DataFrame,
    bands: list[str] | tuple[str, ...],
    score_prefix: str = "pS_",
    score_suffix: str | None = None,
    label_col: str = "external_label",
    file_path: str | Path = "../plots/",
    name: str | None = None,
    score_label: str = "pS",
) -> pd.DataFrame:
    """Plot an overall band-by-band summary using medians and 16-84 percentiles."""

    summary = summarize_multiband_scores(
        df,
        bands,
        score_prefix=score_prefix,
        score_suffix=score_suffix,
        label_col=label_col,
    )
    if len(summary) == 0:
        raise ValueError("No multiband score data available for summary plot.")

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(bands), dtype=float)
    offsets = {"star": -0.06, "galaxy": 0.06}
    styles = {
        "star": dict(color="black", marker="o"),
        "galaxy": dict(color="tab:blue", marker="s"),
    }

    for label in ["star", "galaxy"]:
        sub = summary[summary["label"] == label].set_index("band").reindex(bands)
        if sub["count"].fillna(0).sum() == 0:
            continue
        y = sub["median"].to_numpy(dtype=float)
        yerr = np.vstack(
            [
                y - sub["p16"].to_numpy(dtype=float),
                sub["p84"].to_numpy(dtype=float) - y,
            ]
        )
        ax.errorbar(
            x + offsets[label],
            y,
            yerr=yerr,
            capsize=3,
            lw=1.6,
            ms=6,
            label=label,
            **styles[label],
        )

    ax.set_xticks(x, bands)
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel(f"{score_label} median with 16-84 percentile range")
    ax.set_xlabel("Band")
    ax.set_title(f"ECDFS Rubin {score_label} summary across ugrizy")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    plt.tight_layout()

    if name is not None:
        out = Path(file_path) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
    plt.show()
    return summary


def plot_pooled_multiband_score_summary(
    df: pd.DataFrame,
    bands: list[str] | tuple[str, ...],
    score_prefix: str = "pS_",
    score_suffix: str | None = None,
    label_col: str = "external_label",
    bins: np.ndarray | None = None,
    file_path: str | Path = "../plots/",
    name: str | None = None,
    score_label: str = "pS",
) -> pd.DataFrame:
    """Plot a pooled summary using all available band-score pairs together."""

    long_df = build_multiband_score_long_table(
        df,
        bands,
        score_prefix=score_prefix,
        score_suffix=score_suffix,
        label_col=label_col,
    )
    if len(long_df) == 0:
        raise ValueError("No multiband score data available for pooled summary plot.")

    if bins is None:
        bins = np.linspace(0, 1, 51)

    summary = (
        long_df.groupby("label")["score"]
        .agg(
            count="size",
            mean="mean",
            median="median",
            std="std",
            p16=lambda x: np.nanpercentile(x, 16),
            p84=lambda x: np.nanpercentile(x, 84),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax_hist, ax_box = axes

    for label, color in [("star", "black"), ("galaxy", "tab:blue")]:
        sub = long_df.loc[long_df["label"] == label, "score"]
        if len(sub) == 0:
            continue
        ax_hist.hist(sub, bins=bins, histtype="step", density=True, lw=1.8, label=label, color=color)
        ax_hist.axvline(np.nanmedian(sub), color=color, lw=1.2, alpha=0.8, ls="--")

    ax_hist.set_xlabel(score_label)
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("All available bands pooled")
    ax_hist.grid(True, alpha=0.2)
    ax_hist.legend()

    box_data = [long_df.loc[long_df["label"] == label, "score"].to_numpy() for label in ["star", "galaxy"]]
    ax_box.boxplot(
        box_data,
        tick_labels=["star", "galaxy"],
        patch_artist=True,
        boxprops=dict(facecolor="white", edgecolor="0.3"),
        medianprops=dict(color="tab:red", lw=1.5),
    )

    rng = np.random.default_rng(12345)
    for xpos, label, color in [(1, "star", "black"), (2, "galaxy", "tab:blue")]:
        sub = long_df.loc[long_df["label"] == label, "score"].to_numpy()
        if len(sub) == 0:
            continue
        n_show = min(len(sub), 1200)
        idx = rng.choice(len(sub), size=n_show, replace=False)
        jitter = rng.normal(scale=0.05, size=n_show)
        ax_box.scatter(np.full(n_show, xpos) + jitter, sub[idx], s=8, c=color, alpha=0.12)

    ax_box.set_ylabel(score_label)
    ax_box.set_ylim(-0.02, 1.02)
    ax_box.set_title("Pooled score distribution")
    ax_box.grid(True, axis="y", alpha=0.2)

    fig.suptitle(f"ECDFS Rubin {score_label} summary pooled over ugrizy", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if name is not None:
        out = Path(file_path) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
    plt.show()
    return summary


def plot_multiband_score_vs_mag_grid(
    df: pd.DataFrame,
    bands: list[str] | tuple[str, ...],
    score_prefix: str = "pS_",
    score_suffix: str | None = None,
    label_col: str = "external_label",
    file_path: str | Path = "../plots/",
    name: str | None = None,
    score_label: str = "pS",
) -> None:
    """Plot a 2x3 grid of score-vs-magnitude panels across multiple bands."""

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), sharey=True)
    axes = axes.ravel()

    for ax, band in zip(axes, bands):
        score_col = _score_col_for_band(band, score_prefix=score_prefix, score_suffix=score_suffix)
        mag_col = f"{band}_modelFlux_mag"
        if score_col not in df.columns or mag_col not in df.columns:
            ax.set_visible(False)
            continue

        use = df[
            np.isfinite(df[score_col]) & np.isfinite(df[mag_col]) & df[label_col].isin(["star", "galaxy"])
        ].copy()

        for label, color, alpha, size in [("galaxy", "tab:blue", 0.30, 10), ("star", "black", 0.80, 18)]:
            sub = use[use[label_col] == label]
            if len(sub) == 0:
                continue
            ax.scatter(sub[mag_col], sub[score_col], c=color, alpha=alpha, s=size, label=label)

        ax.set_title(f"{band}-band")
        ax.set_xlabel(f"{band}_modelFlux_mag")
        ax.set_ylabel(score_col)
        ax.invert_xaxis()
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.2)

    for ax in axes[len(bands):]:
        ax.set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(f"ECDFS Rubin {score_label} vs magnitude across ugrizy", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    if name is not None:
        out = Path(file_path) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
    plt.show()


def plot_pooled_multiband_score_vs_mag(
    df: pd.DataFrame,
    bands: list[str] | tuple[str, ...],
    score_prefix: str = "pS_",
    score_suffix: str | None = None,
    label_col: str = "external_label",
    file_path: str | Path = "../plots/",
    name: str | None = None,
    score_label: str = "pS",
) -> pd.DataFrame:
    """Plot all band score-magnitude pairs together, split by external label."""

    long_df = build_multiband_score_mag_long_table(
        df,
        bands,
        score_prefix=score_prefix,
        score_suffix=score_suffix,
        label_col=label_col,
    )
    if len(long_df) == 0:
        raise ValueError("No multiband score/magnitude data available for pooled plot.")

    band_colors = {
        "u": "tab:purple",
        "g": "tab:green",
        "r": "tab:red",
        "i": "tab:orange",
        "z": "tab:blue",
        "y": "tab:brown",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)
    for ax, label in zip(axes, ["star", "galaxy"]):
        sub = long_df[long_df["label"] == label]
        for band in bands:
            band_sub = sub[sub["band"] == band]
            if len(band_sub) == 0:
                continue
            ax.scatter(
                band_sub["mag"],
                band_sub["score"],
                s=10,
                alpha=0.28,
                c=band_colors.get(band, "0.4"),
                label=band,
            )
        ax.set_title(f"{label}: all available bands pooled")
        ax.set_xlabel("modelFlux_mag")
        ax.set_ylabel(score_label)
        ax.set_ylim(-0.02, 1.02)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        dedup = dict(zip(labels, handles))
        fig.legend(dedup.values(), dedup.keys(), loc="upper center", ncol=min(len(dedup), 6), frameon=False)
    fig.suptitle(f"ECDFS Rubin {score_label} vs magnitude pooled over ugrizy", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    if name is not None:
        out = Path(file_path) / name
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
    plt.show()
    return long_df
