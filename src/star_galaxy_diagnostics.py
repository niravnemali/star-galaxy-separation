#!/usr/bin/env python3
"""Generate Rubin star/galaxy diagnostics for the Rubin star/galaxy separation workflow.

This script works from an ECDFS-style matched catalog with external truth
labels, reuses the existing Rubin-side feature construction when possible, and
produces:

1. four-panel i vs g-i CMD diagnostics,
2. completeness/contamination vs i magnitude,
3. truth-based and method-vs-method comparisons,
4. a README and summary tables under results/star_galaxy_diagnostics/.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


DEFAULT_CATALOG = REPO_ROOT / "outputs" / "hst_dp1_matched_clean_ellipse.csv"
DEFAULT_DP1 = REPO_ROOT / "data" / "ECDFS.fits"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "star_galaxy_diagnostics"
DEFAULT_BUTLER_CATALOG = REPO_ROOT / "outputs" / "dp1_ecdfs_analysis_table_butler.parquet"
DEFAULT_TRUTH_CATALOG = REPO_ROOT / "outputs" / "hst_dp1_matched_clean_ellipse.csv"
DEFAULT_BUTLER_MATCHED = REPO_ROOT / "outputs" / "hst_dp1_matched_butler_analysis.parquet"
BANDS = ["u", "g", "r", "i", "z", "y"]

STAR_STRINGS = {"star", "stellar", "point", "pointsource", "point_source", "psf"}
GALAXY_STRINGS = {"galaxy", "gal", "extended", "resolved", "nonstellar", "non_stellar"}


@dataclass
class LabelResult:
    labels: pd.Series
    column: str
    description: str
    threshold: float | None = None


@dataclass
class ComparisonSpec:
    reference_labels: pd.Series
    predicted_labels: pd.Series
    reference_name: str
    predicted_name: str
    title_detail: str
    output_stub: str
    source_columns: dict[str, str]
    common_valid_mask: pd.Series | None = None


def nJy_to_mag(flux) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    out = np.full(flux.shape, np.nan, dtype=float)
    ok = np.isfinite(flux) & (flux > 0)
    out[ok] = -2.5 * np.log10((flux[ok] * 1e-9) / 3631.0)
    return out


def slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_")


def safe_divide(num: float, den: float) -> float:
    return float(num / den) if den else np.nan


def binomial_error(p: float, n: int) -> float:
    if not np.isfinite(p) or n <= 0:
        return np.nan
    return float(np.sqrt(p * (1.0 - p) / n))


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format for {path}; expected CSV or parquet.")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
    elif suffix in {".csv", ".txt"}:
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output table format for {path}; expected CSV or parquet.")


def _contains_all(name: str, tokens: list[str]) -> bool:
    lower = name.lower()
    return all(tok.lower() in lower for tok in tokens)


def _find_matches(df: pd.DataFrame, token_groups: list[list[str]], reject_tokens: list[str] | None = None) -> list[str]:
    reject_tokens = [tok.lower() for tok in (reject_tokens or [])]
    matches: list[str] = []
    for col in df.columns:
        lower = col.lower()
        if any(tok in lower for tok in reject_tokens):
            continue
        if any(_contains_all(col, tokens) for tokens in token_groups):
            matches.append(col)
    return matches


def _format_available_columns(df: pd.DataFrame, max_cols: int = 80) -> str:
    cols = list(df.columns)
    if len(cols) <= max_cols:
        return ", ".join(cols)
    head = ", ".join(cols[:max_cols])
    return f"{head}, ... ({len(cols)} total columns)"


def deduplicate_catalog(
    df: pd.DataFrame,
    object_col: str,
    sep_col: str | None = "match_sep_arcsec",
) -> pd.DataFrame:
    if object_col not in df.columns:
        return df.copy()

    out = df.copy()
    before = len(out)
    duplicated = int(out.duplicated(subset=[object_col]).sum())
    if duplicated == 0:
        print(f"No duplicate {object_col} rows found.")
        return out

    sort_cols = [object_col]
    if sep_col and sep_col in out.columns:
        sort_cols.append(sep_col)
    out = out.sort_values(sort_cols, kind="mergesort")
    out = out.drop_duplicates(subset=[object_col], keep="first").reset_index(drop=True)
    after = len(out)
    print(
        f"Deduplicated matched catalog by {object_col}: "
        f"{before} -> {after} rows (removed {before - after}, smallest separation kept)."
    )
    return out


def detect_object_id_column(df: pd.DataFrame) -> str:
    candidates = ["dp1_objectId", "objectId", "object_id", "id"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        "Could not find an object ID column. "
        f"Available columns include: {_format_available_columns(df)}"
    )


def detect_mag_column(df: pd.DataFrame, band: str, explicit: str | None = None) -> tuple[str, pd.Series]:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"Requested magnitude column '{explicit}' is missing. "
                f"Available columns: {_format_available_columns(df)}"
            )
        return explicit, pd.to_numeric(df[explicit], errors="coerce")

    priority = [
        f"dp1_{band}_free_cModelMag",
        f"{band}_modelFlux_mag",
        f"cmodel_mag_{band}",
        f"{band}_cmodel_mag",
        f"{band}_mag",
        f"mag_{band}",
        f"{band}Mag",
    ]
    for col in priority:
        if col in df.columns:
            return col, pd.to_numeric(df[col], errors="coerce")

    mag_matches = _find_matches(
        df,
        [
            [band, "cmodel", "mag"],
            [band, "modelflux", "mag"],
            [band, "model", "mag"],
            [band, "mag"],
        ],
        reject_tokens=["err", "psf", "flag"],
    )
    if mag_matches:
        col = mag_matches[0]
        return col, pd.to_numeric(df[col], errors="coerce")

    flux_priority = [
        f"dp1_{band}_free_cModelFlux",
        f"{band}_modelFlux",
        f"cmodel_flux_{band}",
        f"{band}_cmodel_flux",
    ]
    for col in flux_priority:
        if col in df.columns:
            return f"{col}_derived_mag", pd.Series(nJy_to_mag(df[col]), index=df.index)

    flux_matches = _find_matches(
        df,
        [
            [band, "cmodel", "flux"],
            [band, "modelflux"],
            [band, "model", "flux"],
        ],
        reject_tokens=["err", "flag"],
    )
    if flux_matches:
        col = flux_matches[0]
        return f"{col}_derived_mag", pd.Series(nJy_to_mag(df[col]), index=df.index)

    raise ValueError(
        f"Could not find a usable {band}-band magnitude column or flux column. "
        f"Available columns: {_format_available_columns(df)}"
    )


def _infer_binary_from_strings(values: pd.Series) -> pd.Series:
    lower = values.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=values.index, dtype=float)
    out[lower.isin(STAR_STRINGS)] = 1.0
    out[lower.isin(GALAXY_STRINGS)] = 0.0
    return out


def make_binary_labels(values: pd.Series, convention: str) -> pd.Series:
    out = pd.Series(np.nan, index=values.index, dtype=float)
    if convention == "star_strings":
        out = _infer_binary_from_strings(values)
    elif convention == "is_star_bool":
        out[:] = pd.Series(values).astype("boolean").astype(float)
    elif convention == "is_galaxy_bool":
        out[:] = 1.0 - pd.Series(values).astype("boolean").astype(float)
    elif convention == "numeric_one_is_star":
        numeric = pd.to_numeric(values, errors="coerce")
        ok = numeric.isin([0, 1])
        out[ok] = numeric[ok].astype(float)
    elif convention == "numeric_one_is_galaxy":
        numeric = pd.to_numeric(values, errors="coerce")
        ok = numeric.isin([0, 1])
        out[ok] = 1.0 - numeric[ok].astype(float)
    else:
        raise ValueError(f"Unsupported label convention: {convention}")
    return out


def infer_truth_labels(df: pd.DataFrame, column_name: str | None = None) -> LabelResult:
    candidates = [column_name] if column_name else [
        "hst_label",
        "external_label",
        "truth",
        "truth_label",
        "truth_type",
        "true_class",
        "object_type",
        "class",
        "is_star",
        "is_galaxy",
    ]

    for col in candidates:
        if col is None or col not in df.columns:
            continue
        series = df[col]
        lower_name = col.lower()

        if pd.api.types.is_bool_dtype(series):
            if "gal" in lower_name and "star" not in lower_name:
                labels = make_binary_labels(series, "is_galaxy_bool")
                return LabelResult(labels=labels, column=col, description="boolean is_galaxy convention")
            labels = make_binary_labels(series, "is_star_bool")
            return LabelResult(labels=labels, column=col, description="boolean is_star convention")

        string_labels = _infer_binary_from_strings(series)
        if int(string_labels.notna().sum()) > 0:
            return LabelResult(labels=string_labels, column=col, description="string star/galaxy labels")

        numeric = pd.to_numeric(series, errors="coerce")
        uniq = set(numeric.dropna().unique().tolist())
        if uniq and uniq.issubset({0, 1}):
            if "gal" in lower_name and "star" not in lower_name:
                labels = make_binary_labels(series, "numeric_one_is_galaxy")
                return LabelResult(labels=labels, column=col, description="numeric 1=galaxy, 0=star convention")
            labels = make_binary_labels(series, "numeric_one_is_star")
            return LabelResult(labels=labels, column=col, description="numeric 1=star, 0=galaxy convention")

    raise ValueError(
        "Could not infer truth labels. "
        f"Available columns: {_format_available_columns(df)}"
    )


def detect_explicit_ps_column(df: pd.DataFrame, explicit: str | None = None) -> str | None:
    if explicit is not None:
        if explicit not in df.columns:
            raise ValueError(
                f"Requested pS/probability column '{explicit}' is missing. "
                f"Available columns: {_format_available_columns(df)}"
            )
        return explicit

    priority = ["pS_r", "pS", "ps", "p_star", "prob_star", "star_probability", "Nirav_pS"]
    for col in priority:
        if col in df.columns:
            return col

    regexes = [
        re.compile(r"^ps$", re.IGNORECASE),
        re.compile(r"^ps_[ugrizy]$", re.IGNORECASE),
        re.compile(r"^psr$", re.IGNORECASE),
        re.compile(r"^p_s$", re.IGNORECASE),
        re.compile(r"^pstar$", re.IGNORECASE),
        re.compile(r"^p_star$", re.IGNORECASE),
        re.compile(r"^prob_star$", re.IGNORECASE),
        re.compile(r"^star_probability$", re.IGNORECASE),
        re.compile(r"^nirav_ps$", re.IGNORECASE),
        re.compile(r"^nirav_ps_[ugrizy]$", re.IGNORECASE),
    ]
    for col in df.columns:
        if any(rx.match(col) for rx in regexes):
            return col
    return None


def infer_pS_labels(
    df: pd.DataFrame,
    column_name: str | None = None,
    threshold: float | None = None,
) -> LabelResult:
    col = detect_explicit_ps_column(df, explicit=column_name)
    if col is None:
        raise ValueError(
            "Could not find a pS/probability column. "
            f"Available columns: {_format_available_columns(df)}"
        )
    thr = 0.5 if threshold is None else float(threshold)
    prob = pd.to_numeric(df[col], errors="coerce")
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    finite = np.isfinite(prob)
    labels.loc[finite] = (prob.loc[finite] >= thr).astype(float)
    return LabelResult(
        labels=labels,
        column=col,
        description=f"probability threshold: star if {col} >= {thr:.3f}",
        threshold=thr,
    )


def find_extendedness_columns(df: pd.DataFrame) -> list[tuple[str, str, str | None]]:
    results: list[tuple[str, str, str | None]] = []
    for band in BANDS:
        candidates = [
            f"dp1_{band}_extendedness",
            f"{band}_extendedness",
            f"extendedness_{band}",
            f"{band}Extendedness",
        ]
        col = next((c for c in candidates if c in df.columns), None)
        if col is None:
            matches = _find_matches(
                df,
                [[band, "extendedness"]],
                reject_tokens=["flag", "sizeextendedness"],
            )
            col = matches[0] if matches else None
        if col is None:
            continue

        flag = None
        flag_candidates = [
            f"dp1_{band}_extendedness_flag",
            f"{band}_extendedness_flag",
            f"extendedness_flag_{band}",
        ]
        flag = next((c for c in flag_candidates if c in df.columns), None)
        results.append((band, col, flag))
    return results


def infer_extendedness_labels(
    df: pd.DataFrame,
    column_name: str | None = None,
    threshold: float | None = None,
) -> LabelResult:
    if column_name is None:
        available = find_extendedness_columns(df)
        if not available:
            raise ValueError(
                "Could not find an extendedness column. "
                f"Available columns: {_format_available_columns(df)}"
            )
        _, col, flag_col = available[0]
    else:
        if column_name not in df.columns:
            raise ValueError(
                f"Requested extendedness column '{column_name}' is missing. "
                f"Available columns: {_format_available_columns(df)}"
            )
        col = column_name
        flag_candidates = [f"{column_name}_flag", column_name.replace("_extendedness", "_extendedness_flag")]
        for band in BANDS:
            if column_name == f"extendedness_{band}":
                flag_candidates.append(f"extendedness_flag_{band}")
            if column_name == f"{band}_extendedness":
                flag_candidates.append(f"{band}_extendedness_flag")
        flag_col = next((c for c in flag_candidates if c in df.columns), None)

    values = pd.to_numeric(df[col], errors="coerce")
    if flag_col is not None:
        flags = pd.to_numeric(df[flag_col], errors="coerce")
        values = values.where(flags.fillna(1).eq(0))

    finite_vals = values[np.isfinite(values)]
    uniq = set(np.unique(finite_vals))
    labels = pd.Series(np.nan, index=df.index, dtype=float)
    if uniq and uniq.issubset({0, 1}):
        labels.loc[np.isfinite(values)] = (values.loc[np.isfinite(values)] == 0).astype(float)
        description = "Rubin binary extendedness convention: 0=star, 1=galaxy"
        thr = None
    else:
        thr = 0.5 if threshold is None else float(threshold)
        labels.loc[np.isfinite(values)] = (values.loc[np.isfinite(values)] < thr).astype(float)
        description = f"continuous extendedness threshold: star if {col} < {thr:.3f}"

    return LabelResult(labels=labels, column=col, description=description, threshold=thr)


def make_confusion_masks(reference_label: pd.Series, predicted_label: pd.Series) -> tuple[dict[str, pd.Series], dict[str, int]]:
    ref = pd.to_numeric(reference_label, errors="coerce")
    pred = pd.to_numeric(predicted_label, errors="coerce")
    valid = ref.isin([0, 1]) & pred.isin([0, 1])
    ref_valid = ref.loc[valid].astype(int)
    pred_valid = pred.loc[valid].astype(int)

    masks = {
        "a_correct_stars": valid & (ref == 1) & (pred == 1),
        "b_correct_galaxies": valid & (ref == 0) & (pred == 0),
        "c_misclassified_stars": valid & (ref == 1) & (pred == 0),
        "d_misclassified_galaxies": valid & (ref == 0) & (pred == 1),
    }
    counts = {key[0]: int(mask.sum()) for key, mask in masks.items()}
    total = int(valid.sum())
    if sum(counts.values()) != total:
        raise RuntimeError(
            f"Confusion masks do not partition the valid sample: "
            f"a+b+c+d={sum(counts.values())}, total={total}"
        )
    return masks, counts


def build_mag_bins(magnitude: pd.Series, mag_bin_width: float | None, n_mag_bins: int | None) -> np.ndarray:
    finite = pd.to_numeric(magnitude, errors="coerce")
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        raise ValueError("Cannot define magnitude bins because no finite i magnitudes are available.")

    lo = float(np.floor(finite.min() * 2) / 2.0)
    hi = float(np.ceil(finite.max() * 2) / 2.0)
    if hi <= lo:
        hi = lo + 1.0

    if mag_bin_width is not None:
        step = float(mag_bin_width)
        edges = np.arange(lo, hi + step, step)
    else:
        n_bins = int(n_mag_bins or 20)
        edges = np.linspace(lo, hi, n_bins + 1)

    if len(edges) < 2:
        edges = np.array([lo, lo + 0.5])
    print(f"Magnitude binning: {len(edges) - 1} bins spanning i = {edges[0]:.2f} to {edges[-1]:.2f}")
    return edges


def histogram_contours(ax, x: np.ndarray, y: np.ndarray, color: str, bins: int = 60, linewidths: float = 1.2, alpha: float = 0.85) -> None:
    if len(x) < 20:
        return
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[finite]
    y = np.asarray(y)[finite]
    if len(x) < 20:
        return
    x_edges = np.linspace(np.nanpercentile(x, 0.5), np.nanpercentile(x, 99.5), bins + 1)
    y_edges = np.linspace(np.nanpercentile(y, 0.5), np.nanpercentile(y, 99.5), bins + 1)
    H, xe, ye = np.histogram2d(x, y, bins=[x_edges, y_edges])
    if not np.any(H > 0):
        return
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    positive = np.unique(H[H > 0])
    if len(positive) == 1:
        levels = positive
    else:
        q = np.quantile(positive, [0.45, 0.65, 0.82, 0.93])
        levels = np.unique(q[q > 0])
    if len(levels) == 0:
        return
    ax.contour(xc, yc, H.T, levels=levels, colors=color, linewidths=linewidths, alpha=alpha)


def choose_plot_mode(n_subsample: int, forced_mode: str) -> str:
    if forced_mode != "auto":
        return forced_mode
    return "scatter" if n_subsample <= 1500 else "contour"


def plot_cmd_four_panel(
    df: pd.DataFrame,
    color_col: str,
    mag_col: str,
    masks: dict[str, pd.Series],
    method_name: str,
    reference_name: str,
    output_path: Path,
    plot_mode: str = "auto",
    subtitle: str = "",
) -> None:
    use = df[np.isfinite(df[color_col]) & np.isfinite(df[mag_col])].copy()
    if len(use) == 0:
        raise ValueError(f"No finite CMD data available for {method_name} vs {reference_name}.")

    x = pd.to_numeric(use[color_col], errors="coerce").to_numpy()
    y = pd.to_numeric(use[mag_col], errors="coerce").to_numpy()
    x_lo, x_hi = np.nanpercentile(x, [0.5, 99.5])
    y_lo, y_hi = np.nanpercentile(y, [0.5, 99.5])
    x_pad = 0.08 * max(x_hi - x_lo, 0.2)
    y_pad = 0.05 * max(y_hi - y_lo, 0.5)
    xlim = (x_lo - x_pad, x_hi + x_pad)
    ylim = (y_lo - y_pad, y_hi + y_pad)

    if reference_name == "truth":
        panel_specs = [
            ("a_correct_stars", "Correct stars (a)", "black"),
            ("b_correct_galaxies", "Correct galaxies (b)", "tab:blue"),
            ("c_misclassified_stars", "Misclassified stars (c)", "tab:orange"),
            ("d_misclassified_galaxies", "Misclassified galaxies (d)", "tab:red"),
        ]
    else:
        panel_specs = [
            ("a_correct_stars", "Agreement on stars (a)", "black"),
            ("b_correct_galaxies", "Agreement on galaxies (b)", "tab:blue"),
            ("c_misclassified_stars", "Reference star / predicted galaxy (c)", "tab:orange"),
            ("d_misclassified_galaxies", "Reference galaxy / predicted star (d)", "tab:red"),
        ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.0), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, (mask_key, title, color) in zip(axes, panel_specs):
        background_contours_x = use[color_col].to_numpy(dtype=float)
        background_contours_y = use[mag_col].to_numpy(dtype=float)
        histogram_contours(ax, background_contours_x, background_contours_y, color="0.45", bins=72, linewidths=1.0, alpha=0.65)

        sub = use.loc[masks[mask_key].reindex(use.index, fill_value=False)]
        n_sub = len(sub)
        mode = choose_plot_mode(n_sub, plot_mode)

        if n_sub > 0:
            sx = sub[color_col].to_numpy(dtype=float)
            sy = sub[mag_col].to_numpy(dtype=float)
            if mode == "scatter":
                size = 10 if n_sub > 500 else 14
                alpha = 0.40 if n_sub > 500 else 0.75
                ax.scatter(sx, sy, s=size, c=color, alpha=alpha, linewidths=0)
            elif mode == "hexbin":
                ax.hexbin(sx, sy, gridsize=45, mincnt=1, cmap="viridis", linewidths=0.0)
            else:
                histogram_contours(ax, sx, sy, color=color, bins=55, linewidths=1.6, alpha=0.9)

        ax.set_title(f"{title}\nN = {n_sub}")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.18)
        ax.text(
            0.03,
            0.97,
            f"mode: {mode}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.65, edgecolor="none"),
        )

    fig.supxlabel("g - i")
    fig.supylabel("i")
    fig.suptitle(
        f"{method_name} vs {reference_name}\n{subtitle}".strip(),
        y=0.98,
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    for suffix in [".png", ".pdf"]:
        path = output_path.with_suffix(suffix)
        if not path.exists():
            raise RuntimeError(f"Failed to save plot: {path}")


def compute_metrics_vs_magnitude(
    reference_label: pd.Series,
    predicted_label: pd.Series,
    magnitude: pd.Series,
    bins: np.ndarray,
) -> pd.DataFrame:
    ref = pd.to_numeric(reference_label, errors="coerce")
    pred = pd.to_numeric(predicted_label, errors="coerce")
    mag = pd.to_numeric(magnitude, errors="coerce")
    valid = ref.isin([0, 1]) & pred.isin([0, 1]) & np.isfinite(mag)

    rows: list[dict[str, float]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        in_bin = valid & (mag >= lo) & (mag < hi)
        sub_ref = ref.loc[in_bin].astype(int)
        sub_pred = pred.loc[in_bin].astype(int)

        a = int(((sub_ref == 1) & (sub_pred == 1)).sum())
        b = int(((sub_ref == 0) & (sub_pred == 0)).sum())
        c = int(((sub_ref == 1) & (sub_pred == 0)).sum())
        d = int(((sub_ref == 0) & (sub_pred == 1)).sum())

        n_star_ref = a + c
        n_gal_ref = b + d
        n_star_pred = a + d
        n_gal_pred = b + c

        star_completeness = safe_divide(a, n_star_ref)
        star_contamination = safe_divide(d, n_star_pred)
        galaxy_completeness = safe_divide(b, n_gal_ref)
        galaxy_contamination = safe_divide(c, n_gal_pred)

        rows.append(
            {
                "mag_lo": float(lo),
                "mag_hi": float(hi),
                "mag_center": float(0.5 * (lo + hi)),
                "n_total": int(in_bin.sum()),
                "a_correct_stars": a,
                "b_correct_galaxies": b,
                "c_misclassified_stars": c,
                "d_misclassified_galaxies": d,
                "star_completeness": star_completeness,
                "star_completeness_err": binomial_error(star_completeness, n_star_ref),
                "star_contamination": star_contamination,
                "star_contamination_err": binomial_error(star_contamination, n_star_pred),
                "galaxy_completeness": galaxy_completeness,
                "galaxy_completeness_err": binomial_error(galaxy_completeness, n_gal_ref),
                "galaxy_contamination": galaxy_contamination,
                "galaxy_contamination_err": binomial_error(galaxy_contamination, n_gal_pred),
            }
        )
    return pd.DataFrame(rows)


def plot_metrics_vs_magnitude(
    metrics_df: pd.DataFrame,
    method_name: str,
    reference_name: str,
    output_path: Path,
    subtitle: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    x = metrics_df["mag_center"].to_numpy()

    series_specs = [
        ("star_completeness", "star_completeness_err", "Star completeness", "black", "o"),
        ("star_contamination", "star_contamination_err", "Star contamination", "tab:red", "s"),
        ("galaxy_completeness", "galaxy_completeness_err", "Galaxy completeness", "tab:blue", "^"),
        ("galaxy_contamination", "galaxy_contamination_err", "Galaxy contamination", "tab:orange", "D"),
    ]
    for col, err_col, label, color, marker in series_specs:
        y = metrics_df[col].to_numpy(dtype=float)
        yerr = metrics_df[err_col].to_numpy(dtype=float)
        ax.errorbar(x, y, yerr=yerr, color=color, marker=marker, lw=1.6, ms=5, capsize=2.5, label=label)

    ax.set_xlabel("i magnitude bin center")
    ax.set_ylabel("Metric value")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, ncol=2)
    ax.set_title(f"{method_name} vs {reference_name}\n{subtitle}".strip())
    fig.tight_layout()
    fig.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    for suffix in [".png", ".pdf"]:
        path = output_path.with_suffix(suffix)
        if not path.exists():
            raise RuntimeError(f"Failed to save plot: {path}")


def summarize_method(name: str, label_result: LabelResult) -> str:
    labels = label_result.labels
    n_star = int((labels == 1).sum())
    n_gal = int((labels == 0).sum())
    lines = [
        f"Method: {name}\n",
        f"  column: {label_result.column}\n",
        f"  convention: {label_result.description}\n",
        f"  predicted stars: {n_star}\n",
        f"  predicted galaxies: {n_gal}\n",
    ]
    return "".join(lines)


def overall_metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    a = counts["a"]
    b = counts["b"]
    c = counts["c"]
    d = counts["d"]
    return {
        "star_completeness": safe_divide(a, a + c),
        "star_contamination": safe_divide(d, a + d),
        "galaxy_completeness": safe_divide(b, b + d),
        "galaxy_contamination": safe_divide(c, b + c),
    }


def add_butler_diagnostic_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Add notebook-style aliases to the Butler prepared table.

    The Butler-prepared table keeps analysis-friendly names such as
    `cmodel_mag_r` and `psf_minus_cmodel_mag_r`.  Existing pS and diagnostics
    code expects the older notebook-style names `r_modelFlux_mag` and `r_diff`.
    This adapter adds aliases without removing or overwriting the original
    Butler-derived columns.
    """

    out = df.copy()
    core_aliases = {
        "object_id": "objectId",
        "ra": "coord_ra",
        "dec": "coord_dec",
    }
    for src, dst in core_aliases.items():
        if src in out.columns and dst not in out.columns:
            out[dst] = out[src]

    for band in BANDS:
        aliases = {
            f"cmodel_mag_{band}": f"{band}_modelFlux_mag",
            f"psf_mag_{band}": f"{band}_psfFlux_mag",
            f"psf_minus_cmodel_mag_{band}": f"{band}_diff",
            f"extendedness_{band}": f"{band}_extendedness",
            f"extendedness_flag_{band}": f"{band}_extendedness_flag",
        }
        for src, dst in aliases.items():
            if src in out.columns and dst not in out.columns:
                out[dst] = out[src]

        psf_flux = f"psf_flux_{band}"
        psf_err = f"psf_flux_err_{band}"
        relerr = f"{band}_psfRelErr"
        if relerr not in out.columns and psf_flux in out.columns and psf_err in out.columns:
            flux = pd.to_numeric(out[psf_flux], errors="coerce")
            err = pd.to_numeric(out[psf_err], errors="coerce")
            values = np.full(len(out), np.nan, dtype=float)
            ok = np.isfinite(flux) & np.isfinite(err) & (flux > 0)
            values[ok] = 1.09 * (err[ok] / flux[ok])
            out[relerr] = values
    return out


def load_truth_for_butler_join(truth_catalog: Path) -> pd.DataFrame:
    truth = pd.read_csv(truth_catalog)
    required = ["dp1_objectId", "hst_label"]
    missing = [col for col in required if col not in truth.columns]
    if missing:
        raise ValueError(f"Truth catalog is missing required columns for Butler join: {missing}")

    keep_cols = ["dp1_objectId"]
    keep_cols.extend(
        [
            col
            for col in truth.columns
            if col.startswith("hst_")
            or col.startswith("match_")
            or col.startswith("delta_")
            or col.startswith("clean_")
            or col == "matched_within_arcsec"
        ]
    )
    keep_cols = list(dict.fromkeys(keep_cols))
    out = truth[keep_cols].copy()
    out["_match_object_id"] = pd.to_numeric(out["dp1_objectId"], errors="coerce").astype("Int64")
    out = out[out["_match_object_id"].notna()].copy()
    sort_cols = ["_match_object_id"]
    if "match_sep_arcsec" in out.columns:
        sort_cols.append("match_sep_arcsec")
    out = out.sort_values(sort_cols, kind="mergesort")
    out = out.drop_duplicates(subset=["_match_object_id"], keep="first").reset_index(drop=True)
    return out


def compute_butler_pS_r(full_butler: pd.DataFrame, matched: pd.DataFrame) -> tuple[pd.DataFrame, str, int]:
    if "pS_r" in matched.columns and int(pd.to_numeric(matched["pS_r"], errors="coerce").notna().sum()) > 0:
        finite = int(pd.to_numeric(matched["pS_r"], errors="coerce").notna().sum())
        return matched, "reused existing `pS_r` in Butler matched table", finite

    try:
        from build_ecdfs_probability_models import compute_current_pSr
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import `compute_current_pSr`; run inside an environment with the repo src/ path "
            "and scipy available, or provide a precomputed pS_r column."
        ) from exc

    full_alias = add_butler_diagnostic_aliases(full_butler)
    matched_alias = add_butler_diagnostic_aliases(matched)
    required = ["r_diff", "r_modelFlux_mag"]
    missing_full = [col for col in required if col not in full_alias.columns]
    missing_matched = [col for col in required if col not in matched_alias.columns]
    if missing_full or missing_matched:
        raise ValueError(
            "Cannot recompute pS_r from Butler table. "
            f"Missing in full table: {missing_full}; missing in matched table: {missing_matched}"
        )

    matched_alias["pS_r"] = compute_current_pSr(full_alias, matched_alias)
    finite = int(pd.to_numeric(matched_alias["pS_r"], errors="coerce").notna().sum())
    return matched_alias, "recomputed `pS_r` from Butler `r_diff` and `r_modelFlux_mag` using existing pS workflow", finite


def build_butler_matched_analysis_sample(
    butler_catalog: Path,
    truth_catalog: Path,
    matched_output: Path,
) -> tuple[Path, dict[str, object]]:
    butler = read_table(butler_catalog)
    if "object_id" not in butler.columns:
        raise ValueError(f"Butler analysis table is missing `object_id`: {butler_catalog}")
    butler = add_butler_diagnostic_aliases(butler)
    butler["_match_object_id"] = pd.to_numeric(butler["object_id"], errors="coerce").astype("Int64")
    butler = butler[butler["_match_object_id"].notna()].copy()
    if butler["_match_object_id"].duplicated().any():
        before = len(butler)
        butler = butler.drop_duplicates(subset=["_match_object_id"], keep="first").copy()
        print(f"Deduplicated Butler table by object_id: {before} -> {len(butler)}")

    truth = load_truth_for_butler_join(truth_catalog)
    matched = butler.merge(truth, on="_match_object_id", how="inner", validate="one_to_one")
    if "dp1_objectId" in matched.columns and "objectId" not in matched.columns:
        matched["objectId"] = matched["object_id"]

    matched, ps_status, finite_ps = compute_butler_pS_r(butler, matched)
    if "_match_object_id" in matched.columns:
        matched = matched.drop(columns=["_match_object_id"])

    write_table(matched, matched_output)
    csv_output = matched_output.with_suffix(".csv")
    matched.to_csv(csv_output, index=False)

    context = {
        "butler_catalog": str(butler_catalog),
        "truth_catalog": str(truth_catalog),
        "total_butler_rows": int(len(butler)),
        "truth_rows_after_dedup": int(len(truth)),
        "matched_rows": int(len(matched)),
        "matched_output": str(matched_output),
        "matched_csv_output": str(csv_output),
        "pS_status": ps_status,
        "finite_pS_r": finite_ps,
        "pS_threshold": 0.5,
    }
    print(f"Butler-HST matched sample rows: {len(matched):,}")
    print(f"pS_r status: {ps_status}; finite values: {finite_ps:,}; threshold: 0.5")
    return matched_output, context


def augment_with_ps(
    df: pd.DataFrame,
    object_col: str,
    dp1_catalog: Path | None,
    ps_column_override: str | None,
) -> tuple[pd.DataFrame, LabelResult | None, str]:
    work = df.copy()

    explicit_ps = detect_explicit_ps_column(work, explicit=ps_column_override)
    if explicit_ps is not None:
        ps_result = infer_pS_labels(work, column_name=explicit_ps, threshold=None)
        print(summarize_method("pS", ps_result), end="")
        return work, ps_result, f"Using existing probability column `{ps_result.column}`."

    if dp1_catalog is None:
        print("No explicit pS column found and no DP1 catalog provided for recomputing pS_r.")
        return work, None, "No pS method available."

    if not dp1_catalog.exists():
        raise ValueError(f"DP1 catalog does not exist: {dp1_catalog}")

    try:
        from build_ecdfs_probability_models import compute_current_pSr, load_ecdfs_processed_table
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Could not import the existing ECDFS pS reconstruction code. "
            "Recomputing Nirav's pS_r requires the Rubin/LSST Python environment "
            "(including astropy and the repo's DP1 utilities). "
            "Either run this script inside that environment or provide an explicit "
            "--ps-column already present in the matched catalog."
        ) from exc

    print(f"Recomputing Nirav-style pS_r from {dp1_catalog}")
    processed = load_ecdfs_processed_table(dp1_catalog)
    if "objectId" not in processed.columns:
        raise ValueError(
            f"Processed DP1 catalog from {dp1_catalog} does not contain objectId. "
            f"Available columns: {_format_available_columns(processed)}"
        )
    processed = processed.copy()
    processed["pS_r"] = compute_current_pSr(processed, processed)

    keep = ["objectId", "pS_r"]
    for band in ["g", "i"]:
        col = f"{band}_modelFlux_mag"
        if col in processed.columns:
            keep.append(col)
    processed_keep = processed[keep].drop_duplicates(subset=["objectId"], keep="first")
    work = work.merge(processed_keep, left_on=object_col, right_on="objectId", how="left", validate="one_to_one")

    ps_result = infer_pS_labels(work, column_name="pS_r", threshold=0.5)
    print(summarize_method("pS_r", ps_result), end="")
    return work, ps_result, "Recomputed Nirav-style pS_r from DP1 morphology model; star if pS_r >= 0.5."


def prepare_base_sample(
    catalog_path: Path,
    truth_col: str | None,
    g_col: str | None,
    i_col: str | None,
    dp1_catalog: Path | None,
    ps_column_override: str | None,
    extendedness_bands: set[str] | None = None,
) -> tuple[pd.DataFrame, str, str, LabelResult, LabelResult | None, list[tuple[str, LabelResult]], dict[str, str]]:
    df = read_table(catalog_path)
    print(f"Loaded catalog: {catalog_path}")
    print(f"Rows before cleaning: {len(df)}")

    object_col = detect_object_id_column(df)
    df = deduplicate_catalog(df, object_col=object_col, sep_col="match_sep_arcsec")

    truth = infer_truth_labels(df, column_name=truth_col)
    print(f"Truth labels: column `{truth.column}` using {truth.description}")

    g_mag_col, g_mag = detect_mag_column(df, "g", explicit=g_col)
    i_mag_col, i_mag = detect_mag_column(df, "i", explicit=i_col)
    df = df.copy()
    df["cmd_g_mag"] = g_mag
    df["cmd_i_mag"] = i_mag
    df["cmd_g_minus_i"] = df["cmd_g_mag"] - df["cmd_i_mag"]

    finite_cmd = np.isfinite(df["cmd_g_mag"]) & np.isfinite(df["cmd_i_mag"]) & np.isfinite(df["cmd_g_minus_i"])
    plausible_mag = (
        (df["cmd_g_mag"] > 0.0)
        & (df["cmd_g_mag"] < 40.0)
        & (df["cmd_i_mag"] > 0.0)
        & (df["cmd_i_mag"] < 40.0)
    )
    truth_valid = truth.labels.isin([0, 1])
    base_mask = finite_cmd & plausible_mag & truth_valid
    if "cmd_gi_cmodel_clean" in df.columns:
        cmd_clean = df["cmd_gi_cmodel_clean"].fillna(False).astype(bool)
        base_mask &= cmd_clean
        print(f"Rows passing Butler CMD clean mask: {int(cmd_clean.sum())}")
    print(
        "Rows after finite/plausible g, i, g-i, and truth-label cuts: "
        f"{int(base_mask.sum())}"
    )
    df = df.loc[base_mask].copy().reset_index(drop=True)
    truth = LabelResult(labels=truth.labels.loc[base_mask].reset_index(drop=True), column=truth.column, description=truth.description)

    df, ps_result, ps_note = augment_with_ps(df, object_col=object_col, dp1_catalog=dp1_catalog, ps_column_override=ps_column_override)

    ext_results: list[tuple[str, LabelResult]] = []
    for band, ext_col, _flag_col in find_extendedness_columns(df):
        if extendedness_bands is not None and band not in extendedness_bands:
            continue
        ext_result = infer_extendedness_labels(df, column_name=ext_col, threshold=None)
        ext_results.append((band, ext_result))
        print(summarize_method(f"extendedness_{band}", ext_result), end="")

    if not ext_results:
        print("No usable extendedness bands detected after CMD cuts.")

    method_notes = {
        "truth": f"Truth column `{truth.column}` ({truth.description})",
        "pS": ps_note,
        "g_mag": g_mag_col,
        "i_mag": i_mag_col,
    }
    return df, "cmd_g_minus_i", "cmd_i_mag", truth, ps_result, ext_results, method_notes


def build_comparisons(
    df: pd.DataFrame,
    truth: LabelResult,
    ps_result: LabelResult | None,
    ext_results: list[tuple[str, LabelResult]],
    strict_common_sample: bool = False,
) -> list[ComparisonSpec]:
    comparisons: list[ComparisonSpec] = []

    if strict_common_sample:
        ext_i = next((result for band, result in ext_results if band == "i"), None)
        if ps_result is None or ext_i is None:
            return comparisons
        ext_i_valid = ext_i.labels.isin([0, 1])
        ps_valid = ps_result.labels.isin([0, 1])
        truth_valid = truth.labels.isin([0, 1])
        common_valid = ext_i_valid & ps_valid & truth_valid
        common_note = (
            "BUTLER COMMON SAMPLE: restricted to objects with valid truth, pS_r, "
            "extendedness_i, g-i, i, and CMD-clean photometry."
        )
        comparisons.extend(
            [
                ComparisonSpec(
                    reference_labels=truth.labels,
                    predicted_labels=ps_result.labels,
                    reference_name="truth",
                    predicted_name="pS",
                    title_detail=f"{ps_result.description}; {common_note}",
                    output_stub="pS_vs_truth",
                    source_columns={"reference": truth.column, "predicted": ps_result.column},
                    common_valid_mask=common_valid,
                ),
                ComparisonSpec(
                    reference_labels=truth.labels,
                    predicted_labels=ext_i.labels,
                    reference_name="truth",
                    predicted_name="extendedness_i",
                    title_detail=f"{ext_i.description}; {common_note}",
                    output_stub="extendedness_i_vs_truth",
                    source_columns={"reference": truth.column, "predicted": ext_i.column},
                    common_valid_mask=common_valid,
                ),
                ComparisonSpec(
                    reference_labels=ext_i.labels,
                    predicted_labels=ps_result.labels,
                    reference_name="extendedness_i",
                    predicted_name="pS",
                    title_detail=f"reference: {ext_i.description}; predicted: {ps_result.description}; {common_note}",
                    output_stub="pS_vs_extendedness_i",
                    source_columns={"reference": ext_i.column, "predicted": ps_result.column},
                    common_valid_mask=common_valid,
                ),
                ComparisonSpec(
                    reference_labels=ps_result.labels,
                    predicted_labels=ext_i.labels,
                    reference_name="pS",
                    predicted_name="extendedness_i",
                    title_detail=f"reference: {ps_result.description}; predicted: {ext_i.description}; {common_note}",
                    output_stub="extendedness_i_vs_pS",
                    source_columns={"reference": ps_result.column, "predicted": ext_i.column},
                    common_valid_mask=common_valid,
                ),
            ]
        )
        return comparisons

    if ps_result is not None:
        comparisons.append(
            ComparisonSpec(
                reference_labels=truth.labels,
                predicted_labels=ps_result.labels,
                reference_name="truth",
                predicted_name="pS",
                title_detail=ps_result.description,
                output_stub="pS_vs_truth",
                source_columns={"reference": truth.column, "predicted": ps_result.column},
            )
        )

    for band, ext_result in ext_results:
        comparisons.append(
            ComparisonSpec(
                reference_labels=truth.labels,
                predicted_labels=ext_result.labels,
                reference_name="truth",
                predicted_name=f"extendedness_{band}",
                title_detail=ext_result.description,
                output_stub=f"extendedness_{band}_vs_truth",
                source_columns={"reference": truth.column, "predicted": ext_result.column},
            )
        )

    ext_i = next((result for band, result in ext_results if band == "i"), None)
    if ps_result is not None and ext_i is not None:
        ext_i_valid = ext_i.labels.isin([0, 1])
        ps_valid = ps_result.labels.isin([0, 1])
        common_note = (
            "COMMON-SAMPLE truth comparison: restricted to objects with valid "
            "truth, pS, extendedness_i, g, and i values."
        )
        comparisons.append(
            ComparisonSpec(
                reference_labels=truth.labels,
                predicted_labels=ps_result.labels,
                reference_name="truth",
                predicted_name="pS",
                title_detail=f"{ps_result.description}; {common_note}",
                output_stub="pS_vs_truth_common_with_extendedness_i",
                source_columns={"reference": truth.column, "predicted": ps_result.column},
                common_valid_mask=ext_i_valid,
            )
        )
        comparisons.append(
            ComparisonSpec(
                reference_labels=truth.labels,
                predicted_labels=ext_i.labels,
                reference_name="truth",
                predicted_name="extendedness_i",
                title_detail=f"{ext_i.description}; {common_note}",
                output_stub="extendedness_i_vs_truth_common_with_pS",
                source_columns={"reference": truth.column, "predicted": ext_i.column},
                common_valid_mask=ps_valid,
            )
        )
        comparisons.append(
            ComparisonSpec(
                reference_labels=ext_i.labels,
                predicted_labels=ps_result.labels,
                reference_name="extendedness_i",
                predicted_name="pS",
                title_detail=f"reference: {ext_i.description}; predicted: {ps_result.description}",
                output_stub="pS_vs_extendedness_i",
                source_columns={"reference": ext_i.column, "predicted": ps_result.column},
            )
        )
        comparisons.append(
            ComparisonSpec(
                reference_labels=ps_result.labels,
                predicted_labels=ext_i.labels,
                reference_name="pS",
                predicted_name="extendedness_i",
                title_detail=f"reference: {ps_result.description}; predicted: {ext_i.description}",
                output_stub="extendedness_i_vs_pS",
                source_columns={"reference": ps_result.column, "predicted": ext_i.column},
            )
        )
    return comparisons


def write_readme(
    output_dir: Path,
    run_command: str,
    method_notes: dict[str, str],
    comparison_rows: pd.DataFrame,
    butler_context: dict[str, object] | None = None,
) -> None:
    if len(comparison_rows):
        comparison_md = [
            "| comparison | reference_column | predicted_column | valid_objects |\n",
            "|---|---|---|---:|\n",
        ]
        for _, row in comparison_rows.iterrows():
            comparison_md.append(
                f"| {row['comparison']} | {row['reference_column']} | "
                f"{row['predicted_column']} | {row['valid_objects']} |\n"
            )
        comparison_md_text = "".join(comparison_md)
    else:
        comparison_md_text = "No comparisons generated.\n"

    lines = [
        "# Star/Galaxy Diagnostics\n\n",
        "This directory contains CMD-based diagnostics for the Rubin star/galaxy separation workflow for the Rubin star/galaxy separation workflow.\n\n",
        "## Definitions\n\n",
        "- `a`: true/reference star predicted as star\n",
        "- `b`: true/reference galaxy predicted as galaxy\n",
        "- `c`: true/reference star predicted as galaxy\n",
        "- `d`: true/reference galaxy predicted as star\n\n",
        "Metrics are defined as:\n\n",
        "- star completeness = `a / (a + c)`\n",
        "- star contamination = `d / (a + d)`\n",
        "- galaxy completeness = `b / (b + d)`\n",
        "- galaxy contamination = `c / (b + c)`\n\n",
        (
            "All comparisons in this Butler run use the same common sample before computing panels and metrics.\n\n"
            if butler_context is not None
            else "Files with `_common_with_...` in the name are fair common-sample truth comparisons; they restrict both truth-based methods to the same objects before computing the panels and metrics.\n\n"
        ),
        "## CMD plots\n\n",
        "Each `cmd_four_panel_*.png/.pdf` file shows a 2x2 i vs g-i CMD with full-sample background contours and one confusion subset overlaid.\n\n",
        "Truth comparisons use:\n\n",
        "1. correct stars\n",
        "2. correct galaxies\n",
        "3. misclassified stars\n",
        "4. misclassified galaxies\n\n",
        "Method-vs-method comparisons use agreement/disagreement labels instead of truth wording.\n\n",
        "The y-axis is i magnitude and is inverted so brighter objects appear higher.\n\n",
        "## Columns and conventions used\n\n",
    ]
    for key, note in method_notes.items():
        lines.append(f"- {key}: {note}\n")

    if butler_context is not None:
        lines.extend(
            [
                "\n## Butler-based run notes\n\n",
                "- These diagnostics use the Butler-based ECDFS analysis table.\n",
                f"- Butler catalog: `{butler_context['butler_catalog']}`\n",
                f"- Truth labels come from: `{butler_context['truth_catalog']}`\n",
                f"- Matched Butler/HST sample: `{butler_context['matched_output']}`\n",
                f"- pS_r status: {butler_context['pS_status']}\n",
                f"- pS_r threshold for star classification: `{butler_context['pS_threshold']}`\n",
                "- All requested Butler comparisons use the common sample with valid truth, pS_r, extendedness_i, g-i, i, and CMD-clean photometry.\n",
                "- Remaining caveat: final clean cuts should be reviewed before final science interpretation before final science interpretation.\n",
            ]
        )

    lines.extend(
        [
            "\n## Comparisons generated\n\n",
            comparison_md_text,
            "\n\n## Run command\n\n",
            "```bash\n",
            f"{run_command}\n",
            "```\n",
        ]
    )
    (output_dir / "README.md").write_text("".join(lines))


def write_common_sample_summary(
    output_dir: Path,
    matched_catalog_path: Path,
    prepared_df: pd.DataFrame,
    truth: LabelResult,
    ps_result: LabelResult | None,
    ext_results: list[tuple[str, LabelResult]],
    color_col: str,
    mag_col: str,
    butler_context: dict[str, object],
) -> None:
    matched_full = read_table(matched_catalog_path)
    ext_i = next((result for band, result in ext_results if band == "i"), None)
    if ps_result is None or ext_i is None:
        final_common = pd.Series(False, index=prepared_df.index)
        valid_ext_i_count = 0
    else:
        valid_ext_i = ext_i.labels.isin([0, 1])
        valid_ps = ps_result.labels.isin([0, 1])
        finite_cmd = np.isfinite(prepared_df[color_col]) & np.isfinite(prepared_df[mag_col])
        final_common = truth.labels.isin([0, 1]) & valid_ps & valid_ext_i & finite_cmd
        valid_ext_i_count = int(valid_ext_i.sum())

    truth_counts = matched_full["hst_label"].value_counts(dropna=False).to_dict() if "hst_label" in matched_full else {}
    prepared_truth_counts = {
        "star": int((truth.labels == 1).sum()),
        "galaxy": int((truth.labels == 0).sum()),
    }
    lines = [
        "# Butler common sample summary\n\n",
        f"- Total Butler table rows: {butler_context['total_butler_rows']:,}\n",
        f"- HST truth rows after object-ID deduplication: {butler_context['truth_rows_after_dedup']:,}\n",
        f"- Butler rows matched to HST truth: {butler_context['matched_rows']:,}\n",
        f"- Rows after finite CMD, truth, and CMD-clean cuts: {len(prepared_df):,}\n",
        f"- Rows with finite/valid pS_r: {butler_context['finite_pS_r']:,} in matched table; {int(ps_result.labels.isin([0, 1]).sum()) if ps_result is not None else 0:,} after base cuts\n",
        f"- Rows with valid extendedness_i: {valid_ext_i_count:,} after base cuts\n",
        f"- Final common sample rows: {int(final_common.sum()):,}\n",
        f"- Truth counts in matched table: {truth_counts}\n",
        f"- Truth counts after base cuts: {prepared_truth_counts}\n",
        "\n## pS_r status\n\n",
        f"- Status: {butler_context['pS_status']}\n",
        f"- Finite `pS_r` values in matched table: {butler_context['finite_pS_r']:,}\n",
        f"- Classification threshold: star if `pS_r >= {butler_context['pS_threshold']}`\n",
        "\n## Required common-sample columns\n\n",
        f"- Truth label column: `{truth.column}`\n",
        f"- pS column: `{ps_result.column if ps_result is not None else 'missing'}`\n",
        f"- extendedness_i column: `{ext_i.column if ext_i is not None else 'missing'}`\n",
        f"- color column: `{color_col}`\n",
        f"- i magnitude column: `{mag_col}`\n",
        "- CMD clean mask: `cmd_gi_cmodel_clean` when present\n",
    ]
    (output_dir / "common_sample_summary.md").write_text("".join(lines))


def write_fits_vs_butler_comparison(output_dir: Path, old_dir: Path) -> None:
    old_summary = old_dir / "summary.csv"
    new_summary = output_dir / "summary.csv"
    lines = [
        "# FITS vs Butler diagnostic comparison\n\n",
        "This is a short bookkeeping comparison only. It should not be treated as a final science interpretation.\n\n",
    ]
    if not old_summary.exists() or not new_summary.exists():
        lines.append(f"- Old summary exists: `{old_summary.exists()}`\n")
        lines.append(f"- New summary exists: `{new_summary.exists()}`\n")
        lines.append("- Comparison skipped because one summary table is missing.\n")
        (output_dir / "fits_vs_butler_diagnostic_comparison.md").write_text("".join(lines))
        return

    old = pd.read_csv(old_summary)
    new = pd.read_csv(new_summary)
    targets = ["pS vs truth", "extendedness_i vs truth", "pS vs extendedness_i"]
    lines.extend(
        [
            "| comparison | FITS valid objects | Butler valid objects | FITS star completeness | Butler star completeness | FITS star contamination | Butler star contamination |\n",
            "|---|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for comp in targets:
        old_rows = old[old["comparison"] == comp]
        new_rows = new[new["comparison"] == comp]
        if old_rows.empty or new_rows.empty:
            lines.append(f"| {comp} | missing | missing | missing | missing | missing | missing |\n")
            continue
        old_row = old_rows.iloc[-1]
        new_row = new_rows.iloc[0]
        lines.append(
            f"| {comp} | {int(old_row['n_valid']):,} | {int(new_row['n_valid']):,} | "
            f"{float(old_row['star_completeness']):.3f} | {float(new_row['star_completeness']):.3f} | "
            f"{float(old_row['star_contamination']):.3f} | {float(new_row['star_contamination']):.3f} |\n"
        )
    lines.extend(
        [
            "\n## Notes\n\n",
            "- The old FITS diagnostics used the historical FITS/export photometry names and sample construction.\n",
            "- The Butler diagnostics use `outputs/dp1_ecdfs_analysis_table_butler.parquet` joined by object ID to the HST matched truth table.\n",
            "- Differences can come from schema photometry differences, common-sample filtering, and the newly recomputed Butler-based `pS_r`.\n",
            "- Broad consistency should be judged with plots and reviewed sample cuts, not this table alone.\n",
        ]
    )
    (output_dir / "fits_vs_butler_diagnostic_comparison.md").write_text("".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG, help="Matched catalog used for truth-based Rubin diagnostics.")
    parser.add_argument("--dp1-catalog", type=Path, default=DEFAULT_DP1, help="DP1 FITS catalog used to recompute Nirav-style pS_r.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for diagnostic plots and tables.")
    parser.add_argument(
        "--use-butler-table",
        action="store_true",
        help="Treat --catalog as the Butler-based prepared ECDFS table and join it to the HST truth catalog by object ID.",
    )
    parser.add_argument("--truth-catalog", type=Path, default=DEFAULT_TRUTH_CATALOG, help="HST/external matched truth catalog for --use-butler-table.")
    parser.add_argument("--matched-output", type=Path, default=DEFAULT_BUTLER_MATCHED, help="Output path for the Butler/HST matched analysis sample.")
    parser.add_argument("--truth-column", type=str, default=None, help="Override truth-label column name.")
    parser.add_argument("--g-col", type=str, default=None, help="Override g-band magnitude column name.")
    parser.add_argument("--i-col", type=str, default=None, help="Override i-band magnitude column name.")
    parser.add_argument("--ps-column", type=str, default=None, help="Override pS/probability column name. If absent, the script recomputes pS_r.")
    parser.add_argument(
        "--extendedness-bands",
        type=str,
        default=None,
        help="Comma-separated extendedness bands to diagnose. Defaults to all bands, or i only with --use-butler-table.",
    )
    parser.add_argument(
        "--filename-suffix",
        type=str,
        default=None,
        help="Suffix appended before plot/metric extensions. Defaults to `_butler` with --use-butler-table.",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["auto", "scatter", "contour", "hexbin"],
        default="auto",
        help="How to draw the confusion subsets in CMD panels.",
    )
    parser.add_argument("--mag-bin-width", type=float, default=0.5, help="i-magnitude bin width. Set to 0 or a negative value to disable.")
    parser.add_argument("--n-mag-bins", type=int, default=None, help="Number of equal-width bins if --mag-bin-width is disabled.")
    args = parser.parse_args()

    ensure_output_dir(args.output_dir)

    mag_bin_width = args.mag_bin_width if args.mag_bin_width and args.mag_bin_width > 0 else None
    butler_context: dict[str, object] | None = None
    catalog_path = args.catalog
    dp1_catalog: Path | None = args.dp1_catalog
    ps_column = args.ps_column
    truth_column = args.truth_column
    g_col = args.g_col
    i_col = args.i_col
    filename_suffix = args.filename_suffix if args.filename_suffix is not None else ("_butler" if args.use_butler_table else "")

    if args.extendedness_bands:
        extendedness_bands = {band.strip() for band in args.extendedness_bands.split(",") if band.strip()}
    elif args.use_butler_table:
        extendedness_bands = {"i"}
    else:
        extendedness_bands = None

    if args.use_butler_table:
        if catalog_path == DEFAULT_CATALOG:
            catalog_path = DEFAULT_BUTLER_CATALOG
        catalog_path, butler_context = build_butler_matched_analysis_sample(
            butler_catalog=catalog_path,
            truth_catalog=args.truth_catalog,
            matched_output=args.matched_output,
        )
        dp1_catalog = None
        ps_column = ps_column or "pS_r"
        truth_column = truth_column or "hst_label"
        g_col = g_col or "cmodel_mag_g"
        i_col = i_col or "cmodel_mag_i"

    df, color_col, mag_col, truth, ps_result, ext_results, method_notes = prepare_base_sample(
        catalog_path=catalog_path,
        truth_col=truth_column,
        g_col=g_col,
        i_col=i_col,
        dp1_catalog=dp1_catalog,
        ps_column_override=ps_column,
        extendedness_bands=extendedness_bands,
    )
    if butler_context is not None:
        method_notes["pS"] = f"{butler_context['pS_status']}; star if pS_r >= {butler_context['pS_threshold']}"

    if not truth.labels.isin([0, 1]).all():
        raise RuntimeError("Truth labels are not fully standardized to binary star/galaxy values.")

    bins = build_mag_bins(df[mag_col], mag_bin_width=mag_bin_width, n_mag_bins=args.n_mag_bins)
    comparisons = build_comparisons(df, truth, ps_result, ext_results, strict_common_sample=args.use_butler_table)
    if not comparisons:
        raise RuntimeError("No comparisons could be constructed. Need at least truth and one classification method.")

    summary_lines = [
        "Star/Galaxy diagnostics summary\n",
        "===============================\n\n",
        f"Catalog: {catalog_path}\n",
        f"Rows in final finite-CMD truth sample: {len(df)}\n",
        f"g magnitude column: {method_notes['g_mag']}\n",
        f"i magnitude column: {method_notes['i_mag']}\n",
        f"Truth: {method_notes['truth']}\n",
        f"pS: {method_notes['pS']}\n\n",
    ]
    summary_rows: list[dict[str, object]] = []
    comparison_rows_for_readme: list[dict[str, str]] = []

    for comp in comparisons:
        ref = comp.reference_labels
        pred = comp.predicted_labels
        valid = ref.isin([0, 1]) & pred.isin([0, 1]) & np.isfinite(df[color_col]) & np.isfinite(df[mag_col])
        if comp.common_valid_mask is not None:
            common_mask = comp.common_valid_mask.reindex(df.index, fill_value=False)
            valid &= common_mask
        use = df.loc[valid].copy().reset_index(drop=True)
        ref_use = ref.loc[valid].reset_index(drop=True)
        pred_use = pred.loc[valid].reset_index(drop=True)

        if len(use) == 0:
            print(f"Skipping {comp.output_stub}: no valid comparison rows.")
            continue

        masks, counts = make_confusion_masks(ref_use, pred_use)
        overall = overall_metrics_from_counts(counts)

        if args.use_butler_table:
            subtitle = (
                f"Butler common sample; reference={comp.reference_name} "
                f"({comp.source_columns['reference']}), predicted={comp.predicted_name} "
                f"({comp.source_columns['predicted']})"
            )
            metrics_subtitle = "Butler common sample"
        else:
            subtitle = (
                f"reference={comp.reference_name} ({comp.source_columns['reference']}), "
                f"predicted={comp.predicted_name} ({comp.source_columns['predicted']}); {comp.title_detail}"
            )
            metrics_subtitle = comp.title_detail
        plot_cmd_four_panel(
            use,
            color_col=color_col,
            mag_col=mag_col,
            masks=masks,
            method_name=comp.predicted_name,
            reference_name=comp.reference_name,
            output_path=args.output_dir / f"cmd_four_panel_{comp.output_stub}{filename_suffix}",
            plot_mode=args.plot_mode,
            subtitle=subtitle,
        )

        metrics_df = compute_metrics_vs_magnitude(ref_use, pred_use, use[mag_col], bins=bins)
        metrics_path = args.output_dir / f"metrics_vs_i_{comp.output_stub}{filename_suffix}.csv"
        metrics_df.to_csv(metrics_path, index=False)
        plot_metrics_vs_magnitude(
            metrics_df,
            method_name=comp.predicted_name,
            reference_name=comp.reference_name,
            output_path=args.output_dir / f"metrics_vs_i_{comp.output_stub}{filename_suffix}",
            subtitle=metrics_subtitle,
        )

        comparison_rows_for_readme.append(
            {
                "comparison": f"{comp.predicted_name} vs {comp.reference_name}",
                "reference_column": comp.source_columns["reference"],
                "predicted_column": comp.source_columns["predicted"],
                "valid_objects": str(len(use)),
            }
        )

        print(f"\nComparison: {comp.predicted_name} vs {comp.reference_name}")
        print(f"Total valid objects: {len(use)}")
        print(f"a correct stars: {counts['a']}")
        print(f"b correct galaxies: {counts['b']}")
        print(f"c misclassified stars: {counts['c']}")
        print(f"d misclassified galaxies: {counts['d']}")
        print(f"overall star completeness: {overall['star_completeness']:.4f}" if np.isfinite(overall["star_completeness"]) else "overall star completeness: NaN")
        print(f"overall star contamination: {overall['star_contamination']:.4f}" if np.isfinite(overall["star_contamination"]) else "overall star contamination: NaN")
        print(f"overall galaxy completeness: {overall['galaxy_completeness']:.4f}" if np.isfinite(overall["galaxy_completeness"]) else "overall galaxy completeness: NaN")
        print(f"overall galaxy contamination: {overall['galaxy_contamination']:.4f}" if np.isfinite(overall["galaxy_contamination"]) else "overall galaxy contamination: NaN")

        summary_lines.extend(
            [
                f"Comparison: {comp.predicted_name} vs {comp.reference_name}\n",
                f"Total valid objects: {len(use)}\n",
                f"a correct stars: {counts['a']}\n",
                f"b correct galaxies: {counts['b']}\n",
                f"c misclassified stars: {counts['c']}\n",
                f"d misclassified galaxies: {counts['d']}\n",
                f"overall star completeness: {overall['star_completeness']:.6f}\n",
                f"overall star contamination: {overall['star_contamination']:.6f}\n",
                f"overall galaxy completeness: {overall['galaxy_completeness']:.6f}\n",
                f"overall galaxy contamination: {overall['galaxy_contamination']:.6f}\n",
                f"Reference column: {comp.source_columns['reference']}\n",
                f"Predicted column: {comp.source_columns['predicted']}\n",
                f"Rule: {comp.title_detail}\n\n",
            ]
        )

        summary_rows.append(
            {
                "comparison": f"{comp.predicted_name} vs {comp.reference_name}",
                "reference_column": comp.source_columns["reference"],
                "predicted_column": comp.source_columns["predicted"],
                "n_valid": len(use),
                "a_correct_stars": counts["a"],
                "b_correct_galaxies": counts["b"],
                "c_misclassified_stars": counts["c"],
                "d_misclassified_galaxies": counts["d"],
                "star_completeness": overall["star_completeness"],
                "star_contamination": overall["star_contamination"],
                "galaxy_completeness": overall["galaxy_completeness"],
                "galaxy_contamination": overall["galaxy_contamination"],
                "rule": comp.title_detail,
            }
        )

    summary_txt = args.output_dir / "summary.txt"
    summary_csv = args.output_dir / "summary.csv"
    summary_txt.write_text("".join(summary_lines))
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    run_command = (
        f"python scripts/star_galaxy_diagnostics.py "
        f"--catalog {args.catalog} --dp1-catalog {args.dp1_catalog}"
    )
    if args.use_butler_table:
        run_command = (
            "python scripts/star_galaxy_diagnostics.py "
            f"--catalog {args.catalog} --use-butler-table "
            f"--truth-catalog {args.truth_catalog} --matched-output {args.matched_output} "
            f"--output-dir {args.output_dir}"
        )
    write_readme(
        args.output_dir,
        run_command=run_command,
        method_notes=method_notes,
        comparison_rows=pd.DataFrame(comparison_rows_for_readme),
        butler_context=butler_context,
    )
    if args.use_butler_table and butler_context is not None:
        write_common_sample_summary(
            args.output_dir,
            matched_catalog_path=catalog_path,
            prepared_df=df,
            truth=truth,
            ps_result=ps_result,
            ext_results=ext_results,
            color_col=color_col,
            mag_col=mag_col,
            butler_context=butler_context,
        )
        write_fits_vs_butler_comparison(args.output_dir, DEFAULT_OUTPUT_DIR)

    print(f"\nWrote summary: {summary_txt}")
    print(f"Wrote summary CSV: {summary_csv}")
    print(f"Wrote README: {args.output_dir / 'README.md'}")


if __name__ == "__main__":
    main()
