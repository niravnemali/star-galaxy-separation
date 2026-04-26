"""Build ECDFS Rubin star-probability models using photometry + morphology.

This is a clean extension beyond the existing single-band morphology-based pS
notebook workflow.  It keeps the old notebook untouched and builds a new
ECDFS-only probability modeling chain driven by the clean HST/3D-HST matched
sample.
"""

from __future__ import annotations

import argparse
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from config import filters, model_flux_mag, model_flux_diff, rel_error
from external_label_tools import attach_external_labels
from pipeline import load_csv_dp1, magnitude_calculations
from psf_cmodel_fit import abstract_cmodel_dependence, compute_pS, fit_slices


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
DEFAULT_DP1 = DATA_DIR / "ECDFS.fits"
DEFAULT_MATCHED = OUTPUT_DIR / "hst_dp1_matched_clean_ellipse.csv"


MAG_FEATURES = [f"{band}{model_flux_mag}" for band in filters]
COLOR_FEATURES = ["ug", "gr", "ri", "iz", "zy", "gi"]
DIFF_FEATURES = [f"{band}{model_flux_diff}" for band in filters]
RELERR_FEATURES = [f"{band}{rel_error}" for band in filters]
EXTENDEDNESS_FEATURES = [f"{band}_extendedness" for band in filters]
SIZEEXT_FEATURES = [f"{band}_sizeExtendedness" for band in filters]
EXTENDEDNESS_FLAG_FEATURES = [f"{band}_extendedness_flag" for band in filters]
SIZEEXT_FLAG_FEATURES = [f"{band}_sizeExtendedness_flag" for band in filters]


ALL_CANDIDATE_FEATURES = (
    MAG_FEATURES
    + COLOR_FEATURES
    + DIFF_FEATURES
    + RELERR_FEATURES
    + EXTENDEDNESS_FEATURES
    + SIZEEXT_FEATURES
    + EXTENDEDNESS_FLAG_FEATURES
    + SIZEEXT_FLAG_FEATURES
)


def _output_paths(output_dir: Path) -> dict[str, object]:
    return {
        "audit": output_dir / "ecdfs_feature_audit_for_probability_model.md",
        "master_sample": output_dir / "ecdfs_probability_model_master_sample.csv",
        "master_summary": output_dir / "ecdfs_probability_model_master_sample_summary.txt",
        "comparison": output_dir / "ecdfs_current_pSr_vs_new_models_comparison.txt",
        "professor_summary": output_dir / "ecdfs_professor_requested_probability_summary.txt",
        "comparison_plots": {
            "roc": output_dir / "ecdfs_model_comparison_roc.png",
            "pr": output_dir / "ecdfs_model_comparison_pr.png",
            "metrics": output_dir / "ecdfs_model_comparison_metrics.png",
        },
        "models": {
            "A": {
                "summary": output_dir / "ecdfs_modelA_morphology_only_summary.txt",
                "hist": output_dir / "ecdfs_modelA_pstar_hist.png",
                "roc": output_dir / "ecdfs_modelA_roc.png",
                "pr": output_dir / "ecdfs_modelA_pr.png",
                "pstar_vs_mag": output_dir / "ecdfs_modelA_pstar_vs_mag.png",
                "calibration": output_dir / "ecdfs_modelA_calibration.png",
            },
            "B": {
                "summary": output_dir / "ecdfs_modelB_photometry_only_summary.txt",
                "hist": output_dir / "ecdfs_modelB_pstar_hist.png",
                "roc": output_dir / "ecdfs_modelB_roc.png",
                "pr": output_dir / "ecdfs_modelB_pr.png",
                "pstar_vs_mag": output_dir / "ecdfs_modelB_pstar_vs_mag.png",
                "calibration": output_dir / "ecdfs_modelB_calibration.png",
            },
            "C": {
                "summary": output_dir / "ecdfs_modelC_combined_summary.txt",
                "hist": output_dir / "ecdfs_modelC_pstar_hist.png",
                "roc": output_dir / "ecdfs_modelC_roc.png",
                "pr": output_dir / "ecdfs_modelC_pr.png",
                "pstar_vs_mag": output_dir / "ecdfs_modelC_pstar_vs_mag.png",
                "calibration": output_dir / "ecdfs_modelC_calibration.png",
            },
        },
    }


@dataclass(frozen=True)
class ModelSpec:
    key: str
    name: str
    feature_cols: list[str]
    rationale: str


MODEL_SPECS = [
    ModelSpec(
        key="A",
        name="Model A: morphology-only baseline",
        feature_cols=["r_diff", "i_diff"],
        rationale="Use the cleanest Rubin morphology-like PSF-CModel features without colors.",
    ),
    ModelSpec(
        key="B",
        name="Model B: photometry-only model",
        feature_cols=["gr", "ri", "iz", "zy", "r_modelFlux_mag", "i_modelFlux_mag"],
        rationale="Use Rubin colors plus r/i magnitudes, but no morphology diff features.",
    ),
    ModelSpec(
        key="C",
        name="Model C: combined photometry + morphology",
        feature_cols=["gr", "ri", "iz", "zy", "r_modelFlux_mag", "i_modelFlux_mag", "r_diff", "i_diff"],
        rationale="Combine Rubin colors, magnitudes, and morphology-like diff features.",
    ),
]


def _clean_numeric_frame(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in dict.fromkeys(cols):
        if col in out.columns:
            series = out[col]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            if pd.api.types.is_bool_dtype(series):
                numeric = series.astype(float)
            else:
                numeric = pd.to_numeric(series, errors="coerce").astype(float)
            numeric.loc[np.isinf(numeric)] = np.nan
            out[col] = numeric
    return out


def _series_count_percent(series: pd.Series) -> tuple[int, float]:
    count = int(series.notna().sum())
    frac = float(series.notna().mean()) if len(series) else np.nan
    return count, frac


def load_ecdfs_processed_table(dp1_path: Path) -> pd.DataFrame:
    """Rebuild the notebook-style ECDFS feature table from the Rubin FITS file."""

    df = load_csv_dp1(dp1_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        df = magnitude_calculations(df, df["ebv"] * 3.10 / 1.20)
    return df


def build_master_sample(processed_df: pd.DataFrame, matched_path: Path) -> pd.DataFrame:
    """Attach clean HST labels to the notebook-style ECDFS Rubin table."""

    merged = attach_external_labels(processed_df, matched_path)
    merged = merged.rename(
        columns={
            "external_label": "hst_label",
            "external_label_source": "hst_label_source",
        }
    )
    master = merged[merged["hst_label"].isin(["star", "galaxy"])].copy()
    master = master.reset_index(drop=True)
    return master


def feature_availability_table(df: pd.DataFrame, feature_cols: list[str], dataset_name: str) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if col not in df.columns:
            rows.append(
                {
                    "dataset": dataset_name,
                    "feature": col,
                    "present": False,
                    "non_null_count": 0,
                    "non_null_fraction": np.nan,
                }
            )
            continue
        count, frac = _series_count_percent(df[col])
        rows.append(
            {
                "dataset": dataset_name,
                "feature": col,
                "present": True,
                "non_null_count": count,
                "non_null_fraction": frac,
            }
        )
    return pd.DataFrame(rows)


def write_feature_audit(processed_df: pd.DataFrame, master_df: pd.DataFrame, out_path: Path) -> None:
    master_audit = feature_availability_table(master_df, ALL_CANDIDATE_FEATURES, "clean_labeled_master")

    def _format_rows(table: pd.DataFrame, cols: list[str]) -> list[str]:
        lines = ["| feature | present | non-null count | non-null fraction |\n", "|---|---:|---:|---:|\n"]
        for col in cols:
            row = table[table["feature"] == col].iloc[0]
            present = "yes" if bool(row["present"]) else "no"
            frac = "NA" if not np.isfinite(row["non_null_fraction"]) else f"{row['non_null_fraction']:.3f}"
            lines.append(f"| `{col}` | {present} | {int(row['non_null_count'])} | {frac} |\n")
        return lines

    text = [
        "# ECDFS feature audit for probability model\n\n",
        "This audit uses the notebook-style ECDFS feature construction:\n",
        "1. load `data/ECDFS.fits` via `pipeline.load_csv_dp1`\n",
        "2. run `pipeline.magnitude_calculations(df, df['ebv']*3.10/1.20)`\n",
        "3. attach clean HST/3D-HST labels from `outputs/hst_dp1_matched_clean_ellipse.csv`\n\n",
        "## Candidate feature families\n\n",
        "- Magnitudes: `u/g/r/i/z/y_modelFlux_mag`\n",
        "- Colors: `ug`, `gr`, `ri`, `iz`, `zy`, `gi`\n",
        "- Morphology-like PSF-CModel quantities: `u/g/r/i/z/y_diff`\n",
        "- Relative photometric errors: `u/g/r/i/z/y_psfRelErr`\n",
        "- Extendedness / sizeExtendedness columns where present\n",
        "- Related flag columns where present\n\n",
        f"Processed ECDFS rows: {len(processed_df)}\n\n",
        f"Clean labeled ECDFS master rows: {len(master_df)}\n",
        f"Clean labeled stars: {int((master_df['hst_label'] == 'star').sum())}\n",
        f"Clean labeled galaxies: {int((master_df['hst_label'] == 'galaxy').sum())}\n\n",
        "## Availability in the clean labeled master sample\n\n",
    ]

    for title, cols in [
        ("Magnitudes", MAG_FEATURES),
        ("Colors", COLOR_FEATURES),
        ("Morphology-like diff features", DIFF_FEATURES),
        ("Relative PSF error features", RELERR_FEATURES),
        ("Extendedness features", EXTENDEDNESS_FEATURES),
        ("SizeExtendedness features", SIZEEXT_FEATURES),
    ]:
        text.append(f"### {title}\n\n")
        text.extend(_format_rows(master_audit, cols))
        text.append("\n")

    text.extend(
        [
            "## Modeling decisions\n\n",
            "- `ug`, `u_*`, `z_*`, and `y_*` features are available but have lower completeness in the clean ECDFS labeled sample.\n",
            "- The baseline photometry model therefore uses the more complete color chain `gr`, `ri`, `iz`, `zy` plus `r/i` magnitudes.\n",
            "- `extendedness` and `sizeExtendedness` are present but substantially sparser than the `*_diff` features, so they are audited but not part of the first A/B/C baseline models.\n",
            "- Missing values are preserved in the master table; model pipelines handle them explicitly with median imputation.\n",
        ]
    )
    out_path.write_text("".join(text))


def write_master_sample(master_df: pd.DataFrame, out_csv: Path, out_summary: Path) -> None:
    keep_cols = ["objectId", "coord_ra", "coord_dec", "hst_label", "hst_label_source", "match_sep_arcsec"]
    keep_cols += [col for col in ALL_CANDIDATE_FEATURES if col in master_df.columns]
    keep_cols = [col for col in keep_cols if col in master_df.columns]

    master_out = _clean_numeric_frame(master_df[keep_cols].copy(), [c for c in keep_cols if c not in {"hst_label", "hst_label_source"}])
    master_out.to_csv(out_csv, index=False)

    missingness = []
    for col in keep_cols:
        if col in {"hst_label", "hst_label_source"}:
            continue
        count, frac = _series_count_percent(master_out[col])
        missingness.append((col, count, frac))
    missingness.sort(key=lambda t: (-t[2], t[0]))

    lines = [
        "ECDFS probability model master sample summary\n",
        "============================================\n\n",
        f"Rows: {len(master_out)}\n",
        f"Stars: {int((master_out['hst_label'] == 'star').sum())}\n",
        f"Galaxies: {int((master_out['hst_label'] == 'galaxy').sum())}\n",
        "Starting point: clean ECDFS matched sample `outputs/hst_dp1_matched_clean_ellipse.csv`\n",
        "Rubin features: rebuilt from `data/ECDFS.fits` using the notebook-style magnitude calculations.\n",
        "Duplicate objectId handling: keep the smallest-separation clean-ellipse match per Rubin object.\n",
        "Missing-value handling: preserve NaNs in the master table; downstream models use median imputation inside the sklearn pipeline.\n\n",
        "Columns included:\n",
    ]
    for col in keep_cols:
        lines.append(f"  - {col}\n")
    lines.append("\nTop non-null feature coverage in the master sample:\n")
    for col, count, frac in missingness[:20]:
        lines.append(f"  - {col}: {count} non-null ({frac:.3f})\n")
    out_summary.write_text("".join(lines))


def compute_current_pSr(processed_df: pd.DataFrame, master_df: pd.DataFrame) -> pd.Series:
    """Recompute the current notebook-style single-band pS_r score."""

    fit_results = fit_slices(processed_df, "r", n_resolved=2)
    abs_model_r = abstract_cmodel_dependence(fit_results, poly_deg=2)
    pSr = pd.Series(np.nan, index=master_df.index, dtype=float)
    valid = master_df["r_diff"].notna() & master_df["r_modelFlux_mag"].notna()
    if valid.any():
        pSr.loc[valid] = compute_pS(
            master_df.loc[valid, "r_diff"].to_numpy(),
            master_df.loc[valid, "r_modelFlux_mag"].to_numpy(),
            abs_model_r,
        )
    return pSr


def build_model_sample(df: pd.DataFrame, feature_cols: list[str], label_col: str = "hst_label") -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build a labeled feature sample while preserving rows with partial missingness."""

    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    base_cols = [label_col, "objectId", "coord_ra", "coord_dec", "match_sep_arcsec", "r_modelFlux_mag", "i_modelFlux_mag"]
    selected_cols = list(dict.fromkeys(base_cols + feature_cols))
    work = df[selected_cols].copy()
    work = _clean_numeric_frame(work, feature_cols + ["coord_ra", "coord_dec", "match_sep_arcsec", "r_modelFlux_mag", "i_modelFlux_mag"])
    mask = work[label_col].isin(["star", "galaxy"])
    mask &= work[feature_cols].notna().any(axis=1)
    sample = work.loc[mask].reset_index(drop=True)
    y = (sample[label_col] == "star").astype(int).to_numpy()
    X = sample[feature_cols].to_numpy(dtype=float)
    return sample, X, y


def _threshold_metrics(y_true: np.ndarray, p_star: np.ndarray, threshold: float) -> dict[str, float]:
    pred_star = p_star >= threshold
    truth_star = y_true == 1
    truth_gal = y_true == 0

    tp = int(np.sum(pred_star & truth_star))
    fp = int(np.sum(pred_star & truth_gal))
    tn = int(np.sum((~pred_star) & truth_gal))
    fn = int(np.sum((~pred_star) & truth_star))

    star_recall = tp / (tp + fn) if (tp + fn) else np.nan
    star_precision = tp / (tp + fp) if (tp + fp) else np.nan
    gal_recall = tn / (tn + fp) if (tn + fp) else np.nan
    gal_precision = tn / (tn + fn) if (tn + fn) else np.nan
    balanced_accuracy = 0.5 * (star_recall + gal_recall) if np.isfinite(star_recall) and np.isfinite(gal_recall) else np.nan

    return {
        "threshold": float(threshold),
        "tp_star": tp,
        "fp_star": fp,
        "tn_galaxy": tn,
        "fn_star": fn,
        "star_completeness": star_recall,
        "star_purity": star_precision,
        "galaxy_completeness": gal_recall,
        "galaxy_purity": gal_precision,
        "balanced_accuracy": balanced_accuracy,
    }


def evaluate_logistic_model(sample: pd.DataFrame, X: np.ndarray, y: np.ndarray, feature_cols: list[str], random_state: int = 42) -> dict:
    from sklearn.calibration import calibration_curve
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=4000, solver="lbfgs")),
        ]
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    p_oof = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba", n_jobs=None)[:, 1]
    roc_auc = float(roc_auc_score(y, p_oof))
    avg_precision = float(average_precision_score(y, p_oof))
    fpr, tpr, roc_thresholds = roc_curve(y, p_oof)
    precision, recall, pr_thresholds = precision_recall_curve(y, p_oof)
    frac_pos, mean_pred = calibration_curve(y, p_oof, n_bins=10, strategy="quantile")

    thresholds = np.linspace(0.05, 0.95, 181)
    threshold_table = pd.DataFrame([_threshold_metrics(y, p_oof, thr) for thr in thresholds])
    best_idx = threshold_table["balanced_accuracy"].astype(float).idxmax()
    best_metrics = threshold_table.loc[int(best_idx)].to_dict()
    metrics_05 = _threshold_metrics(y, p_oof, 0.5)

    final_pipeline = pipeline.fit(X, y)
    final_p = final_pipeline.predict_proba(X)[:, 1]

    return {
        "sample": sample.copy(),
        "y": y,
        "p_oof": p_oof,
        "p_fit_all": final_p,
        "feature_cols": feature_cols,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "calibration_frac_pos": frac_pos,
        "calibration_mean_pred": mean_pred,
        "threshold_table": threshold_table,
        "threshold_metrics_05": metrics_05,
        "best_threshold_metrics": best_metrics,
    }


def _choose_reference_mag(sample: pd.DataFrame) -> str:
    for col in ["i_modelFlux_mag", "r_modelFlux_mag", "z_modelFlux_mag", "g_modelFlux_mag"]:
        if col in sample.columns and sample[col].notna().sum() > 0:
            return col
    raise ValueError("No usable reference magnitude column for plotting.")


def plot_probability_hist(result: dict, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    sample = result["sample"].copy()
    sample["p_star"] = result["p_oof"]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    bins = np.linspace(0, 1, 51)
    for label, color in [("star", "black"), ("galaxy", "tab:blue")]:
        vals = sample.loc[sample["hst_label"] == label, "p_star"].to_numpy(dtype=float)
        ax.hist(vals, bins=bins, histtype="step", density=True, lw=1.8, label=label, color=color)
    ax.set_xlabel("P_star")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_roc(result: dict, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot(result["fpr"], result["tpr"], lw=1.8, label=f"ROC AUC = {result['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pr(result: dict, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot(result["recall"], result["precision"], lw=1.8, label=f"AP = {result['average_precision']:.3f}")
    ax.set_xlabel("Star recall")
    ax.set_ylabel("Star precision")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pstar_vs_mag(result: dict, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    sample = result["sample"].copy()
    sample["p_star"] = result["p_oof"]
    mag_col = _choose_reference_mag(sample)
    plot_df = sample[sample[mag_col].notna()].copy()

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    for label, color, alpha, size in [("galaxy", "tab:blue", 0.25, 10), ("star", "black", 0.75, 18)]:
        sub = plot_df[plot_df["hst_label"] == label]
        ax.scatter(sub[mag_col], sub["p_star"], c=color, alpha=alpha, s=size, label=label)
    ax.set_xlabel(mag_col)
    ax.set_ylabel("P_star")
    ax.set_ylim(-0.02, 1.02)
    ax.invert_xaxis()
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_calibration(result: dict, out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.plot(result["calibration_mean_pred"], result["calibration_frac_pos"], "o-", label="model")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
    ax.set_xlabel("Mean predicted P_star")
    ax.set_ylabel("Observed star fraction")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_model_comparison_curves(
    pSr_result: dict,
    model_results: dict[str, dict],
    out_paths: dict[str, Path],
) -> None:
    import matplotlib.pyplot as plt

    curve_specs = [
        ("Current pS_r", pSr_result, "0.35"),
        ("Model A", model_results["A"], "tab:green"),
        ("Model B", model_results["B"], "tab:orange"),
        ("Model C", model_results["C"], "tab:blue"),
    ]

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for label, result, color in curve_specs:
        ax.plot(result["fpr"], result["tpr"], lw=1.8, color=color, label=f"{label} (AUC={result['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ECDFS model comparison: ROC")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_paths["roc"], dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for label, result, color in curve_specs:
        ax.plot(
            result["recall"],
            result["precision"],
            lw=1.8,
            color=color,
            label=f"{label} (AP={result['average_precision']:.3f})",
        )
    ax.set_xlabel("Star recall")
    ax.set_ylabel("Star precision")
    ax.set_title("ECDFS model comparison: precision-recall")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_paths["pr"], dpi=180)
    plt.close(fig)

    metric_rows = []
    for label, result, _ in curve_specs:
        metric_rows.append(
            {
                "label": label,
                "roc_auc": result["roc_auc"],
                "average_precision": result["average_precision"],
                "balanced_accuracy": result["best_threshold_metrics"]["balanced_accuracy"],
            }
        )
    metric_df = pd.DataFrame(metric_rows)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.5), sharey=False)
    metrics = [
        ("roc_auc", "ROC AUC"),
        ("average_precision", "Average precision"),
        ("balanced_accuracy", "Best balanced accuracy"),
    ]
    colors = ["0.35", "tab:green", "tab:orange", "tab:blue"]
    for ax, (metric_col, title) in zip(axes, metrics):
        ax.bar(metric_df["label"], metric_df[metric_col], color=colors)
        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.grid(True, axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("ECDFS model comparison metrics", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_paths["metrics"], dpi=180)
    plt.close(fig)


def write_model_summary(spec: ModelSpec, result: dict, out_path: Path) -> None:
    sample = result["sample"]
    frac_missing = sample[spec.feature_cols].isna().mean().sort_values(ascending=False)
    lines = [
        f"{spec.name}\n",
        "=" * len(spec.name) + "\n\n",
        f"Rationale: {spec.rationale}\n",
        f"Features: {', '.join(spec.feature_cols)}\n",
        "Missing-value handling: median imputation inside a sklearn pipeline, followed by standard scaling and logistic regression with class_weight='balanced'.\n",
        "Validation strategy: 5-fold stratified cross-validated out-of-fold probabilities.\n\n",
        f"Rows used: {len(sample)}\n",
        f"Stars: {int((sample['hst_label'] == 'star').sum())}\n",
        f"Galaxies: {int((sample['hst_label'] == 'galaxy').sum())}\n",
        f"ROC AUC: {result['roc_auc']:.4f}\n",
        f"Average precision: {result['average_precision']:.4f}\n\n",
        "Threshold metrics at P_star >= 0.5:\n",
    ]
    for key, val in result["threshold_metrics_05"].items():
        if key == "threshold":
            lines.append(f"  - {key}: {val:.3f}\n")
        elif isinstance(val, float):
            lines.append(f"  - {key}: {val:.4f}\n")
        else:
            lines.append(f"  - {key}: {val}\n")
    lines.append("\nThreshold metrics at best balanced-accuracy threshold:\n")
    for key, val in result["best_threshold_metrics"].items():
        if key == "threshold":
            lines.append(f"  - {key}: {val:.3f}\n")
        elif isinstance(val, float):
            lines.append(f"  - {key}: {val:.4f}\n")
        else:
            lines.append(f"  - {key}: {val}\n")
    lines.append("\nPer-feature missing fraction in the model sample:\n")
    for col, frac in frac_missing.items():
        lines.append(f"  - {col}: {frac:.4f}\n")
    out_path.write_text("".join(lines))


def evaluate_current_pSr(master_df: pd.DataFrame) -> dict:
    use = master_df[master_df["current_pS_r"].notna() & master_df["hst_label"].isin(["star", "galaxy"])].copy()
    y = (use["hst_label"] == "star").astype(int).to_numpy()
    p = use["current_pS_r"].to_numpy(dtype=float)

    from sklearn.calibration import calibration_curve
    from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

    roc_auc = float(roc_auc_score(y, p))
    avg_precision = float(average_precision_score(y, p))
    fpr, tpr, roc_thresholds = roc_curve(y, p)
    precision, recall, pr_thresholds = precision_recall_curve(y, p)
    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    threshold_table = pd.DataFrame([_threshold_metrics(y, p, thr) for thr in np.linspace(0.05, 0.95, 181)])
    best_idx = threshold_table["balanced_accuracy"].astype(float).idxmax()
    best_metrics = threshold_table.loc[int(best_idx)].to_dict()

    return {
        "sample": use,
        "y": y,
        "p": p,
        "roc_auc": roc_auc,
        "average_precision": avg_precision,
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "calibration_frac_pos": frac_pos,
        "calibration_mean_pred": mean_pred,
        "threshold_table": threshold_table,
        "best_threshold_metrics": best_metrics,
        "threshold_metrics_05": _threshold_metrics(y, p, 0.5),
    }


def write_current_vs_new_models(master_df: pd.DataFrame, pSr_result: dict, model_results: dict[str, dict], out_path: Path) -> None:
    lines = [
        "ECDFS current pS_r vs new models comparison\n",
        "==========================================\n\n",
        "Current notebook score:\n",
        "- `current_pS_r` is the existing Rubin single-band morphology + magnitude probability-like score.\n",
        "- It is derived from the notebook-style r-band PSF-CModel mixture model and does not include Rubin colors.\n\n",
        f"Current pS_r rows with usable score: {len(pSr_result['sample'])}\n",
        f"Current pS_r ROC AUC: {pSr_result['roc_auc']:.4f}\n",
        f"Current pS_r average precision: {pSr_result['average_precision']:.4f}\n",
        f"Current pS_r best balanced-accuracy threshold: {pSr_result['best_threshold_metrics']['threshold']:.3f}\n",
        f"Current pS_r best balanced accuracy: {pSr_result['best_threshold_metrics']['balanced_accuracy']:.4f}\n\n",
    ]

    for spec in MODEL_SPECS:
        result = model_results[spec.key]
        lines += [
            f"{spec.name}\n",
            f"  - rows: {len(result['sample'])}\n",
            f"  - ROC AUC: {result['roc_auc']:.4f}\n",
            f"  - average precision: {result['average_precision']:.4f}\n",
            f"  - best balanced-accuracy threshold: {result['best_threshold_metrics']['threshold']:.3f}\n",
            f"  - best balanced accuracy: {result['best_threshold_metrics']['balanced_accuracy']:.4f}\n\n",
        ]

    lines += [
        "Interpretation\n",
        "--------------\n",
        "1. The current `pS_r` should be compared most directly against Model A, because both are mainly morphology-driven.\n",
        "2. Model B is the first true Rubin photometry model in this chain because it uses multi-band colors and magnitudes without morphology diff features.\n",
        "3. Model C is the best match to the professor's wording if it improves over both Model A and `pS_r`, because it explicitly uses Rubin photometry plus morphology.\n\n",
    ]

    best_model = max(MODEL_SPECS, key=lambda spec: model_results[spec.key]["roc_auc"])
    lines.append(f"Best ROC AUC among the new models: {best_model.name} ({model_results[best_model.key]['roc_auc']:.4f}).\n")

    if model_results["B"]["roc_auc"] > model_results["A"]["roc_auc"]:
        lines.append("Adding Rubin colors improves over the morphology-only baseline in ROC AUC.\n")
    else:
        lines.append("Adding Rubin colors does not improve over the morphology-only baseline in ROC AUC for this ECDFS sample.\n")

    if model_results["C"]["roc_auc"] > pSr_result["roc_auc"]:
        lines.append("The combined model outperforms the current single-band pS_r score in ROC AUC.\n")
    else:
        lines.append("The combined model does not outperform the current single-band pS_r score in ROC AUC.\n")
    out_path.write_text("".join(lines))


def write_professor_summary(master_df: pd.DataFrame, pSr_result: dict, model_results: dict[str, dict], out_path: Path) -> None:
    best_model = max(MODEL_SPECS, key=lambda spec: model_results[spec.key]["roc_auc"])
    best_result = model_results[best_model.key]
    text = textwrap.dedent(
        f"""
        ECDFS professor-requested Rubin probability summary
        ================================================

        This ECDFS extension uses the clean HST/3D-HST matched sample as an external
        reference catalog and compares Rubin-side probability models against those
        labels.

        The existing notebook score `pS_r` is mainly a morphology + magnitude score:
        it uses the Rubin r-band PSF-CModel-like quantity together with r-band
        magnitude, but it does not use Rubin multi-band colors.

        The new probability-model extension builds three explicit baselines:
        - Model A: morphology-only (`r_diff`, `i_diff`)
        - Model B: photometry-only (Rubin colors + r/i magnitudes)
        - Model C: combined photometry + morphology

        HST labels are used only as the external reference target:
        star = HST/3D-HST star label, galaxy = HST/3D-HST galaxy label.

        ECDFS clean master sample:
        - rows: {len(master_df)}
        - stars: {int((master_df['hst_label'] == 'star').sum())}
        - galaxies: {int((master_df['hst_label'] == 'galaxy').sum())}

        Current notebook-style pS_r:
        - ROC AUC: {pSr_result['roc_auc']:.4f}
        - average precision: {pSr_result['average_precision']:.4f}

        Best new model:
        - {best_model.name}
        - ROC AUC: {best_result['roc_auc']:.4f}
        - average precision: {best_result['average_precision']:.4f}

        Photometric colors {'do' if model_results['B']['roc_auc'] > model_results['A']['roc_auc'] else 'do not'} improve over the morphology-only baseline in this ECDFS sample.
        The combined model {'does' if model_results['C']['roc_auc'] > pSr_result['roc_auc'] else 'does not'} outperform the current single-band pS_r score in ROC AUC.

        Recommended ECDFS main result going forward:
        - keep `pS_r` as the notebook-compatible morphology baseline
        - treat the combined model as the more professor-aligned Rubin photometry probability result
          whenever it matches or outperforms the single-band baseline.
        """
    ).strip() + "\n"
    out_path.write_text(text)


def run_ecdfs_probability_models(
    dp1_path: Path = DEFAULT_DP1,
    matched_path: Path = DEFAULT_MATCHED,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    from sklearn.exceptions import UndefinedMetricWarning

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
    output_dir.mkdir(parents=True, exist_ok=True)
    out = _output_paths(output_dir)

    processed_df = load_ecdfs_processed_table(dp1_path)
    master_df = build_master_sample(processed_df, matched_path)
    master_df["current_pS_r"] = compute_current_pSr(processed_df, master_df)

    write_feature_audit(processed_df, master_df, out["audit"])
    write_master_sample(master_df, out["master_sample"], out["master_summary"])

    model_results: dict[str, dict] = {}
    for spec in MODEL_SPECS:
        sample, X, y = build_model_sample(master_df, spec.feature_cols)
        result = evaluate_logistic_model(sample, X, y, spec.feature_cols)
        model_results[spec.key] = result
        outputs = out["models"][spec.key]
        plot_probability_hist(result, outputs["hist"], f"{spec.name} - P_star distributions")
        plot_roc(result, outputs["roc"], f"{spec.name} - ROC")
        plot_pr(result, outputs["pr"], f"{spec.name} - precision-recall")
        plot_pstar_vs_mag(result, outputs["pstar_vs_mag"], f"{spec.name} - P_star vs magnitude")
        plot_calibration(result, outputs["calibration"], f"{spec.name} - calibration")
        write_model_summary(spec, result, outputs["summary"])

    pSr_result = evaluate_current_pSr(master_df)
    plot_model_comparison_curves(pSr_result, model_results, out["comparison_plots"])
    write_current_vs_new_models(master_df, pSr_result, model_results, out["comparison"])
    write_professor_summary(master_df, pSr_result, model_results, out["professor_summary"])

    return {
        "processed_df": processed_df,
        "master_df": master_df,
        "pSr_result": pSr_result,
        "model_results": model_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp1", type=Path, default=DEFAULT_DP1)
    parser.add_argument("--matched", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    run_ecdfs_probability_models(args.dp1, args.matched, args.output_dir)


if __name__ == "__main__":
    main()
