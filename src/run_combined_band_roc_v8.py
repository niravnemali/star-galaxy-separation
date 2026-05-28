#!/usr/bin/env python3
"""Generate combined-band v8 pS ROC diagnostics.

The ROC scores are validation diagnostics built from existing external-label
matched catalogs. The likelihood-combination score is an empirical
log-likelihood ratio, not a calibrated posterior probability.

This script reads prepared parquet/CSV products only. It does not read or modify
private FITS catalogs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PLOTS = REPO_ROOT / "plots"
RESULTS = REPO_ROOT / "results"
PAPER = REPO_ROOT / "paper"
VERSION = "v8"
BANDS = ("u", "g", "r", "i", "z", "y")
REFERENCE_MAG_COL = "dp2_cmodel_mag_r"
PS_BINS = 50
EPSILON = 1e-6
RANDOM_SEED = 20260527

MAG_BINS = (
    ("mag < 22", -np.inf, 22.0, 21.5),
    ("22 <= mag < 23", 22.0, 23.0, 22.5),
    ("23 <= mag < 24", 23.0, 24.0, 23.5),
    ("24 <= mag < 24.5", 24.0, 24.5, 24.25),
    ("24.5 <= mag < 25", 24.5, 25.0, 24.75),
    ("25 <= mag < 25.5", 25.0, 25.5, 25.25),
    ("25.5 <= mag < 26", 25.5, 26.0, 25.75),
    ("26 <= mag < 26.5", 26.0, 26.5, 26.25),
)
COMBINATIONS = {
    "r_only": ("r",),
    "gri": ("g", "r", "i"),
    "ugrizy": ("u", "g", "r", "i", "z", "y"),
}
COMBO_LABELS = {
    "r_only": "r only",
    "gri": "g+r+i",
    "ugrizy": "u+g+r+i+z+y",
}
COMBO_COLORS = {
    "r_only": "tab:blue",
    "gri": "tab:orange",
    "ugrizy": "tab:green",
}


@dataclass(frozen=True)
class FieldConfig:
    field: str
    source: str
    ps_path: Path
    matched_path: Path


CONFIGS = (
    FieldConfig(
        field="ECDFS",
        source="HST",
        ps_path=REPO_ROOT / "outputs" / "dp2_ecdfs_ps_v8.parquet",
        matched_path=REPO_ROOT / "outputs" / "dp2_ecdfs_hst_matched.parquet",
    ),
    FieldConfig(
        field="COSMOS",
        source="COSMOS2020",
        ps_path=REPO_ROOT / "outputs" / "dp2_cosmos_ps_v8.parquet",
        matched_path=REPO_ROOT / "outputs" / "dp2_cosmos_cosmos2020_farmer_matched.parquet",
    ),
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ps_col(band: str) -> str:
    return f"pS_{band}"


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def read_parquet_existing(path: Path, columns: list[str]) -> pd.DataFrame:
    import pyarrow.parquet as pq

    available = set(pq.ParquetFile(path).schema.names)
    keep = [c for c in columns if c in available]
    return pd.read_parquet(path, columns=keep)


def valid_png(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False


def save_or_skip(fig: plt.Figure, path: Path) -> str:
    ensure_dir(path.parent)
    if valid_png(path):
        plt.close(fig)
        print(f"[SKIP] {rel(path)}")
        return "skipped"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {rel(path)}")
    return "saved"


def truth_binary(df: pd.DataFrame) -> pd.Series:
    if "truth_binary" in df.columns:
        return numeric(df["truth_binary"])
    return df["truth_label"].astype(str).str.lower().map({"star": 1, "galaxy": 0})


def mag_bin_mask(mag: pd.Series, lo: float, hi: float) -> pd.Series:
    mask = np.isfinite(mag) & (mag < hi)
    if np.isfinite(lo):
        mask &= mag >= lo
    return mask


def load_sample(config: FieldConfig) -> tuple[pd.DataFrame, list[str]]:
    notes: list[str] = []
    ps_columns = ["object_id"] + [ps_col(b) for b in BANDS] + ["ps_version"]
    matched_columns = ["object_id", "dp2_object_id", "truth_binary", "truth_label", "external_label_source", REFERENCE_MAG_COL]
    ps = read_parquet_existing(config.ps_path, ps_columns)
    matched = read_parquet_existing(config.matched_path, matched_columns)
    if "object_id" not in matched.columns and "dp2_object_id" in matched.columns:
        matched = matched.rename(columns={"dp2_object_id": "object_id"})
    if ps["object_id"].duplicated().any():
        notes.append(f"{config.ps_path.name} has duplicate object_id rows; kept first occurrence.")
        ps = ps.drop_duplicates("object_id", keep="first")
    sample = matched.merge(ps, on="object_id", how="left", suffixes=("", "_ps"))
    sample["truth_binary"] = truth_binary(sample)
    sample = sample[sample["truth_binary"].isin([0, 1])].copy()
    sample[REFERENCE_MAG_COL] = numeric(sample[REFERENCE_MAG_COL])
    for band in BANDS:
        col = ps_col(band)
        if col not in sample.columns:
            notes.append(f"missing pS column: {col}")
        else:
            sample[col] = numeric(sample[col])
    if REFERENCE_MAG_COL not in sample.columns:
        notes.append(f"missing reference magnitude column: {REFERENCE_MAG_COL}")
    versions = sorted(map(str, sample.get("ps_version", pd.Series(dtype=str)).dropna().unique()))
    if versions and VERSION not in versions:
        notes.append(f"pS version values are {versions}; expected {VERSION}.")
    return sample, notes


def compute_roc(y_true: np.ndarray, score: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(-score, kind="mergesort")
    y = y_true[order].astype(int)
    s = score[order]
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        raise ValueError("one class missing")
    idx = np.r_[np.where(s[1:] != s[:-1])[0], len(s) - 1]
    tps = np.cumsum(y == 1)[idx]
    fps = np.cumsum(y == 0)[idx]
    tpr = np.r_[0.0, tps / pos]
    fpr = np.r_[0.0, fps / neg]
    auc = float(np.trapezoid(tpr, fpr))
    return fpr, tpr, auc


def density_from_values(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(0.0, 1.0, PS_BINS + 1)
    hist, _ = np.histogram(np.clip(values, 0, 1), bins=edges, density=True)
    hist = np.maximum(hist.astype(float), EPSILON)
    return hist, edges


def density_lookup(values: np.ndarray, density: np.ndarray, edges: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(edges, np.clip(values, 0, 1), side="right") - 1
    idx = np.clip(idx, 0, len(density) - 1)
    return density[idx]


def build_band_densities(train: pd.DataFrame, band: str) -> tuple[dict[str, np.ndarray] | None, dict]:
    col = ps_col(band)
    star_vals = train.loc[train["truth_binary"].eq(1), col].dropna().to_numpy(float)
    gal_vals = train.loc[train["truth_binary"].eq(0), col].dropna().to_numpy(float)
    diag = {
        "N_star_train": int(len(star_vals)),
        "N_galaxy_train": int(len(gal_vals)),
        "density_method": "histogram_density",
        "pS_bin_count": PS_BINS,
        "epsilon_floor": EPSILON,
        "density_built_successfully": bool(len(star_vals) > 0 and len(gal_vals) > 0),
    }
    if len(star_vals) == 0 or len(gal_vals) == 0:
        return None, diag
    star_density, edges = density_from_values(star_vals)
    gal_density, _ = density_from_values(gal_vals)
    return {"star": star_density, "galaxy": gal_density, "edges": edges}, diag


def stratified_fold_ids(df: pd.DataFrame, n_folds: int, seed: int) -> pd.Series:
    fold = pd.Series(-1, index=df.index, dtype=int)
    rng = np.random.default_rng(seed)
    for cls in (0, 1):
        idx = df.index[df["truth_binary"].eq(cls)].to_numpy()
        rng.shuffle(idx)
        for i, item in enumerate(idx):
            fold.loc[item] = i % n_folds
    return fold


def add_mean_scores(sample: pd.DataFrame) -> pd.DataFrame:
    out = sample.copy()
    for combo, bands in COMBINATIONS.items():
        cols = [ps_col(b) for b in bands]
        out[f"score_{combo}_mean"] = out[cols].mean(axis=1, skipna=True)
        out[f"n_bands_{combo}_mean"] = out[cols].notna().sum(axis=1)
        out.loc[out[f"n_bands_{combo}_mean"].eq(0), f"score_{combo}_mean"] = np.nan
    return out


def add_likelihood_scores(sample: pd.DataFrame, config: FieldConfig) -> tuple[pd.DataFrame, list[dict]]:
    out = sample.copy()
    diagnostics: list[dict] = []
    for combo in COMBINATIONS:
        out[f"score_{combo}_likelihood"] = np.nan
        out[f"n_bands_{combo}_likelihood"] = 0
        out[f"cv_used_{combo}_likelihood"] = False
        out[f"notes_{combo}_likelihood"] = ""

    for bin_index, (label, lo, hi, _) in enumerate(MAG_BINS):
        in_bin = mag_bin_mask(out[REFERENCE_MAG_COL], lo, hi)
        bin_df = out.loc[in_bin].copy()
        n_star = int(bin_df["truth_binary"].eq(1).sum())
        n_gal = int(bin_df["truth_binary"].eq(0).sum())
        if n_star == 0 or n_gal == 0:
            reason = "N_star=0" if n_star == 0 else "N_galaxy=0"
            for band in BANDS:
                diagnostics.append(
                    {
                        "field": config.field,
                        "external_label_source": config.source,
                        "magnitude_bin": label,
                        "fold": "not_built",
                        "band": band,
                        "N_star_train": n_star,
                        "N_galaxy_train": n_gal,
                        "density_method": "histogram_density",
                        "pS_bin_count": PS_BINS,
                        "epsilon_floor": EPSILON,
                        "density_built_successfully": False,
                        "cross_validation_used": False,
                        "notes": reason,
                    }
                )
            for combo in COMBINATIONS:
                out.loc[in_bin, f"notes_{combo}_likelihood"] = reason
            continue

        min_class = min(n_star, n_gal)
        use_cv = min_class >= 2
        n_folds = min(5, min_class) if use_cv else 1
        if use_cv:
            fold_ids = stratified_fold_ids(bin_df, n_folds, RANDOM_SEED + bin_index)
            fold_iter = list(range(n_folds))
        else:
            fold_ids = pd.Series(0, index=bin_df.index)
            fold_iter = [0]

        for fold in fold_iter:
            if use_cv:
                train = bin_df.loc[fold_ids.ne(fold)]
                held_idx = bin_df.index[fold_ids.eq(fold)]
                fold_label = str(fold)
                note = f"{n_folds}-fold out-of-fold likelihood"
            else:
                train = bin_df
                held_idx = bin_df.index
                fold_label = "all"
                note = "same-bin density fallback because one class has fewer than 2 objects"

            densities: dict[str, dict[str, np.ndarray]] = {}
            for band in BANDS:
                density, diag = build_band_densities(train, band)
                diag.update(
                    {
                        "field": config.field,
                        "external_label_source": config.source,
                        "magnitude_bin": label,
                        "fold": fold_label,
                        "band": band,
                        "cross_validation_used": use_cv,
                        "notes": note if diag["density_built_successfully"] else "density not built because one class has no valid pS values",
                    }
                )
                diagnostics.append(diag)
                if density is not None:
                    densities[band] = density

            held = out.loc[held_idx]
            for combo, bands in COMBINATIONS.items():
                score = pd.Series(0.0, index=held_idx)
                n_used = pd.Series(0, index=held_idx, dtype=int)
                for band in bands:
                    if band not in densities:
                        continue
                    values = numeric(held[ps_col(band)])
                    valid = np.isfinite(values)
                    if not valid.any():
                        continue
                    density = densities[band]
                    star_d = density_lookup(values[valid].to_numpy(float), density["star"], density["edges"])
                    gal_d = density_lookup(values[valid].to_numpy(float), density["galaxy"], density["edges"])
                    score.loc[valid.index[valid]] += np.log(star_d) - np.log(gal_d)
                    n_used.loc[valid.index[valid]] += 1
                valid_score = n_used.gt(0)
                out.loc[held_idx[valid_score], f"score_{combo}_likelihood"] = score.loc[valid_score]
                out.loc[held_idx, f"n_bands_{combo}_likelihood"] = n_used
                out.loc[held_idx, f"cv_used_{combo}_likelihood"] = use_cv
                out.loc[held_idx, f"notes_{combo}_likelihood"] = note

    return out, diagnostics


def summarize_method(sample: pd.DataFrame, config: FieldConfig, method: str) -> list[dict]:
    rows: list[dict] = []
    for label, lo, hi, _ in MAG_BINS:
        in_bin = mag_bin_mask(sample[REFERENCE_MAG_COL], lo, hi) & sample["truth_binary"].isin([0, 1])
        for combo in COMBINATIONS:
            score_col = f"score_{combo}_{method}"
            n_col = f"n_bands_{combo}_{method}"
            cv_col = f"cv_used_{combo}_{method}"
            notes_col = f"notes_{combo}_{method}"
            base = sample.loc[in_bin].copy()
            valid = np.isfinite(base[score_col])
            eval_df = base.loc[valid]
            y = eval_df["truth_binary"].to_numpy(int)
            score = eval_df[score_col].to_numpy(float)
            n_star = int(np.sum(y == 1))
            n_gal = int(np.sum(y == 0))
            row = {
                "field": config.field,
                "external_label_source": config.source,
                "method": method,
                "combination": combo,
                "magnitude_bin": label,
                "reference_magnitude_column": REFERENCE_MAG_COL,
                "N_total": int(len(base)),
                "N_star": n_star,
                "N_galaxy": n_gal,
                "N_valid_score": int(len(eval_df)),
                "AUC": np.nan,
                "roc_computed": False,
                "not_computed_reason": "",
                "score_direction": "larger score is more star-like",
                "number_of_bands_used_min": np.nan,
                "number_of_bands_used_median": np.nan,
                "number_of_bands_used_max": np.nan,
                "cross_validation_used": bool(eval_df[cv_col].any()) if method == "likelihood" and cv_col in eval_df else False,
                "notes": "",
            }
            if len(eval_df):
                used = eval_df[n_col]
                row["number_of_bands_used_min"] = int(used.min())
                row["number_of_bands_used_median"] = float(used.median())
                row["number_of_bands_used_max"] = int(used.max())
                if method == "likelihood" and notes_col in eval_df:
                    row["notes"] = "; ".join(sorted(set(str(x) for x in eval_df[notes_col].dropna().unique() if str(x))))
            if n_star == 0 or n_gal == 0:
                if n_star == 0 and n_gal == 0:
                    row["not_computed_reason"] = "one class missing"
                elif n_star == 0:
                    row["not_computed_reason"] = "N_star=0"
                else:
                    row["not_computed_reason"] = "N_galaxy=0"
            else:
                _, _, auc = compute_roc(y, score)
                row["AUC"] = auc
                row["roc_computed"] = True
            rows.append(row)
    return rows


def plot_combined_roc(summary: pd.DataFrame, samples: dict[str, pd.DataFrame], configs: dict[str, FieldConfig], method: str) -> list[Path]:
    paths: list[Path] = []
    for field, sample in samples.items():
        config = configs[field]
        fig, axes = plt.subplots(2, 4, figsize=(20, 9.5), sharex=True, sharey=True)
        for ax, (label, lo, hi, _) in zip(axes.ravel(), MAG_BINS):
            ax.plot([0, 1], [0, 1], color="0.65", ls="--", lw=1.0, label="random")
            notes: list[str] = []
            in_bin = mag_bin_mask(sample[REFERENCE_MAG_COL], lo, hi) & sample["truth_binary"].isin([0, 1])
            for combo in COMBINATIONS:
                score_col = f"score_{combo}_{method}"
                eval_df = sample.loc[in_bin & np.isfinite(sample[score_col])]
                y = eval_df["truth_binary"].to_numpy(int)
                score = eval_df[score_col].to_numpy(float)
                n_star = int(np.sum(y == 1))
                n_gal = int(np.sum(y == 0))
                sub = summary[
                    summary["field"].eq(field)
                    & summary["method"].eq(method)
                    & summary["combination"].eq(combo)
                    & summary["magnitude_bin"].eq(label)
                ].iloc[0]
                if not bool(sub["roc_computed"]):
                    notes.append(f"{COMBO_LABELS[combo]}: {sub['not_computed_reason']}")
                    continue
                fpr, tpr, auc = compute_roc(y, score)
                ax.plot(
                    fpr,
                    tpr,
                    lw=2.0,
                    color=COMBO_COLORS[combo],
                    label=f"{COMBO_LABELS[combo]}: AUC={auc:.2f} (N_star={n_star}, N_gal={n_gal})",
                )
            if notes:
                ax.text(
                    0.03,
                    0.05,
                    "not computed:\n" + "\n".join(notes),
                    transform=ax.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=7.5,
                    bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.9, "pad": 2},
                )
            ax.set_title(label, fontsize=10)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.22)
            ax.legend(fontsize=7.2, loc="lower right", framealpha=0.88)
        for ax in axes[-1, :]:
            ax.set_xlabel("False positive rate (contamination)")
        for ax in axes[:, 0]:
            ax.set_ylabel("True positive rate (completeness)")
        fig.suptitle(
            f"DP2 {field} v8 combined-band ROC ({method}); external labels: {config.source}; reference magnitude: {REFERENCE_MAG_COL}",
            y=0.995,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.955])
        path = PLOTS / f"combined_roc_v8_{method}_DP2_{field}_{config.source}_8magbins.png"
        save_or_skip(fig, path)
        paths.append(path)
    return paths


def plot_auc_vs_mag(summary: pd.DataFrame, configs: dict[str, FieldConfig], method: str) -> list[Path]:
    paths: list[Path] = []
    centers = np.array([x[3] for x in MAG_BINS])
    labels = [x[0] for x in MAG_BINS]
    for field, config in configs.items():
        fig, ax = plt.subplots(figsize=(9.5, 5.8))
        for combo in COMBINATIONS:
            sub = summary[summary["field"].eq(field) & summary["method"].eq(method) & summary["combination"].eq(combo)].copy()
            auc = []
            for label in labels:
                row = sub[sub["magnitude_bin"].eq(label)]
                auc.append(float(row["AUC"].iloc[0]) if len(row) else np.nan)
            ax.plot(centers, auc, marker="o", lw=2.0, color=COMBO_COLORS[combo], label=COMBO_LABELS[combo])
        ax.set_xlabel(f"{REFERENCE_MAG_COL} bin center")
        ax.set_ylabel("AUC")
        ax.set_ylim(0, 1.03)
        ax.grid(True, alpha=0.25)
        ax.legend()
        ax.set_title(f"DP2 {field} v8 combined-band AUC vs magnitude ({method}); external labels: {config.source}")
        path = PLOTS / f"combined_auc_vs_mag_v8_{method}_DP2_{field}_{config.source}.png"
        save_or_skip(fig, path)
        paths.append(path)
    return paths


def write_input_summary(samples: dict[str, pd.DataFrame], notes: dict[str, list[str]]) -> Path:
    lines = [
        "# Combined-Band ROC v8 Input Summary",
        "",
        "- pS version: `v8`.",
        f"- Reference magnitude column: `{REFERENCE_MAG_COL}`.",
        "- Positive class: external-label star.",
        "- pS columns used: `pS_u`, `pS_g`, `pS_r`, `pS_i`, `pS_z`, `pS_y`.",
        "- Combined scores: simple mean and empirical histogram-density log-likelihood ratio.",
        "",
    ]
    for config in CONFIGS:
        sample = samples[config.field]
        valid_ref = sample[REFERENCE_MAG_COL].notna()
        lines.extend(
            [
                f"## {config.field}",
                "",
                f"- External label source: `{config.source}`",
                f"- pS file: `{rel(config.ps_path)}`",
                f"- matched label file: `{rel(config.matched_path)}`",
                f"- rows with external labels: {len(sample):,}",
                f"- rows with valid `{REFERENCE_MAG_COL}`: {int(valid_ref.sum()):,}",
                f"- external stars with valid reference magnitude: {int((valid_ref & sample['truth_binary'].eq(1)).sum()):,}",
                f"- external galaxies with valid reference magnitude: {int((valid_ref & sample['truth_binary'].eq(0)).sum()):,}",
            ]
        )
        if notes.get(config.field):
            lines.append("- Notes:")
            lines.extend([f"  - {note}" for note in notes[config.field]])
        lines.append("")
    path = RESULTS / "combined_roc_v8_input_summary.md"
    path.write_text("\n".join(lines).rstrip() + "\n")
    return path


def update_paper_candidates() -> Path:
    ensure_dir(PAPER)
    path = PAPER / "figure_candidates.md"
    existing = path.read_text() if path.exists() else ""
    marker = "## Combined-Band pS Validation"
    if marker in existing:
        existing = existing.split(marker)[0].rstrip()
    lines = [
        "",
        marker,
        "",
        "These figures compare r-only, g+r+i, and u+g+r+i+z+y combined-band v8 pS validation diagnostics. The reference magnitude for all combinations is `dp2_cmodel_mag_r`, so the curves compare the same magnitude-selected samples.",
        "",
        "### Mean-score combination",
        "- `plots/combined_roc_v8_mean_DP2_ECDFS_HST_8magbins.png`: ECDFS/HST ROC curves for simple mean combined scores in eight r-band magnitude bins.",
        "- `plots/combined_roc_v8_mean_DP2_COSMOS_COSMOS2020_8magbins.png`: COSMOS/COSMOS2020 ROC curves for simple mean combined scores in eight r-band magnitude bins.",
        "- `plots/combined_auc_vs_mag_v8_mean_DP2_ECDFS_HST.png`: ECDFS/HST AUC vs r-band magnitude for the three mean-score combinations.",
        "- `plots/combined_auc_vs_mag_v8_mean_DP2_COSMOS_COSMOS2020.png`: COSMOS/COSMOS2020 AUC vs r-band magnitude for the three mean-score combinations.",
        "",
        "### Empirical likelihood combination",
        "- `plots/combined_roc_v8_likelihood_DP2_ECDFS_HST_8magbins.png`: ECDFS/HST ROC curves for empirical log-likelihood-ratio combined scores.",
        "- `plots/combined_roc_v8_likelihood_DP2_COSMOS_COSMOS2020_8magbins.png`: COSMOS/COSMOS2020 ROC curves for empirical log-likelihood-ratio combined scores.",
        "- `plots/combined_auc_vs_mag_v8_likelihood_DP2_ECDFS_HST.png`: ECDFS/HST AUC vs r-band magnitude for likelihood-combined scores.",
        "- `plots/combined_auc_vs_mag_v8_likelihood_DP2_COSMOS_COSMOS2020.png`: COSMOS/COSMOS2020 AUC vs r-band magnitude for likelihood-combined scores.",
        "",
        "Caveats: ECDFS has small HST star counts in some bins, while COSMOS has better statistics. The likelihood-combination score is empirical and calibrated from external labels; later simulation/prior code can improve posterior calibration.",
    ]
    path.write_text(existing.rstrip() + "\n" + "\n".join(lines).rstrip() + "\n")
    return path


def main() -> int:
    ensure_dir(PLOTS)
    ensure_dir(RESULTS)
    configs = {config.field: config for config in CONFIGS}
    samples: dict[str, pd.DataFrame] = {}
    input_notes: dict[str, list[str]] = {}
    density_rows: list[dict] = []
    summary_rows: list[dict] = []

    for config in CONFIGS:
        sample, notes = load_sample(config)
        input_notes[config.field] = notes
        sample = add_mean_scores(sample)
        sample, diagnostics = add_likelihood_scores(sample, config)
        samples[config.field] = sample
        density_rows.extend(diagnostics)
        for method in ("mean", "likelihood"):
            summary_rows.extend(summarize_method(sample, config, method))

    summary = pd.DataFrame(summary_rows)
    density = pd.DataFrame(density_rows)
    summary_path = RESULTS / "combined_roc_v8_summary.csv"
    density_path = RESULTS / "combined_likelihood_density_diagnostics_v8.csv"
    summary.to_csv(summary_path, index=False)
    density.to_csv(density_path, index=False)
    input_path = write_input_summary(samples, input_notes)

    plot_paths: list[Path] = []
    for method in ("mean", "likelihood"):
        plot_paths.extend(plot_combined_roc(summary, samples, configs, method))
        plot_paths.extend(plot_auc_vs_mag(summary, configs, method))
    paper_path = update_paper_candidates()

    manifest_path = RESULTS / "combined_roc_v8_generated_files.txt"
    manifest_path.write_text(
        "\n".join(
            [
                "Generated or checked plots:",
                *[rel(p) for p in plot_paths],
                "",
                "Generated tables/summaries:",
                rel(summary_path),
                rel(density_path),
                rel(input_path),
                rel(paper_path),
            ]
        )
        + "\n"
    )

    print(f"summary_rows={len(summary)}")
    print(f"density_diagnostic_rows={len(density)}")
    print(f"roc_not_computed={int((~summary['roc_computed']).sum())}")
    print(f"wrote {rel(summary_path)}")
    print(f"wrote {rel(density_path)}")
    print(f"wrote {rel(input_path)}")
    print(f"wrote {rel(manifest_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
