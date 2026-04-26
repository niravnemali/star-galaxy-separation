"""Build a first Rubin-side P_star model using a clean labeled Rubin matched sample.

This is intentionally a simple baseline model.  It uses external star/galaxy
labels as truth-like supervision and Rubin-side photometry/morphology as predictors.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from validate_dp1_against_hst import (
    DEFAULT_MATCHED,
    OUTPUT_DIR,
    clean_morphology_sample,
    compute_psf_cmodel_features,
)


def add_color_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple Rubin color features from derived CModel magnitudes."""
    work = df.copy()
    color_defs = {
        "g_minus_r": ("g_cmodel_mag", "r_cmodel_mag"),
        "r_minus_i": ("r_cmodel_mag", "i_cmodel_mag"),
        "i_minus_z": ("i_cmodel_mag", "z_cmodel_mag"),
    }
    for color, (left, right) in color_defs.items():
        if left in work.columns and right in work.columns:
            work[color] = pd.to_numeric(work[left], errors="coerce") - pd.to_numeric(work[right], errors="coerce")
    return work


def build_feature_sample(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return rows with valid features and binary labels, where star=1 and galaxy=0."""
    required = ["hst_label"] + feature_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for model: {missing}")
    work = df.copy()
    for col in feature_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    mask = work[feature_cols].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    mask &= work["hst_label"].isin(["star", "galaxy"])
    sample = work.loc[mask].copy().reset_index(drop=True)
    y = (sample["hst_label"] == "star").astype(int).to_numpy()
    x = sample[feature_cols].to_numpy(dtype=float)
    return sample, x, y


def evaluate_logistic_model(sample: pd.DataFrame, x: np.ndarray, y: np.ndarray, feature_cols: list[str], model_name: str) -> dict:
    """Train/test a class-weighted logistic regression model."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for the baseline probability model.") from exc

    n_star = int(y.sum())
    n_gal = int(len(y) - n_star)
    if len(y) < 50 or n_star < 10 or n_gal < 10:
        raise ValueError(f"Not enough labeled rows for {model_name}: n={len(y)}, stars={n_star}, galaxies={n_gal}")

    x_train, x_test, y_train, y_test, idx_train, idx_test = train_test_split(
        x,
        y,
        np.arange(len(y)),
        test_size=0.3,
        random_state=42,
        stratify=y,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs"),
    )
    model.fit(x_train, y_train)
    p_test = model.predict_proba(x_test)[:, 1]
    roc_auc = roc_auc_score(y_test, p_test)
    avg_precision = average_precision_score(y_test, p_test)
    fpr, tpr, roc_thresholds = roc_curve(y_test, p_test)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, p_test)

    final_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs"),
    )
    final_model.fit(x, y)
    p_all = final_model.predict_proba(x)[:, 1]

    return {
        "model_name": model_name,
        "feature_cols": feature_cols,
        "sample": sample,
        "y": y,
        "p_all": p_all,
        "x_test": x_test,
        "y_test": y_test,
        "p_test": p_test,
        "idx_test": idx_test,
        "roc_auc": float(roc_auc),
        "average_precision": float(avg_precision),
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "precision": precision,
        "recall": recall,
        "pr_thresholds": pr_thresholds,
        "n_total": len(y),
        "n_star": n_star,
        "n_galaxy": n_gal,
    }


def plot_probability_outputs(result: dict, output_dir: Path, label_description: str = "ECDFS: HST/3D-HST GOODS-S truth labels") -> None:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    y_test = result["y_test"]
    p_test = result["p_test"]

    plt.figure(figsize=(7, 4))
    for label, truth, color in [(f"{label_description}: stars", 1, "black"), (f"{label_description}: galaxies", 0, "tab:blue")]:
        vals = p_test[y_test == truth]
        plt.hist(vals, bins=np.linspace(0, 1, 51), histtype="step", lw=1.8, density=True, label=label, color=color)
    plt.xlabel("P_star")
    plt.ylabel("Density")
    plt.title(f"{label_description} - P_star distributions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pstar_histograms_by_hst_label.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(result["fpr"], result["tpr"], lw=1.8, label=f"ROC AUC = {result['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"{label_description} - P_star ROC")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pstar_roc_curve.png", dpi=180)
    plt.close()

    plt.figure(figsize=(5, 5))
    plt.plot(result["recall"], result["precision"], lw=1.8, label=f"AP = {result['average_precision']:.3f}")
    plt.xlabel("Star recall")
    plt.ylabel("Star precision")
    plt.title(f"{label_description} - P_star precision-recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pstar_precision_recall_curve.png", dpi=180)
    plt.close()

    frac_pos, mean_pred = calibration_curve(y_test, p_test, n_bins=8, strategy="quantile")
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, "o-", label="model")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8, label="ideal")
    plt.xlabel("Mean predicted P_star")
    plt.ylabel("Observed star fraction")
    plt.title(f"{label_description} - P_star calibration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "pstar_calibration.png", dpi=180)
    plt.close()


def run_probability_model(
    matched_path: Path = DEFAULT_MATCHED,
    output_dir: Path = OUTPUT_DIR,
    label_description: str = "ECDFS: HST/3D-HST GOODS-S truth labels",
) -> pd.DataFrame | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not matched_path.exists():
        msg = f"Clean matched table not found: {matched_path}\nRun src/match_hst_to_dp1.py first.\n"
        (output_dir / "rubin_star_probability_model_summary.txt").write_text(msg)
        print(msg)
        return None

    df = pd.read_csv(matched_path)
    df, _ = compute_psf_cmodel_features(df)
    _, clean_df, _ = clean_morphology_sample(df, output_dir, label_description=label_description)
    clean_df = add_color_features(clean_df)

    model_specs = [
        ("Model A: morphology + magnitude", ["i_psf_minus_cmodel", "i_cmodel_mag"]),
        ("Model B: morphology + magnitude + colors", ["i_psf_minus_cmodel", "i_cmodel_mag", "g_minus_r", "r_minus_i", "i_minus_z"]),
    ]
    results = []
    lines = [
        "Rubin star probability model summary\n",
        "====================================\n\n",
        f"Matched table: {matched_path}\n",
        f"Truth/label source: {label_description}\n",
        f"Clean morphology rows available: {len(clean_df)}\n",
        "Class convention: label star = 1, label galaxy = 0.\n",
        "The model uses class_weight='balanced' to reduce sensitivity to class imbalance.\n\n",
    ]

    for model_name, feature_cols in model_specs:
        try:
            sample, x, y = build_feature_sample(clean_df, feature_cols)
            result = evaluate_logistic_model(sample, x, y, feature_cols, model_name)
            results.append(result)
            lines += [
                f"{model_name}\n",
                f"  features: {', '.join(feature_cols)}\n",
                f"  training/evaluation rows: {result['n_total']}\n",
                f"  label stars: {result['n_star']}\n",
                f"  label galaxies: {result['n_galaxy']}\n",
                f"  test ROC AUC: {result['roc_auc']:.4f}\n",
                f"  test average precision: {result['average_precision']:.4f}\n\n",
            ]
        except Exception as exc:
            lines += [f"{model_name}\n  skipped: {exc}\n\n"]

    if not results:
        lines.append("No probability model could be trained.\n")
        (output_dir / "rubin_star_probability_model_summary.txt").write_text("".join(lines))
        print("".join(lines))
        return None

    best = max(results, key=lambda r: r["roc_auc"])
    plot_probability_outputs(best, output_dir, label_description=label_description)
    out = best["sample"].copy()
    out["p_star"] = best["p_all"]
    out["p_star_model"] = best["model_name"]
    out.to_csv(output_dir / "hst_dp1_matched_with_pstar.csv", index=False)

    lines += [
        "Selected baseline model\n",
        "-----------------------\n",
        f"{best['model_name']} was selected by held-out ROC AUC.\n",
        f"Selected model ROC AUC: {best['roc_auc']:.4f}\n",
        f"Selected model average precision: {best['average_precision']:.4f}\n",
        f"Wrote probability table: {output_dir / 'hst_dp1_matched_with_pstar.csv'}\n",
    ]
    summary = "".join(lines)
    (output_dir / "rubin_star_probability_model_summary.txt").write_text(summary)

    comparison = (output_dir / "morphology_quantity_comparison_summary.txt").read_text() if (output_dir / "morphology_quantity_comparison_summary.txt").exists() else ""
    interpretation_v2 = [
        "Rubin morphology/probability interpretation summary v2\n",
        "=====================================================\n\n",
        f"Truth/label source: {label_description}\n",
        "1. Clean matching results\n",
        f"The probability model uses {matched_path.name}, the recentered 3-sigma elliptical clean match sample when available.\n\n",
        "2. Morphology signal strength and magnitude dependence\n",
        comparison,
        "\n3. First Rubin-side probability model\n",
        summary,
        "\n4. Recommended next science step\n",
        "Treat this as a baseline, not a final classifier.  Next, inspect performance by magnitude and sky region, then compare against another DP1 FITS table or a held-out area.\n",
    ]
    (output_dir / "dp1_vs_hst_interpretation_summary_v2.txt").write_text("".join(interpretation_v2))
    print(summary)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matched", type=Path, default=DEFAULT_MATCHED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--label-description", default="ECDFS: HST/3D-HST GOODS-S truth labels")
    args = parser.parse_args()
    run_probability_model(args.matched, args.output_dir, label_description=args.label_description)


if __name__ == "__main__":
    main()
