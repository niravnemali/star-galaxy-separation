#!/usr/bin/env python3
"""Validate and prepare the Butler-derived DP1 ECDFS Object catalog."""

from __future__ import annotations

import argparse
import ast
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "outputs" / "dp1_ecdfs_object_butler_alltracts.parquet"
DEFAULT_COLUMNS = REPO_ROOT / "results" / "dp1_butler_column_list.txt"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_ANALYSIS_OUTPUT = REPO_ROOT / "outputs" / "dp1_ecdfs_analysis_table_butler.parquet"
DEFAULT_LIST_NAME = "target_butler_column_names"
BANDS = ["u", "g", "r", "i", "z", "y"]
AB_ZEROPOINT_JY = 3631.0
ECDFS_RA_MIN = 52.35
ECDFS_RA_MAX = 53.93
ECDFS_DEC_MIN = -28.76
ECDFS_DEC_MAX = -27.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Butler Object parquet table.")
    parser.add_argument("--columns", type=Path, default=DEFAULT_COLUMNS, help="Column-list Python file.")
    parser.add_argument("--list-name", default=DEFAULT_LIST_NAME, help="Column list name to validate.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--analysis-output", type=Path, default=DEFAULT_ANALYSIS_OUTPUT)
    return parser.parse_args()


def read_column_list(path: Path, list_name: str) -> list[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == list_name:
                    if isinstance(node.value, ast.Name):
                        return read_column_list(path, node.value.id)
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                        raise ValueError(f"`{list_name}` in {path} must be a list of strings.")
                    return value
    raise ValueError(f"Could not find `{list_name}` in {path}.")


def flux_to_mag(flux, zeropoint: float | None = None) -> np.ndarray:
    """Convert nJy flux to AB magnitude; non-positive flux becomes NaN."""
    zeropoint_jy = AB_ZEROPOINT_JY if zeropoint is None else float(zeropoint)
    values = np.asarray(flux, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    ok = np.isfinite(values) & (values > 0)
    out[ok] = -2.5 * np.log10((values[ok] * 1e-9) / zeropoint_jy)
    return out


def flux_err_to_mag_err(flux, flux_err) -> np.ndarray:
    values = np.asarray(flux, dtype=float)
    errors = np.asarray(flux_err, dtype=float)
    out = np.full(values.shape, np.nan, dtype=float)
    ok = np.isfinite(values) & np.isfinite(errors) & (values > 0) & (errors >= 0)
    out[ok] = 2.5 / np.log(10.0) * (errors[ok] / values[ok])
    return out


def bool_false(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return ~series.fillna(True)
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["false", "0", "0.0", "", "nan", "none"])


def bool_true(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    text = series.astype(str).str.strip().str.lower()
    return text.isin(["true", "1", "1.0"])


def finite_positive(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.isfinite(values) & (values > 0), index=series.index)


def finite_nonnegative(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.isfinite(values) & (values >= 0), index=series.index)


def summarize_numeric(series: pd.Series) -> dict[str, float | int]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return {"finite": 0, "min": np.nan, "median": np.nan, "max": np.nan}
    return {
        "finite": int(len(finite)),
        "min": float(np.nanmin(finite)),
        "median": float(np.nanmedian(finite)),
        "max": float(np.nanmax(finite)),
    }


def build_analysis_table(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    core_map = {
        "objectId": "object_id",
        "coord_ra": "ra",
        "coord_dec": "dec",
        "ebv": "ebv",
        "refBand": "ref_band",
        "refExtendedness": "ref_extendedness",
        "detect_fromBlend": "detect_from_blend",
        "detect_isIsolated": "detect_is_isolated",
        "parentObjectId": "parent_object_id",
    }
    for raw, clean in core_map.items():
        if raw in df.columns:
            out[clean] = df[raw]

    base_clean = pd.Series(True, index=df.index)
    if "object_id" in out:
        base_clean &= out["object_id"].notna()
    if "ra" in out:
        base_clean &= pd.to_numeric(out["ra"], errors="coerce").between(ECDFS_RA_MIN, ECDFS_RA_MAX)
    if "dec" in out:
        base_clean &= pd.to_numeric(out["dec"], errors="coerce").between(ECDFS_DEC_MIN, ECDFS_DEC_MAX)
    out["base_clean"] = base_clean.astype(bool)

    if "detect_is_isolated" in out:
        isolated = bool_true(out["detect_is_isolated"])
    else:
        isolated = pd.Series(True, index=df.index)
    if "detect_from_blend" in out:
        not_from_blend = bool_false(out["detect_from_blend"])
    else:
        not_from_blend = pd.Series(True, index=df.index)
    if "parent_object_id" in out:
        parent_zero = pd.to_numeric(out["parent_object_id"], errors="coerce").fillna(-1).eq(0)
    else:
        parent_zero = pd.Series(True, index=df.index)
    out["isolated_unblended_clean"] = (out["base_clean"] & isolated & not_from_blend & parent_zero).astype(bool)

    for band in BANDS:
        psf_flux = f"{band}_psfFlux"
        psf_err = f"{band}_psfFluxErr"
        psf_flag = f"{band}_psfFlux_flag"
        cmodel_flux = f"{band}_cModelFlux"
        cmodel_err = f"{band}_cModelFluxErr"
        cmodel_flag = f"{band}_cModel_flag"
        extendedness = f"{band}_extendedness"
        extendedness_flag = f"{band}_extendedness_flag"
        blendedness = f"{band}_blendedness"
        blendedness_flag = f"{band}_blendedness_flag"
        psf_center_flag = f"{band}_pixelFlags_inexact_psfCenter"

        if psf_flux in df:
            out[f"psf_flux_{band}"] = pd.to_numeric(df[psf_flux], errors="coerce")
        if psf_err in df:
            out[f"psf_flux_err_{band}"] = pd.to_numeric(df[psf_err], errors="coerce")
        if psf_flag in df:
            out[f"psf_flux_flag_{band}"] = df[psf_flag].astype("boolean")
        if cmodel_flux in df:
            out[f"cmodel_flux_{band}"] = pd.to_numeric(df[cmodel_flux], errors="coerce")
        if cmodel_err in df:
            out[f"cmodel_flux_err_{band}"] = pd.to_numeric(df[cmodel_err], errors="coerce")
        if cmodel_flag in df:
            out[f"cmodel_flux_flag_{band}"] = df[cmodel_flag].astype("boolean")
        if extendedness in df:
            out[f"extendedness_{band}"] = pd.to_numeric(df[extendedness], errors="coerce")
        if extendedness_flag in df:
            out[f"extendedness_flag_{band}"] = df[extendedness_flag].astype("boolean")
        if blendedness in df:
            out[f"blendedness_{band}"] = pd.to_numeric(df[blendedness], errors="coerce")
        if blendedness_flag in df:
            out[f"blendedness_flag_{band}"] = df[blendedness_flag].astype("boolean")
        if psf_center_flag in df:
            out[f"pixel_flags_inexact_psf_center_{band}"] = df[psf_center_flag].astype("boolean")

        psf_flux_clean = f"psf_flux_{band}"
        psf_err_clean = f"psf_flux_err_{band}"
        psf_flag_clean = f"psf_flux_flag_{band}"
        cmodel_flux_clean = f"cmodel_flux_{band}"
        cmodel_err_clean = f"cmodel_flux_err_{band}"
        cmodel_flag_clean = f"cmodel_flux_flag_{band}"
        ext_clean = f"extendedness_{band}"
        ext_flag_clean = f"extendedness_flag_{band}"
        center_flag_clean = f"pixel_flags_inexact_psf_center_{band}"

        if psf_flux_clean in out:
            out[f"psf_mag_{band}"] = flux_to_mag(out[psf_flux_clean])
            if psf_err_clean in out:
                out[f"psf_mag_err_{band}"] = flux_err_to_mag_err(out[psf_flux_clean], out[psf_err_clean])
            mask = finite_positive(out[psf_flux_clean])
            if psf_err_clean in out:
                mask &= finite_nonnegative(out[psf_err_clean])
            if psf_flag_clean in out:
                mask &= bool_false(out[psf_flag_clean])
            if center_flag_clean in out:
                mask &= bool_false(out[center_flag_clean])
            out[f"psf_flux_valid_{band}"] = mask.astype(bool)

        if cmodel_flux_clean in out:
            out[f"cmodel_mag_{band}"] = flux_to_mag(out[cmodel_flux_clean])
            if cmodel_err_clean in out:
                out[f"cmodel_mag_err_{band}"] = flux_err_to_mag_err(out[cmodel_flux_clean], out[cmodel_err_clean])
            mask = finite_positive(out[cmodel_flux_clean])
            if cmodel_err_clean in out:
                mask &= finite_nonnegative(out[cmodel_err_clean])
            if cmodel_flag_clean in out:
                mask &= bool_false(out[cmodel_flag_clean])
            out[f"cmodel_flux_valid_{band}"] = mask.astype(bool)

        if psf_flux_clean in out and cmodel_flux_clean in out:
            psf = pd.to_numeric(out[psf_flux_clean], errors="coerce").to_numpy(dtype=float)
            cmodel = pd.to_numeric(out[cmodel_flux_clean], errors="coerce").to_numpy(dtype=float)
            diff = np.full(len(out), np.nan, dtype=float)
            ratio = np.full(len(out), np.nan, dtype=float)
            ok = np.isfinite(psf) & np.isfinite(cmodel) & (psf > 0) & (cmodel > 0)
            ratio[ok] = psf[ok] / cmodel[ok]
            diff[ok] = 2.5 * np.log10(cmodel[ok] / psf[ok])
            out[f"psf_over_cmodel_flux_ratio_{band}"] = ratio
            out[f"psf_minus_cmodel_mag_{band}"] = diff

        if ext_clean in out:
            mask = pd.to_numeric(out[ext_clean], errors="coerce").notna()
            if ext_flag_clean in out:
                mask &= bool_false(out[ext_flag_clean])
            out[f"extendedness_valid_{band}"] = mask.astype(bool)

        if f"psf_flux_valid_{band}" in out and f"cmodel_flux_valid_{band}" in out:
            out[f"psf_cmodel_ready_{band}"] = (
                out["base_clean"] & out[f"psf_flux_valid_{band}"] & out[f"cmodel_flux_valid_{band}"]
            ).astype(bool)
            out[f"photometry_clean_{band}"] = (
                out["isolated_unblended_clean"]
                & out[f"psf_flux_valid_{band}"]
                & out[f"cmodel_flux_valid_{band}"]
            ).astype(bool)

    if {"cmodel_mag_g", "cmodel_mag_i"}.issubset(out.columns):
        out["cmodel_color_g_i"] = out["cmodel_mag_g"] - out["cmodel_mag_i"]
        out["gi"] = out["cmodel_color_g_i"]
    if {"psf_mag_g", "psf_mag_i"}.issubset(out.columns):
        out["psf_color_g_i"] = out["psf_mag_g"] - out["psf_mag_i"]

    if {"cmodel_flux_valid_g", "cmodel_flux_valid_i", "cmodel_color_g_i", "cmodel_mag_i"}.issubset(out.columns):
        out["cmd_gi_cmodel_clean"] = (
            out["base_clean"]
            & out["cmodel_flux_valid_g"]
            & out["cmodel_flux_valid_i"]
            & np.isfinite(pd.to_numeric(out["cmodel_color_g_i"], errors="coerce"))
            & np.isfinite(pd.to_numeric(out["cmodel_mag_i"], errors="coerce"))
        ).astype(bool)
    if {"psf_cmodel_ready_i", "extendedness_valid_i"}.issubset(out.columns):
        out["pS_extendedness_i_ready"] = (
            out["psf_cmodel_ready_i"] & out["extendedness_valid_i"]
        ).astype(bool)
    if {"psf_cmodel_ready_r", "extendedness_valid_r"}.issubset(out.columns):
        out["pS_extendedness_r_ready"] = (
            out["psf_cmodel_ready_r"] & out["extendedness_valid_r"]
        ).astype(bool)
    ready_cols = [f"psf_cmodel_ready_{band}" for band in BANDS if f"psf_cmodel_ready_{band}" in out]
    if ready_cols:
        mask = out["base_clean"].copy()
        for col in ready_cols:
            mask &= out[col]
        out["all_bands_psf_cmodel_ready"] = mask.astype(bool)

    return out


def make_ra_dec_plot(df: pd.DataFrame, path_png: Path) -> None:
    path_pdf = path_png.with_suffix(".pdf")
    ra = pd.to_numeric(df["coord_ra"], errors="coerce")
    dec = pd.to_numeric(df["coord_dec"], errors="coerce")
    ok = np.isfinite(ra) & np.isfinite(dec)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    hb = ax.hexbin(ra[ok], dec[ok], gridsize=140, bins="log", mincnt=1, cmap="viridis")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title("DP1 ECDFS Butler Object Footprint")
    ax.invert_xaxis()
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("log10(N objects)")
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_pdf)
    plt.close(fig)


def make_cmd_plot(analysis: pd.DataFrame, path_png: Path) -> None:
    path_pdf = path_png.with_suffix(".pdf")
    if "cmd_gi_cmodel_clean" in analysis:
        mask = analysis["cmd_gi_cmodel_clean"].astype(bool)
    else:
        mask = pd.Series(True, index=analysis.index)
    color = pd.to_numeric(analysis["cmodel_color_g_i"], errors="coerce")
    mag_i = pd.to_numeric(analysis["cmodel_mag_i"], errors="coerce")
    ok = mask & np.isfinite(color) & np.isfinite(mag_i)
    ok &= color.between(-2.0, 6.0) & mag_i.between(14.0, 32.0)

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    hb = ax.hexbin(color[ok], mag_i[ok], gridsize=120, bins="log", mincnt=1, cmap="magma")
    ax.set_xlabel("g - i (CModel, flux-derived AB mag)")
    ax.set_ylabel("i (CModel, flux-derived AB mag)")
    ax.set_title("DP1 ECDFS Butler First-Look CMD")
    ax.invert_yaxis()
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("log10(N objects)")
    fig.savefig(path_png, dpi=180)
    fig.savefig(path_png.with_suffix(".pdf"))
    plt.close(fig)


def write_validation_report(
    path: Path,
    input_path: Path,
    df: pd.DataFrame,
    analysis: pd.DataFrame,
    target_columns: list[str],
) -> None:
    missing = [col for col in target_columns if col not in df.columns]
    extra = [col for col in df.columns if col not in target_columns]
    ra = pd.to_numeric(df["coord_ra"], errors="coerce")
    dec = pd.to_numeric(df["coord_dec"], errors="coerce")
    ecdfs_like = bool(
        len(df) > 0
        and ra.between(ECDFS_RA_MIN - 0.01, ECDFS_RA_MAX + 0.01).all()
        and dec.between(ECDFS_DEC_MIN - 0.01, ECDFS_DEC_MAX + 0.01).all()
    )

    lines = [
        "# Butler ECDFS extraction validation\n\n",
        "## Input\n\n",
        f"- Butler output: `{input_path}`\n",
        f"- File exists: `{input_path.exists()}`\n",
        f"- Rows: {len(df):,}\n",
        f"- Columns: {len(df.columns):,}\n",
        f"- Target columns present: {len(target_columns) - len(missing)}/{len(target_columns)}\n",
        f"- Missing target columns: {', '.join(f'`{col}`' for col in missing) if missing else 'none'}\n",
        f"- Extra columns beyond target list: {', '.join(f'`{col}`' for col in extra) if extra else 'none'}\n",
        "\n## Coordinate validation\n\n",
        "- RA column: `coord_ra`\n",
        "- Dec column: `coord_dec`\n",
        f"- RA range: {float(ra.min()):.6f} to {float(ra.max()):.6f} deg\n",
        f"- Dec range: {float(dec.min()):.6f} to {float(dec.max()):.6f} deg\n",
        f"- Within requested ECDFS box: `{bool(ra.between(ECDFS_RA_MIN, ECDFS_RA_MAX).all() and dec.between(ECDFS_DEC_MIN, ECDFS_DEC_MAX).all())}`\n",
        f"- Footprint consistent with ECDFS: `{ecdfs_like}`\n",
        "\n## Flux/error finite-positive checks\n\n",
        "| band | PSF flux finite | PSF flux >0 | PSF err finite | CModel flux finite | CModel flux >0 | CModel err finite |\n",
        "|---|---:|---:|---:|---:|---:|---:|\n",
    ]

    for band in BANDS:
        psf_flux = pd.to_numeric(df[f"{band}_psfFlux"], errors="coerce")
        psf_err = pd.to_numeric(df[f"{band}_psfFluxErr"], errors="coerce")
        cmodel_flux = pd.to_numeric(df[f"{band}_cModelFlux"], errors="coerce")
        cmodel_err = pd.to_numeric(df[f"{band}_cModelFluxErr"], errors="coerce")
        lines.append(
            f"| `{band}` | {int(np.isfinite(psf_flux).sum()):,} | {int((psf_flux > 0).sum()):,} | "
            f"{int(np.isfinite(psf_err).sum()):,} | {int(np.isfinite(cmodel_flux).sum()):,} | "
            f"{int((cmodel_flux > 0).sum()):,} | {int(np.isfinite(cmodel_err).sum()):,} |\n"
        )

    lines.extend(
        [
            "\n## Basic flag counts\n\n",
            "| flag column | false/0 | true/1 | null |\n",
            "|---|---:|---:|---:|\n",
        ]
    )
    flag_columns = [col for col in df.columns if col.endswith("_flag") or "pixelFlags" in col or col.startswith("detect_")]
    for col in flag_columns:
        series = df[col]
        false_count = int(bool_false(series).sum())
        true_count = int(bool_true(series).sum())
        null_count = int(series.isna().sum())
        lines.append(f"| `{col}` | {false_count:,} | {true_count:,} | {null_count:,} |\n")

    lines.extend(
        [
            "\n## Derived analysis table\n\n",
            f"- Output: `outputs/{Path(DEFAULT_ANALYSIS_OUTPUT).name}`\n",
            f"- Rows: {len(analysis):,}\n",
            f"- Columns: {len(analysis.columns):,}\n",
            "- Magnitudes are flux-derived from nJy using AB zeropoint 3631 Jy; non-positive fluxes become `NaN`.\n",
            "- CMD color: `cmodel_color_g_i = cmodel_mag_g - cmodel_mag_i`.\n",
            "- PSF-CModel morphology quantity: `psf_minus_cmodel_mag_<band> = psf_mag_<band> - cmodel_mag_<band>`.\n",
            "\n## Clean-mask counts\n\n",
            "| mask | true count |\n",
            "|---|---:|\n",
        ]
    )
    mask_cols = [col for col in analysis.columns if col.endswith("_clean") or col.endswith("_ready") or col.endswith("_valid")]
    for col in mask_cols:
        lines.append(f"| `{col}` | {int(analysis[col].fillna(False).astype(bool).sum()):,} |\n")

    path.write_text("".join(lines))


def main() -> None:
    args = parse_args()
    warnings.simplefilter("ignore", category=PerformanceWarning)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.analysis_output.parent.mkdir(parents=True, exist_ok=True)

    target_columns = read_column_list(args.columns, args.list_name)
    df = pd.read_parquet(args.input)
    analysis = build_analysis_table(df)
    analysis.to_parquet(args.analysis_output, index=False)

    make_ra_dec_plot(df, args.results_dir / "butler_ecdfs_ra_dec.png")
    make_cmd_plot(analysis, args.results_dir / "butler_ecdfs_cmd_firstlook.png")
    write_validation_report(
        args.results_dir / "butler_ecdfs_extraction_validation.md",
        args.input,
        df,
        analysis,
        target_columns,
    )

    print(f"Input rows: {len(df):,}")
    print(f"Input columns: {len(df.columns):,}")
    print(f"Analysis output: {args.analysis_output}")
    print(f"Analysis rows: {len(analysis):,}")
    print(f"Analysis columns: {len(analysis.columns):,}")
    print(f"Validation report: {args.results_dir / 'butler_ecdfs_extraction_validation.md'}")
    print(f"RA/Dec plot: {args.results_dir / 'butler_ecdfs_ra_dec.png'}")
    print(f"CMD plot: {args.results_dir / 'butler_ecdfs_cmd_firstlook.png'}")


if __name__ == "__main__":
    main()
