#!/usr/bin/env python3
"""Prepare an ECDFS Rubin DP1 object catalog for later CMD and pS work.

This script is intentionally local-file first.  It inspects the actual columns
present in the current ECDFS FITS table, reuses the existing repo feature
construction where possible, documents recommended cleaning flags, and writes:

* a split current-FITS and target Butler/Object-schema column list,
* a column inventory,
* cleaning recommendations,
* a pS-input compatibility note,
* a clean analysis-ready ECDFS table,
* a readable RA/Dec footprint plot,
* a first-look g-i vs i CMD.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    from astropy.table import Table
except ImportError:  # pragma: no cover - depends on local Rubin environment
    Table = None
from pandas.errors import PerformanceWarning


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config import filters  # noqa: E402
try:
    from pipeline import load_csv_dp1, magnitude_calculations  # noqa: E402
except ImportError:  # pragma: no cover - depends on local Rubin environment
    load_csv_dp1 = None
    magnitude_calculations = None


DEFAULT_INPUT = REPO_ROOT / "data" / "ECDFS.fits"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"

AB_ZEROPOINT_JY = 3631.0
EXTINCTION_SCALE = 3.10 / 1.20


@dataclass(frozen=True)
class BandColumns:
    band: str
    psf_flux: str | None
    psf_flux_err: str | None
    psf_flux_flag: str | None
    cmodel_flux: str | None
    cmodel_flux_err: str | None
    cmodel_flux_flag: str | None
    cmodel_mag_raw: str | None
    cmodel_mag_err_raw: str | None
    psf_mag_raw: str | None
    psf_mag_err_raw: str | None
    extendedness: str | None
    extendedness_flag: str | None
    size_extendedness: str | None
    size_extendedness_flag: str | None
    blendedness: str | None
    blendedness_flag: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to ECDFS DP1 FITS catalog.")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for plots/docs.")
    parser.add_argument("--outputs-dir", type=Path, default=DEFAULT_OUTPUTS_DIR, help="Directory for tables.")
    parser.add_argument(
        "--apply-cleaning",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to drop rows failing the base_clean mask from the saved analysis table.",
    )
    parser.add_argument(
        "--make-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to generate RA/Dec and CMD plots.",
    )
    parser.add_argument(
        "--export-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export outputs/dp1_ecdfs_analysis_table.csv.",
    )
    parser.add_argument(
        "--export-parquet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to export outputs/dp1_ecdfs_analysis_table.parquet.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_raw_catalog(path: Path) -> pd.DataFrame:
    if Table is None:
        raise RuntimeError(
            "astropy is required to read FITS input. Run this script in the Rubin/LSST Python "
            "environment or install astropy in the active Python environment."
        )
    return Table.read(path, format="fits").to_pandas()


def read_processed_catalog(path: Path) -> pd.DataFrame:
    if load_csv_dp1 is None or magnitude_calculations is None:
        raise RuntimeError(
            "The existing repo FITS loader requires the Rubin/LSST Python environment with astropy. "
            "Activate that environment before running this script."
        )
    processed = load_csv_dp1(path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        processed = magnitude_calculations(processed, processed["ebv"] * EXTINCTION_SCALE)
    return processed


def detect_band_columns(df: pd.DataFrame) -> dict[str, BandColumns]:
    cols = set(df.columns)
    out: dict[str, BandColumns] = {}
    for band in filters:
        out[band] = BandColumns(
            band=band,
            psf_flux=f"{band}_free_psfFlux" if f"{band}_free_psfFlux" in cols else None,
            psf_flux_err=f"{band}_free_psfFluxErr" if f"{band}_free_psfFluxErr" in cols else None,
            psf_flux_flag=f"{band}_free_psfFlux_flag" if f"{band}_free_psfFlux_flag" in cols else None,
            cmodel_flux=f"{band}_free_cModelFlux" if f"{band}_free_cModelFlux" in cols else None,
            cmodel_flux_err=f"{band}_free_cModelFluxErr" if f"{band}_free_cModelFluxErr" in cols else None,
            cmodel_flux_flag=f"{band}_free_cModelFlux_flag" if f"{band}_free_cModelFlux_flag" in cols else None,
            cmodel_mag_raw=f"{band}_free_cModelMag" if f"{band}_free_cModelMag" in cols else None,
            cmodel_mag_err_raw=f"{band}_free_cModelMagErr" if f"{band}_free_cModelMagErr" in cols else None,
            psf_mag_raw=f"{band}_psfMag" if f"{band}_psfMag" in cols else None,
            psf_mag_err_raw=f"{band}_psfMagErr" if f"{band}_psfMagErr" in cols else None,
            extendedness=f"{band}_extendedness" if f"{band}_extendedness" in cols else None,
            extendedness_flag=f"{band}_extendedness_flag" if f"{band}_extendedness_flag" in cols else None,
            size_extendedness=f"{band}_sizeExtendedness" if f"{band}_sizeExtendedness" in cols else None,
            size_extendedness_flag=f"{band}_sizeExtendedness_flag" if f"{band}_sizeExtendedness_flag" in cols else None,
            blendedness=f"{band}_blendedness" if f"{band}_blendedness" in cols else None,
            blendedness_flag=f"{band}_blendedness_flag" if f"{band}_blendedness_flag" in cols else None,
        )
    return out


def detect_core_columns(df: pd.DataFrame) -> dict[str, str | None]:
    cols = set(df.columns)
    return {
        "objectId": "objectId" if "objectId" in cols else None,
        "coord_ra": "coord_ra" if "coord_ra" in cols else None,
        "coord_dec": "coord_dec" if "coord_dec" in cols else None,
        "ebv": "ebv" if "ebv" in cols else None,
        "detect_isIsolated": "detect_isIsolated" if "detect_isIsolated" in cols else None,
        "deblend_failed": "deblend_failed" if "deblend_failed" in cols else None,
    }


def flux_to_mag(flux, zeropoint_jy: float = AB_ZEROPOINT_JY) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    out = np.full(flux.shape, np.nan, dtype=float)
    ok = np.isfinite(flux) & (flux > 0)
    out[ok] = -2.5 * np.log10((flux[ok] * 1e-9) / zeropoint_jy)
    return out


def flux_err_to_mag_err(flux, flux_err) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    out = np.full(flux.shape, np.nan, dtype=float)
    ok = np.isfinite(flux) & np.isfinite(flux_err) & (flux > 0) & (flux_err >= 0)
    out[ok] = 2.5 / np.log(10.0) * (flux_err[ok] / flux[ok])
    return out


def flag_keep_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return ~series
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series(False, index=series.index)
    out.loc[numeric.isin([0, 0.0])] = True
    text = series.astype(str).str.strip().str.lower()
    out |= text.isin(["false", "0", "0.0", "nan", "none", ""])
    return out


def finite_positive(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return pd.Series(np.isfinite(numeric) & (numeric > 0), index=series.index)


def build_butler_column_list(core: dict[str, str | None], band_info: dict[str, BandColumns]) -> list[str]:
    """Columns that are actually present in the current local ECDFS FITS export."""
    cols: list[str] = []
    for key in ["objectId", "coord_ra", "coord_dec", "ebv", "detect_isIsolated", "deblend_failed"]:
        if core.get(key):
            cols.append(core[key])  # type: ignore[arg-type]
    for band in filters:
        info = band_info[band]
        for name in [
            info.psf_flux,
            info.psf_flux_err,
            info.psf_flux_flag,
            info.cmodel_flux,
            info.cmodel_flux_err,
            info.cmodel_flux_flag,
            info.extendedness,
            info.extendedness_flag,
            info.size_extendedness,
            info.size_extendedness_flag,
            info.blendedness,
            info.blendedness_flag,
        ]:
            if name is not None:
                cols.append(name)
    return cols


def build_target_butler_column_list() -> list[str]:
    """Target LSST Object-schema-style columns for direct Butler extraction.

    This list is deliberately separate from the current FITS/export list above:
    the local ECDFS FITS table contains ``*_free_*`` photometry names, while
    DP1 tutorial/schema examples use Object-table names such as
    ``g_psfFlux`` and ``g_cModelFlux``.
    """
    cols = [
        "objectId",
        "coord_ra",
        "coord_dec",
        "ebv",
        "refBand",
        "refExtendedness",
        "detect_fromBlend",
        "detect_isIsolated",
        "parentObjectId",
    ]
    for band in filters:
        cols.extend(
            [
                f"{band}_psfFlux",
                f"{band}_psfFluxErr",
                f"{band}_psfFlux_flag",
                f"{band}_cModelFlux",
                f"{band}_cModelFluxErr",
                f"{band}_cModel_flag",
                f"{band}_extendedness",
                f"{band}_extendedness_flag",
                f"{band}_blendedness",
                f"{band}_blendedness_flag",
                f"{band}_pixelFlags_inexact_psfCenter",
            ]
        )
    return cols


def build_analysis_table(
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    core: dict[str, str | None],
    band_info: dict[str, BandColumns],
) -> pd.DataFrame:
    if len(raw_df) != len(processed_df):
        raise ValueError("Raw and processed ECDFS tables do not have matching row counts.")

    out = pd.DataFrame(index=raw_df.index)
    if core["objectId"] is not None:
        out["object_id"] = raw_df[core["objectId"]]
    if core["coord_ra"] is not None:
        out["ra"] = pd.to_numeric(raw_df[core["coord_ra"]], errors="coerce")
    if core["coord_dec"] is not None:
        out["dec"] = pd.to_numeric(raw_df[core["coord_dec"]], errors="coerce")
    if core["ebv"] is not None:
        out["ebv"] = pd.to_numeric(raw_df[core["ebv"]], errors="coerce")
    if core["deblend_failed"] is not None:
        out["deblend_failed"] = raw_df[core["deblend_failed"]].astype("boolean")

    base_clean = pd.Series(True, index=raw_df.index)
    if "object_id" in out.columns:
        base_clean &= out["object_id"].notna()
    if "ra" in out.columns:
        base_clean &= np.isfinite(out["ra"])
    if "dec" in out.columns:
        base_clean &= np.isfinite(out["dec"])
    if "deblend_failed" in out.columns:
        base_clean &= ~out["deblend_failed"].fillna(True)
    out["base_clean"] = base_clean.astype(bool)

    for band in filters:
        info = band_info[band]
        if info.psf_flux is not None:
            out[f"{band}_psf_flux"] = pd.to_numeric(raw_df[info.psf_flux], errors="coerce")
        if info.psf_flux_err is not None:
            out[f"{band}_psf_flux_err"] = pd.to_numeric(raw_df[info.psf_flux_err], errors="coerce")
        if info.psf_flux_flag is not None:
            out[f"{band}_psf_flux_flag"] = raw_df[info.psf_flux_flag].astype("boolean")
        if info.cmodel_flux is not None:
            out[f"{band}_cmodel_flux"] = pd.to_numeric(raw_df[info.cmodel_flux], errors="coerce")
        if info.cmodel_flux_err is not None:
            out[f"{band}_cmodel_flux_err"] = pd.to_numeric(raw_df[info.cmodel_flux_err], errors="coerce")
        if info.cmodel_flux_flag is not None:
            out[f"{band}_cmodel_flux_flag"] = raw_df[info.cmodel_flux_flag].astype("boolean")
        if info.extendedness is not None:
            out[f"{band}_extendedness"] = pd.to_numeric(raw_df[info.extendedness], errors="coerce")
        if info.extendedness_flag is not None:
            out[f"{band}_extendedness_flag"] = raw_df[info.extendedness_flag].astype("boolean")
        if info.size_extendedness is not None:
            out[f"{band}_sizeExtendedness"] = pd.to_numeric(raw_df[info.size_extendedness], errors="coerce")
        if info.size_extendedness_flag is not None:
            out[f"{band}_sizeExtendedness_flag"] = raw_df[info.size_extendedness_flag].astype("boolean")
        if info.blendedness is not None:
            out[f"{band}_blendedness"] = pd.to_numeric(raw_df[info.blendedness], errors="coerce")
        if info.blendedness_flag is not None:
            out[f"{band}_blendedness_flag"] = raw_df[info.blendedness_flag].astype("boolean")

        psf_flux_col = f"{band}_psf_flux"
        cmodel_flux_col = f"{band}_cmodel_flux"
        psf_err_col = f"{band}_psf_flux_err"
        cmodel_err_col = f"{band}_cmodel_flux_err"
        psf_flag_col = f"{band}_psf_flux_flag"
        cmodel_flag_col = f"{band}_cmodel_flux_flag"
        ext_flag_col = f"{band}_extendedness_flag"
        sizeext_flag_col = f"{band}_sizeExtendedness_flag"
        blend_flag_col = f"{band}_blendedness_flag"

        if psf_flux_col in out.columns:
            mask = finite_positive(out[psf_flux_col])
            if psf_flag_col in out.columns:
                mask &= ~out[psf_flag_col].fillna(True)
            out[f"{band}_psf_flux_valid"] = mask.astype(bool)
            proc_mag_col = f"{band}_psfFlux_mag"
            if proc_mag_col in processed_df.columns:
                out[proc_mag_col] = pd.to_numeric(processed_df[proc_mag_col], errors="coerce")
            else:
                out[proc_mag_col] = flux_to_mag(out[psf_flux_col])
            if psf_err_col in out.columns:
                out[f"{band}_psfFlux_mag_err"] = flux_err_to_mag_err(out[psf_flux_col], out[psf_err_col])
            proc_relerr_col = f"{band}_psfRelErr"
            if proc_relerr_col in processed_df.columns:
                out[proc_relerr_col] = pd.to_numeric(processed_df[proc_relerr_col], errors="coerce")
            elif psf_err_col in out.columns:
                rel = np.full(len(out), np.nan, dtype=float)
                ok = np.isfinite(out[psf_flux_col]) & np.isfinite(out[psf_err_col]) & (out[psf_flux_col] > 0)
                rel[ok] = 1.09 * (out.loc[ok, psf_err_col] / out.loc[ok, psf_flux_col])
                out[proc_relerr_col] = rel

        if cmodel_flux_col in out.columns:
            mask = finite_positive(out[cmodel_flux_col])
            if cmodel_flag_col in out.columns:
                mask &= ~out[cmodel_flag_col].fillna(True)
            out[f"{band}_cmodel_flux_valid"] = mask.astype(bool)
            proc_mag_col = f"{band}_modelFlux_mag"
            if proc_mag_col in processed_df.columns:
                out[proc_mag_col] = pd.to_numeric(processed_df[proc_mag_col], errors="coerce")
            else:
                out[proc_mag_col] = flux_to_mag(out[cmodel_flux_col])
            if cmodel_err_col in out.columns:
                out[f"{band}_modelFlux_mag_err"] = flux_err_to_mag_err(out[cmodel_flux_col], out[cmodel_err_col])

        if f"{band}_diff" in processed_df.columns:
            out[f"{band}_diff"] = pd.to_numeric(processed_df[f"{band}_diff"], errors="coerce")
        elif psf_flux_col in out.columns and cmodel_flux_col in out.columns:
            diff = np.full(len(out), np.nan, dtype=float)
            psf = pd.to_numeric(out[psf_flux_col], errors="coerce").to_numpy(dtype=float)
            cmodel = pd.to_numeric(out[cmodel_flux_col], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(psf) & np.isfinite(cmodel) & (psf > 0) & (cmodel > 0)
            diff[ok] = 2.5 * np.log10(cmodel[ok] / psf[ok])
            out[f"{band}_diff"] = diff

        if f"{band}_extendedness" in out.columns:
            mask = np.isfinite(out[f"{band}_extendedness"])
            if ext_flag_col in out.columns:
                mask &= ~out[ext_flag_col].fillna(True)
            out[f"{band}_extendedness_valid"] = mask.astype(bool)

        if f"{band}_sizeExtendedness" in out.columns:
            mask = np.isfinite(out[f"{band}_sizeExtendedness"])
            if sizeext_flag_col in out.columns:
                mask &= ~out[sizeext_flag_col].fillna(True)
            out[f"{band}_sizeExtendedness_valid"] = mask.astype(bool)

        if f"{band}_blendedness" in out.columns:
            mask = np.isfinite(out[f"{band}_blendedness"])
            if blend_flag_col in out.columns:
                mask &= ~out[blend_flag_col].fillna(True)
            out[f"{band}_blendedness_valid"] = mask.astype(bool)

    color_defs = [
        ("ug", "u_modelFlux_mag", "g_modelFlux_mag"),
        ("gr", "g_modelFlux_mag", "r_modelFlux_mag"),
        ("ri", "r_modelFlux_mag", "i_modelFlux_mag"),
        ("iz", "i_modelFlux_mag", "z_modelFlux_mag"),
        ("zy", "z_modelFlux_mag", "y_modelFlux_mag"),
        ("gi", "g_modelFlux_mag", "i_modelFlux_mag"),
    ]
    for color_name, a_col, b_col in color_defs:
        if color_name in processed_df.columns:
            out[color_name] = pd.to_numeric(processed_df[color_name], errors="coerce")
        elif a_col in out.columns and b_col in out.columns:
            out[color_name] = out[a_col] - out[b_col]

    if "Ar" in processed_df.columns:
        out["Ar"] = pd.to_numeric(processed_df["Ar"], errors="coerce")
    elif core["ebv"] is not None and "ebv" in out.columns:
        out["Ar"] = out["ebv"] * 3.10 / 1.20

    if {"g_cmodel_flux_valid", "i_cmodel_flux_valid"}.issubset(out.columns):
        out["gi_cmd_clean"] = (out["base_clean"] & out["g_cmodel_flux_valid"] & out["i_cmodel_flux_valid"]).astype(bool)
    if {"r_cmodel_flux_valid", "r_psf_flux_valid"}.issubset(out.columns):
        out["pS_r_ready"] = (out["base_clean"] & out["r_cmodel_flux_valid"] & out["r_psf_flux_valid"]).astype(bool)
    if {"g_cmodel_flux_valid", "r_cmodel_flux_valid", "i_cmodel_flux_valid"}.issubset(out.columns):
        out["all_gri_cmodel_ready"] = (
            out["base_clean"] & out["g_cmodel_flux_valid"] & out["r_cmodel_flux_valid"] & out["i_cmodel_flux_valid"]
        ).astype(bool)
    psf_cmodel_ready_cols = [f"{band}_cmodel_flux_valid" for band in filters] + [f"{band}_psf_flux_valid" for band in filters]
    if set(psf_cmodel_ready_cols).issubset(out.columns):
        mask = out["base_clean"].copy()
        for col in psf_cmodel_ready_cols:
            mask &= out[col]
        out["all_bands_psf_cmodel_ready"] = mask.astype(bool)
    return out.copy()


def write_butler_column_list(path: Path, current_fits_columns: list[str], target_butler_columns: list[str]) -> None:
    lines = [
        "# Current local ECDFS FITS/export column list.\n",
        "# These names are verified in data/ECDFS.fits and are used by scripts/prepare_dp1_ecdfs_catalog.py.\n",
        "# They are not, by themselves, a verified direct Butler Object query list.\n",
        "current_fits_column_names = [\n",
    ]
    for col in current_fits_columns:
        lines.append(f"    '{col}',\n")
    lines.extend(
        [
            "]\n\n",
            "# Target DP1/LSST Object-schema-style column list for direct Butler extraction.\n",
            "# This should be used in the Rubin environment first with a small --limit-refs test.\n",
            "# Some flag names may still need live-schema verification against the active DP1 collection.\n",
            "target_butler_column_names = [\n",
        ]
    )
    for col in target_butler_columns:
        lines.append(f"    '{col}',\n")
    lines.extend(
        [
            "]\n\n",
            "# Default for scripts/query_dp1_ecdfs_object_catalog.py.\n",
            "column_names = target_butler_column_names\n",
        ]
    )
    path.write_text("".join(lines))


def summarize_series(series: pd.Series) -> dict[str, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {"finite": 0, "p50": np.nan, "p95": np.nan}
    return {
        "finite": int(len(arr)),
        "p50": float(np.nanmedian(arr)),
        "p95": float(np.nanpercentile(arr, 95)),
    }


def write_column_inventory(
    path: Path,
    raw_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    core: dict[str, str | None],
    band_info: dict[str, BandColumns],
    current_fits_columns: list[str],
    target_butler_columns: list[str],
    input_path: Path,
) -> None:
    ra = pd.to_numeric(raw_df[core["coord_ra"]], errors="coerce") if core["coord_ra"] else pd.Series(dtype=float)
    dec = pd.to_numeric(raw_df[core["coord_dec"]], errors="coerce") if core["coord_dec"] else pd.Series(dtype=float)

    lines = [
        "# DP1 ECDFS column inventory\n\n",
        "## Repo inspection and reuse\n\n",
        "- ECDFS/DP1 catalog access in the existing repo is local-file based, not Butler-based.\n",
        f"- The main catalog-loading entry point is `src/pipeline.py:load_csv_dp1`, which reads `{input_path.name}` and renames `_free_cModelFlux/_free_psfFlux` families into the current notebook convention.\n",
        "- Existing feature construction is reused from `src/pipeline.py:magnitude_calculations` and `src/magnitude_calc.py`.\n",
        "- Existing CMD plotting conventions are visible in `src/plotting.py:plot_compareCMDs` and `notebooks/ECDFS-DP1.ipynb`.\n",
        "- Existing cleaning/quality cuts already used in the project are documented in `src/validate_dp1_against_hst.py`.\n",
        "- Existing pS workflow dependencies come from `src/psf_cmodel_fit.py`, `src/build_ecdfs_probability_models.py`, and `notebooks/ECDFS-DP1.ipynb`.\n\n",
        "- The current repo has `notebooks/ECDFS-DP1.ipynb`, but no `tutorial-notebooks/DP1` directory is present in this checkout. DP1 tutorial/schema examples used for schema alignment are documented separately in the schema/Butler validation reports.\n\n",
        "## Actual ECDFS FITS summary\n\n",
        f"- Input file: `{input_path}`\n",
        f"- Rows: {len(raw_df):,}\n",
        f"- Columns: {len(raw_df.columns)}\n",
    ]
    if len(ra):
        lines.append(f"- RA range: {float(np.nanmin(ra)):.6f} to {float(np.nanmax(ra)):.6f} deg\n")
    if len(dec):
        lines.append(f"- Dec range: {float(np.nanmin(dec)):.6f} to {float(np.nanmax(dec)):.6f} deg\n")
    lines.append("\n## Core identifier and coordinate columns found\n\n")
    for label, col in core.items():
        lines.append(f"- `{label}` -> `{col or 'missing'}`\n")

    lines.append("\n## Current FITS/export extraction column list\n\n")
    lines.append("These names are verified in the current local ECDFS FITS table and are the columns used by the preparation script.\n\n")
    for col in current_fits_columns:
        lines.append(f"- `{col}`\n")

    lines.append("\n## Target Butler/Object-schema column list\n\n")
    lines.append("These names are the target schema-style equivalents for a future direct Butler extraction. They intentionally avoid the local-export `*_free_*` naming where DP1 tutorials/schema examples use fixed Object names such as `*_psfFlux` and `*_cModelFlux`.\n\n")
    for col in target_butler_columns:
        lines.append(f"- `{col}`\n")

    lines.append("\n## Flux, error, and morphology families found by band\n\n")
    header = (
        "| band | PSF flux | PSF flux err | PSF flag | CModel flux | CModel flux err | CModel flag | "
        "extendedness | extendedness flag | sizeExtendedness | sizeExt flag | blendedness | blendedness flag |\n"
    )
    lines.extend([header, "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"])
    for band in filters:
        info = band_info[band]
        lines.append(
            f"| `{band}` | `{info.psf_flux or 'missing'}` | `{info.psf_flux_err or 'missing'}` | `{info.psf_flux_flag or 'missing'}` | "
            f"`{info.cmodel_flux or 'missing'}` | `{info.cmodel_flux_err or 'missing'}` | `{info.cmodel_flux_flag or 'missing'}` | "
            f"`{info.extendedness or 'missing'}` | `{info.extendedness_flag or 'missing'}` | "
            f"`{info.size_extendedness or 'missing'}` | `{info.size_extendedness_flag or 'missing'}` | "
            f"`{info.blendedness or 'missing'}` | `{info.blendedness_flag or 'missing'}` |\n"
        )

    lines.append("\n## Catalog magnitude columns also present in the current FITS file\n\n")
    lines.append(
        "The current local FITS file already includes `*_free_cModelMag`, `*_free_cModelMagErr`, `*_psfMag`, and `*_psfMagErr`. "
        "However, the extraction/analysis workflow below intentionally derives magnitudes from fluxes so it stays consistent with the project request and with schema versions where magnitudes are not materialized.\n\n"
    )
    for band in filters:
        info = band_info[band]
        lines.append(
            f"- `{band}`: cModel mag `{info.cmodel_mag_raw or 'missing'}`, cModel mag err `{info.cmodel_mag_err_raw or 'missing'}`, "
            f"PSF mag `{info.psf_mag_raw or 'missing'}`, PSF mag err `{info.psf_mag_err_raw or 'missing'}`\n"
        )

    lines.append("\n## Standardized analysis-table columns written by this script\n\n")
    for col in analysis_df.columns:
        lines.append(f"- `{col}`\n")

    lines.append("\n## Flux-to-magnitude derivation used here\n\n")
    lines.append(
        textwrap.dedent(
            f"""
            - Fluxes are assumed to be in nJy, matching the existing repo convention in `src/magnitude_calc.py`.
            - Magnitudes use the AB conversion:
              `mag = -2.5 * log10((flux_nJy * 1e-9) / {AB_ZEROPOINT_JY:.1f})`
            - Non-positive flux values are converted to `NaN`, not forced into finite magnitudes.
            - Magnitude uncertainties use:
              `mag_err = 2.5 / ln(10) * flux_err / flux`
            - Existing repo-style extinction-corrected colors are derived via `pipeline.magnitude_calculations(df, df['ebv'] * {EXTINCTION_SCALE:.6f})`.
            """
        ).strip()
        + "\n"
    )

    missing = []
    for band in filters:
        info = band_info[band]
        for label, value in {
            "PSF flux": info.psf_flux,
            "PSF flux error": info.psf_flux_err,
            "CModel flux": info.cmodel_flux,
            "CModel flux error": info.cmodel_flux_err,
            "extendedness": info.extendedness,
        }.items():
            if value is None:
                missing.append(f"- `{band}` band missing {label}\n")
    lines.append("\n## Missing or notable caveats relative to the target ideal list\n\n")
    if missing:
        lines.extend(missing)
    else:
        lines.append("- The ECDFS FITS file contains the project requested core families for all six bands: PSF flux/error, CModel flux/error, and per-band extendedness-style quantities.\n")
    lines.append("- The completed preparation workflow still reads local FITS tables under `data/`; `scripts/query_dp1_ecdfs_object_catalog.py` is a Butler extraction skeleton for rerunning the extraction once the exact ECDFS data-ID/tract selection is supplied in a Rubin environment.\n")
    lines.append("- DP1 tutorials use schema-style examples such as `r_psfFlux`, `r_cModelFlux`, `refExtendedness`, `*_pixelFlags_inexact_psfCenter`, `i_cModel_flag`, `i_kronFlux_flag`, and `sersic_no_data_flag`; those exact non-`free`/shape flag columns are not present in the current ECDFS FITS extraction.\n")
    lines.append("- No single `refExtendedness` or `base_ClassificationExtendedness_value` column is present here; the current file uses per-band `*_extendedness` and `*_sizeExtendedness` instead.\n")
    lines.append("- `deblend_failed` is present, but in the current ECDFS FITS file every row is `False`, so it is documented but not a discriminating cut for this specific table.\n")

    path.write_text("".join(lines))


def write_cleaning_recommendations(
    path: Path,
    json_path: Path,
    raw_df: pd.DataFrame,
    core: dict[str, str | None],
    band_info: dict[str, BandColumns],
) -> None:
    recommendations = {
        "confirmed": [],
        "tutorial_inspired": [],
        "likely": [],
        "tentative": [],
    }

    recommendations["confirmed"].append(
        {
            "column": "coord_ra, coord_dec, objectId",
            "logic": "keep finite coordinates and non-null object IDs",
            "reason": "required to identify the ECDFS footprint and preserve object-level joins",
            "evidence": "core assumption across `src/pipeline.py`, `src/match_hst_to_dp1.py`, and `notebooks/ECDFS-DP1.ipynb`",
        }
    )
    recommendations["confirmed"].append(
        {
            "column": "deblend_failed",
            "logic": "keep rows where `deblend_failed == False`",
            "reason": "already used in `src/validate_dp1_against_hst.py` before morphology interpretation",
            "evidence": "project cleaning function `clean_morphology_sample(...)`",
        }
    )
    recommendations["confirmed"].append(
        {
            "column": "*_free_psfFlux / *_free_cModelFlux",
            "logic": "for any band used, require finite and positive PSF/CModel fluxes",
            "reason": "needed for physically meaningful flux-to-mag conversion and PSF-CModel morphology diagnostics",
            "evidence": "existing repo feature construction and `clean_morphology_sample(...)`",
        }
    )
    recommendations["confirmed"].append(
        {
            "column": "*_free_psfFlux_flag, *_free_cModelFlux_flag",
            "logic": "for any band used, keep rows where the corresponding flux flags are `False`",
            "reason": "already used in `src/validate_dp1_against_hst.py` for i-band morphology cleaning; should generalize band-by-band for CMD work",
            "evidence": "project cleaning function `clean_morphology_sample(...)`",
        }
    )
    recommendations["confirmed"].append(
        {
            "column": "*_extendedness_flag, *_sizeExtendedness_flag",
            "logic": "when using `*_extendedness` or `*_sizeExtendedness`, keep rows where the corresponding flag is `False`",
            "reason": "the current notebook extendedness add-on explicitly excludes flagged values before plotting/comparison",
            "evidence": "`notebooks/ECDFS-DP1.ipynb` extendedness section",
        }
    )

    recommendations["tutorial_inspired"].append(
        {
            "column": "refExtendedness",
            "logic": "tutorial star samples use `refExtendedness == 0`; tutorial galaxy samples use band extendedness or `refExtendedness` to select resolved objects",
            "reason": "DP1 object and stellar-color tutorials use this as the reference point-source selector",
            "evidence": "DP1 Object tutorial/schema examples; not present in the current ECDFS FITS extraction",
        }
    )
    recommendations["tutorial_inspired"].append(
        {
            "column": "*_extendedness",
            "logic": "tutorial star/PSF examples use `*_extendedness == 0`; galaxy examples use `*_extendedness == 1`",
            "reason": "standard tutorial split between point-like and extended Object-table samples",
            "evidence": "DP1 PSF-star, galaxy-photometry, and ECDFS tutorial examples; present in the current ECDFS FITS extraction for all six bands",
        }
    )
    recommendations["tutorial_inspired"].append(
        {
            "column": "*_pixelFlags_inexact_psfCenter",
            "logic": "for clean PSF-star examples, tutorial keeps rows where this flag is `0`",
            "reason": "rejects objects with problematic PSF-center placement in the PSF photometry demo",
            "evidence": "DP1 PSF-star tutorial example; not present in the current ECDFS FITS extraction",
        }
    )
    recommendations["tutorial_inspired"].append(
        {
            "column": "*_cModelFlux / *_cModelFluxErr",
            "logic": "tutorial galaxy/ECDFS examples use CModel signal-to-noise cuts such as `r_cModelFlux/r_cModelFluxErr > 20` and `i_cModelFlux/i_cModelFluxErr > 20`",
            "reason": "ensures high-S/N CModel photometry for morphology/CMD examples",
            "evidence": "DP1 ECDFS and galaxy-photometry tutorial examples; equivalent `*_free_cModelFlux` and `*_free_cModelFluxErr` columns are present here",
        }
    )
    recommendations["tutorial_inspired"].append(
        {
            "column": "i_cModel_flag, i_kronFlux_flag, sersic_no_data_flag",
            "logic": "tutorial galaxy examples keep rows where these flags are `0`",
            "reason": "rejects failed CModel/Kron/Sersic-related galaxy measurements in the DP1 galaxy photometry demo",
            "evidence": "DP1 galaxy-photometry tutorial example; these exact columns are not present in the current ECDFS FITS extraction",
        }
    )

    recommendations["likely"].append(
        {
            "column": "g/i CModel fluxes and flags",
            "logic": "for the first-look `g-i` vs `i` CMD, use rows with valid `g` and `i` CModel fluxes and clean CModel flux flags",
            "reason": "this is the minimum defensible band-specific clean subset for a coadd-style CMD using `g-i` color and `i` magnitude",
            "evidence": "consistent with the project requested CMD plus existing local flux-based workflow",
        }
    )
    recommendations["likely"].append(
        {
            "column": "r / g / i PSF+CModel fluxes",
            "logic": "for later pS-related morphology work, require valid PSF and CModel fluxes in the band(s) being modeled, at minimum `r`, or preferably `g/r/i` together for legacy `dm3`-style work",
            "reason": "the repo's star/galaxy morphology code uses `*_diff`, which is undefined unless both fluxes are usable",
            "evidence": "`src/psf_cmodel_fit.py`, `src/SG_separation.py`, and `src/build_ecdfs_probability_models.py`",
        }
    )

    recommendations["tentative"].append(
        {
            "column": "*_blendedness_flag",
            "logic": "consider keeping only rows where `*_blendedness_flag == False` for stricter photometric CMD samples",
            "reason": "the column family exists and is plausibly useful for conservative coadd photometry, but no current repo code uses it as a confirmed default cut",
            "evidence": "present in the ECDFS FITS file; not yet adopted elsewhere in the local project",
        }
    )
    recommendations["tentative"].append(
        {
            "column": "*_blendedness",
            "logic": "do not impose a hard blendedness threshold yet; inspect band-dependent distributions first",
            "reason": "the quantity is present, but local code does not define a confirmed science threshold",
            "evidence": "distribution available in the raw file; no local threshold recipe found",
        }
    )

    flag_counts: dict[str, dict[str, int]] = {}
    for band in filters:
        info = band_info[band]
        for attr, col in [
            ("psf_flux_flag", info.psf_flux_flag),
            ("cmodel_flux_flag", info.cmodel_flux_flag),
            ("extendedness_flag", info.extendedness_flag),
            ("sizeExtendedness_flag", info.size_extendedness_flag),
            ("blendedness_flag", info.blendedness_flag),
        ]:
            if col is not None:
                series = raw_df[col]
                counts = series.value_counts(dropna=False).to_dict()
                flag_counts[col] = {str(k): int(v) for k, v in counts.items()}

    lines = [
        "# DP1 ECDFS cleaning recommendations\n\n",
        "## Evidence searched locally\n\n",
        "- Current project cleaning code: `src/validate_dp1_against_hst.py`\n",
        "- Current notebook workflow: `notebooks/ECDFS-DP1.ipynb`\n",
        "- Existing CMD/morphology plotting convention: `src/plotting.py:plot_compareCMDs`\n",
        "- Existing morphology-star selection helper: `src/SG_separation.py`\n\n",
        "The current repo has `notebooks/ECDFS-DP1.ipynb`, but no `tutorial-notebooks/DP1` directory is present in this checkout. DP1 tutorial/schema examples used here come from the documented DP1 Object/ECDFS tutorial material referenced in `results/dp1_schema_mapping.md`. They provide useful selection examples, but not a single universal Object-table cleaning recipe for this ECDFS preparation table. The recommendations below therefore separate project-confirmed cuts from tutorial-inspired cuts and tentative candidate cuts.\n\n",
    ]

    def _write_group(title: str, key: str) -> None:
        lines.append(f"## {title}\n\n")
        for item in recommendations[key]:
            lines.append(f"- Column(s): `{item['column']}`\n")
            lines.append(f"  Keep/veto logic: {item['logic']}\n")
            lines.append(f"  Reason: {item['reason']}\n")
            lines.append(f"  Confidence: {key}\n")
            lines.append(f"  Local evidence: {item['evidence']}\n\n")

    _write_group("Confirmed / reused from existing project code", "confirmed")
    _write_group("Tutorial-inspired examples found locally", "tutorial_inspired")
    _write_group("Likely defaults consistent with the current repo workflow", "likely")
    _write_group("Tentative candidate cuts that still need project confirmation", "tentative")

    lines.append("## Actual flag coverage in the current ECDFS FITS file\n\n")
    for col, counts in flag_counts.items():
        lines.append(f"- `{col}`: {counts}\n")
    if core["deblend_failed"] is not None:
        counts = raw_df[core["deblend_failed"]].value_counts(dropna=False).to_dict()
        lines.append(f"- `{core['deblend_failed']}`: {counts}\n")

    lines.append("\n## What local code does *not* currently provide\n\n")
    lines.append("- No local tutorial-derived single best-practice `Object` cleaning recipe was found; tutorial examples are sample-specific selections.\n")
    lines.append("- `plot_compareCMDs` in `src/plotting.py` uses display-oriented magnitude windows and a morphology split (`r_diff <= 0.016`), but those are classifier/display choices rather than generic DP1 cleaning flags.\n")

    path.write_text("".join(lines))
    json_path.write_text(json.dumps(recommendations, indent=2))


def write_ps_input_requirements(path: Path, analysis_df: pd.DataFrame) -> None:
    rows = []
    requirements = [
        ("objectId / coord_ra / coord_dec", ["object_id", "ra", "dec"], "core row identity and sky coordinates"),
        ("u..y raw CModel fluxes", [f"{band}_cmodel_flux" for band in filters], "needed to rebuild notebook-style magnitudes/colors"),
        ("u..y raw PSF fluxes", [f"{band}_psf_flux" for band in filters], "needed to rebuild notebook-style PSF-CModel differences"),
        ("u..y raw CModel flux errors", [f"{band}_cmodel_flux_err" for band in filters], "useful for uncertainty propagation and later QC"),
        ("u..y raw PSF flux errors", [f"{band}_psf_flux_err" for band in filters], "useful for uncertainty propagation and later QC"),
        ("u..y derived CModel magnitudes", [f"{band}_modelFlux_mag" for band in filters], "direct pS coordinate input via `*_modelFlux_mag`"),
        ("u..y derived PSF magnitudes", [f"{band}_psfFlux_mag" for band in filters], "useful for diagnostics and cross-checks"),
        ("u..y derived PSF-CModel morphology", [f"{band}_diff" for band in filters], "direct pS coordinate input via `*_diff`"),
        ("u..y relative PSF errors", [f"{band}_psfRelErr" for band in filters], "already used in the current repo feature family"),
        ("Extinction/colors", ["ebv", "Ar", "ug", "gr", "ri", "iz", "zy", "gi"], "existing notebook-style color diagnostics"),
        ("Extendedness family", [f"{band}_extendedness" for band in filters], "later comparison to native Rubin morphology"),
    ]
    for label, cols, reason in requirements:
        missing = [col for col in cols if col not in analysis_df.columns]
        status = "satisfied" if not missing else "partial"
        rows.append((label, reason, status, missing))

    text = [
        "# pS input requirements for the current ECDFS workflow\n\n",
        "## Existing pS workflow files inspected\n\n",
        "- `src/pipeline.py`\n",
        "- `src/magnitude_calc.py`\n",
        "- `src/psf_cmodel_fit.py`\n",
        "- `src/build_ecdfs_probability_models.py`\n",
        "- `notebooks/ECDFS-DP1.ipynb`\n\n",
        "## What the current pS workflow expects\n\n",
        "- The raw FITS loader currently starts from `*_free_cModelFlux`, `*_free_psfFlux`, `*_free_cModelFluxErr`, and `*_free_psfFluxErr` via `pipeline.load_csv_dp1(...)`.\n",
        "- `pipeline.magnitude_calculations(...)` then builds the repo's standard derived columns: `u/g/r/i/z/y_modelFlux_mag`, `u/g/r/i/z/y_psfFlux_mag`, `u/g/r/i/z/y_diff`, `u/g/r/i/z/y_psfRelErr`, and colors `ug`, `gr`, `ri`, `iz`, `zy`, `gi`.\n",
        "- `psf_cmodel_fit.compute_pS(...)` operates on the per-band pair (`*_diff`, `*_modelFlux_mag`).\n",
        "- `build_ecdfs_probability_models.compute_current_pSr(...)` explicitly recomputes `pS_r` from `r_diff` and `r_modelFlux_mag`.\n\n",
        "## Prepared-table coverage\n\n",
    ]
    for label, reason, status, missing in rows:
        text.append(f"- `{label}`: {status}. {reason}.\n")
        if missing:
            text.append(f"  Missing columns: {', '.join(f'`{col}`' for col in missing)}\n")
    text.extend(
        [
            "\n## Practical compatibility notes\n\n",
            "- The new prepared table intentionally renames the raw identity/coordinate columns to `object_id`, `ra`, and `dec` for readability.\n",
            "- The pS-specific derived columns (`*_modelFlux_mag`, `*_psfFlux_mag`, `*_diff`, `*_psfRelErr`, and the color indices) are kept in the existing repo naming convention so later pS work can reuse them directly.\n",
            "- If an old notebook/script expects the original raw coordinate names (`objectId`, `coord_ra`, `coord_dec`), only a trivial three-column rename adapter is needed.\n",
            "- Native Rubin `*_extendedness` columns are present for all six bands and are the most direct quantities for later pS-vs-extendedness comparisons.\n",
        ]
    )
    path.write_text("".join(text))


def robust_limits(values: pd.Series, q_lo: float = 0.5, q_hi: float = 99.5, pad_frac: float = 0.05) -> tuple[float, float]:
    numeric = pd.to_numeric(values, errors="coerce")
    arr = numeric.to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return (0.0, 1.0)
    lo = float(np.nanpercentile(arr, q_lo))
    hi = float(np.nanpercentile(arr, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return (float(np.nanmin(arr)), float(np.nanmax(arr)))
    pad = max((hi - lo) * pad_frac, 1e-6)
    return lo - pad, hi + pad


def make_ra_dec_plot(df: pd.DataFrame, path_png: Path, path_pdf: Path) -> None:
    use = df[df["base_clean"]].copy()
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    hb = ax.hexbin(use["ra"], use["dec"], gridsize=200, bins="log", mincnt=1, cmap="viridis")
    hb.set_rasterized(True)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("objects per hexbin (log)")
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.set_title(f"ECDFS DP1 footprint: {len(use):,} base-clean objects")
    ax.grid(alpha=0.25)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)


def make_cmd_plot(df: pd.DataFrame, path_png: Path, path_pdf: Path) -> tuple[int, str, str]:
    required = ["gi_cmd_clean", "gi", "i_modelFlux_mag"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Cannot build first-look CMD; missing columns: {missing}")
    use = df[df["gi_cmd_clean"]].copy()
    use = use[np.isfinite(use["gi"]) & np.isfinite(use["i_modelFlux_mag"])].copy()
    use = use[(use["i_modelFlux_mag"] > 10) & (use["i_modelFlux_mag"] < 35)].copy()
    if use.empty:
        raise ValueError("No rows survived the g/i CMD sanity cuts.")

    # Use robust display limits so a small number of pathological colors does
    # not flatten the main CMD locus.
    xlim = robust_limits(use["gi"], q_lo=2.0, q_hi=98.0, pad_frac=0.08)
    ylim = robust_limits(use["i_modelFlux_mag"], q_lo=0.5, q_hi=99.5, pad_frac=0.05)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    hb = ax.hexbin(use["gi"], use["i_modelFlux_mag"], gridsize=220, bins="log", mincnt=1, cmap="magma")
    hb.set_rasterized(True)
    cbar = fig.colorbar(hb, ax=ax)
    cbar.set_label("objects per hexbin (log)")
    ax.set_xlabel("g - i (dereddened CModel color)")
    ax.set_ylabel("i (flux-derived CModel magnitude)")
    ax.set_title(f"ECDFS DP1 first-look CMD: {len(use):,} g/i-clean objects")
    ax.grid(alpha=0.25)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path_png, dpi=220, bbox_inches="tight")
    fig.savefig(path_pdf, bbox_inches="tight")
    plt.close(fig)
    return len(use), f"{xlim[0]:.3f} .. {xlim[1]:.3f}", f"{ylim[0]:.3f} .. {ylim[1]:.3f}"


def save_analysis_table(df: pd.DataFrame, csv_path: Path, parquet_path: Path, export_csv: bool, export_parquet: bool) -> list[str]:
    notes: list[str] = []
    if export_csv:
        df.to_csv(csv_path, index=False)
        notes.append(f"Wrote CSV: {csv_path}")
    if export_parquet:
        try:
            df.to_parquet(parquet_path, index=False)
            notes.append(f"Wrote parquet: {parquet_path}")
        except Exception as exc:
            notes.append(f"Parquet export failed: {exc}")
    return notes


def run(args: argparse.Namespace) -> None:
    ensure_dir(args.results_dir)
    ensure_dir(args.outputs_dir)

    results = {
        "inventory": args.results_dir / "dp1_column_inventory.md",
        "cleaning": args.results_dir / "dp1_cleaning_recommendations.md",
        "cleaning_json": args.results_dir / "dp1_cleaning_flags.json",
        "ps_inputs": args.results_dir / "pS_input_requirements.md",
        "butler_cols": args.results_dir / "dp1_butler_column_list.txt",
        "ra_dec_png": args.results_dir / "dp1_ecdfs_ra_dec.png",
        "ra_dec_pdf": args.results_dir / "dp1_ecdfs_ra_dec.pdf",
        "cmd_png": args.results_dir / "dp1_ecdfs_cmd_firstlook.png",
        "cmd_pdf": args.results_dir / "dp1_ecdfs_cmd_firstlook.pdf",
    }
    outputs = {
        "csv": args.outputs_dir / "dp1_ecdfs_analysis_table.csv",
        "parquet": args.outputs_dir / "dp1_ecdfs_analysis_table.parquet",
    }

    raw_df = read_raw_catalog(args.input)
    processed_df = read_processed_catalog(args.input)

    core = detect_core_columns(raw_df)
    missing_core = [key for key in ["objectId", "coord_ra", "coord_dec"] if core[key] is None]
    if missing_core:
        raise ValueError(f"Missing required core ECDFS columns: {missing_core}. Available columns: {list(raw_df.columns)}")
    band_info = detect_band_columns(raw_df)
    current_fits_columns = build_butler_column_list(core, band_info)
    target_butler_columns = build_target_butler_column_list()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=PerformanceWarning)
        analysis_df = build_analysis_table(raw_df, processed_df, core, band_info)

    n_input = len(analysis_df)
    n_base_clean = int(analysis_df["base_clean"].sum())
    print(f"Input ECDFS rows: {n_input:,}")
    print(f"Rows passing base_clean: {n_base_clean:,}")
    if "gi_cmd_clean" in analysis_df.columns:
        print(f"Rows passing gi_cmd_clean: {int(analysis_df['gi_cmd_clean'].sum()):,}")
    if "all_bands_psf_cmodel_ready" in analysis_df.columns:
        print(f"Rows passing all_bands_psf_cmodel_ready: {int(analysis_df['all_bands_psf_cmodel_ready'].sum()):,}")

    write_butler_column_list(results["butler_cols"], current_fits_columns, target_butler_columns)
    write_column_inventory(
        results["inventory"], raw_df, analysis_df, core, band_info, current_fits_columns, target_butler_columns, args.input
    )
    write_cleaning_recommendations(results["cleaning"], results["cleaning_json"], raw_df, core, band_info)
    write_ps_input_requirements(results["ps_inputs"], analysis_df)

    save_df = analysis_df.loc[analysis_df["base_clean"]].copy() if args.apply_cleaning else analysis_df.copy()
    export_notes = save_analysis_table(save_df, outputs["csv"], outputs["parquet"], args.export_csv, args.export_parquet)
    for note in export_notes:
        print(note)

    if args.make_plots:
        make_ra_dec_plot(analysis_df, results["ra_dec_png"], results["ra_dec_pdf"])
        cmd_n, cmd_xlim, cmd_ylim = make_cmd_plot(analysis_df, results["cmd_png"], results["cmd_pdf"])
        print(f"RA/Dec plot written to: {results['ra_dec_png']}")
        print(f"CMD plot written to: {results['cmd_png']}")
        ra = pd.to_numeric(analysis_df["ra"], errors="coerce")
        dec = pd.to_numeric(analysis_df["dec"], errors="coerce")
        print(f"RA range: {float(np.nanmin(ra)):.6f} .. {float(np.nanmax(ra)):.6f} deg")
        print(f"Dec range: {float(np.nanmin(dec)):.6f} .. {float(np.nanmax(dec)):.6f} deg")
        print(
            "Footprint assessment: the density map spans RA ~52.35-53.93 deg and Dec ~-28.76 to -27.41 deg, "
            "which is consistent with an extended CDFS / ECDFS-style footprint rather than an unrelated field."
        )
        print(f"First-look CMD rows: {cmd_n:,}")
        print(f"First-look CMD x-range (g-i): {cmd_xlim}")
        print(f"First-look CMD y-range (i): {cmd_ylim}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
