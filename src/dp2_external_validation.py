"""Shared DP2 external-validation utilities.

The functions in this module are intentionally field-agnostic.  They support
the DP2 ECDFS and COSMOS FITS exports now in the repo, plus external validation
catalogs such as GOODS-S/3D-HST and COSMOS2020 FARMER.  COSMOS2020 labels are
handled as external validation labels, not as perfect truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

BANDS = ("u", "g", "r", "i", "z", "y")
NANOMAGGY_ZEROPOINT = 31.4
DEFAULT_COLOR_RANGE = (-1.0, 5.0)
DEFAULT_MAG_RANGE = (15.0, 28.0)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def flux_to_mag(flux, zeropoint: float = NANOMAGGY_ZEROPOINT):
    """Convert nJy flux to AB magnitude, returning NaN for non-positive flux."""

    arr = np.asarray(flux, dtype="float64")
    out = np.full(arr.shape, np.nan, dtype="float64")
    good = np.isfinite(arr) & (arr > 0)
    out[good] = zeropoint - 2.5 * np.log10(arr[good])
    if np.ndim(flux) == 0:
        return float(out)
    return out


def flux_err_to_mag_err(flux, flux_err):
    """Propagate flux error to magnitude error."""

    f = np.asarray(flux, dtype="float64")
    ferr = np.asarray(flux_err, dtype="float64")
    out = np.full(np.broadcast(f, ferr).shape, np.nan, dtype="float64")
    good = np.isfinite(f) & np.isfinite(ferr) & (f > 0) & (ferr >= 0)
    out[good] = (2.5 / np.log(10.0)) * ferr[good] / f[good]
    if np.ndim(flux) == 0 and np.ndim(flux_err) == 0:
        return float(out)
    return out


def read_table(path: str | Path, columns: Iterable[str] | None = None) -> pd.DataFrame:
    """Read FITS/parquet/csv with optional FITS column selection."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    if suffix == ".csv":
        return pd.read_csv(path, usecols=list(columns) if columns else None)
    if suffix in {".fits", ".fit", ".fz"}:
        from astropy.io import fits

        with fits.open(path, memmap=True) as hdul:
            hdu = next(h for h in hdul if hasattr(h, "columns") and h.data is not None)
            available = list(hdu.columns.names)
            selected = list(columns) if columns is not None else available
            missing = [col for col in selected if col not in available]
            if missing:
                raise ValueError(f"FITS table is missing requested columns: {missing}")
            data = {}
            for col in selected:
                arr = np.asarray(hdu.data[col])
                if arr.dtype.byteorder not in ("=", "|"):
                    arr = arr.byteswap().view(arr.dtype.newbyteorder("="))
                if arr.dtype.kind == "S":
                    arr = arr.astype(str)
                data[col] = arr
        return pd.DataFrame(data)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: str | Path, write_csv: bool = False, max_csv_rows: int = 250_000) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if path.suffix.lower() in {".parquet", ".pq"}:
        df.to_parquet(path, index=False)
        if write_csv and len(df) <= max_csv_rows:
            df.to_csv(path.with_suffix(".csv"), index=False)
    elif path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def _as_bool_clean(series: pd.Series) -> pd.Series:
    """Return True where a Rubin flag column is clean/False/0."""

    if series.dtype == bool:
        return ~series.fillna(True)
    vals = pd.to_numeric(series, errors="coerce")
    return vals.fillna(1).eq(0)


def standardize_dp2_table(df: pd.DataFrame, field: str | None = None) -> pd.DataFrame:
    """Add analysis-friendly DP2 aliases and derived columns without dropping raw columns."""

    out = df.copy()
    if "objectId" in out.columns:
        out["object_id"] = pd.to_numeric(out["objectId"], errors="coerce").astype("Int64")
    if "coord_ra" in out.columns:
        out["ra"] = pd.to_numeric(out["coord_ra"], errors="coerce")
    if "coord_dec" in out.columns:
        out["dec"] = pd.to_numeric(out["coord_dec"], errors="coerce")
    if field is not None:
        out["field"] = field

    for band in BANDS:
        raw_map = {
            f"{band}_psfFlux": f"psf_flux_{band}",
            f"{band}_psfFluxErr": f"psf_flux_err_{band}",
            f"{band}_psfFlux_flag": f"psf_flux_flag_{band}",
            f"{band}_cModelFlux": f"cmodel_flux_{band}",
            f"{band}_cModelFluxErr": f"cmodel_flux_err_{band}",
            f"{band}_cModel_flag": f"cmodel_flux_flag_{band}",
            f"{band}_extendedness": f"extendedness_{band}",
            f"{band}_extendedness_flag": f"extendedness_flag_{band}",
            f"{band}_blendedness": f"blendedness_{band}",
            f"{band}_blendedness_flag": f"blendedness_flag_{band}",
        }
        for src, dest in raw_map.items():
            if src in out.columns and dest not in out.columns:
                out[dest] = out[src]

        # Existing notebook conventions used by Nirav's pS code.
        if f"{band}_cModelFlux" in out.columns and f"{band}_modelFlux" not in out.columns:
            out[f"{band}_modelFlux"] = out[f"{band}_cModelFlux"]
        if f"{band}_cModelFluxErr" in out.columns and f"{band}_modelErr" not in out.columns:
            out[f"{band}_modelErr"] = out[f"{band}_cModelFluxErr"]

        cflux = f"cmodel_flux_{band}"
        pflux = f"psf_flux_{band}"
        cerr = f"cmodel_flux_err_{band}"
        perr = f"psf_flux_err_{band}"
        if cflux in out.columns:
            out[f"cmodel_mag_{band}"] = flux_to_mag(out[cflux])
            out[f"{band}_modelFlux_mag"] = out[f"cmodel_mag_{band}"]
        if pflux in out.columns:
            out[f"psf_mag_{band}"] = flux_to_mag(out[pflux])
            out[f"{band}_psfFlux_mag"] = out[f"psf_mag_{band}"]
        if cflux in out.columns and cerr in out.columns:
            out[f"cmodel_mag_err_{band}"] = flux_err_to_mag_err(out[cflux], out[cerr])
        if pflux in out.columns and perr in out.columns:
            out[f"psf_mag_err_{band}"] = flux_err_to_mag_err(out[pflux], out[perr])
        if cflux in out.columns and pflux in out.columns:
            out[f"psf_minus_cmodel_{band}"] = out[f"psf_mag_{band}"] - out[f"cmodel_mag_{band}"]
            out[f"{band}_diff"] = out[f"psf_minus_cmodel_{band}"]
        if pflux in out.columns and perr in out.columns:
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"psf_rel_err_{band}"] = 1.09 * out[perr].astype("float64") / out[pflux].astype("float64")

    for b1, b2 in [("g", "i"), ("g", "r"), ("r", "i"), ("u", "g"), ("i", "z"), ("z", "y")]:
        m1, m2 = f"cmodel_mag_{b1}", f"cmodel_mag_{b2}"
        if m1 in out.columns and m2 in out.columns:
            out[f"cmodel_color_{b1}_{b2}"] = out[m1] - out[m2]
            out[f"{b1}{b2}"] = out[f"cmodel_color_{b1}_{b2}"]
    if "cmodel_color_g_i" in out.columns:
        out["cmd_g_minus_i"] = out["cmodel_color_g_i"]
    if "cmodel_mag_i" in out.columns:
        out["cmd_i_mag"] = out["cmodel_mag_i"]

    add_clean_masks(out)
    return out


def band_flux_flag_clean(df: pd.DataFrame, band: str, flux_type: str) -> pd.Series:
    flag = f"{flux_type}_flux_flag_{band}"
    if flag in df.columns:
        return _as_bool_clean(df[flag])
    raw = f"{band}_{'psfFlux' if flux_type == 'psf' else 'cModel'}_flag"
    if raw in df.columns:
        return _as_bool_clean(df[raw])
    return pd.Series(True, index=df.index)


def add_clean_masks(df: pd.DataFrame) -> pd.DataFrame:
    """Add task-specific clean masks in-place and return df."""

    if {"ra", "dec"}.issubset(df.columns):
        df["base_clean"] = np.isfinite(df["ra"]) & np.isfinite(df["dec"])
    else:
        df["base_clean"] = True

    all_ready = []
    for band in BANDS:
        cflux = f"cmodel_flux_{band}"
        pflux = f"psf_flux_{band}"
        cmag = f"cmodel_mag_{band}"
        pmag = f"psf_mag_{band}"
        ext = f"extendedness_{band}"
        extflag = f"extendedness_flag_{band}"
        if cflux in df.columns:
            c_values = pd.to_numeric(df[cflux], errors="coerce")
            c_ok = np.isfinite(c_values) & (c_values > 0) & band_flux_flag_clean(df, band, "cmodel")
        else:
            c_ok = pd.Series(False, index=df.index)
        if pflux in df.columns:
            p_values = pd.to_numeric(df[pflux], errors="coerce")
            p_ok = np.isfinite(p_values) & (p_values > 0) & band_flux_flag_clean(df, band, "psf")
        else:
            p_ok = pd.Series(False, index=df.index)
        ready = c_ok & p_ok
        df[f"pS_{band}_ready"] = ready & np.isfinite(df.get(f"{band}_diff", np.nan)) & np.isfinite(df.get(cmag, np.nan))
        if ext in df.columns:
            ext_ok = np.isfinite(pd.to_numeric(df[ext], errors="coerce"))
            if extflag in df.columns:
                ext_ok &= _as_bool_clean(df[extflag])
            df[f"extendedness_{band}_ready"] = ext_ok
        else:
            df[f"extendedness_{band}_ready"] = False
        all_ready.append(ready)

    if "cmd_g_minus_i" in df.columns and "cmd_i_mag" in df.columns:
        df["cmd_gi_cmodel_clean"] = (
            df["base_clean"]
            & df["pS_g_ready"]
            & df["pS_i_ready"]
            & np.isfinite(df["cmd_g_minus_i"])
            & np.isfinite(df["cmd_i_mag"])
        )
    else:
        df["cmd_gi_cmodel_clean"] = False

    if {"gr", "ri", "cmd_i_mag"}.issubset(df.columns):
        df["cmd_gr_ri_cmodel_clean"] = (
            df["base_clean"]
            & df["pS_g_ready"]
            & df["pS_r_ready"]
            & df["pS_i_ready"]
            & np.isfinite(df["gr"])
            & np.isfinite(df["ri"])
        )
    else:
        df["cmd_gr_ri_cmodel_clean"] = False

    df["all_bands_psf_cmodel_ready"] = np.logical_and.reduce(all_ready) if all_ready else False
    return df


def make_basic_field_plots(df: pd.DataFrame, field: str, plot_dir: str | Path) -> dict[str, int]:
    """Write footprint/CMD/color/extendedness sanity plots."""

    import matplotlib.pyplot as plt

    plot_dir = ensure_dir(plot_dir)
    counts = {
        "rows": int(len(df)),
        "base_clean": int(df.get("base_clean", pd.Series(True, index=df.index)).sum()),
        "cmd_gi_cmodel_clean": int(df.get("cmd_gi_cmodel_clean", pd.Series(False, index=df.index)).sum()),
        "cmd_gr_ri_cmodel_clean": int(df.get("cmd_gr_ri_cmodel_clean", pd.Series(False, index=df.index)).sum()),
        "all_bands_psf_cmodel_ready": int(df.get("all_bands_psf_cmodel_ready", pd.Series(False, index=df.index)).sum()),
    }

    def save(fig, stem: str):
        fig.savefig(plot_dir / f"{stem}.png", dpi=200, bbox_inches="tight")
        fig.savefig(plot_dir / f"{stem}.pdf", bbox_inches="tight")
        plt.close(fig)

    use = df[np.isfinite(df["ra"]) & np.isfinite(df["dec"])]
    fig, ax = plt.subplots(figsize=(7, 5.8))
    hb = ax.hexbin(use["ra"], use["dec"], gridsize=180, bins="log", mincnt=1, cmap="viridis")
    ax.set_xlabel("RA (deg)")
    ax.set_ylabel("Dec (deg)")
    ax.set_title(f"DP2 {field} footprint")
    fig.colorbar(hb, ax=ax, label="log10(N)")
    save(fig, "footprint")

    if {"cmd_g_minus_i", "cmd_i_mag"}.issubset(df.columns):
        full = df[
            np.isfinite(df["cmd_g_minus_i"])
            & np.isfinite(df["cmd_i_mag"])
            & df["cmd_g_minus_i"].between(*DEFAULT_COLOR_RANGE)
            & df["cmd_i_mag"].between(*DEFAULT_MAG_RANGE)
        ]
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        hb = ax.hexbin(full["cmd_g_minus_i"], full["cmd_i_mag"], gridsize=180, bins="log", mincnt=1, cmap="magma")
        ax.set_xlabel("g - i (CModel AB)")
        ax.set_ylabel("i (CModel AB)")
        ax.set_xlim(*DEFAULT_COLOR_RANGE)
        ax.set_ylim(DEFAULT_MAG_RANGE[1], DEFAULT_MAG_RANGE[0])
        ax.set_title(f"DP2 {field} first-look CMD")
        fig.colorbar(hb, ax=ax, label="log10(N)")
        save(fig, "cmd_full")

        clean = df[
            df["cmd_gi_cmodel_clean"]
            & df["cmd_g_minus_i"].between(*DEFAULT_COLOR_RANGE)
            & df["cmd_i_mag"].between(*DEFAULT_MAG_RANGE)
        ]
        counts["cmd_gi_zoom_display"] = int(len(clean))
        fig, ax = plt.subplots(figsize=(6.4, 6.0))
        hb = ax.hexbin(clean["cmd_g_minus_i"], clean["cmd_i_mag"], gridsize=160, bins="log", mincnt=1, cmap="magma")
        ax.set_xlabel("g - i (CModel AB)")
        ax.set_ylabel("i (CModel AB)")
        ax.set_xlim(*DEFAULT_COLOR_RANGE)
        ax.set_ylim(DEFAULT_MAG_RANGE[1], DEFAULT_MAG_RANGE[0])
        ax.set_title(f"DP2 {field} clean/zoom CMD")
        fig.colorbar(hb, ax=ax, label="log10(N)")
        save(fig, "cmd_clean_zoom")

    if {"gr", "ri"}.issubset(df.columns):
        cc = df[
            df["cmd_gr_ri_cmodel_clean"]
            & np.isfinite(df["gr"])
            & np.isfinite(df["ri"])
            & df["gr"].between(*DEFAULT_COLOR_RANGE)
            & df["ri"].between(*DEFAULT_COLOR_RANGE)
        ]
        fig, ax = plt.subplots(figsize=(6.2, 5.6))
        hb = ax.hexbin(cc["gr"], cc["ri"], gridsize=150, bins="log", mincnt=1, cmap="cividis")
        ax.set_xlabel("g - r (CModel AB)")
        ax.set_ylabel("r - i (CModel AB)")
        ax.set_xlim(*DEFAULT_COLOR_RANGE)
        ax.set_ylim(*DEFAULT_COLOR_RANGE)
        ax.set_title(f"DP2 {field} color-color")
        fig.colorbar(hb, ax=ax, label="log10(N)")
        save(fig, "color_color_gr_ri")

    if {"extendedness_r", "cmodel_mag_r"}.issubset(df.columns):
        ext = df[
            df["extendedness_r_ready"]
            & np.isfinite(df["extendedness_r"])
            & np.isfinite(df["cmodel_mag_r"])
            & df["cmodel_mag_r"].between(*DEFAULT_MAG_RANGE)
        ]
        fig, ax = plt.subplots(figsize=(6.5, 5.3))
        hb = ax.hexbin(ext["cmodel_mag_r"], ext["extendedness_r"], gridsize=130, bins="log", mincnt=1, cmap="viridis")
        ax.set_xlabel("r (CModel AB)")
        ax.set_ylabel("r extendedness")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(DEFAULT_MAG_RANGE[1], DEFAULT_MAG_RANGE[0])
        ax.set_title(f"DP2 {field} r-band extendedness sanity")
        fig.colorbar(hb, ax=ax, label="log10(N)")
        save(fig, "r_extendedness_sanity")

    summary = pd.DataFrame([{"mask": k, "count": v, "fraction": v / len(df) if len(df) else np.nan} for k, v in counts.items()])
    summary.to_csv(plot_dir / "clean_mask_summary.csv", index=False)
    lines = [f"# DP2 {field} clean-mask summary\n\n", f"Input rows: {len(df):,}\n\n"]
    for k, v in counts.items():
        lines.append(f"- `{k}`: {v:,}\n")
    lines.append("\nMasks are task-specific first-pass diagnostics, not final science cuts.\n")
    (plot_dir / "clean_mask_summary.md").write_text("".join(lines))
    return counts


@dataclass
class MatchResult:
    matched: pd.DataFrame
    summary: dict[str, object]


def load_hst_ecdfs_truth(path: str | Path) -> pd.DataFrame:
    truth = read_table(path)
    required = {"ra", "dec", "label"}
    missing = required - set(truth.columns)
    if missing:
        raise ValueError(f"HST ECDFS truth catalog missing columns: {sorted(missing)}")
    out = truth.copy()
    out["external_id"] = out.get("id", pd.RangeIndex(len(out))).astype(str)
    out["external_ra"] = pd.to_numeric(out["ra"], errors="coerce")
    out["external_dec"] = pd.to_numeric(out["dec"], errors="coerce")
    out["truth_label"] = out["label"].astype(str).str.lower()
    out["truth_binary"] = out["truth_label"].map({"galaxy": 0, "star": 1})
    out["external_label_source"] = out.get("label_source", "GOODS-S/3D-HST")
    keep = ["external_id", "external_ra", "external_dec", "truth_label", "truth_binary", "external_label_source"]
    for col in ["quality_flag", "notes"]:
        if col in out.columns:
            keep.append(col)
    return out[keep].dropna(subset=["external_ra", "external_dec"]).reset_index(drop=True)


def inspect_cosmos2020_farmer_columns(path: str | Path) -> dict[str, object]:
    from astropy.io import fits

    with fits.open(path, memmap=True) as hdul:
        hdu = next(h for h in hdul if hasattr(h, "columns"))
        cols = list(hdu.columns.names)
        return {
            "path": str(path),
            "rows": int(len(hdu.data)),
            "n_columns": int(len(cols)),
            "columns": cols,
            "has_acs_mu_class": "ACS_MU_CLASS" in cols,
            "has_lp_type": "lp_type" in cols,
            "has_flag_combined": "FLAG_COMBINED" in cols,
        }


def load_cosmos2020_farmer_validation(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        truth = read_table(path)
        required = {"ra", "dec", "label"}
        if required.issubset(truth.columns):
            out = truth.copy()
            out["external_id"] = out.get("id", pd.RangeIndex(len(out))).astype(str)
            out["external_ra"] = pd.to_numeric(out["ra"], errors="coerce")
            out["external_dec"] = pd.to_numeric(out["dec"], errors="coerce")
            out["truth_label"] = out["label"].astype(str).str.lower()
            out["truth_binary"] = out["truth_label"].map({"galaxy": 0, "star": 1})
            out["external_label_source"] = out.get("label_source", "COSMOS2020 FARMER ACS_MU_CLASS")
            keep = ["external_id", "external_ra", "external_dec", "truth_label", "truth_binary", "external_label_source"]
            for col in ["quality_flag", "notes", "acs_mu_class", "flag_combined", "lp_type", "lp_type_label", "acs_f814w_mag"]:
                if col in out.columns:
                    keep.append(col)
            return out[keep].dropna(subset=["external_ra", "external_dec"]).reset_index(drop=True)

    cols = [
        "ID",
        "ALPHA_J2000",
        "DELTA_J2000",
        "FARMER_ID",
        "FLAG_COMBINED",
        "ID_ACS",
        "ACS_F814W_MAG",
        "ACS_MU_CLASS",
        "lp_type",
    ]
    info = inspect_cosmos2020_farmer_columns(path)
    available = [c for c in cols if c in info["columns"]]
    missing_required = [c for c in ["ID", "ALPHA_J2000", "DELTA_J2000"] if c not in available]
    if missing_required:
        raise ValueError(f"COSMOS2020 FARMER missing required coordinate/id columns: {missing_required}")
    df = read_table(path, columns=available)
    out = pd.DataFrame()
    out["external_id"] = df["ID"].astype(str)
    out["external_ra"] = pd.to_numeric(df["ALPHA_J2000"], errors="coerce")
    out["external_dec"] = pd.to_numeric(df["DELTA_J2000"], errors="coerce")
    for col in ["FARMER_ID", "FLAG_COMBINED", "ID_ACS", "ACS_F814W_MAG", "ACS_MU_CLASS", "lp_type"]:
        if col in df.columns:
            out[col] = df[col]

    # COSMOS2020 documentation convention requested for external validation.
    if "ACS_MU_CLASS" in out.columns:
        acs = pd.to_numeric(out["ACS_MU_CLASS"], errors="coerce")
        out["truth_label"] = np.select([acs.eq(1), acs.eq(2), acs.eq(3)], ["galaxy", "star", "fake"], default=pd.NA)
        out["truth_binary"] = pd.Series(out["truth_label"]).map({"galaxy": 0, "star": 1})
        out["external_label_source"] = "COSMOS2020 FARMER ACS_MU_CLASS; fake/spurious excluded from truth metrics"
    else:
        out["truth_label"] = pd.NA
        out["truth_binary"] = pd.NA
        out["external_label_source"] = "COSMOS2020 FARMER; no ACS_MU_CLASS found"

    if "lp_type" in out.columns:
        lp = pd.to_numeric(out["lp_type"], errors="coerce")
        out["lp_type_label"] = np.select([lp.eq(0), lp.eq(1), lp.eq(2), lp.eq(-9)], ["galaxy", "star", "xray", "failed"], default=pd.NA)

    return out.dropna(subset=["external_ra", "external_dec"]).reset_index(drop=True)


def match_external_to_dp2(
    dp2: pd.DataFrame,
    external: pd.DataFrame,
    max_sep_arcsec: float = 0.3,
    extra_dp2_cols: list[str] | None = None,
) -> MatchResult:
    """Nearest-neighbor RA/Dec match with one-to-one deduplication."""

    from astropy import units as u
    from astropy.coordinates import SkyCoord

    needed = {"object_id", "ra", "dec"}
    missing = needed - set(dp2.columns)
    if missing:
        raise ValueError(f"DP2 table missing standardized columns: {sorted(missing)}")
    needed_ext = {"external_id", "external_ra", "external_dec"}
    missing_ext = needed_ext - set(external.columns)
    if missing_ext:
        raise ValueError(f"External table missing columns: {sorted(missing_ext)}")

    dp2_use = dp2.dropna(subset=["ra", "dec", "object_id"]).copy()
    ext_use = external.dropna(subset=["external_ra", "external_dec"]).copy()
    dp2_coord = SkyCoord(dp2_use["ra"].to_numpy() * u.deg, dp2_use["dec"].to_numpy() * u.deg)
    ext_coord = SkyCoord(ext_use["external_ra"].to_numpy() * u.deg, ext_use["external_dec"].to_numpy() * u.deg)
    idx, sep2d, _ = ext_coord.match_to_catalog_sky(dp2_coord)
    sep_arcsec = sep2d.arcsec
    within = sep_arcsec <= max_sep_arcsec

    ext_m = ext_use.loc[within].reset_index(drop=True)
    nearest = dp2_use.iloc[idx[within]].reset_index(drop=True)
    ext_m["match_sep_arcsec"] = sep_arcsec[within]
    dec_ref = np.deg2rad(ext_m["external_dec"].astype("float64").to_numpy())
    ext_m["delta_ra_cosdec_arcsec"] = (nearest["ra"].to_numpy() - ext_m["external_ra"].to_numpy()) * np.cos(dec_ref) * 3600.0
    ext_m["delta_dec_arcsec"] = (nearest["dec"].to_numpy() - ext_m["external_dec"].to_numpy()) * 3600.0

    if extra_dp2_cols is None:
        keep_dp2 = [
            c
            for c in dp2_use.columns
            if c in {"object_id", "ra", "dec", "field", "refExtendedness", "refBand"}
            or c.startswith(("cmodel_", "psf_", "extendedness_", "pS_", "cmd_", "gr", "ri", "ug", "iz", "zy"))
            or c.endswith("_ready")
            or c in {"base_clean", "cmd_gi_cmodel_clean", "cmd_gr_ri_cmodel_clean", "all_bands_psf_cmodel_ready"}
        ]
    else:
        keep_dp2 = ["object_id", "ra", "dec"] + extra_dp2_cols
    keep_dp2 = [c for c in dict.fromkeys(keep_dp2) if c in nearest.columns]
    dp2_part = nearest[keep_dp2].add_prefix("dp2_")
    matched = pd.concat([ext_m.reset_index(drop=True), dp2_part.reset_index(drop=True)], axis=1)
    matched["object_id"] = matched["dp2_object_id"]

    raw_count = len(matched)
    matched = matched.sort_values(["object_id", "match_sep_arcsec"], kind="mergesort")
    dup_dp2 = int(matched["object_id"].duplicated().sum())
    matched = matched.drop_duplicates(subset=["object_id"], keep="first")
    matched = matched.sort_values(["external_id", "match_sep_arcsec"], kind="mergesort")
    dup_ext = int(matched["external_id"].duplicated().sum())
    matched = matched.drop_duplicates(subset=["external_id"], keep="first").reset_index(drop=True)

    summary = {
        "external_rows": int(len(external)),
        "dp2_rows": int(len(dp2)),
        "max_sep_arcsec": float(max_sep_arcsec),
        "raw_matches_within_radius": int(raw_count),
        "duplicates_by_dp2_object_removed": dup_dp2,
        "duplicates_by_external_id_removed": dup_ext,
        "final_matches": int(len(matched)),
        "match_fraction_external": float(len(matched) / len(external)) if len(external) else np.nan,
        "median_sep_arcsec": float(np.nanmedian(matched["match_sep_arcsec"])) if len(matched) else np.nan,
        "p95_sep_arcsec": float(np.nanpercentile(matched["match_sep_arcsec"], 95)) if len(matched) else np.nan,
        "max_sep_arcsec_found": float(np.nanmax(matched["match_sep_arcsec"])) if len(matched) else np.nan,
    }
    if "truth_label" in matched.columns:
        summary["truth_label_counts"] = matched["truth_label"].value_counts(dropna=False).to_dict()
    if "FLAG_COMBINED" in matched.columns:
        summary["FLAG_COMBINED_counts"] = matched["FLAG_COMBINED"].value_counts(dropna=False).head(20).to_dict()
    return MatchResult(matched=matched, summary=summary)


def save_match_diagnostics(matched: pd.DataFrame, summary: dict[str, object], output_dir: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt

    output_dir = ensure_dir(output_dir)
    pd.DataFrame([summary]).to_csv(output_dir / "match_summary.csv", index=False)
    lines = [f"# {title} match summary\n\n"]
    for key, val in summary.items():
        lines.append(f"- `{key}`: {val}\n")
    lines.append("\nLabels are external validation labels; they should not be treated as perfect truth without review.\n")
    (output_dir / "match_summary.md").write_text("".join(lines))

    if len(matched):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.hist(matched["match_sep_arcsec"], bins=60, range=(0, float(summary["max_sep_arcsec"])), histtype="stepfilled", alpha=0.65)
        ax.set_xlabel("match separation (arcsec)")
        ax.set_ylabel("N")
        ax.set_title(title)
        fig.savefig(output_dir / "match_sep_hist.png", dpi=200, bbox_inches="tight")
        fig.savefig(output_dir / "match_sep_hist.pdf", bbox_inches="tight")
        plt.close(fig)

        if {"external_ra", "external_dec"}.issubset(matched.columns):
            fig, ax = plt.subplots(figsize=(6.2, 5.4))
            ax.scatter(matched["external_ra"], matched["external_dec"], s=1, alpha=0.25)
            ax.set_xlabel("external RA (deg)")
            ax.set_ylabel("external Dec (deg)")
            ax.set_title(f"{title} footprint")
            fig.savefig(output_dir / "match_footprint.png", dpi=200, bbox_inches="tight")
            fig.savefig(output_dir / "match_footprint.pdf", bbox_inches="tight")
            plt.close(fig)

        if "truth_label" in matched.columns:
            counts = matched["truth_label"].value_counts(dropna=False)
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            counts.plot(kind="bar", ax=ax)
            ax.set_ylabel("N")
            ax.set_title(f"{title} label counts")
            fig.savefig(output_dir / "external_label_counts.png", dpi=200, bbox_inches="tight")
            fig.savefig(output_dir / "external_label_counts.pdf", bbox_inches="tight")
            plt.close(fig)


def infer_truth_binary(df: pd.DataFrame) -> pd.Series:
    if "truth_binary" in df.columns:
        return pd.to_numeric(df["truth_binary"], errors="coerce")
    if "truth_label" in df.columns:
        return df["truth_label"].astype(str).str.lower().map({"galaxy": 0, "star": 1})
    if "external_label" in df.columns:
        return df["external_label"].astype(str).str.lower().map({"galaxy": 0, "star": 1})
    raise ValueError("No truth/external label column found.")


def confusion_counts(reference_binary: pd.Series, predicted_star: pd.Series) -> dict[str, int]:
    ref = pd.to_numeric(reference_binary, errors="coerce")
    pred = predicted_star.astype("boolean")
    valid = ref.isin([0, 1]) & pred.notna()
    ref = ref[valid].astype(int)
    pred = pred[valid].astype(bool)
    a = int(((ref == 1) & pred).sum())
    b = int(((ref == 0) & ~pred).sum())
    c = int(((ref == 1) & ~pred).sum())
    d = int(((ref == 0) & pred).sum())
    return {"a": a, "b": b, "c": c, "d": d, "n": a + b + c + d}


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    a, b, c, d = counts["a"], counts["b"], counts["c"], counts["d"]
    div = lambda x, y: float(x / y) if y else np.nan
    return {
        **counts,
        "star_completeness": div(a, a + c),
        "star_purity": div(a, a + d),
        "star_contamination": div(d, a + d),
        "galaxy_completeness": div(b, b + d),
        "galaxy_purity": div(b, b + c),
        "galaxy_contamination": div(c, b + c),
    }


def build_common_sample(df: pd.DataFrame, score_col: str, ext_col: str = "dp2_extendedness_i") -> pd.DataFrame:
    truth = infer_truth_binary(df)
    masks = [
        truth.isin([0, 1]),
        np.isfinite(pd.to_numeric(df.get(score_col, np.nan), errors="coerce")),
        np.isfinite(pd.to_numeric(df.get(ext_col, np.nan), errors="coerce")),
    ]
    for col in ["dp2_cmd_g_minus_i", "dp2_cmd_i_mag"]:
        if col in df.columns:
            masks.append(np.isfinite(pd.to_numeric(df[col], errors="coerce")))
    if "dp2_cmd_gi_cmodel_clean" in df.columns:
        masks.append(df["dp2_cmd_gi_cmodel_clean"].astype(bool))
    mask = np.logical_and.reduce(masks)
    return df.loc[mask].copy().reset_index(drop=True)


def threshold_scan(df: pd.DataFrame, score_col: str, thresholds: np.ndarray | None = None) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.linspace(0, 1, 101)
    truth = infer_truth_binary(df)
    score = pd.to_numeric(df[score_col], errors="coerce")
    rows = []
    for thr in thresholds:
        rows.append({"threshold": float(thr), **metrics_from_counts(confusion_counts(truth, score >= thr))})
    return pd.DataFrame(rows)


def metrics_vs_mag(df: pd.DataFrame, predicted_star: pd.Series, mag_col: str = "dp2_cmd_i_mag") -> pd.DataFrame:
    truth = infer_truth_binary(df)
    mag = pd.to_numeric(df[mag_col], errors="coerce")
    finite = np.isfinite(mag) & truth.isin([0, 1]) & predicted_star.notna()
    if not finite.any():
        return pd.DataFrame()
    lo = max(DEFAULT_MAG_RANGE[0], np.floor(mag[finite].min() * 2) / 2)
    hi = min(DEFAULT_MAG_RANGE[1], np.ceil(mag[finite].max() * 2) / 2)
    if hi <= lo:
        return pd.DataFrame()
    bins = np.arange(lo, hi + 0.5, 0.5)
    rows = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = finite & (mag >= b0) & (mag < b1)
        if int(m.sum()) == 0:
            continue
        rows.append({"mag_lo": b0, "mag_hi": b1, "mag_center": 0.5 * (b0 + b1), **metrics_from_counts(confusion_counts(truth[m], predicted_star[m]))})
    return pd.DataFrame(rows)


def plot_four_panel_cmd(
    df: pd.DataFrame,
    predicted_star: pd.Series,
    output_path: str | Path,
    title: str,
    reference_binary: pd.Series | None = None,
) -> dict[str, int]:
    import matplotlib.pyplot as plt

    if reference_binary is None:
        reference_binary = infer_truth_binary(df)
    color = pd.to_numeric(df["dp2_cmd_g_minus_i"], errors="coerce")
    mag = pd.to_numeric(df["dp2_cmd_i_mag"], errors="coerce")
    pred = predicted_star.astype("boolean")
    ref = pd.to_numeric(reference_binary, errors="coerce")
    valid = color.notna() & mag.notna() & ref.isin([0, 1]) & pred.notna()
    ref = ref[valid].astype(int)
    pred = pred[valid].astype(bool)
    color = color[valid]
    mag = mag[valid]

    panels = [
        ("Correct/agreement stars (a)", (ref == 1) & pred, "black"),
        ("Correct/agreement galaxies (b)", (ref == 0) & ~pred, "tab:blue"),
        ("Reference stars predicted galaxies (c)", (ref == 1) & ~pred, "tab:orange"),
        ("Reference galaxies predicted stars (d)", (ref == 0) & pred, "tab:red"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.4), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, (name, mask, color_name) in zip(axes, panels):
        if len(color) > 10:
            ax.hexbin(color, mag, gridsize=80, bins="log", mincnt=1, cmap="Greys", alpha=0.22)
        subx, suby = color[mask], mag[mask]
        if len(subx) > 500:
            ax.hexbin(subx, suby, gridsize=70, bins="log", mincnt=1, cmap="Reds" if color_name == "tab:red" else "Blues")
        else:
            ax.scatter(subx, suby, s=8, alpha=0.65, c=color_name)
        ax.set_title(f"{name}\nN={int(mask.sum()):,}", fontsize=10)
        ax.grid(True, alpha=0.18)
    for ax in axes[2:]:
        ax.set_xlabel("g - i (CModel AB)")
    for ax in axes[::2]:
        ax.set_ylabel("i (CModel AB)")
    axes[0].set_xlim(*DEFAULT_COLOR_RANGE)
    axes[0].set_ylim(DEFAULT_MAG_RANGE[1], DEFAULT_MAG_RANGE[0])
    fig.suptitle(title)
    fig.tight_layout()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return confusion_counts(ref, pred)


def plot_metrics_vs_mag(metrics: pd.DataFrame, output_path: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if metrics.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for col, label in [
        ("star_completeness", "star completeness"),
        ("star_contamination", "star contamination"),
        ("galaxy_completeness", "galaxy completeness"),
        ("galaxy_contamination", "galaxy contamination"),
    ]:
        if col in metrics.columns:
            ax.plot(metrics["mag_center"], metrics[col], marker="o", ms=3, label=label)
    ax.set_xlabel("i magnitude bin center")
    ax.set_ylabel("metric")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_purity_completeness(scan: pd.DataFrame, output_path: str | Path, title: str) -> None:
    import matplotlib.pyplot as plt

    if scan.empty:
        return
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.plot(scan["star_completeness"], scan["star_purity"], marker=".", lw=1.0, label="star")
    ax.plot(scan["galaxy_completeness"], scan["galaxy_purity"], marker=".", lw=1.0, label="galaxy")
    ax.set_xlabel("completeness")
    ax.set_ylabel("purity")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Small dependency-free Markdown table writer."""

    if df.empty:
        return ""
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
        else:
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(map(str, row)) + " |" for row in text_df.to_numpy()]
    return "\n".join([header, sep, *rows])


def run_validation_suite(
    matched: pd.DataFrame,
    output_dir: str | Path,
    field: str,
    ps_version: str,
    score_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run pS/extendedness external-validation diagnostics for available scores."""

    output_dir = ensure_dir(output_dir)
    if score_cols is None:
        score_cols = [
            c
            for c in matched.columns
            if not c.endswith("_ready") and ((c.startswith("pS_") and c[-1:] in BANDS) or (c.startswith("dp2_pS_") and c[-1:] in BANDS))
        ]
    rows = []
    truth = infer_truth_binary(matched)

    ext_candidates = [c for c in ["dp2_extendedness_i", "extendedness_i", "i_extendedness"] if c in matched.columns]
    if ext_candidates:
        ext_col = ext_candidates[0]
        ext = pd.to_numeric(matched[ext_col], errors="coerce")
        use = build_common_sample(matched, score_cols[0] if score_cols else ext_col, ext_col=ext_col) if score_cols else matched[truth.isin([0, 1]) & np.isfinite(ext)].copy()
        pred_star = pd.to_numeric(use[ext_col], errors="coerce") < 0.5
        counts = plot_four_panel_cmd(use, pred_star, output_dir / "cmd_four_panel_extendedness_i_vs_external", f"DP2 {field} {ps_version}: extendedness_i vs external labels")
        row = {"comparison": "extendedness_i_vs_external", "score_col": ext_col, **metrics_from_counts(counts)}
        rows.append(row)
        mm = metrics_vs_mag(use, (pd.to_numeric(use[ext_col], errors="coerce") < 0.5))
        mm.to_csv(output_dir / "metrics_vs_i_extendedness_i_vs_external.csv", index=False)
        plot_metrics_vs_mag(mm, output_dir / "metrics_vs_i_extendedness_i_vs_external", f"DP2 {field}: extendedness_i metrics")

    for score_col in score_cols:
        if score_col not in matched.columns:
            continue
        score = pd.to_numeric(matched[score_col], errors="coerce")
        valid = truth.isin([0, 1]) & np.isfinite(score)
        if "dp2_cmd_gi_cmodel_clean" in matched.columns:
            valid &= matched["dp2_cmd_gi_cmodel_clean"].astype(bool)
        use = matched.loc[valid].copy()
        if len(use) == 0:
            continue
        pred_star = pd.to_numeric(use[score_col], errors="coerce") >= 0.5
        safe = score_col.replace("dp2_", "").replace("/", "_")
        counts = plot_four_panel_cmd(use, pred_star, output_dir / f"cmd_four_panel_{safe}_vs_external", f"DP2 {field} {ps_version}: {score_col} vs external labels")
        rows.append({"comparison": f"{safe}_vs_external", "score_col": score_col, **metrics_from_counts(counts)})
        mm = metrics_vs_mag(use, pred_star)
        mm.to_csv(output_dir / f"metrics_vs_i_{safe}_vs_external.csv", index=False)
        plot_metrics_vs_mag(mm, output_dir / f"metrics_vs_i_{safe}_vs_external", f"DP2 {field}: {score_col} metrics")
        scan = threshold_scan(use, score_col)
        scan.to_csv(output_dir / f"threshold_scan_{safe}.csv", index=False)
        plot_purity_completeness(scan, output_dir / f"purity_completeness_{safe}", f"DP2 {field}: {score_col}")

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    lines = [
        f"# DP2 {field} external-validation summary ({ps_version})\n\n",
        f"- Input matched rows: {len(matched):,}\n",
        f"- External star/galaxy rows: {int(infer_truth_binary(matched).isin([0, 1]).sum()):,}\n",
        f"- pS score columns used: {score_cols if score_cols else 'none found'}\n",
        "- COSMOS2020 labels, when used, are external validation labels and not perfect truth.\n",
    ]
    if summary.empty:
        lines.append("\nNo pS score columns were found; only available baseline diagnostics were attempted.\n")
    else:
        lines.append("\n## Metric Summary\n\n")
        lines.append(dataframe_to_markdown(summary))
        lines.append("\n")
    (output_dir / "README.md").write_text("".join(lines))
    return summary
