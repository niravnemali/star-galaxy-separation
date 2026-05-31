"""COSMOS/COSMOS2020 sample selection helpers for paper figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

try:
    from magnitude_calc import extcoeff
except ImportError as exc:  # pragma: no cover - fail loudly rather than invent coefficients
    raise ImportError(f"Could not import existing repo extinction code from {_REPO_SRC}") from exc

from paper_plot_style import BANDS

COSMOS_RA_RANGE = (149.50413, 150.99170)
COSMOS_DEC_RANGE = (1.48760, 2.97521)
PAPER_RMAG_RANGE = (16.0, 26.0)
EXTINCTION_SCALE = 3.10 / 1.20


@dataclass(frozen=True)
class PaperPaths:
    repo_root: Path
    dp2_table: Path
    matched_table: Path
    external_truth: Path
    external_farmer_fits: Path | None = None

    @classmethod
    def from_repo_root(cls, repo_root: Path | str) -> "PaperPaths":
        root = Path(repo_root)
        return cls(
            repo_root=root,
            dp2_table=root / "outputs/dp2_cosmos_analysis_table.parquet",
            matched_table=root / "outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet",
            external_truth=root / "data/cosmos2020_farmer_truth_catalog_github.csv",
            external_farmer_fits=find_cosmos2020_farmer_fits(root),
        )


def find_cosmos2020_farmer_fits(repo_root: Path) -> Path | None:
    """Return a local full COSMOS2020 FARMER FITS path if one is available.

    The compact GitHub truth catalog only stores IDs, coordinates, labels, and
    flags. Fig 1.1b can include a COSMOS2020-only r-band curve only when the
    full FARMER photometry table is available locally. This function does not
    copy or move the large FITS file; it only records a readable local path.
    """

    candidates = [
        repo_root / "data/external/COSMOS2020_FARMER_R1_v2.2_p3.fits",
        repo_root.parent / "COSMOS2020.R1_v2.2-nochi2/COSMOS2020_FARMER_R1_v2.2_p3.fits",
        repo_root.parent / "star-galaxy-separation_old_backup/data/external/COSMOS2020_FARMER_R1_v2.2_p3.fits",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _existing_columns(path: Path) -> list[str]:
    try:
        import pyarrow.parquet as pq

        return pq.read_schema(path).names
    except Exception:
        return pd.read_parquet(path).columns.tolist()


def load_cosmos_dp2(paths: PaperPaths) -> pd.DataFrame:
    """Load only columns needed for COSMOS paper sample diagnostics."""

    needed = [
        "object_id",
        "ra",
        "dec",
        "ebv",
        "refExtendedness",
        "cmodel_flux_r",
        "cmodel_flux_err_r",
        "extendedness_r",
        "extendedness_flag_r",
    ]
    for band in BANDS:
        needed += [f"cmodel_mag_{band}", f"psf_mag_{band}", f"psf_minus_cmodel_{band}"]
    columns = [c for c in needed if c in _existing_columns(paths.dp2_table)]
    return pd.read_parquet(paths.dp2_table, columns=columns)


def load_cosmos_matched(paths: PaperPaths) -> pd.DataFrame:
    """Load the COSMOS2020-matched DP2 table and attach DP2 EBV."""

    needed = [
        "external_id",
        "external_ra",
        "external_dec",
        "truth_label",
        "truth_binary",
        "external_label_source",
        "acs_mu_class",
        "flag_combined",
        "match_sep_arcsec",
        "dp2_object_id",
        "object_id",
        "dp2_ra",
        "dp2_dec",
        "dp2_refExtendedness",
        "dp2_cmodel_flux_r",
        "dp2_cmodel_flux_err_r",
        "dp2_extendedness_r",
        "dp2_extendedness_flag_r",
    ]
    for band in BANDS:
        needed += [
            f"dp2_cmodel_mag_{band}",
            f"dp2_psf_mag_{band}",
            f"dp2_psf_minus_cmodel_{band}",
        ]
    cols = [c for c in needed if c in _existing_columns(paths.matched_table)]
    matched = pd.read_parquet(paths.matched_table, columns=cols)
    dp2_ebv = pd.read_parquet(paths.dp2_table, columns=["object_id", "ebv"])
    return matched.merge(dp2_ebv.rename(columns={"ebv": "dp2_ebv"}), on="object_id", how="left")


def load_cosmos_external_truth(paths: PaperPaths) -> pd.DataFrame:
    return pd.read_csv(paths.external_truth)


def load_cosmos_farmer_rmag(paths: PaperPaths) -> pd.DataFrame:
    """Load COSMOS2020/FARMER HSC r-band magnitude columns from full FITS.

    Returns an empty frame when the full local FARMER FITS is not available.
    The returned magnitude is explicitly named `external_hsc_r_mag` because it
    is a COSMOS2020 HSC/FARMER magnitude, not a Rubin DP2 CModel magnitude.
    """

    if paths.external_farmer_fits is None:
        return pd.DataFrame()
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Loading full COSMOS2020 FARMER FITS requires astropy.") from exc

    columns = [
        "ID",
        "ALPHA_J2000",
        "DELTA_J2000",
        "HSC_r_MAG",
        "HSC_r_VALID",
        "ACS_MU_CLASS",
        "FLAG_COMBINED",
    ]
    with fits.open(paths.external_farmer_fits, memmap=True) as hdul:
        data = hdul[1].data
        available = [col for col in columns if col in data.names]
        raw = {}
        for col in available:
            arr = np.asarray(data[col])
            if arr.dtype.byteorder not in ("=", "|"):
                arr = arr.astype(arr.dtype.newbyteorder("="))
            raw[col] = arr
    if not raw:
        return pd.DataFrame()
    out = pd.DataFrame(raw)
    out = out.rename(
        columns={
            "ID": "id",
            "ALPHA_J2000": "farmer_ra",
            "DELTA_J2000": "farmer_dec",
            "HSC_r_MAG": "external_hsc_r_mag",
            "HSC_r_VALID": "external_hsc_r_valid",
            "ACS_MU_CLASS": "farmer_acs_mu_class",
            "FLAG_COMBINED": "farmer_flag_combined",
        }
    )
    out["id"] = pd.to_numeric(out["id"], errors="coerce")
    out["external_hsc_r_mag"] = pd.to_numeric(out["external_hsc_r_mag"], errors="coerce")
    if "external_hsc_r_valid" in out:
        out["external_hsc_r_valid"] = out["external_hsc_r_valid"].astype(bool)
    return out


def coordinate_mask(df: pd.DataFrame, ra_col: str, dec_col: str) -> pd.Series:
    ra_min, ra_max = COSMOS_RA_RANGE
    dec_min, dec_max = COSMOS_DEC_RANGE
    return (
        pd.to_numeric(df[ra_col], errors="coerce").between(ra_min, ra_max)
        & pd.to_numeric(df[dec_col], errors="coerce").between(dec_min, dec_max)
    )


def paper_mag_mask(df: pd.DataFrame, mag_col: str) -> pd.Series:
    lo, hi = PAPER_RMAG_RANGE
    mag = pd.to_numeric(df[mag_col], errors="coerce")
    return mag.gt(lo) & mag.lt(hi)


def truth_masks(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    truth = pd.to_numeric(df["truth_binary"], errors="coerce")
    return truth.eq(1), truth.eq(0)


def add_dust_corrected_colors(
    df: pd.DataFrame,
    mag_prefix: str,
    ebv_col: str,
    output_prefix: str = "",
) -> pd.DataFrame:
    """Add dust-corrected CModel colors using the repo extinction coefficients.

    The input magnitude columns are treated as uncorrected magnitudes. The
    correction follows the existing project convention:
    Ar = ebv * 3.10 / 1.20 and A_band / A_r from magnitude_calc.extcoeff().
    """

    out = df.copy()
    coeff = extcoeff()
    ar = pd.to_numeric(out[ebv_col], errors="coerce") * EXTINCTION_SCALE
    pairs = [("u", "g", "ug"), ("g", "r", "gr"), ("r", "i", "ri"), ("i", "z", "iz"), ("z", "y", "zy")]
    for b1, b2, name in pairs:
        m1 = pd.to_numeric(out[f"{mag_prefix}{b1}"], errors="coerce")
        m2 = pd.to_numeric(out[f"{mag_prefix}{b2}"], errors="coerce")
        out[f"{output_prefix}{name}"] = (m1 - m2) - (coeff[b1] - coeff[b2]) * ar
    out[f"{output_prefix}gi"] = out[f"{output_prefix}gr"] + out[f"{output_prefix}ri"]
    out[f"{output_prefix}Ar"] = ar
    return out


def prepare_phase1_samples(repo_root: Path | str) -> dict[str, pd.DataFrame]:
    """Prepare COSMOS DP2, matched, DP2-only, and external-only samples."""

    paths = PaperPaths.from_repo_root(repo_root)
    dp2 = load_cosmos_dp2(paths)
    matched = load_cosmos_matched(paths)
    external = load_cosmos_external_truth(paths)
    farmer_rmag = load_cosmos_farmer_rmag(paths)
    if not farmer_rmag.empty:
        farmer_cols = [
            "id",
            "external_hsc_r_mag",
            "external_hsc_r_valid",
        ]
        farmer_cols = [col for col in farmer_cols if col in farmer_rmag.columns]
        external = external.merge(
            farmer_rmag[farmer_cols],
            on="id",
            how="left",
        )

    dp2_coord = dp2.loc[coordinate_mask(dp2, "ra", "dec")].copy()
    dp2_paper = dp2_coord.loc[paper_mag_mask(dp2_coord, "cmodel_mag_r")].copy()

    matched_coord = matched.loc[coordinate_mask(matched, "dp2_ra", "dp2_dec")].copy()
    matched_paper = matched_coord.loc[paper_mag_mask(matched_coord, "dp2_cmodel_mag_r")].copy()
    matched_paper = add_dust_corrected_colors(
        matched_paper,
        mag_prefix="dp2_cmodel_mag_",
        ebv_col="dp2_ebv",
        output_prefix="color_",
    )

    external_coord = external.loc[coordinate_mask(external, "ra", "dec")].copy()
    matched_external_ids = set(pd.to_numeric(matched_coord["external_id"], errors="coerce").dropna().astype("int64"))
    external_only = external_coord.loc[
        ~pd.to_numeric(external_coord["id"], errors="coerce").isin(matched_external_ids)
    ].copy()

    matched_object_ids = set(pd.to_numeric(matched_paper["object_id"], errors="coerce").dropna().astype("int64"))
    dp2_only = dp2_paper.loc[
        ~pd.to_numeric(dp2_paper["object_id"], errors="coerce").isin(matched_object_ids)
    ].copy()

    dp2_paper = add_dust_corrected_colors(
        dp2_paper,
        mag_prefix="cmodel_mag_",
        ebv_col="ebv",
        output_prefix="color_",
    )

    return {
        "dp2": dp2,
        "dp2_coord": dp2_coord,
        "dp2_paper": dp2_paper,
        "matched": matched,
        "matched_coord": matched_coord,
        "matched_paper": matched_paper,
        "external": external,
        "external_coord": external_coord,
        "external_only": external_only,
        "farmer_rmag": farmer_rmag,
        "dp2_only": dp2_only,
    }


def build_sample_counts(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    matched_paper = samples["matched_paper"]
    stars, galaxies = truth_masks(matched_paper)
    rows = [
        ("N_DP2_total_before_cut", len(samples["dp2"]), ""),
        ("N_DP2_after_coordinate_cut", len(samples["dp2_coord"]), ""),
        ("N_DP2_after_mag_cut", len(samples["dp2_paper"]), "16 < cmodel_mag_r < 26"),
        ("N_external_total", len(samples["external"]), ""),
        ("N_external_after_coordinate_cut", len(samples["external_coord"]), ""),
        ("N_matched_total", len(matched_paper), "coordinate and r magnitude cuts"),
        ("N_matched_star", int(stars.sum()), "truth_binary == 1"),
        ("N_matched_galaxy", int(galaxies.sum()), "truth_binary == 0"),
        ("N_DP2_only", len(samples["dp2_only"]), "DP2 paper sample not in matched table"),
        ("N_external_only", len(samples["external_only"]), "external objects in coordinate cut without DP2 match"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value", "notes"])


def verify_phase1_conventions(samples: dict[str, pd.DataFrame]) -> dict[str, str]:
    matched = samples["matched"]
    labels = matched.groupby(["truth_binary", "truth_label"]).size().reset_index(name="count")
    return {
        "truth_binary": labels.to_string(index=False),
        "extinction": (
            "CModel magnitude columns are treated as uncorrected; colors are corrected "
            "with magnitude_calc.extcoeff and Ar = ebv * 3.10 / 1.20. No pre-existing "
            "dust-corrected color columns were used. Existing cmodel_color columns were "
            "checked and match raw CModel magnitude differences, so the paper color columns "
            "are recomputed from magnitudes before applying the repo extinction correction."
        ),
        "cosmos2020_only_magnitude": (
            "data/cosmos2020_farmer_truth_catalog_github.csv contains id, ra, dec, label, "
            "truth_binary, acs_mu_class, flag_combined only. When a local full FARMER FITS "
            "is available, `HSC_r_MAG` is used for the COSMOS2020-only r-band count curve "
            "and is labelled separately from DP2 r CModelMag."
        ),
    }
