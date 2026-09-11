#!/usr/bin/env python3
"""Shared machinery for building cutout-inspection samples from DP2 COSMOS.

A *sample* is a selection of DP2 objects, sorted, numbered 1..N, and written out
as a CSV (for the mosaic notebook), an aligned TXT (for reading and emailing),
and a short summary of how many objects survived each cut.

Individual samples live in their own thin scripts (``make_sample_*.py``) that
declare filters and call :func:`build_sample`; the logic lives here so a new
sample is a few lines rather than a copy of this file.

Defining a new sample
---------------------
Write a script that builds a list of ``Cut`` objects and calls ``build_sample``::

    from cutout_samples import Cut, build_sample, load_labeled_catalog

    df = load_labeled_catalog()
    cuts = [
        Cut("dm > 0.05", lambda d: d["r_psf_minus_cModel"] > 0.05),
        Cut("20 < r < 22", lambda d: d["r_cModelMag"].between(20, 22, inclusive="neither")),
    ]
    build_sample(df, cuts, name="paper", outdir=...)

The base selection (HST-unresolved, Rubin-resolved, quality flags) is applied by
:func:`load_labeled_catalog` and is common to every sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits

BANDS = ("u", "g", "r", "i", "z", "y")
NANOJANSKY_ZEROPOINT = 31.4

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT.parent / "star-galaxy-DP2-data" / "DP2_COSMOS_objects.fits"
DEFAULT_TRUTH = REPO_ROOT / "data" / "cosmos2020_farmer_truth_catalog_github.csv"
DEFAULT_OUTDIR = REPO_ROOT / "outputs" / "cutout_samples"

MAX_SEP_ARCSEC = 0.3
MAX_BLENDEDNESS = 0.1

# Carried through to every sample for inspection alongside the cutouts.
DIAGNOSTIC_COLUMNS = [
    "objectId",
    "refBand",
    "refExtendedness",
    "detect_fromBlend",
    "detect_isIsolated",
    "parentObjectId",
]

# Columns shown in the human-readable TXT, as (column, header, format).
TXT_COLUMNS: Sequence[tuple[str, str, str]] = (
    ("obj_num", "N", "{:>4d}"),
    ("objectId", "objectId", "{:>19d}"),
    ("RA", "RA", "{:>11.6f}"),
    ("DEC", "DEC", "{:>10.6f}"),
    ("r_cModelMag", "r_mag", "{:>7.3f}"),
    ("r_psf_minus_cModel", "dm", "{:>7.3f}"),
    ("u_cModelMag", "u_mag", "{:>7.3f}"),
    ("g_cModelMag", "g_mag", "{:>7.3f}"),
    ("i_cModelMag", "i_mag", "{:>7.3f}"),
    ("z_cModelMag", "z_mag", "{:>7.3f}"),
    ("y_cModelMag", "y_mag", "{:>7.3f}"),
    ("detect_fromBlend", "blend", "{:>5}"),
    ("detect_isIsolated", "isol", "{:>5}"),
    ("hst_sep_arcsec", "hst_sep", "{:>8.4f}"),
)


@dataclass(frozen=True)
class Cut:
    """A named boolean filter over the catalog, for per-stage reporting."""

    label: str
    func: Callable[[pd.DataFrame], "pd.Series[bool]"]


# --------------------------------------------------------------------------
# Catalog loading (base selection shared by all samples)
# --------------------------------------------------------------------------

def flux_to_mag(flux: np.ndarray) -> np.ndarray:
    out = np.full(flux.shape, np.nan, dtype="float64")
    good = np.isfinite(flux) & (flux > 0)
    out[good] = NANOJANSKY_ZEROPOINT - 2.5 * np.log10(flux[good])
    return out


def read_fits_to_dataframe(path: Path) -> pd.DataFrame:
    with fits.open(path) as hdul:
        table = hdul[1].data
        data = {}
        for col in table.columns.names:
            arr = np.asarray(table[col])
            if arr.dtype.byteorder == ">":
                # Byte-swap big-endian FITS columns into native order for pandas.
                arr = arr.astype(arr.dtype.newbyteorder("="))
            data[col] = arr
        return pd.DataFrame(data)


def match_to_truth(dp2: pd.DataFrame, truth_path: Path) -> pd.DataFrame:
    """Nearest-neighbour match to COSMOS2020 FARMER.

    Adds ``hst_label`` (0 = star/unresolved, 1 = galaxy, -1 = no match) and the
    match provenance. Each truth source is used at most once, keeping the
    closest DP2 counterpart.
    """
    truth = pd.read_csv(truth_path)

    dp2_coord = SkyCoord(
        np.asarray(dp2["coord_ra"], dtype="float64") * u.deg,
        np.asarray(dp2["coord_dec"], dtype="float64") * u.deg,
    )
    truth_coord = SkyCoord(
        np.asarray(truth["ra"], dtype="float64") * u.deg,
        np.asarray(truth["dec"], dtype="float64") * u.deg,
    )

    idx, sep2d, _ = dp2_coord.match_to_catalog_sky(truth_coord)
    within = sep2d.arcsec <= MAX_SEP_ARCSEC

    result = dp2.copy()
    for col, fill in [("hst_label", -1), ("hst_id", -1), ("hst_sep_arcsec", np.nan),
                      ("hst_acs_mu_class", -1), ("hst_flag_combined", -1)]:
        result[col] = fill

    matched = truth.iloc[idx[within]]
    dedup = pd.DataFrame({
        "_dp2_idx": np.where(within)[0],
        "_truth_id": matched["id"].to_numpy(),
        "_sep": sep2d.arcsec[within],
        "_label": pd.Series(matched["label"].to_numpy()).map({"star": 0, "galaxy": 1}).to_numpy(),
        "_mu_class": matched["acs_mu_class"].to_numpy(),
        "_flag": matched["flag_combined"].to_numpy(),
    })
    dedup = dedup.dropna(subset=["_label"])
    dedup = dedup.sort_values("_sep").drop_duplicates(subset=["_truth_id"], keep="first")

    rows = dedup["_dp2_idx"].to_numpy()
    for col, src, dtype in [("hst_label", "_label", int), ("hst_id", "_truth_id", None),
                            ("hst_sep_arcsec", "_sep", None), ("hst_acs_mu_class", "_mu_class", None),
                            ("hst_flag_combined", "_flag", None)]:
        values = dedup[src].to_numpy()
        if dtype is not None:
            values = values.astype(dtype)
        result.iloc[rows, result.columns.get_loc(col)] = values

    return result


def base_cuts(max_blendedness: float = MAX_BLENDEDNESS) -> list[Cut]:
    """The selection common to every sample: the HST/Rubin disagreement itself."""
    return [
        Cut("HST unresolved (ACS_MU_CLASS=2)", lambda d: d["hst_label"] == 0),
        Cut("Rubin resolved (refExtendedness=1)", lambda d: d["refExtendedness"] == 1),
        Cut("r cModel flag clear", lambda d: ~d["r_cModel_flag"].astype(bool)),
        Cut("r extendedness flag clear", lambda d: ~d["r_extendedness_flag"].astype(bool)),
        Cut(f"r blendedness < {max_blendedness}",
            lambda d: d["r_blendedness"].astype("float64").lt(max_blendedness).fillna(False)),
    ]


def build_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten to the per-object columns the samples and notebook use."""
    out = pd.DataFrame(index=df.index)
    out["RA"] = np.asarray(df["coord_ra"], dtype="float64")
    out["DEC"] = np.asarray(df["coord_dec"], dtype="float64")

    for col in DIAGNOSTIC_COLUMNS:
        out[col] = df[col].to_numpy()

    for band in BANDS:
        out[f"{band}_cModelMag"] = flux_to_mag(np.asarray(df[f"{band}_cModelFlux"], dtype="float64"))
        out[f"{band}_psfMag"] = flux_to_mag(np.asarray(df[f"{band}_psfFlux"], dtype="float64"))
        # psf - cModel ("dm") is the point-source discriminant; ~0 means point-like.
        out[f"{band}_psf_minus_cModel"] = out[f"{band}_psfMag"] - out[f"{band}_cModelMag"]
        out[f"{band}_extendedness"] = np.asarray(df[f"{band}_extendedness"], dtype="float64")
        out[f"{band}_blendedness"] = np.asarray(df[f"{band}_blendedness"], dtype="float64")

    for col in ["hst_id", "hst_label", "hst_sep_arcsec", "hst_acs_mu_class", "hst_flag_combined"]:
        out[col] = df[col].to_numpy()

    return out


def load_labeled_catalog(
    input_path: Path = DEFAULT_INPUT,
    truth_path: Path = DEFAULT_TRUTH,
    verbose: bool = True,
) -> pd.DataFrame:
    """Read DP2 COSMOS, match to HST truth, apply the base cuts, flatten columns.

    Returns the pool every sample draws from.
    """
    if verbose:
        print(f"Reading {input_path} ...")
    raw = read_fits_to_dataframe(input_path)
    if verbose:
        print(f"  {len(raw):,} rows")
        print(f"Matching to {truth_path.name} (max sep {MAX_SEP_ARCSEC}\") ...")

    labeled = match_to_truth(raw, truth_path)
    if verbose:
        n_star = int((labeled["hst_label"] == 0).sum())
        n_gal = int((labeled["hst_label"] == 1).sum())
        print(f"  {n_star:,} HST stars, {n_gal:,} HST galaxies")
        print("Base selection:")

    mask = pd.Series(True, index=labeled.index)
    if verbose:
        print(f"  {'all DP2 COSMOS objects':<40s} {len(labeled):>8,d}")
    for cut in base_cuts():
        mask &= cut.func(labeled).to_numpy()
        if verbose:
            print(f"  {cut.label:<40s} {int(mask.sum()):>8,d}")

    return build_columns(labeled[mask]).reset_index(drop=True)


# --------------------------------------------------------------------------
# Sample construction
# --------------------------------------------------------------------------

def write_txt(df: pd.DataFrame, path: Path, title: str, provenance: Sequence[str]) -> None:
    """Write a fixed-width, column-aligned text version of the sample."""
    present = [(c, h, f) for c, h, f in TXT_COLUMNS if c in df.columns]
    widths = [max(len(h), len(f.format(0 if "d" in f else 0.0).strip()) + 1) for _, h, f in present]

    lines = [f"# {title}", "#"]
    lines += [f"# {line}" for line in provenance]
    lines += ["#", f"# {len(df)} objects", "#"]

    header = "  ".join(h.rjust(w) for (_, h, _), w in zip(present, widths))
    lines.append("# " + header)
    lines.append("# " + "-" * len(header))

    for _, row in df.iterrows():
        cells = []
        for (col, _, fmt), w in zip(present, widths):
            value = row[col]
            try:
                if isinstance(value, (bool, np.bool_)):
                    cell = ("Y" if value else "N")
                elif pd.isna(value):
                    cell = "--"
                else:
                    cell = fmt.format(value).strip()
            except (TypeError, ValueError):
                cell = str(value)
            cells.append(cell.rjust(w))
        lines.append("  " + "  ".join(cells))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_sample(
    pool: pd.DataFrame,
    cuts: Iterable[Cut],
    name: str,
    description: str,
    outdir: Path = DEFAULT_OUTDIR,
    sort_by: str = "r_psf_minus_cModel",
    ascending: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """Apply ``cuts`` to ``pool``, sort, number 1..N, and write CSV + TXT.

    Numbering is per-sample and assigned *after* sorting, so ``obj_num`` always
    matches the position in that sample's mosaics.
    """
    cuts = list(cuts)
    outdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\nSample {name!r}: {description}")
        print(f"  {'base sample':<40s} {len(pool):>8,d}")

    mask = pd.Series(True, index=pool.index)
    provenance = [f"Sample: {name} - {description}", "Cuts applied:"]
    provenance.append(f"    {'base sample':<40s} {len(pool):>8,d}")
    for cut in cuts:
        mask &= cut.func(pool).to_numpy()
        if verbose:
            print(f"  {cut.label:<40s} {int(mask.sum()):>8,d}")
        provenance.append(f"    {cut.label:<40s} {int(mask.sum()):>8,d}")

    sample = pool[mask].copy()
    sample = sample.sort_values(sort_by, ascending=ascending).reset_index(drop=True)
    sample.insert(0, "obj_num", np.arange(1, len(sample) + 1))

    order = "ascending" if ascending else "descending"
    provenance.append(f"Sorted by {sort_by} ({order}); obj_num is 1..N in that order.")

    csv_path = outdir / f"sample_{name}.csv"
    txt_path = outdir / f"sample_{name}.txt"
    sample.to_csv(csv_path, index=False)
    write_txt(sample, txt_path, f"DP2 COSMOS cutout sample: {name}", provenance)

    if verbose:
        print(f"  -> {csv_path}")
        print(f"  -> {txt_path}")
        if len(sample):
            print(f"     {len(sample)} objects | "
                  f"r {sample['r_cModelMag'].min():.2f}-{sample['r_cModelMag'].max():.2f} | "
                  f"dm {sample['r_psf_minus_cModel'].min():.3f}-{sample['r_psf_minus_cModel'].max():.3f}")
            print(f"     {int(np.ceil(len(sample) / 25))} mosaic page(s) at 25/page")

    return sample
