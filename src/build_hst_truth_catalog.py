"""Build a conservative HST-based star/galaxy truth catalog.

The first-pass product is intentionally simple and conservative:
ambiguous objects are excluded rather than forced into star/galaxy labels.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "outputs"

RA_CANDIDATES = ["ra", "RA", "ALPHA_J2000", "alpha_j2000", "coord_ra", "RAJ2000", "X_WORLD"]
DEC_CANDIDATES = ["dec", "DEC", "DELTA_J2000", "delta_j2000", "coord_dec", "DEJ2000", "Y_WORLD"]
ID_CANDIDATES = ["id", "ID", "object_id", "objectId", "source_id", "NUMBER", "catalog_id"]
LABEL_CANDIDATES = [
    "star_flag",
    "STAR_FLAG",
    "use_star",
    "is_star",
    "z_type",
    "z_best_s",
    "CLASS_STAR",
    "ACS_MU_CLASS",
    "type",
    "class",
]


def first_existing_col(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first candidate present in ``columns``, case-insensitive."""
    colset = {c: c for c in columns}
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate in colset:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def find_local_hst_catalog() -> Path | None:
    """Search common local locations for a GOODS-S / HST-like catalog."""
    patterns = [
        "*goodss*3dhst*IR.cat",
        "*goodss*3dhst*.IR.cat",
        "*3D*HST*GOOD*S*.fits",
        "*3dhst*good*s*.fits",
        "*GOODS*S*.fits",
        "*GOODSS*.fits",
        "*CAND*ELS*GOOD*S*.fits",
        "*hst*good*s*.fits",
        "*3D*HST*GOOD*S*.csv",
        "*GOODS*S*.csv",
    ]
    roots = [REPO_ROOT / "data", REPO_ROOT / "external", REPO_ROOT, REPO_ROOT.parent]
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]
    return None


def read_catalog(path: Path):
    """Read FITS/CSV/parquet into a pandas DataFrame."""
    suffix = path.suffix.lower()
    if suffix in {".fits", ".fit", ".fz"}:
        try:
            from astropy.table import Table
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("astropy is required to read FITS catalogs.") from exc
        return Table.read(path, format="fits").to_pandas()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".cat":
        try:
            from astropy.table import Table
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("astropy is required to read SExtractor .cat catalogs.") from exc
        return Table.read(path, format="ascii.sextractor").to_pandas()
    if suffix in {".dat", ".sfr", ".rf", ".fout"}:
        try:
            from astropy.table import Table
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("astropy is required to read ASCII 3D-HST catalogs.") from exc
        return Table.read(path, format="ascii.commented_header").to_pandas()
    raise ValueError(f"Unsupported HST catalog format: {path}")


def attach_3dhst_ztype_if_available(df: pd.DataFrame, input_path: Path) -> tuple[pd.DataFrame, str | None]:
    """Join 3D-HST IR coordinates to z_type labels when companion files exist.

    The GOODS-S 3D-HST release stores positions in ``*.IR.cat`` and the
    redshift/source-type flag in companion files such as ``*.zbest.sfr`` or
    ``*.zbest.fits``.  In that convention, ``z_type = 0`` denotes stars and
    ``z_type = 1,2,3`` denote galaxies with spectroscopic, grism, or
    photometric redshifts.
    """
    if "z_type" in df.columns or "z_best_s" in df.columns:
        return df, None

    id_col = first_existing_col(df.columns, ["NUMBER", "id", "phot_id"])
    if id_col is None:
        return df, None

    parent = input_path.parent
    companion_candidates = [
        parent / "goodss_3dhst.v4.1.5.zbest.sfr",
        parent / "goodss_3dhst.v4.1.5.zbest.fits",
        parent / "goodss_3dhst.v4.1.5.zbest.dat",
    ]
    for companion in companion_candidates:
        if not companion.exists():
            continue
        try:
            companion_df = read_catalog(companion)
        except Exception:
            continue

        comp_id = first_existing_col(companion_df.columns, ["id", "phot_id", "NUMBER"])
        label_col = first_existing_col(companion_df.columns, ["z_type", "z_best_s"])
        if comp_id is None or label_col is None:
            continue

        small = companion_df[[comp_id, label_col]].copy()
        small = small.rename(columns={comp_id: id_col})
        merged = df.merge(small, on=id_col, how="left", validate="one_to_one")
        note = (
            f"Joined {input_path.name} to companion {companion.name} using "
            f"{id_col} -> {comp_id}; using {label_col} as the conservative "
            "3D-HST source-type label."
        )
        return merged, note

    return df, None


def derive_labels(df: pd.DataFrame) -> tuple[pd.Series, str]:
    """Return conservative star/galaxy labels and a text rule description."""
    label_col = first_existing_col(df.columns, LABEL_CANDIDATES)
    if label_col is None:
        raise ValueError(
            "No label-like column found. Expected one of: "
            + ", ".join(LABEL_CANDIDATES)
        )

    values = df[label_col]
    labels = pd.Series(pd.NA, index=df.index, dtype="object")

    if label_col.lower() == "star_flag":
        numeric = pd.to_numeric(values, errors="coerce")
        labels[numeric == 1] = "star"
        labels[numeric == 0] = "galaxy"
        rule = (
            f"Used {label_col}: star_flag == 1 -> star; "
            "star_flag == 0 -> galaxy; all other/missing values excluded."
        )
    elif label_col.lower() in {"use_star", "is_star"}:
        numeric = pd.to_numeric(values, errors="coerce")
        labels[numeric == 1] = "star"
        labels[numeric == 0] = "galaxy"
        rule = (
            f"Used {label_col}: value == 1 -> star; value == 0 -> galaxy; "
            "all other/missing values excluded."
        )
    elif label_col.lower() in {"z_type", "z_best_s"}:
        numeric = pd.to_numeric(values, errors="coerce")
        labels[numeric == 0] = "star"
        labels[numeric.isin([1, 2, 3])] = "galaxy"
        rule = (
            f"Used 3D-HST {label_col}: 0 -> star; 1/2/3 -> galaxy "
            "(spectroscopic/grism/photometric redshift source types); "
            "all other/missing values excluded."
        )
    elif label_col.lower() == "class_star":
        numeric = pd.to_numeric(values, errors="coerce")
        labels[numeric >= 0.95] = "star"
        labels[numeric <= 0.05] = "galaxy"
        rule = (
            f"Used {label_col}: CLASS_STAR >= 0.95 -> star; "
            "CLASS_STAR <= 0.05 -> galaxy; intermediate values excluded."
        )
    else:
        text = values.astype(str).str.strip().str.lower()
        labels[text.isin(["star", "stellar", "1"])] = "star"
        labels[text.isin(["galaxy", "gal", "0"])] = "galaxy"
        rule = (
            f"Used {label_col} as a string/class column: star-like values -> star; "
            "galaxy-like values -> galaxy; all others excluded. "
            "If this is ACS_MU_CLASS, confirm its exact meaning from the source catalog documentation."
        )

    return labels, rule


def build_truth_catalog(input_path: Path | None, output_dir: Path = OUTPUT_DIR) -> pd.DataFrame | None:
    """Build and save the conservative HST truth catalog."""
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path is None:
        input_path = find_local_hst_catalog()

    if input_path is None or not input_path.exists():
        message = (
            "No local HST/3D-HST GOODS-S truth catalog was found.\n\n"
            "Expected input: a GOODS-S / E-CDFS HST-based catalog, preferably 3D-HST, "
            "with columns for RA, Dec, and a star/galaxy indicator such as star_flag.\n\n"
            "Required columns:\n"
            "  - RA column, e.g. ra, RA, ALPHA_J2000\n"
            "  - Dec column, e.g. dec, DEC, DELTA_J2000\n"
            "  - label column, preferably star_flag where 1=secure star and 0=secure galaxy\n"
        )
        (output_dir / "hst_truth_catalog_missing_input.txt").write_text(message)
        (output_dir / "hst_truth_catalog_summary.txt").write_text(message)
        print(message)
        return None

    df = read_catalog(input_path)
    df, companion_note = attach_3dhst_ztype_if_available(df, input_path)
    ra_col = first_existing_col(df.columns, RA_CANDIDATES)
    dec_col = first_existing_col(df.columns, DEC_CANDIDATES)
    id_col = first_existing_col(df.columns, ID_CANDIDATES)
    if ra_col is None or dec_col is None:
        raise ValueError(
            f"Could not identify RA/Dec columns in {input_path}. "
            f"Found columns: {list(df.columns)[:80]}"
        )

    labels, label_rule = derive_labels(df)
    finite_pos = np.isfinite(pd.to_numeric(df[ra_col], errors="coerce")) & np.isfinite(
        pd.to_numeric(df[dec_col], errors="coerce")
    )
    secure = finite_pos & labels.isin(["star", "galaxy"])

    out = pd.DataFrame(
        {
            "id": df[id_col].astype(str) if id_col else np.arange(len(df)).astype(str),
            "ra": pd.to_numeric(df[ra_col], errors="coerce"),
            "dec": pd.to_numeric(df[dec_col], errors="coerce"),
            "label": labels,
            "label_source": input_path.name,
            "quality_flag": "secure_hst_label",
            "notes": label_rule,
        }
    ).loc[secure].reset_index(drop=True)

    out_path = output_dir / "hst_truth_catalog.csv"
    out.to_csv(out_path, index=False)

    n_star = int((out["label"] == "star").sum())
    n_galaxy = int((out["label"] == "galaxy").sum())
    n_excluded = int(len(df) - len(out))
    summary = (
        "HST truth catalog summary\n"
        "=========================\n\n"
        f"External catalog used: {input_path}\n"
        f"Input objects: {len(df)}\n"
        f"Secure stars: {n_star}\n"
        f"Secure galaxies: {n_galaxy}\n"
        f"Excluded ambiguous/non-finite objects: {n_excluded}\n"
        f"RA column: {ra_col}\n"
        f"Dec column: {dec_col}\n"
        f"ID column: {id_col or 'generated row index'}\n"
        f"Companion label join: {companion_note or 'not used'}\n"
        f"Label rule: {label_rule}\n"
    )
    (output_dir / "hst_truth_catalog_summary.txt").write_text(summary)
    print(summary)
    print(f"Wrote {out_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None, help="Path to HST/3D-HST GOODS-S catalog.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    build_truth_catalog(args.input, args.output_dir)


if __name__ == "__main__":
    main()
