"""Audit the repo FITS files used as Rubin DP1 source catalogs.

This script is intentionally local-file first.  It follows the project
instruction that the Rubin-side catalogs for this validation phase are the
FITS files already present under ``data/``.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"

FIELD_FILES = {
    "ECDFS": DATA_DIR / "ECDFS.fits",
    "EDFS": DATA_DIR / "EDFS.fits",
    "Rubin_SV_95_25": DATA_DIR / "Rubin_SV_95_25.fits",
}

PRIMARY_FIELD = "ECDFS"
PRIMARY_DP1_CATALOG = FIELD_FILES[PRIMARY_FIELD]


def is_git_lfs_pointer(path: Path) -> tuple[bool, str | None]:
    """Return whether ``path`` is a Git LFS pointer and its declared size."""
    if not path.exists():
        return False, None
    head = path.read_bytes()[:512]
    if not head.startswith(b"version https://git-lfs.github.com/spec/v1"):
        return False, None
    text = head.decode("utf-8", errors="replace")
    match = re.search(r"size\s+(\d+)", text)
    return True, match.group(1) if match else None


def _safe_import_astropy_table():
    try:
        from astropy.table import Table
    except Exception as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "astropy is required to read real FITS tables. "
            "Run in the LSST/Rubin Python environment."
        ) from exc
    return Table


def read_fits_columns(path: Path) -> tuple[int, list[str], dict[str, str]]:
    """Read a FITS table and return row count, column names, and dtypes."""
    Table = _safe_import_astropy_table()
    table = Table.read(path, format="fits")
    dtypes = {name: str(table[name].dtype) for name in table.colnames}
    return len(table), list(table.colnames), dtypes


def _contains_any(name: str, tokens: Iterable[str]) -> bool:
    lower = name.lower()
    return any(tok.lower() in lower for tok in tokens)


def categorize_columns(columns: list[str]) -> dict[str, list[str]]:
    """Group columns by their likely role for matching and validation."""
    categories = {
        "ra_dec": [],
        "object_id": [],
        "rubin_photometry": [],
        "psf_quantities": [],
        "cmodel_quantities": [],
        "morphology_indicators": [],
        "probability_or_classification": [],
    }

    for col in columns:
        lower = col.lower()
        if lower in {"ra", "dec", "coord_ra", "coord_dec"} or lower.endswith("_ra") or lower.endswith("_dec"):
            categories["ra_dec"].append(col)
        if _contains_any(col, ["objectid", "sourceid", "id"]):
            categories["object_id"].append(col)
        if _contains_any(col, ["flux", "mag", "err"]) and any(lower.startswith(f"{b}_") or lower.startswith(b) for b in "ugrizy"):
            categories["rubin_photometry"].append(col)
        if _contains_any(col, ["psfflux", "psf_flux", "psf", "free_psfflux"]):
            categories["psf_quantities"].append(col)
        if _contains_any(col, ["cmodelflux", "cmodel_flux", "cmodel", "free_cmodelflux", "model"]):
            categories["cmodel_quantities"].append(col)
        if _contains_any(col, ["extendedness", "refextendedness", "shape", "blend", "dm3"]):
            categories["morphology_indicators"].append(col)
        if _contains_any(col, ["prob", "pstar", "p_star", "class", "label", "sg"]):
            categories["probability_or_classification"].append(col)

    return categories


def _format_list(values: list[str], max_items: int = 40) -> str:
    if not values:
        return "  (none found)\n"
    shown = values[:max_items]
    lines = "".join(f"  - {v}\n" for v in shown)
    if len(values) > max_items:
        lines += f"  ... {len(values) - max_items} more\n"
    return lines


def audit_catalogs(output_dir: Path = OUTPUT_DIR) -> dict[str, dict]:
    """Audit all repo FITS files and write project-facing summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    audits: dict[str, dict] = {}

    for field, path in FIELD_FILES.items():
        pointer, lfs_size = is_git_lfs_pointer(path)
        info = {
            "field": field,
            "path": path,
            "exists": path.exists(),
            "is_lfs_pointer": pointer,
            "lfs_declared_size": lfs_size,
            "n_rows": None,
            "columns": [],
            "dtypes": {},
            "categories": {},
            "error": None,
        }
        if path.exists() and not pointer:
            try:
                n_rows, columns, dtypes = read_fits_columns(path)
                info.update(
                    n_rows=n_rows,
                    columns=columns,
                    dtypes=dtypes,
                    categories=categorize_columns(columns),
                )
            except Exception as exc:
                info["error"] = str(exc)
        audits[field] = info

    write_matching_summary(audits, output_dir / "dp1_catalog_for_matching.txt")
    write_column_audit(audits, output_dir / "dp1_column_audit.txt")
    return audits


def write_matching_summary(audits: dict[str, dict], path: Path) -> None:
    """Write the short DP1 catalog selection summary requested by the project."""
    primary = audits[PRIMARY_FIELD]
    lines = [
        "DP1 catalog selected for HST positional matching\n",
        "=================================================\n\n",
        f"Primary Rubin DP1 matching catalog: data/{PRIMARY_DP1_CATALOG.name}\n",
        "Reason: notebooks/ECDFS-DP1.ipynb loads ECDFS, EDFS, and Rubin_SV_95_25 "
        "from data/*.fits; this phase targets the E-CDFS / GOODS-S region, so "
        "ECDFS.fits is the primary matching table.\n\n",
        "Auxiliary repo FITS files:\n",
        "  - data/EDFS.fits: auxiliary comparison field, not primary for GOODS-S/E-CDFS matching.\n",
        "  - data/Rubin_SV_95_25.fits: auxiliary comparison field, not primary for GOODS-S/E-CDFS matching.\n\n",
    ]

    if primary["is_lfs_pointer"]:
        lines += [
            "Local file status\n",
            "-----------------\n",
            "The local ECDFS.fits file is currently a Git LFS pointer, not the materialized FITS table.\n",
            f"The pointer declares a real file size of {primary['lfs_declared_size']} bytes.\n",
            "Column-level auditing will complete automatically once the real LFS object is present.\n\n",
            "Expected useful DP1-side column families from the existing notebook/code\n",
            "---------------------------------------------------------------------\n",
            "RA / Dec:\n",
            "  - expected coordinate columns include coord_ra and coord_dec, based on src/config.py.\n",
            "Object ID:\n",
            "  - expected ID-like columns will be detected from objectId/sourceId/id-like names.\n",
            "Rubin photometry:\n",
            "  - src/pipeline.py expects per-band flux columns ending in _free_cModelFlux, _free_psfFlux,\n",
            "    _free_cModelFluxErr, and _free_psfFluxErr, then renames them to model/PSF flux names.\n",
            "PSF-related quantities:\n",
            "  - columns containing psfFlux / PSF / psf-related names will be used.\n",
            "CModel-related quantities:\n",
            "  - columns containing cModelFlux / model / cModel-related names will be used.\n",
            "Morphology indicators:\n",
            "  - refExtendedness / extendedness / shape / blend columns will be used if present.\n",
            "Probability / classification columns:\n",
            "  - probability or classifier columns will be detected from prob / pstar / class-like names.\n",
        ]
    else:
        lines += [
            "Detected useful columns in primary catalog\n",
            "-----------------------------------------\n",
            f"Rows: {primary['n_rows']}\n",
        ]
        categories = primary.get("categories", {})
        for label, cols in categories.items():
            lines.append(f"\n{label}:\n")
            lines.append(_format_list(cols))

    path.write_text("".join(lines))


def write_column_audit(audits: dict[str, dict], path: Path) -> None:
    """Write a detailed audit of every repo FITS file."""
    lines = ["DP1 FITS column audit\n", "======================\n\n"]
    for field, info in audits.items():
        rel = info["path"].relative_to(REPO_ROOT)
        lines += [f"## {field}: {rel}\n"]
        lines.append(f"Exists: {info['exists']}\n")
        lines.append(f"Git LFS pointer: {info['is_lfs_pointer']}\n")
        if info["lfs_declared_size"]:
            lines.append(f"Declared LFS size: {info['lfs_declared_size']} bytes\n")
        if info["error"]:
            lines.append(f"Read error: {info['error']}\n")
        if info["columns"]:
            lines.append(f"Rows: {info['n_rows']}\n")
            lines.append(f"Column count: {len(info['columns'])}\n\n")
            lines.append("Columns:\n")
            for col in info["columns"]:
                lines.append(f"  - {col} [{info['dtypes'].get(col, 'unknown')}]\n")
        else:
            lines.append("Columns: not available from the local file in its current state.\n")
        lines.append("\n")
    path.write_text("".join(lines))


def main() -> None:
    audit_catalogs()
    print(f"Wrote {OUTPUT_DIR / 'dp1_catalog_for_matching.txt'}")
    print(f"Wrote {OUTPUT_DIR / 'dp1_column_audit.txt'}")


if __name__ == "__main__":
    main()
