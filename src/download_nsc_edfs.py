"""Download a NOIRLab Source Catalog DR2 regional subset for the EDFS field.

This script uses the official NOIRLab Astro Data Lab TAP service and derives
the query bounds from ``data/EDFS.fits`` at runtime.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUT_DIR = REPO_ROOT / "outputs"
EXTERNAL_DIR = DATA_DIR / "external"
DEFAULT_EDFS = DATA_DIR / "EDFS.fits"
NSC_TAP_URL = "https://datalab.noirlab.edu/tap"

NSC_COLUMNS = [
    "id",
    "ra",
    "dec",
    "gmag",
    "rmag",
    "imag",
    "zmag",
    "ymag",
    "gerr",
    "rerr",
    "ierr",
    "zerr",
    "yerr",
    "class_star",
    "fwhm",
    "flags",
]


def find_coord_columns(columns: list[str]) -> tuple[str, str]:
    ra_candidates = ["coord_ra", "ra", "RA", "ALPHA_J2000", "alpha_j2000"]
    dec_candidates = ["coord_dec", "dec", "DEC", "DELTA_J2000", "delta_j2000"]
    ra_col = next((c for c in ra_candidates if c in columns), None)
    dec_col = next((c for c in dec_candidates if c in columns), None)
    if ra_col is None or dec_col is None:
        raise ValueError(f"Could not identify RA/Dec columns. Available columns include: {columns[:80]}")
    return ra_col, dec_col


def edfs_bounds(edfs_path: Path = DEFAULT_EDFS, padding_deg: float = 0.05) -> dict[str, float | str | int]:
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    import astropy.units as u

    table = Table.read(edfs_path, format="fits")
    ra_col, dec_col = find_coord_columns(list(table.colnames))
    ra = np.asarray(table[ra_col], dtype=float)
    dec = np.asarray(table[dec_col], dtype=float)
    ok = np.isfinite(ra) & np.isfinite(dec)
    ra = ra[ok]
    dec = dec[ok]
    ra_min = float(np.min(ra))
    ra_max = float(np.max(ra))
    dec_min = float(np.min(dec))
    dec_max = float(np.max(dec))
    center_ra = 0.5 * (ra_min + ra_max)
    center_dec = 0.5 * (dec_min + dec_max)
    center = SkyCoord(center_ra * u.deg, center_dec * u.deg)
    coords = SkyCoord(ra * u.deg, dec * u.deg)
    radius = float(np.max(center.separation(coords).deg) + padding_deg)
    return {
        "input_file": str(edfs_path.relative_to(REPO_ROOT)),
        "n_rows": len(table),
        "n_finite_coords": len(ra),
        "ra_col": ra_col,
        "dec_col": dec_col,
        "ra_min": ra_min,
        "ra_max": ra_max,
        "dec_min": dec_min,
        "dec_max": dec_max,
        "rect_ra_min": ra_min - padding_deg,
        "rect_ra_max": ra_max + padding_deg,
        "rect_dec_min": dec_min - padding_deg,
        "rect_dec_max": dec_max + padding_deg,
        "center_ra": center_ra,
        "center_dec": center_dec,
        "radius": radius,
    }


def nsc_sql(bounds: dict[str, float | str | int], max_rows: int | None = None) -> str:
    top = f"TOP {max_rows} " if max_rows else ""
    columns = ",\n    ".join(NSC_COLUMNS)
    return f"""
SELECT {top}
    {columns}
FROM nsc_dr2.object
WHERE ra BETWEEN {bounds['rect_ra_min']:.8f} AND {bounds['rect_ra_max']:.8f}
  AND dec BETWEEN {bounds['rect_dec_min']:.8f} AND {bounds['rect_dec_max']:.8f}
""".strip()


def write_instruction_file(bounds: dict[str, float | str | int], query: str, output_path: Path, reason: str) -> None:
    text = f"""NOIRLab NSC DR2 EDFS regional download instructions
==================================================

Automatic download status:
  {reason}

Official service:
  NOIRLab Astro Data Lab TAP
  {NSC_TAP_URL}

Sky region derived from:
  {bounds['input_file']}

Rectangle used:
  RA  = {bounds['rect_ra_min']:.8f} to {bounds['rect_ra_max']:.8f} deg
  Dec = {bounds['rect_dec_min']:.8f} to {bounds['rect_dec_max']:.8f} deg

SQL/ADQL query:

{query}

Recommended manual steps:
  1. Open the NOIRLab Astro Data Lab query interface.
  2. Use schema/table nsc_dr2.object.
  3. Paste the query above.
  4. Download the result as FITS or CSV.
  5. Save it as data/external/nsc_edfs_region.fits or data/external/nsc_edfs_region.csv.
"""
    output_path.write_text(text)


def download_nsc_edfs(edfs_path: Path = DEFAULT_EDFS, output_dir: Path = EXTERNAL_DIR, max_rows: int | None = None) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bounds = edfs_bounds(edfs_path)
    query = nsc_sql(bounds, max_rows=max_rows)
    instruction_path = OUTPUT_DIR / "nsc_download_instructions.txt"
    fits_path = output_dir / "nsc_edfs_region.fits"
    csv_path = output_dir / "nsc_edfs_region.csv"

    try:
        import pyvo
        service = pyvo.dal.TAPService(NSC_TAP_URL)
        job = service.submit_job(query, language="ADQL")
        job.run()
        job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=1800)
        if job.phase != "COMPLETED":
            raise RuntimeError(f"TAP job ended with phase {job.phase}")
        table = job.fetch_result().to_table()
        table.write(fits_path, overwrite=True)
        table.write(csv_path, format="csv", overwrite=True)
        write_instruction_file(bounds, query, instruction_path, f"Success. Downloaded {len(table)} NSC DR2 rows automatically.")
        print(f"Downloaded {len(table)} NSC DR2 rows to {fits_path}")
        return fits_path
    except Exception as exc:
        write_instruction_file(bounds, query, instruction_path, f"Automatic download failed: {type(exc).__name__}: {exc}")
        print(f"NSC automatic download failed. Wrote fallback instructions to {instruction_path}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edfs", type=Path, default=DEFAULT_EDFS)
    parser.add_argument("--output-dir", type=Path, default=EXTERNAL_DIR)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional TOP limit for testing.")
    args = parser.parse_args()
    download_nsc_edfs(args.edfs, args.output_dir, max_rows=args.max_rows)


if __name__ == "__main__":
    main()
