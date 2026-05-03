#!/usr/bin/env python3
"""Token-authenticated Rubin RSP TAP checks for the DP1 ECDFS Object catalog.

This script does not use the Butler and does not require ``lsst.daf.butler``.
It validates the TAP-accessible ``dp1.Object`` schema and can retrieve a small
ECDFS test sample through TAP/ADQL. The Rubin token is read only from an
environment variable and is never printed or written to disk.
"""

from __future__ import annotations

import argparse
import ast
import csv
import io
import os
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TAP_URL = "https://data.lsst.cloud/api/tap"
DEFAULT_TOKEN_ENV = "RUBIN_TOKEN"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_OUTPUTS_DIR = REPO_ROOT / "outputs"
DEFAULT_COLUMN_LIST = REPO_ROOT / "results" / "dp1_butler_column_list.txt"

ECDFS_RA_MIN = 52.35
ECDFS_RA_MAX = 53.93
ECDFS_DEC_MIN = -28.76
ECDFS_DEC_MAX = -27.40


def ecdfs_polygon_adql() -> str:
    return textwrap.dedent(
        f"""
        CONTAINS(
            POINT('ICRS', coord_ra, coord_dec),
            POLYGON(
                'ICRS',
                {ECDFS_RA_MIN:.2f}, {ECDFS_DEC_MIN:.2f},
                {ECDFS_RA_MAX:.2f}, {ECDFS_DEC_MIN:.2f},
                {ECDFS_RA_MAX:.2f}, {ECDFS_DEC_MAX:.2f},
                {ECDFS_RA_MIN:.2f}, {ECDFS_DEC_MAX:.2f}
            )
        ) = 1
        """
    ).strip()


TRACT_PATCH_QUERY = f"""
SELECT DISTINCT
    tract,
    patch
FROM dp1.Object
WHERE {ecdfs_polygon_adql()}
ORDER BY tract, patch
""".strip()


SCHEMA_QUERY = """
SELECT
    column_name,
    datatype,
    unit,
    description
FROM TAP_SCHEMA.columns
WHERE table_name = 'dp1.Object'
ORDER BY column_name
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tap-url", default=DEFAULT_TAP_URL, help=f"Rubin TAP URL. Default: {DEFAULT_TAP_URL}")
    parser.add_argument("--token-env", default=DEFAULT_TOKEN_ENV, help=f"Environment variable containing token. Default: {DEFAULT_TOKEN_ENV}")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum rows for --extract-small-sample.")
    parser.add_argument("--extract-small-sample", action="store_true", help="Extract a small ECDFS Object sample via TAP.")
    parser.add_argument("--skip-column-validation", action="store_true", help="Skip TAP_SCHEMA target-column validation.")
    parser.add_argument("--skip-tract-query", action="store_true", help="Skip ECDFS tract/patch discovery query.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory for TAP validation outputs.")
    parser.add_argument("--sample-output-dir", type=Path, default=DEFAULT_OUTPUTS_DIR, help="Directory for small sample outputs.")
    parser.add_argument("--column-list", type=Path, default=DEFAULT_COLUMN_LIST, help="Python file containing target_butler_column_names.")
    return parser.parse_args()


def require_token(token_env: str) -> str:
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(
            f"Missing Rubin API token. Set it in your shell without committing it, e.g.:\n"
            f'  export {token_env}="<your-token>"'
        )
    return token


def normalize_tap_url(tap_url: str) -> str:
    return tap_url.rstrip("/")


def run_tap_query_pyvo(tap_url: str, token: str, query: str) -> pd.DataFrame:
    try:
        import pyvo  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyvo is not installed") from exc

    try:
        import requests  # type: ignore
    except ImportError as exc:
        raise RuntimeError("pyvo authentication path requires requests; using stdlib fallback instead") from exc

    session = requests.Session()
    session.headers["Authorization"] = f"Bearer {token}"

    service = pyvo.dal.TAPService(tap_url, session=session)
    result = service.search(query)
    table = result.to_table()
    if hasattr(table, "to_pandas"):
        return table.to_pandas()
    return pd.DataFrame({name: table[name] for name in table.colnames})


def run_tap_query_stdlib(tap_url: str, token: str, query: str) -> pd.DataFrame:
    sync_url = normalize_tap_url(tap_url) + "/sync"
    payload = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "csv",
            "QUERY": query,
        }
    ).encode("utf-8")
    request = urllib.request.Request(sync_url, data=payload, method="POST")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Content-Type", "application/x-www-form-urlencoded")
    request.add_header("Accept", "text/csv")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TAP HTTP error {exc.code}: {exc.reason}\n{body[:1200]}") from exc
    return pd.read_csv(io.StringIO(text))


def run_tap_query(tap_url: str, token: str, query: str) -> tuple[pd.DataFrame, str]:
    try:
        return run_tap_query_pyvo(tap_url, token, query), "pyvo"
    except Exception as pyvo_exc:
        try:
            return run_tap_query_stdlib(tap_url, token, query), "stdlib"
        except Exception as stdlib_exc:
            raise RuntimeError(
                "TAP query failed with both pyvo and stdlib fallback. "
                f"pyvo path: {type(pyvo_exc).__name__}: {pyvo_exc}; "
                f"stdlib path: {type(stdlib_exc).__name__}: {stdlib_exc}"
            ) from stdlib_exc


def read_target_columns(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    assignments: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = node.value

    def resolve(name: str) -> list[str]:
        if name not in assignments:
            raise ValueError(f"Could not find `{name}` in {path}.")
        value = assignments[name]
        if isinstance(value, ast.Name):
            return resolve(value.id)
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, list) or not all(isinstance(col, str) for col in parsed):
            raise ValueError(f"`{name}` in {path} must resolve to a list of strings.")
        return parsed

    return resolve("target_butler_column_names")


def write_tract_patch_outputs(df: pd.DataFrame, output_dir: Path, query: str) -> None:
    csv_path = output_dir / "tap_ecdfs_tract_patch_candidates.csv"
    md_path = output_dir / "tap_ecdfs_tract_patch_summary.md"
    df.to_csv(csv_path, index=False)
    tracts = sorted(df["tract"].dropna().astype(str).unique().tolist()) if "tract" in df.columns else []
    patches = sorted(df["patch"].dropna().astype(str).unique().tolist()) if "patch" in df.columns else []
    lines = [
        "# TAP ECDFS tract/patch candidates\n\n",
        "This is TAP-based catalog selection, not a Butler extraction.\n\n",
        f"- Distinct tract/patch combinations: {len(df):,}\n",
        f"- Tracts: {', '.join(tracts) if tracts else 'none returned'}\n",
        f"- Patches: {', '.join(patches) if patches else 'none returned'}\n",
        f"- Output CSV: `{csv_path}`\n\n",
        "## ADQL used\n\n",
        "```sql\n",
        query,
        "\n```\n",
    ]
    md_path.write_text("".join(lines))


def suggest_alternatives(target: str, available: list[str]) -> str:
    target_lower = target.lower()
    exact_case = [col for col in available if col.lower() == target_lower and col != target]
    if exact_case:
        return "; ".join(exact_case)
    target_norm = target_lower.replace("_", "")
    norm_hits = [col for col in available if col.lower().replace("_", "") == target_norm and col != target]
    if norm_hits:
        return "; ".join(norm_hits[:10])
    if "_" in target:
        band, rest = target.split("_", 1)
        rest_base = rest.lower().replace("fluxerr", "flux").replace("_flag", "").replace("flag", "")
        band_hits = [
            col
            for col in available
            if col.lower().startswith(f"{band.lower()}_") and rest_base[:8] in col.lower().replace("_", "")
        ]
        if band_hits:
            return "; ".join(band_hits[:10])
    return ""


def validate_columns(schema_df: pd.DataFrame, target_columns: list[str], output_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    if "column_name" not in schema_df.columns:
        raise ValueError(f"TAP_SCHEMA query did not return `column_name`. Returned columns: {list(schema_df.columns)}")
    available = schema_df["column_name"].astype(str).tolist()
    available_set = set(available)
    records: list[dict[str, Any]] = []
    valid_columns: list[str] = []
    for col in target_columns:
        exists = col in available_set
        if exists:
            valid_columns.append(col)
        records.append(
            {
                "target_column": col,
                "exists_in_tap_schema": exists,
                "case_insensitive_match": any(a.lower() == col.lower() for a in available),
                "suggested_alternatives": "" if exists else suggest_alternatives(col, available),
            }
        )
    validation = pd.DataFrame(records)
    validation.to_csv(output_dir / "tap_dp1_object_column_validation.csv", index=False)

    missing = validation.loc[~validation["exists_in_tap_schema"], "target_column"].tolist()
    lines = [
        "# TAP dp1.Object column validation\n\n",
        f"- Target columns checked: {len(target_columns):,}\n",
        f"- Columns present in TAP_SCHEMA: {len(valid_columns):,}\n",
        f"- Columns missing from TAP_SCHEMA: {len(missing):,}\n",
        f"- Validation CSV: `{output_dir / 'tap_dp1_object_column_validation.csv'}`\n\n",
        "This validates TAP schema visibility for `dp1.Object`; it does not prove Butler dataset-column availability.\n\n",
        "## Missing target columns\n\n",
    ]
    if missing:
        for _, row in validation.loc[~validation["exists_in_tap_schema"]].iterrows():
            alt = row["suggested_alternatives"] or "none obvious"
            lines.append(f"- `{row['target_column']}`; possible alternatives: {alt}\n")
    else:
        lines.append("- None.\n")
    lines.extend(
        [
            "\n## Present target columns\n\n",
            ", ".join(f"`{col}`" for col in valid_columns) if valid_columns else "none",
            "\n\n## TAP_SCHEMA ADQL used\n\n```sql\n",
            SCHEMA_QUERY,
            "\n```\n",
        ]
    )
    (output_dir / "tap_dp1_object_column_validation.md").write_text("".join(lines))
    return validation, valid_columns


def quote_identifier(column: str) -> str:
    if column.replace("_", "").isalnum() and not column[0].isdigit():
        return column
    return f'"{column}"'


def build_sample_query(valid_columns: list[str], limit: int) -> str:
    required = ["objectId", "coord_ra", "coord_dec"]
    selected = []
    for col in required + valid_columns:
        if col in valid_columns and col not in selected:
            selected.append(col)
    if not selected:
        raise ValueError("No valid columns available for sample query.")
    columns_sql = ",\n    ".join(quote_identifier(col) for col in selected)
    return f"""
SELECT TOP {int(limit)}
    {columns_sql}
FROM dp1.Object
WHERE {ecdfs_polygon_adql()}
""".strip()


def write_sample_outputs(df: pd.DataFrame, sample_output_dir: Path) -> None:
    csv_path = sample_output_dir / "tap_ecdfs_object_test_sample.csv"
    parquet_path = sample_output_dir / "tap_ecdfs_object_test_sample.parquet"
    df.to_csv(csv_path, index=False)
    try:
        df.to_parquet(parquet_path, index=False)
        parquet_note = f"Wrote parquet: {parquet_path}"
    except Exception as exc:
        parquet_note = f"Parquet not written: {type(exc).__name__}: {exc}"
    print(f"Rows retrieved: {len(df):,}")
    if {"coord_ra", "coord_dec"}.issubset(df.columns) and len(df):
        ra = pd.to_numeric(df["coord_ra"], errors="coerce")
        dec = pd.to_numeric(df["coord_dec"], errors="coerce")
        print(f"RA range: {float(ra.min()):.6f} .. {float(ra.max()):.6f} deg")
        print(f"Dec range: {float(dec.min()):.6f} .. {float(dec.max()):.6f} deg")
        is_ecdfs_like = (
            ra.between(ECDFS_RA_MIN - 0.1, ECDFS_RA_MAX + 0.1).all()
            and dec.between(ECDFS_DEC_MIN - 0.1, ECDFS_DEC_MAX + 0.1).all()
        )
        print(f"Footprint roughly ECDFS-like: {bool(is_ecdfs_like)}")
    print(f"Retrieved columns: {', '.join(df.columns.astype(str))}")
    print(f"Wrote CSV: {csv_path}")
    print(parquet_note)


def write_run_instructions(path: Path) -> None:
    path.write_text(
        textwrap.dedent(
            f"""
            # RSP TAP API run instructions

            This workflow uses Rubin RSP TAP/API access with a token. It does not use the Butler and does not require local `lsst.daf.butler`.

            ## Set token

            ```bash
            export RUBIN_TOKEN="<your-token>"
            ```

            Security reminder: never commit the token, never print it, and never paste it into logs.

            ## Tract/patch discovery plus column validation

            ```bash
            python scripts/rsp_tap_ecdfs_query.py
            ```

            ## Column validation only

            ```bash
            python scripts/rsp_tap_ecdfs_query.py --skip-tract-query
            ```

            ## Small ECDFS Object sample extraction

            ```bash
            python scripts/rsp_tap_ecdfs_query.py --extract-small-sample --limit 1000
            ```

            ## Expected outputs

            - `results/tap_ecdfs_tract_patch_candidates.csv`
            - `results/tap_ecdfs_tract_patch_summary.md`
            - `results/tap_dp1_object_column_validation.csv`
            - `results/tap_dp1_object_column_validation.md`
            - `outputs/tap_ecdfs_object_test_sample.csv` when `--extract-small-sample` is used
            - `outputs/tap_ecdfs_object_test_sample.parquet` if a parquet engine is installed

            ## Default endpoint

            The default TAP URL is `{DEFAULT_TAP_URL}`. Override it with `--tap-url` if Rubin changes the endpoint or if you use another RSP deployment.

            TAP success does not imply Butler success. Butler extraction still requires a Rubin/RSP Science Pipelines environment with `lsst.daf.butler`.
            """
        ).strip()
        + "\n"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.sample_output_dir.mkdir(parents=True, exist_ok=True)
    write_run_instructions(args.output_dir / "tap_api_run_instructions.md")

    try:
        token = require_token(args.token_env)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)

    tap_url = normalize_tap_url(args.tap_url)
    print(f"Using TAP URL: {tap_url}")
    print(f"Token source: environment variable `{args.token_env}`")

    if not args.skip_tract_query:
        tract_df, backend = run_tap_query(tap_url, token, TRACT_PATCH_QUERY)
        write_tract_patch_outputs(tract_df, args.output_dir, TRACT_PATCH_QUERY)
        print(f"Tract/patch discovery backend: {backend}")
        print(f"Distinct tract/patch combinations: {len(tract_df):,}")

    valid_columns: list[str] = []
    if not args.skip_column_validation or args.extract_small_sample:
        target_columns = read_target_columns(args.column_list)
        schema_df, backend = run_tap_query(tap_url, token, SCHEMA_QUERY)
        validation, valid_columns = validate_columns(schema_df, target_columns, args.output_dir)
        print(f"Column validation backend: {backend}")
        print(f"Target columns present: {int(validation['exists_in_tap_schema'].sum()):,}/{len(validation):,}")
        missing = validation.loc[~validation["exists_in_tap_schema"], "target_column"].tolist()
        if missing:
            print(f"Missing target columns: {', '.join(missing)}")

    if args.extract_small_sample:
        sample_query = build_sample_query(valid_columns, args.limit)
        sample_df, backend = run_tap_query(tap_url, token, sample_query)
        print(f"Small sample extraction backend: {backend}")
        write_sample_outputs(sample_df, args.sample_output_dir)


if __name__ == "__main__":
    main()
