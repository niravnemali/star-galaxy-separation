#!/usr/bin/env python3
"""Butler extraction skeleton for the DP1 ECDFS Object catalog.

This script is intentionally an explicit adapter around the Butler because the
completed local workflow currently works from materialized FITS files under
``data/``. It uses the target schema-style list in
``results/dp1_butler_column_list.txt`` by default and lets the caller provide
the Butler ``where`` clause that identifies the ECDFS tracts/patches in the
active Rubin environment.

Example
-------
python scripts/query_dp1_ecdfs_object_catalog.py \
    --where "skymap='lsst_cells_v1' AND tract IN (...)" \
    --output outputs/dp1_ecdfs_object_butler.parquet

Installing the Python package that provides ``lsst.daf.butler`` is not by
itself enough for a real DP1 extraction. The caller must also run in an
environment where the DP1 Butler repository alias/path is configured, for
example through ``DAF_BUTLER_REPOSITORIES`` or ``DAF_BUTLER_REPOSITORY_INDEX``,
or inside the Rubin Science Platform Science Pipelines environment. Start with
``--limit-refs 1`` and verify the active DP1 schema accepts the target column
names before running the full ECDFS extraction. If your environment already has
explicit ECDFS tract/patch constraints, prefer passing them directly through
``--where``.
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COLUMNS = REPO_ROOT / "results" / "dp1_butler_column_list.txt"
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "dp1_ecdfs_object_butler.parquet"
DEFAULT_LIST_NAME = "target_butler_column_names"
DEFAULT_REPOSITORY_INDEX = "https://data.lsst.cloud/api/butler/configs/idf-repositories.yaml"
DEFAULT_TOKEN_FILE = Path.home() / ".config" / "rubin" / "butler_token"
ECDFS_RA_MIN = 52.35
ECDFS_RA_MAX = 53.93
ECDFS_DEC_MIN = -28.76
ECDFS_DEC_MAX = -27.40


def read_column_list(path: Path, list_name: str) -> list[str]:
    text = path.read_text()
    tree = ast.parse(text)
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
    raise ValueError(f"Could not find a Python list named `{list_name}` in {path}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="dp1", help="Butler repo name/path, default: dp1.")
    parser.add_argument(
        "--repo-index",
        default=None,
        help=(
            "Optional Butler repository index URI. If omitted and no Butler repo index env var is set, "
            f"the script defaults to {DEFAULT_REPOSITORY_INDEX}."
        ),
    )
    parser.add_argument("--collections", default="LSSTComCam/DP1", help="Butler collection expression.")
    parser.add_argument("--dataset-type", default="object", help="DP1 Object dataset type name.")
    parser.add_argument("--where", required=True, help="Butler query_datasets where clause selecting ECDFS data IDs.")
    parser.add_argument("--columns", type=Path, default=DEFAULT_COLUMNS, help="Python column list file.")
    parser.add_argument(
        "--list-name",
        default=DEFAULT_LIST_NAME,
        help=(
            "Name of the list to read from --columns. Use target_butler_column_names for direct Butler extraction; "
            "current_fits_column_names is for the local FITS/export table only."
        ),
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output parquet or csv path.")
    parser.add_argument("--limit-refs", type=int, default=None, help="Optional limit for testing.")
    parser.add_argument("--row-limit", type=int, default=None, help="Optional row limit after concatenating dataset refs.")
    parser.add_argument(
        "--filter-ecdfs-box",
        action="store_true",
        help=(
            "Filter output to the ECDFS RA/Dec box: "
            f"{ECDFS_RA_MIN} <= coord_ra <= {ECDFS_RA_MAX}, "
            f"{ECDFS_DEC_MIN} <= coord_dec <= {ECDFS_DEC_MAX}."
        ),
    )
    return parser.parse_args()


def configure_butler_environment(repo_index: str | None) -> None:
    """Set safe local defaults for Rubin remote Butler access.

    The token is never printed or written. If the caller has exported
    RUBIN_TOKEN, map it to the token variable names expected by the remote
    Butler client.
    """
    if repo_index:
        os.environ["DAF_BUTLER_REPOSITORY_INDEX"] = repo_index
    elif not os.environ.get("DAF_BUTLER_REPOSITORY_INDEX") and not os.environ.get("DAF_BUTLER_REPOSITORIES"):
        os.environ["DAF_BUTLER_REPOSITORY_INDEX"] = DEFAULT_REPOSITORY_INDEX

    rubin_token = os.environ.get("RUBIN_TOKEN")
    if (
        not rubin_token
        and not os.environ.get("BUTLER_RUBIN_ACCESS_TOKEN")
        and not os.environ.get("ACCESS_TOKEN")
        and DEFAULT_TOKEN_FILE.is_file()
    ):
        rubin_token = DEFAULT_TOKEN_FILE.read_text().strip()
    if rubin_token:
        os.environ.setdefault("BUTLER_RUBIN_ACCESS_TOKEN", rubin_token)
        os.environ.setdefault("ACCESS_TOKEN", rubin_token)


def query_dataset_refs(butler, dataset_type: str, where: str):
    if hasattr(butler, "query_datasets"):
        return list(butler.query_datasets(dataset_type, where=where))
    if hasattr(butler, "registry") and hasattr(butler.registry, "queryDatasets"):
        return list(butler.registry.queryDatasets(dataset_type, where=where))
    raise RuntimeError("Could not find a supported Butler dataset-query API on this Butler object.")


def explain_butler_repo_failure(repo: str, exc: Exception) -> None:
    repo_index = os.environ.get("DAF_BUTLER_REPOSITORY_INDEX")
    repo_aliases = os.environ.get("DAF_BUTLER_REPOSITORIES")
    print(
        "Butler Python is available, but the DP1 Butler repository is not configured for this shell.",
        file=sys.stderr,
    )
    print(f"Requested repo: {repo!r}", file=sys.stderr)
    print(
        "Set `DAF_BUTLER_REPOSITORY_INDEX` or `DAF_BUTLER_REPOSITORIES`, pass a real --repo path/URI, "
        "or run this command inside the Rubin Science Platform Science Pipelines environment.",
        file=sys.stderr,
    )
    print(
        f"Current DAF_BUTLER_REPOSITORY_INDEX: {'set' if repo_index else 'not set'}",
        file=sys.stderr,
    )
    print(
        f"Current DAF_BUTLER_REPOSITORIES: {'set' if repo_aliases else 'not set'}",
        file=sys.stderr,
    )
    print(f"Original Butler error: {type(exc).__name__}: {exc}", file=sys.stderr)


def first_existing(columns: list[str], candidates: list[str]) -> str | None:
    return next((col for col in candidates if col in columns), None)


def apply_ecdfs_box_filter(df: pd.DataFrame) -> pd.DataFrame:
    ra_col = first_existing(list(df.columns), ["coord_ra", "ra"])
    dec_col = first_existing(list(df.columns), ["coord_dec", "dec"])
    if ra_col is None or dec_col is None:
        raise ValueError("Cannot apply ECDFS box filter because no RA/Dec columns were found.")
    ra = pd.to_numeric(df[ra_col], errors="coerce")
    dec = pd.to_numeric(df[dec_col], errors="coerce")
    mask = ra.between(ECDFS_RA_MIN, ECDFS_RA_MAX) & dec.between(ECDFS_DEC_MIN, ECDFS_DEC_MAX)
    return df.loc[mask].copy()


def print_coordinate_summary(df: pd.DataFrame) -> None:
    ra_col = first_existing(list(df.columns), ["coord_ra", "ra"])
    dec_col = first_existing(list(df.columns), ["coord_dec", "dec"])
    if ra_col is None or dec_col is None or df.empty:
        print("RA/Dec range: unavailable")
        return
    ra = pd.to_numeric(df[ra_col], errors="coerce")
    dec = pd.to_numeric(df[dec_col], errors="coerce")
    print(f"RA column: {ra_col}")
    print(f"Dec column: {dec_col}")
    print(f"RA range: {float(ra.min()):.6f} .. {float(ra.max()):.6f} deg")
    print(f"Dec range: {float(dec.min()):.6f} .. {float(dec.max()):.6f} deg")
    is_ecdfs_like = (
        ra.between(ECDFS_RA_MIN - 0.1, ECDFS_RA_MAX + 0.1).all()
        and dec.between(ECDFS_DEC_MIN - 0.1, ECDFS_DEC_MAX + 0.1).all()
    )
    print(f"Footprint roughly ECDFS-like: {bool(is_ecdfs_like)}")


def main() -> None:
    args = parse_args()
    configure_butler_environment(args.repo_index)
    column_names = read_column_list(args.columns, args.list_name)
    if args.list_name == "current_fits_column_names" or any("_free_" in col for col in column_names):
        print(
            "WARNING: the selected list contains local FITS/export-style names. "
            "They may not be accepted as direct Butler Object columns."
        )

    try:
        from lsst.daf.butler import Butler
    except ImportError as exc:  # pragma: no cover - depends on Rubin env
        print(
            "This extractor must be run in a Rubin/LSST environment with lsst.daf.butler available. "
            "Local TAP/API validation can be done with scripts/rsp_tap_ecdfs_query.py, but this Butler test cannot run here.",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    try:
        butler = Butler(args.repo, collections=args.collections)
    except Exception as exc:  # pragma: no cover - depends on local/RSP Butler config
        explain_butler_repo_failure(args.repo, exc)
        raise SystemExit(2) from exc

    try:
        refs = query_dataset_refs(butler, args.dataset_type, args.where)
    except Exception as exc:  # pragma: no cover - depends on live Butler registry
        print(
            "Butler opened, but dataset-ref discovery failed. Check --collections, --dataset-type, "
            "and --where for the active DP1 repository.",
            file=sys.stderr,
        )
        print(f"Original Butler query error: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    if args.limit_refs is not None:
        refs = refs[: args.limit_refs]
    if not refs:
        raise RuntimeError(f"No `{args.dataset_type}` dataset refs matched --where: {args.where}")

    tables = []
    for ref in refs:
        table = butler.get(ref, parameters={"columns": column_names})
        if hasattr(table, "to_pandas"):
            tables.append(table.to_pandas())
        else:
            tables.append(pd.DataFrame(table))

    output = pd.concat(tables, ignore_index=True)
    if "objectId" in output.columns:
        output = output.drop_duplicates(subset=["objectId"], keep="first")
    if args.filter_ecdfs_box:
        n_before = len(output)
        output = apply_ecdfs_box_filter(output)
        print(f"Rows after ECDFS RA/Dec box filter: {len(output):,} / {n_before:,}")
    if args.row_limit is not None:
        output = output.head(args.row_limit).copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".csv":
        output.to_csv(args.output, index=False)
    else:
        output.to_parquet(args.output, index=False)

    print(f"Dataset refs read: {len(refs)}")
    print(f"Column list: {args.list_name} from {args.columns}")
    print(f"Columns requested: {len(column_names)}")
    print(f"Rows written: {len(output):,}")
    print(f"Columns written: {len(output.columns)}")
    print_coordinate_summary(output)
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
