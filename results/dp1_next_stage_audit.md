# DP1 next-stage schema/Butler readiness audit

| Goal item | Status | Evidence | Remaining limitation |
|---|---|---|---|
| Schema mapping exists | complete | `results/dp1_schema_mapping.md` | Exact live schema still needs Rubin-environment verification for a few flags. |
| Butler column list has been validated | complete | `results/dp1_butler_column_validation.md` | Validation is local/doc/tutorial based, not a live Butler query. |
| FITS-column names vs Butler/schema names are clearly distinguished | complete | `results/dp1_butler_column_list.txt` now has `current_fits_column_names` and `target_butler_column_names` | None for documentation; downstream code still needs source adapters. |
| Butler extraction skeleton is clear and usable | complete | `scripts/query_dp1_ecdfs_object_catalog.py` supports `--list-name`, target list default, `--limit-refs`, `--row-limit`, `--filter-ecdfs-box`, coordinate summaries, csv/parquet output, default Rubin repo index, and safe token fallback | Full four-tract Butler extraction has now succeeded locally. |
| Current limitations are explicitly documented | complete | `results/dp1_butler_run_instructions.md`, `results/dp1_butler_column_validation.md`, this audit, `results/butler_ecdfs_extraction_validation.md` | The remaining caveat is scientific review of clean-mask choices, not data availability. |
| No misleading claim remains that FITS columns are automatically Butler columns | complete | Old single-list file replaced by two named lists; inventory generator updated | Existing older downstream outputs may still mention FITS names because they are historical artifacts. |
| Current FITS-based prepared table remains usable | complete | `outputs/dp1_ecdfs_analysis_table.csv` has standardized analysis columns | Parquet could not be inspected locally with default Python if `pyarrow` is absent, but CSV structure was verified. |
| Transition risks are documented | complete | `results/dp1_fits_vs_butler_transition_notes.md`, `scripts/validate_prepare_butler_ecdfs_catalog.py` | Existing FITS workflow remains intact; downstream code should explicitly choose FITS-derived or Butler-derived table. |
| TAP/API validation path exists | complete | `scripts/rsp_tap_ecdfs_query.py`, `results/tap_api_run_instructions.md`, `results/tap_ecdfs_tract_patch_candidates.csv`, `results/tap_dp1_object_column_validation.md`, `outputs/tap_ecdfs_object_test_sample.parquet` | TAP success is catalog/API validation; Butler extraction has now also been tested separately for one dataset ref. |
| Candidate Butler tract selection exists | complete | TAP preflight identified tracts `4848, 4849, 5063, 5064`; instructions now use `skymap='lsst_cells_v1' AND tract IN (4848, 4849, 5063, 5064)` | Patch filtering is not used initially because Object catalogs are tract-level; RA/Dec filtering is applied after extraction. |
| Butler-vs-TAP validation plan exists | complete | `results/butler_vs_tap_validation_plan.md`, `results/butler_ecdfs_extraction_validation.md` | Full output is validated at the table/footprint/column level; object-level TAP overlap checks remain optional. |
| Full four-tract Butler extraction exists | complete | `outputs/dp1_ecdfs_object_butler_alltracts.parquet` has 494,850 rows x 75 columns after ECDFS RA/Dec filtering | None for the current data-preparation request. |
| Butler-based diagnostic plots exist | complete | `results/butler_ecdfs_ra_dec.png`, `results/butler_ecdfs_cmd_firstlook.png` | These are first-look diagnostics, not final science figures. |
| Butler clean/zoom CMD exists | complete | `results/butler_ecdfs_cmd_firstlook_clean_zoom.png`, `results/butler_ecdfs_cmd_firstlook_clean_zoom.pdf`, `results/butler_ecdfs_cmd_firstlook_clean_zoom_summary.md` | This is a display-oriented g/i CMD clean subset, not a final science sample. |
| Butler-based downstream analysis table exists | complete | `outputs/dp1_ecdfs_analysis_table_butler.parquet` has 494,850 rows x 150 columns | Clean masks should be reviewed before final science sample definition. |

## Overall verdict

The next-stage schema-alignment, Butler-readiness, and Butler-based ECDFS data-preparation task is complete. TAP/API preflight succeeded, a one-ref Butler test succeeded, and the full four-tract Butler extraction succeeded locally in the `Python (LSST v29_2_1 local)` environment. The final Butler raw table is `outputs/dp1_ecdfs_object_butler_alltracts.parquet` with 494,850 rows and 75 columns after ECDFS RA/Dec filtering. The Butler-based prepared table is `outputs/dp1_ecdfs_analysis_table_butler.parquet` with 494,850 rows and 150 columns.

## Task-specific cleaning

Cleaning is task-specific, not a single global cut. The footprint uses the full coordinate-valid ECDFS sample. The full CMD remains a broad sanity check. The clean/zoom CMD uses only the g/i CModel CMD mask `cmd_gi_cmodel_clean`, with display limits `-1 < g-i < 4` and `18 < i < 29`. Star/galaxy diagnostics keep their existing common-sample logic. Final science cuts still need project review.
