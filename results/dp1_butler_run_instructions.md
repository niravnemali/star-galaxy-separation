# DP1 ECDFS Butler extraction run instructions

## Current status

The repository's completed ECDFS preparation workflow began as FITS-based, but a first true Butler extraction test has now succeeded in the local `Python (LSST v29_2_1 local)` environment.

TAP/API preflight has succeeded:

- ECDFS polygon returned 83 tract/patch combinations.
- Candidate tracts: `4848, 4849, 5063, 5064`.
- `target_butler_column_names` were validated in TAP `dp1.Object`: 75/75 present.
- A 1000-row TAP ECDFS Object sample was extracted successfully and has an ECDFS-like footprint.

TAP success does not imply Butler success. It does, however, give a concrete starting `where` clause and confirms that the target Object columns exist in TAP.

Butler one-ref test result:

- `outputs/butler_ecdfs_test.parquet`
- 62,720 rows x 75 columns
- RA range: `52.383838..52.990648`
- Dec range: `-28.756059..-28.264464`
- Footprint roughly ECDFS-like: yes

Butler row-limited sanity output:

- `outputs/butler_ecdfs_test_1000.csv`
- 1,000 rows x 75 columns
- RA range: `52.741302..52.990616`
- Dec range: `-28.756059..-28.672650`

Full four-tract Butler extraction result:

- `outputs/dp1_ecdfs_object_butler_alltracts.parquet`
- 494,850 rows x 75 columns after ECDFS RA/Dec box filtering
- RA range: `52.350970..53.929610`
- Dec range: `-28.758696..-27.405178`
- Target columns present: 75/75
- Footprint roughly ECDFS-like: yes

Butler-based downstream analysis table:

- `outputs/dp1_ecdfs_analysis_table_butler.parquet`
- 494,850 rows x 150 columns
- Includes standardized coordinates, fluxes, flux errors, flags, flux-derived magnitudes, `g-i` color, PSF-CModel morphology differences, extendedness columns, and clean masks.

## Required Butler environment

- Rubin Science Platform or another Rubin/LSST Python environment with `lsst.daf.butler`.
- Locally, use the `Python (LSST v29_2_1 local)` environment:

```bash
source "$HOME/Rubin-Work/lsst_stack/loadLSST.zsh"
setup lsst_distrib
```

- Access to DP1 Butler repo `dp1`.
- Collection expression, currently defaulted to `LSSTComCam/DP1`.
- Butler dataset type `object`.
- Rubin authentication through `RUBIN_TOKEN`, `BUTLER_RUBIN_ACCESS_TOKEN`, `ACCESS_TOKEN`, or `~/.config/rubin/butler_token`.

The local Jupyter kernel launcher sets the DP1 repository index. The query script also defaults to:

```text
https://data.lsst.cloud/api/butler/configs/idf-repositories.yaml
```

## Recommended first Butler where clause

Use the tract candidates from TAP:

```text
skymap='lsst_cells_v1' AND tract IN (4848, 4849, 5063, 5064)
```

The Object catalog is tract-level, so patch may not be required for the initial Butler extraction. After extraction, filter rows to the ECDFS RA/Dec polygon or box:

```text
52.35 <= coord_ra <= 53.93
-28.76 <= coord_dec <= -27.40
```

If the table has been normalized by a downstream adapter, use:

```text
52.35 <= ra <= 53.93
-28.76 <= dec <= -27.40
```

## First Butler test command

Run inside the local LSST v29_2_1 environment or Rubin/RSP Science Pipelines environment:

```bash
python scripts/query_dp1_ecdfs_object_catalog.py \
  --where "skymap='lsst_cells_v1' AND tract IN (4848, 4849, 5063, 5064)" \
  --limit-refs 1 \
  --output outputs/butler_ecdfs_test.parquet
```

This reads `target_butler_column_names` from `results/dp1_butler_column_list.txt` by default. This command has succeeded locally and wrote `outputs/butler_ecdfs_test.parquet`.

## Optional small row-limited Butler test

The script also supports a post-read row limit and ECDFS box filter:

```bash
python scripts/query_dp1_ecdfs_object_catalog.py \
  --where "skymap='lsst_cells_v1' AND tract IN (4848, 4849, 5063, 5064)" \
  --limit-refs 1 \
  --filter-ecdfs-box \
  --row-limit 1000 \
  --output outputs/butler_ecdfs_test_1000.csv
```

This is still not the full four-tract ECDFS extraction; it is a fast sanity check for columns, RA/Dec range, and local file writing. This command has succeeded locally and wrote `outputs/butler_ecdfs_test_1000.csv`.

## Full Butler extraction command

After the one-ref test confirms that the Butler dataset and target columns work:

```bash
python scripts/query_dp1_ecdfs_object_catalog.py \
  --where "skymap='lsst_cells_v1' AND tract IN (4848, 4849, 5063, 5064)" \
  --filter-ecdfs-box \
  --output outputs/dp1_ecdfs_object_butler_alltracts.parquet
```

This command has succeeded locally.

## Butler validation and preparation command

After extraction, run:

```bash
python scripts/validate_prepare_butler_ecdfs_catalog.py
```

This writes:

- `results/butler_ecdfs_extraction_validation.md`
- `results/butler_ecdfs_ra_dec.png`
- `results/butler_ecdfs_cmd_firstlook.png`
- `outputs/dp1_ecdfs_analysis_table_butler.parquet`

## Existing TAP preflight outputs

- `results/tap_ecdfs_tract_patch_candidates.csv`
- `results/tap_ecdfs_tract_patch_summary.md`
- `results/tap_dp1_object_column_validation.csv`
- `results/tap_dp1_object_column_validation.md`
- `outputs/tap_ecdfs_object_test_sample.csv`
- `outputs/tap_ecdfs_object_test_sample.parquet`

Use these files to validate the first Butler output against the TAP preflight.

## Expected Butler outputs

- Test output: `outputs/butler_ecdfs_test.parquet`
- Optional row-limited output: `outputs/butler_ecdfs_test_1000.csv`
- Full output: `outputs/dp1_ecdfs_object_butler_alltracts.parquet`
- Butler-based prepared table: `outputs/dp1_ecdfs_analysis_table_butler.parquet`

The full Butler output has been validated in `results/butler_ecdfs_extraction_validation.md`. The Butler-based prepared table is ready for downstream CMD and pS/extendedness analysis without replacing the existing FITS-based workflow.
