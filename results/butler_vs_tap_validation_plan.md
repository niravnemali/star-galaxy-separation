# Butler vs TAP validation plan

## Purpose

TAP preflight has validated the ECDFS sky region and `dp1.Object` target columns. A first true Butler one-ref extraction has now succeeded in the local LSST v29_2_1 environment. TAP success does not prove Butler success, so Butler outputs should still be checked against the TAP results.

## Inputs

- TAP tract/patch candidates: `results/tap_ecdfs_tract_patch_candidates.csv`
- TAP column validation: `results/tap_dp1_object_column_validation.csv`
- TAP sample: `outputs/tap_ecdfs_object_test_sample.csv` and `outputs/tap_ecdfs_object_test_sample.parquet`
- Butler test candidate: `outputs/butler_ecdfs_test.parquet`
- Butler row-limited sanity output: `outputs/butler_ecdfs_test_1000.csv`
- Butler full candidate: `outputs/dp1_ecdfs_object_butler.parquet`

## Completed Butler one-ref test

`outputs/butler_ecdfs_test.parquet` was generated from:

```bash
python scripts/query_dp1_ecdfs_object_catalog.py \
  --where "skymap='lsst_cells_v1' AND tract IN (4848, 4849, 5063, 5064)" \
  --limit-refs 1 \
  --output outputs/butler_ecdfs_test.parquet
```

Observed result:

- Dataset refs read: 1
- Columns requested/written: 75/75
- Rows written: 62,720
- RA range: `52.383838..52.990648`
- Dec range: `-28.756059..-28.264464`
- Footprint roughly ECDFS-like: yes

`outputs/butler_ecdfs_test_1000.csv` was also generated as a smaller inspection file:

- Rows written: 1,000
- Columns written: 75
- RA range: `52.741302..52.990616`
- Dec range: `-28.756059..-28.672650`

## First Butler where clause

```text
skymap='lsst_cells_v1' AND tract IN (4848, 4849, 5063, 5064)
```

Object catalogs are tract-level, so patch filtering may not be needed for the first Butler test. After reading tract-level Object tables, filter rows to the ECDFS RA/Dec box:

```text
52.35 <= coord_ra <= 53.93
-28.76 <= coord_dec <= -27.40
```

## Checks after Butler test extraction

1. Columns retrieved
   - Confirm all 75 `target_butler_column_names` are present.
   - Confirm no requested column was silently dropped.

2. Row count
   - Record raw tract-level row count before ECDFS RA/Dec filtering.
   - Record post-filter ECDFS row count.
   - Compare post-filter count to expectations from the current FITS sample and TAP query.

3. RA/Dec range
   - Confirm post-filter RA is within `52.35..53.93`.
   - Confirm post-filter Dec is within `-28.76..-27.40`.

4. Footprint shape
   - Make or inspect an RA/Dec density plot.
   - Confirm it is ECDFS-like and not an unrelated field.

5. TAP overlap sanity check
   - Compare Butler output columns to `outputs/tap_ecdfs_object_test_sample.csv`.
   - Check whether the Butler output covers the TAP small-sample RA/Dec subregion:
     - TAP sample RA: about `52.6002..52.7271`.
     - TAP sample Dec: about `-28.6662..-28.5883`.
   - If object IDs overlap, verify shared rows agree for core quantities such as `coord_ra`, `coord_dec`, and selected fluxes.

6. Schema semantics
   - Confirm Butler columns use fixed schema names like `g_cModelFlux`, not local FITS `g_free_cModelFlux`.
   - Document any semantic differences before comparing new Butler-derived pS/CMD products to older FITS-derived products.

## Pass/fail criteria

The Butler test is acceptable for the next preparation step if:

- The script runs inside the local LSST v29_2_1 environment or Rubin/RSP Science Pipelines environment.
- The requested target column list is accepted.
- The output has all 75 target columns.
- The ECDFS RA/Dec filter returns a non-empty table.
- The post-filter footprint is ECDFS-like.

The test is incomplete if any of these fail, even though TAP preflight succeeded.

The one-ref test satisfies these criteria. The full four-tract extraction and a full footprint/table comparison remain to be done before replacing the FITS-derived ECDFS analysis table.
