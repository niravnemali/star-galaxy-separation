# DP2 Porting Audit

This audit records the current state before and during the DP1-to-DP2 external-validation port.

## Existing DP1 Analyses

- ECDFS/GOODS-S HST/3D-HST label construction:
  - `src/build_hst_truth_catalog.py`
  - `data/external/hst_truth_catalog.csv`
- DP1 ECDFS positional matching:
  - `src/match_hst_to_dp1.py`
  - `outputs/hst_dp1_matched_clean_ellipse.csv`
  - `data/hst_dp1_matched_clean_ellipse.csv`
- DP1 EDFS Gaia/NSC external-label matching:
  - `src/build_edfs_external_truth_catalog.py`
  - `src/match_edfs_external_to_rubin.py`
  - `outputs/edfs_validation/edfs_external_dp1_matched_clean_ellipse.csv`
- DP1 morphology/extendedness validation:
  - `src/validate_dp1_against_hst.py`
  - `results/star_galaxy_diagnostics/`
- DP1 Butler data-preparation and diagnostics:
  - `scripts/query_dp1_ecdfs_object_catalog.py`
  - `scripts/validate_prepare_butler_ecdfs_catalog.py`
  - `scripts/star_galaxy_diagnostics.py`
  - `results/star_galaxy_diagnostics_butler/`

## DP2 Data Files Found

| File | Rows | Columns | Notes |
|---|---:|---:|---|
| `data/DP2_ECDFS_objects.fits` | 732,402 | 75 | DP2 ECDFS Rubin Object export. |
| `data/DP2_COSMOS_objects.fits` | 865,287 | 81 | DP2 COSMOS Rubin Object export; includes additional `g/r/i_cModelMag` columns. |

Both DP2 FITS exports include:

- `objectId`, `coord_ra`, `coord_dec`
- `ebv`, `refBand`, `refExtendedness`
- `detect_fromBlend`, `detect_isIsolated`, `parentObjectId`
- for all `u,g,r,i,z,y`: `psfFlux`, `psfFluxErr`, `psfFlux_flag`, `cModelFlux`, `cModelFluxErr`, `cModel_flag`, `extendedness`, `extendedness_flag`, `blendedness`, `blendedness_flag`, `pixelFlags_inexact_psfCenter`

## DP2 Notebooks Found

| Notebook | Version intent inferred from contents |
|---|---|
| `notebooks/ECDFS-DP2v1.ipynb` | Smooth pS model via `abstract_cmodel_dependence`. |
| `notebooks/ECDFS-DP2v2.ipynb` | Raw per-bin pS model via `build_binned_model`. |
| `notebooks/ECDFS-DP2v3.ipynb` | Triple CModel magnitude bins. |
| `notebooks/ECDFS-DP2v4.ipynb` | Additional DP2 pS variant. |
| `notebooks/ECDFS-DP2v5.ipynb` | Three resolved components in faint bins. |
| `notebooks/ECDFS-DP2v6.ipynb` | Skew-normal plus three-component faint-bin variant. |

The notebooks now save standardized per-object pS parquet files for every version after the standardized validation section was added and rerun.

## External Validation Catalogs Found

| Catalog | Purpose | Status |
|---|---|---|
| `data/external/hst_truth_catalog.csv` | ECDFS/GOODS-S HST/3D-HST external labels with `ra`, `dec`, `label`. | Usable for DP2 ECDFS RA/Dec matching. |
| `outputs/hst_dp1_matched_clean_ellipse.csv` | Old DP1 matched ECDFS table. | Historical DP1 reference; not the DP2 matching input. |
| `data/external/cosmos2020_farmer_truth_catalog.csv` | Compact COSMOS2020 FARMER star/galaxy external-validation catalog. | GitHub-ready reduced catalog generated from the local FARMER FITS. |

COSMOS2020 FARMER columns inspected:

- coordinates: `ALPHA_J2000`, `DELTA_J2000`
- IDs: `ID`, `FARMER_ID`, `ID_ACS`
- quality: `FLAG_COMBINED`
- primary morphology label candidate: `ACS_MU_CLASS`
- secondary label candidate: `lp_type`
- useful magnitude: `ACS_F814W_MAG`

Label convention to use conservatively:

- `ACS_MU_CLASS = 1`: galaxy
- `ACS_MU_CLASS = 2`: star
- `ACS_MU_CLASS = 3`: fake/spurious, excluded from star/galaxy truth metrics
- `lp_type` is a secondary cross-check only

COSMOS2020 labels are external validation labels, not perfect truth.

## Reusable Code Added/To Reuse

- Existing pS model code:
  - `src/psf_cmodel_fit.py`
  - `src/pipeline.py`
  - `src/external_label_tools.py`
- New DP2 shared validation utilities:
  - `src/dp2_external_validation.py`
- New field-prep and matching scripts:
  - `scripts/prepare_dp2_field_catalog.py`
  - `scripts/match_dp2_to_external_catalog.py`
  - `scripts/match_dp2_ecdfs_to_hst.py`
  - `scripts/match_dp2_cosmos_to_cosmos2020_farmer.py`
  - `scripts/run_dp2_external_validation.py`
  - `scripts/compare_dp2_ps_versions.py`
- DP2 column/flag documentation aligned with the earlier DP1 documentation:
  - `results/dp2_column_list.txt`
  - `results/dp2_column_inventory.md`
  - `results/dp2_cleaning_recommendations.md`

## Generalization Status

- Notebook pS outputs have been standardized to per-object files:
  - `outputs/dp2_ecdfs_ps_v1.parquet`
  - `outputs/dp2_ecdfs_ps_v2.parquet`
  - ...
  - `outputs/dp2_cosmos_ps_v1.parquet`
  - ...
- The same validation section has been appended to every DP2 notebook.
- Full pS-version comparison is now complete because the standardized pS output files exist.

## Current Porting Plan

1. Prepare DP2 ECDFS and COSMOS analysis tables using the same code path.
2. Match ECDFS DP2 to HST/3D-HST by RA/Dec.
3. Match COSMOS DP2 to COSMOS2020 FARMER by RA/Dec.
4. Use matched external labels for extendedness and pS diagnostics.
5. Add standardized pS-output cells to all DP2 notebooks.
6. Compare pS versions after the standardized pS output files are generated.

## Current Completion Status

- DP2 ECDFS analysis table was prepared: `outputs/dp2_ecdfs_analysis_table.parquet` with 732,402 rows.
- DP2 COSMOS analysis table was prepared: `outputs/dp2_cosmos_analysis_table.parquet` with 865,287 rows.
- ECDFS DP2-to-HST external matching was completed with a 0.3 arcsec default radius: 7,730 final one-to-one matches.
- COSMOS DP2-to-COSMOS2020 FARMER star/galaxy truth catalog matching was completed with a 0.3 arcsec default radius: 379,373 final one-to-one matches.
- COSMOS star/galaxy external validation subset was saved: `outputs/dp2_cosmos_external_truth_sample.parquet`.
- Baseline `extendedness_i` external-validation diagnostics were generated for both fields.
- Standardized pS-output and validation-command cells were added to all six DP2 notebooks.
- Full pS-version comparison has been generated under `results/dp2_ps_version_comparison/`.
