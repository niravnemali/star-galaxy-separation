# DP1 ECDFS column inventory

## Repo inspection and reuse

- ECDFS/DP1 catalog access in the existing repo is local-file based, not Butler-based.
- The main catalog-loading entry point is `src/pipeline.py:load_csv_dp1`, which reads `ECDFS.fits` and renames `_free_cModelFlux/_free_psfFlux` families into the current notebook convention.
- A Butler extraction skeleton is now available at `scripts/query_dp1_ecdfs_object_catalog.py`; it uses the target schema-style list in `results/dp1_butler_column_list.txt` but still requires an ECDFS tract/patch `--where` clause in a Rubin environment.
- Existing feature construction is reused from `src/pipeline.py:magnitude_calculations` and `src/magnitude_calc.py`.
- Existing CMD plotting conventions are visible in `src/plotting.py:plot_compareCMDs` and `notebooks/ECDFS-DP1.ipynb`.
- Existing cleaning/quality cuts already used in the project are documented in `src/validate_dp1_against_hst.py`.
- Existing pS workflow dependencies come from `src/psf_cmodel_fit.py`, `src/build_ecdfs_probability_models.py`, and `notebooks/ECDFS-DP1.ipynb`.
- The current repo has `notebooks/ECDFS-DP1.ipynb`, but no `tutorial-notebooks/DP1` directory is present in this checkout. DP1 tutorial/schema examples used for schema alignment are therefore documented separately in `results/dp1_schema_mapping.md` and `results/dp1_butler_column_validation.md`.

## Actual ECDFS FITS summary

- Input file: `/Users/jinshuozhang/Library/CloudStorage/Dropbox/Rubin-Work/Rubin-star-galaxy-classification/star-galaxy-separation/data/ECDFS.fits`
- Rows: 494,851
- Columns: 114
- RA range: 52.350970 to 53.932624 deg
- Dec range: -28.758696 to -27.405178 deg

## Core identifier and coordinate columns found

- `objectId` -> `objectId`
- `coord_ra` -> `coord_ra`
- `coord_dec` -> `coord_dec`
- `ebv` -> `ebv`
- `detect_isIsolated` -> `detect_isIsolated`
- `deblend_failed` -> `deblend_failed`

## Current FITS/export extraction column list

These are verified local ECDFS FITS/export columns. They satisfy the project-requested core quantities in the current file, but they are not automatically valid direct Butler Object query names. The split current-FITS and target-Butler lists are in `results/dp1_butler_column_list.txt`.

- `objectId`
- `coord_ra`
- `coord_dec`
- `ebv`
- `detect_isIsolated`
- `deblend_failed`
- `u_free_psfFlux`
- `u_free_psfFluxErr`
- `u_free_psfFlux_flag`
- `u_free_cModelFlux`
- `u_free_cModelFluxErr`
- `u_free_cModelFlux_flag`
- `u_extendedness`
- `u_extendedness_flag`
- `u_sizeExtendedness`
- `u_sizeExtendedness_flag`
- `u_blendedness`
- `u_blendedness_flag`
- `g_free_psfFlux`
- `g_free_psfFluxErr`
- `g_free_psfFlux_flag`
- `g_free_cModelFlux`
- `g_free_cModelFluxErr`
- `g_free_cModelFlux_flag`
- `g_extendedness`
- `g_extendedness_flag`
- `g_sizeExtendedness`
- `g_sizeExtendedness_flag`
- `g_blendedness`
- `g_blendedness_flag`
- `r_free_psfFlux`
- `r_free_psfFluxErr`
- `r_free_psfFlux_flag`
- `r_free_cModelFlux`
- `r_free_cModelFluxErr`
- `r_free_cModelFlux_flag`
- `r_extendedness`
- `r_extendedness_flag`
- `r_sizeExtendedness`
- `r_sizeExtendedness_flag`
- `r_blendedness`
- `r_blendedness_flag`
- `i_free_psfFlux`
- `i_free_psfFluxErr`
- `i_free_psfFlux_flag`
- `i_free_cModelFlux`
- `i_free_cModelFluxErr`
- `i_free_cModelFlux_flag`
- `i_extendedness`
- `i_extendedness_flag`
- `i_sizeExtendedness`
- `i_sizeExtendedness_flag`
- `i_blendedness`
- `i_blendedness_flag`
- `z_free_psfFlux`
- `z_free_psfFluxErr`
- `z_free_psfFlux_flag`
- `z_free_cModelFlux`
- `z_free_cModelFluxErr`
- `z_free_cModelFlux_flag`
- `z_extendedness`
- `z_extendedness_flag`
- `z_sizeExtendedness`
- `z_sizeExtendedness_flag`
- `z_blendedness`
- `z_blendedness_flag`
- `y_free_psfFlux`
- `y_free_psfFluxErr`
- `y_free_psfFlux_flag`
- `y_free_cModelFlux`
- `y_free_cModelFluxErr`
- `y_free_cModelFlux_flag`
- `y_extendedness`
- `y_extendedness_flag`
- `y_sizeExtendedness`
- `y_sizeExtendedness_flag`
- `y_blendedness`
- `y_blendedness_flag`

## Target Butler/Object-schema column list

The direct Butler target list is now stored as `target_butler_column_names` in `results/dp1_butler_column_list.txt`. It uses schema-style names such as `[f]_psfFlux`, `[f]_cModelFlux`, `refExtendedness`, `refBand`, `detect_fromBlend`, `parentObjectId`, and `[f]_pixelFlags_inexact_psfCenter`, rather than the current FITS `*_free_*` export names.

## Flux, error, and morphology families found by band

| band | PSF flux | PSF flux err | PSF flag | CModel flux | CModel flux err | CModel flag | extendedness | extendedness flag | sizeExtendedness | sizeExt flag | blendedness | blendedness flag |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `u` | `u_free_psfFlux` | `u_free_psfFluxErr` | `u_free_psfFlux_flag` | `u_free_cModelFlux` | `u_free_cModelFluxErr` | `u_free_cModelFlux_flag` | `u_extendedness` | `u_extendedness_flag` | `u_sizeExtendedness` | `u_sizeExtendedness_flag` | `u_blendedness` | `u_blendedness_flag` |
| `g` | `g_free_psfFlux` | `g_free_psfFluxErr` | `g_free_psfFlux_flag` | `g_free_cModelFlux` | `g_free_cModelFluxErr` | `g_free_cModelFlux_flag` | `g_extendedness` | `g_extendedness_flag` | `g_sizeExtendedness` | `g_sizeExtendedness_flag` | `g_blendedness` | `g_blendedness_flag` |
| `r` | `r_free_psfFlux` | `r_free_psfFluxErr` | `r_free_psfFlux_flag` | `r_free_cModelFlux` | `r_free_cModelFluxErr` | `r_free_cModelFlux_flag` | `r_extendedness` | `r_extendedness_flag` | `r_sizeExtendedness` | `r_sizeExtendedness_flag` | `r_blendedness` | `r_blendedness_flag` |
| `i` | `i_free_psfFlux` | `i_free_psfFluxErr` | `i_free_psfFlux_flag` | `i_free_cModelFlux` | `i_free_cModelFluxErr` | `i_free_cModelFlux_flag` | `i_extendedness` | `i_extendedness_flag` | `i_sizeExtendedness` | `i_sizeExtendedness_flag` | `i_blendedness` | `i_blendedness_flag` |
| `z` | `z_free_psfFlux` | `z_free_psfFluxErr` | `z_free_psfFlux_flag` | `z_free_cModelFlux` | `z_free_cModelFluxErr` | `z_free_cModelFlux_flag` | `z_extendedness` | `z_extendedness_flag` | `z_sizeExtendedness` | `z_sizeExtendedness_flag` | `z_blendedness` | `z_blendedness_flag` |
| `y` | `y_free_psfFlux` | `y_free_psfFluxErr` | `y_free_psfFlux_flag` | `y_free_cModelFlux` | `y_free_cModelFluxErr` | `y_free_cModelFlux_flag` | `y_extendedness` | `y_extendedness_flag` | `y_sizeExtendedness` | `y_sizeExtendedness_flag` | `y_blendedness` | `y_blendedness_flag` |

## Catalog magnitude columns also present in the current FITS file

The current local FITS file already includes `*_free_cModelMag`, `*_free_cModelMagErr`, `*_psfMag`, and `*_psfMagErr`. However, the extraction/analysis workflow below intentionally derives magnitudes from fluxes so it stays consistent with the project request and with schema versions where magnitudes are not materialized.

- `u`: cModel mag `u_free_cModelMag`, cModel mag err `u_free_cModelMagErr`, PSF mag `u_psfMag`, PSF mag err `u_psfMagErr`
- `g`: cModel mag `g_free_cModelMag`, cModel mag err `g_free_cModelMagErr`, PSF mag `g_psfMag`, PSF mag err `g_psfMagErr`
- `r`: cModel mag `r_free_cModelMag`, cModel mag err `r_free_cModelMagErr`, PSF mag `r_psfMag`, PSF mag err `r_psfMagErr`
- `i`: cModel mag `i_free_cModelMag`, cModel mag err `i_free_cModelMagErr`, PSF mag `i_psfMag`, PSF mag err `i_psfMagErr`
- `z`: cModel mag `z_free_cModelMag`, cModel mag err `z_free_cModelMagErr`, PSF mag `z_psfMag`, PSF mag err `z_psfMagErr`
- `y`: cModel mag `y_free_cModelMag`, cModel mag err `y_free_cModelMagErr`, PSF mag `y_psfMag`, PSF mag err `y_psfMagErr`

## Standardized analysis-table columns written by this script

- `object_id`
- `ra`
- `dec`
- `ebv`
- `deblend_failed`
- `base_clean`
- `u_psf_flux`
- `u_psf_flux_err`
- `u_psf_flux_flag`
- `u_cmodel_flux`
- `u_cmodel_flux_err`
- `u_cmodel_flux_flag`
- `u_extendedness`
- `u_extendedness_flag`
- `u_sizeExtendedness`
- `u_sizeExtendedness_flag`
- `u_blendedness`
- `u_blendedness_flag`
- `u_psf_flux_valid`
- `u_psfFlux_mag`
- `u_psfFlux_mag_err`
- `u_psfRelErr`
- `u_cmodel_flux_valid`
- `u_modelFlux_mag`
- `u_modelFlux_mag_err`
- `u_diff`
- `u_extendedness_valid`
- `u_sizeExtendedness_valid`
- `u_blendedness_valid`
- `g_psf_flux`
- `g_psf_flux_err`
- `g_psf_flux_flag`
- `g_cmodel_flux`
- `g_cmodel_flux_err`
- `g_cmodel_flux_flag`
- `g_extendedness`
- `g_extendedness_flag`
- `g_sizeExtendedness`
- `g_sizeExtendedness_flag`
- `g_blendedness`
- `g_blendedness_flag`
- `g_psf_flux_valid`
- `g_psfFlux_mag`
- `g_psfFlux_mag_err`
- `g_psfRelErr`
- `g_cmodel_flux_valid`
- `g_modelFlux_mag`
- `g_modelFlux_mag_err`
- `g_diff`
- `g_extendedness_valid`
- `g_sizeExtendedness_valid`
- `g_blendedness_valid`
- `r_psf_flux`
- `r_psf_flux_err`
- `r_psf_flux_flag`
- `r_cmodel_flux`
- `r_cmodel_flux_err`
- `r_cmodel_flux_flag`
- `r_extendedness`
- `r_extendedness_flag`
- `r_sizeExtendedness`
- `r_sizeExtendedness_flag`
- `r_blendedness`
- `r_blendedness_flag`
- `r_psf_flux_valid`
- `r_psfFlux_mag`
- `r_psfFlux_mag_err`
- `r_psfRelErr`
- `r_cmodel_flux_valid`
- `r_modelFlux_mag`
- `r_modelFlux_mag_err`
- `r_diff`
- `r_extendedness_valid`
- `r_sizeExtendedness_valid`
- `r_blendedness_valid`
- `i_psf_flux`
- `i_psf_flux_err`
- `i_psf_flux_flag`
- `i_cmodel_flux`
- `i_cmodel_flux_err`
- `i_cmodel_flux_flag`
- `i_extendedness`
- `i_extendedness_flag`
- `i_sizeExtendedness`
- `i_sizeExtendedness_flag`
- `i_blendedness`
- `i_blendedness_flag`
- `i_psf_flux_valid`
- `i_psfFlux_mag`
- `i_psfFlux_mag_err`
- `i_psfRelErr`
- `i_cmodel_flux_valid`
- `i_modelFlux_mag`
- `i_modelFlux_mag_err`
- `i_diff`
- `i_extendedness_valid`
- `i_sizeExtendedness_valid`
- `i_blendedness_valid`
- `z_psf_flux`
- `z_psf_flux_err`
- `z_psf_flux_flag`
- `z_cmodel_flux`
- `z_cmodel_flux_err`
- `z_cmodel_flux_flag`
- `z_extendedness`
- `z_extendedness_flag`
- `z_sizeExtendedness`
- `z_sizeExtendedness_flag`
- `z_blendedness`
- `z_blendedness_flag`
- `z_psf_flux_valid`
- `z_psfFlux_mag`
- `z_psfFlux_mag_err`
- `z_psfRelErr`
- `z_cmodel_flux_valid`
- `z_modelFlux_mag`
- `z_modelFlux_mag_err`
- `z_diff`
- `z_extendedness_valid`
- `z_sizeExtendedness_valid`
- `z_blendedness_valid`
- `y_psf_flux`
- `y_psf_flux_err`
- `y_psf_flux_flag`
- `y_cmodel_flux`
- `y_cmodel_flux_err`
- `y_cmodel_flux_flag`
- `y_extendedness`
- `y_extendedness_flag`
- `y_sizeExtendedness`
- `y_sizeExtendedness_flag`
- `y_blendedness`
- `y_blendedness_flag`
- `y_psf_flux_valid`
- `y_psfFlux_mag`
- `y_psfFlux_mag_err`
- `y_psfRelErr`
- `y_cmodel_flux_valid`
- `y_modelFlux_mag`
- `y_modelFlux_mag_err`
- `y_diff`
- `y_extendedness_valid`
- `y_sizeExtendedness_valid`
- `y_blendedness_valid`
- `ug`
- `gr`
- `ri`
- `iz`
- `zy`
- `gi`
- `Ar`
- `gi_cmd_clean`
- `pS_r_ready`
- `all_gri_cmodel_ready`
- `all_bands_psf_cmodel_ready`

## Flux-to-magnitude derivation used here

- Fluxes are assumed to be in nJy, matching the existing repo convention in `src/magnitude_calc.py`.
- Magnitudes use the AB conversion:
  `mag = -2.5 * log10((flux_nJy * 1e-9) / 3631.0)`
- Non-positive flux values are converted to `NaN`, not forced into finite magnitudes.
- Magnitude uncertainties use:
  `mag_err = 2.5 / ln(10) * flux_err / flux`
- Existing repo-style extinction-corrected colors are derived via `pipeline.magnitude_calculations(df, df['ebv'] * 2.583333)`.

## Missing or notable caveats relative to the target ideal list

- The ECDFS FITS file contains the project-requested core families for all six bands: PSF flux/error, CModel flux/error, and per-band extendedness-style quantities.
- Existing completed ECDFS work reads local FITS tables under `data/`; `scripts/query_dp1_ecdfs_object_catalog.py` is a Butler extraction skeleton for rerunning extraction in the Rubin environment once the exact ECDFS data-ID/tract selection is supplied.
- DP1 tutorials use schema-style examples such as `r_psfFlux`, `r_cModelFlux`, `refExtendedness`, `*_pixelFlags_inexact_psfCenter`, `i_cModel_flag`, `i_kronFlux_flag`, and `sersic_no_data_flag`; those exact non-`free`/shape flag columns are not present in the current ECDFS FITS extraction.
- No single `refExtendedness` or `base_ClassificationExtendedness_value` column is present here; the current file uses per-band `*_extendedness` and `*_sizeExtendedness` instead.
- `deblend_failed` is present, but in the current ECDFS FITS file every row is `False`, so it is documented but not a discriminating cut for this specific table.
