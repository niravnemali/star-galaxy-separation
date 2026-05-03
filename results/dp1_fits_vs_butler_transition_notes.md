# FITS-to-Butler transition notes

## Current normalized analysis table

`outputs/dp1_ecdfs_analysis_table.csv` already uses downstream-friendly names such as:

- `object_id`, `ra`, `dec`
- `[f]_psf_flux`, `[f]_psf_flux_err`, `[f]_psf_flux_flag`
- `[f]_cmodel_flux`, `[f]_cmodel_flux_err`, `[f]_cmodel_flux_flag`
- `[f]_extendedness`, `[f]_extendedness_flag`
- `[f]_blendedness`, `[f]_blendedness_flag`
- derived `[f]_psfFlux_mag`, `[f]_modelFlux_mag`, `[f]_diff`, colors, and readiness masks

This normalized layer is the right contract for downstream CMD, pS, and extendedness comparison code.

## Columns likely to change when switching source from FITS to Butler

| Downstream normalized column | Current FITS source | Target Butler source | Risk |
|---|---|---|---|
| `[f]_psf_flux` | `[f]_free_psfFlux` | `[f]_psfFlux` | low naming risk; possible free-vs-fixed semantic change |
| `[f]_psf_flux_err` | `[f]_free_psfFluxErr` | `[f]_psfFluxErr` | low naming risk |
| `[f]_psf_flux_flag` | `[f]_free_psfFlux_flag` | `[f]_psfFlux_flag` or another PSF flag | medium; live schema check needed |
| `[f]_cmodel_flux` | `[f]_free_cModelFlux` | `[f]_cModelFlux` | medium; fixed CModel is tutorial-preferred but not identical to `free` export |
| `[f]_cmodel_flux_err` | `[f]_free_cModelFluxErr` | `[f]_cModelFluxErr` | medium due source semantics |
| `[f]_cmodel_flux_flag` | `[f]_free_cModelFlux_flag` | `[f]_cModel_flag` | medium; live schema check needed |
| `[f]_extendedness` | `[f]_extendedness` | `[f]_extendedness` | low |
| `[f]_sizeExtendedness` | `[f]_sizeExtendedness` | no clear target | high; may need to remain FITS-only or be dropped from Butler-driven table |
| `deblend_failed` | `deblend_failed` | `deblend_failed` plus `detect_fromBlend`, `parentObjectId` | low for `deblend_failed`; add blend context |
| no current normalized column | no source | `refExtendedness`, `refBand` | add new columns for reference-band morphology |

## Adapter recommendation

Keep downstream code pointed at normalized analysis names. Add a source adapter with two input maps:

- FITS map: `current_fits_column_names` -> normalized names.
- Butler map: `target_butler_column_names` -> normalized names.

This avoids changing CMD/pS/extendedness scripts when the source switches from `data/ECDFS.fits` to `outputs/dp1_ecdfs_object_butler.parquet`.

## Validation needed after first Butler extraction

- Confirm row count and RA/Dec footprint match the current ECDFS sample expectation.
- Confirm all six bands have PSF and CModel flux/error columns.
- Compare flux distributions between FITS `*_free_*` and Butler fixed `*_psfFlux`/`*_cModelFlux` columns before treating old and new pS results as interchangeable.
- Rebuild derived magnitudes/colors from fluxes and regenerate the first-look CMD.
- Confirm that `refExtendedness` and per-band `[f]_extendedness` are both available and decide which one is the default comparator for pS.
