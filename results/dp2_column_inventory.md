# DP2 Column Inventory
This inventory documents the actual DP2 ECDFS and COSMOS object-catalog columns available locally and the standardized/derived columns created for downstream external-validation work. It is the DP2 counterpart to the older DP1 column inventory.
## Inputs Checked
| Field | Raw input | Raw rows | Raw columns | Prepared table | Prepared rows | Prepared columns | RA range deg | Dec range deg |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| ECDFS | data/DP2_ECDFS_objects.fits | 732,402 | 75 | outputs/dp2_ecdfs_analysis_table.parquet | 732,402 | 235 | 52.258065 .. 53.917050 | -28.264462 .. -26.776860 |
| COSMOS | data/DP2_COSMOS_objects.fits | 865,287 | 81 | outputs/dp2_cosmos_analysis_table.parquet | 865,287 | 241 | 149.504133 .. 150.991735 | 1.487603 .. 2.975206 |

## Core Identifier / Coordinate Columns
| Column | ECDFS raw | COSMOS raw | Standardized analysis name | Role |
|---|---|---|---|---|
| `objectId` | yes | yes | `object_id` | unique object identifier |
| `coord_ra` | yes | yes | `ra` | right ascension in degrees |
| `coord_dec` | yes | yes | `dec` | declination in degrees |
| `ebv` | yes | yes | `ebv` | foreground extinction value from Object table |
| `refBand` | yes | yes | `refBand` | reference band |
| `refExtendedness` | yes | yes | `refExtendedness` | reference-band Rubin extendedness |
| `detect_fromBlend` | yes | yes | `detect_fromBlend` | blend/deblend context flag |
| `detect_isIsolated` | yes | yes | `detect_isIsolated` | isolation flag |
| `parentObjectId` | yes | yes | `parentObjectId` | parent object identifier |

## Per-Band Flux / Error / Morphology Columns
| Band | PSF flux | PSF flux err | PSF flag | CModel flux | CModel flux err | CModel flag | extendedness | extendedness flag | blendedness | blendedness flag | inexact PSF center flag |
|---|---|---|---|---|---|---|---|---|---|---|---|
| u | `u_psfFlux` | `u_psfFluxErr` | `u_psfFlux_flag` | `u_cModelFlux` | `u_cModelFluxErr` | `u_cModel_flag` | `u_extendedness` | `u_extendedness_flag` | `u_blendedness` | `u_blendedness_flag` | `u_pixelFlags_inexact_psfCenter` |
| g | `g_psfFlux` | `g_psfFluxErr` | `g_psfFlux_flag` | `g_cModelFlux` | `g_cModelFluxErr` | `g_cModel_flag` | `g_extendedness` | `g_extendedness_flag` | `g_blendedness` | `g_blendedness_flag` | `g_pixelFlags_inexact_psfCenter` |
| r | `r_psfFlux` | `r_psfFluxErr` | `r_psfFlux_flag` | `r_cModelFlux` | `r_cModelFluxErr` | `r_cModel_flag` | `r_extendedness` | `r_extendedness_flag` | `r_blendedness` | `r_blendedness_flag` | `r_pixelFlags_inexact_psfCenter` |
| i | `i_psfFlux` | `i_psfFluxErr` | `i_psfFlux_flag` | `i_cModelFlux` | `i_cModelFluxErr` | `i_cModel_flag` | `i_extendedness` | `i_extendedness_flag` | `i_blendedness` | `i_blendedness_flag` | `i_pixelFlags_inexact_psfCenter` |
| z | `z_psfFlux` | `z_psfFluxErr` | `z_psfFlux_flag` | `z_cModelFlux` | `z_cModelFluxErr` | `z_cModel_flag` | `z_extendedness` | `z_extendedness_flag` | `z_blendedness` | `z_blendedness_flag` | `z_pixelFlags_inexact_psfCenter` |
| y | `y_psfFlux` | `y_psfFluxErr` | `y_psfFlux_flag` | `y_cModelFlux` | `y_cModelFluxErr` | `y_cModel_flag` | `y_extendedness` | `y_extendedness_flag` | `y_blendedness` | `y_blendedness_flag` | `y_pixelFlags_inexact_psfCenter` |

## COSMOS-Only Extra Raw Columns
- `g_cModelMag`: present in `data/DP2_COSMOS_objects.fits`; not present in ECDFS raw FITS. The DP2 workflow still derives CModel magnitudes from fluxes for both fields for consistency.
- `g_cModelMagErr`: present in `data/DP2_COSMOS_objects.fits`; not present in ECDFS raw FITS. The DP2 workflow still derives CModel magnitudes from fluxes for both fields for consistency.
- `r_cModelMag`: present in `data/DP2_COSMOS_objects.fits`; not present in ECDFS raw FITS. The DP2 workflow still derives CModel magnitudes from fluxes for both fields for consistency.
- `r_cModelMagErr`: present in `data/DP2_COSMOS_objects.fits`; not present in ECDFS raw FITS. The DP2 workflow still derives CModel magnitudes from fluxes for both fields for consistency.
- `i_cModelMag`: present in `data/DP2_COSMOS_objects.fits`; not present in ECDFS raw FITS. The DP2 workflow still derives CModel magnitudes from fluxes for both fields for consistency.
- `i_cModelMagErr`: present in `data/DP2_COSMOS_objects.fits`; not present in ECDFS raw FITS. The DP2 workflow still derives CModel magnitudes from fluxes for both fields for consistency.

## Standardized / Derived Analysis Columns
The preparation script keeps raw schema-style names and adds standardized names. Important derived families include:
- `object_id`, `ra`, `dec`, `field`
- `psf_flux_<band>`, `psf_flux_err_<band>`, `psf_flux_flag_<band>` for u,g,r,i,z,y
- `cmodel_flux_<band>`, `cmodel_flux_err_<band>`, `cmodel_flux_flag_<band>` for u,g,r,i,z,y
- `psf_mag_<band>`, `cmodel_mag_<band>`, and their magnitude errors, derived from fluxes without overwriting raw fluxes
- `psf_minus_cmodel_<band>` and `<band>_diff`, used by pS-style morphology models
- `extendedness_<band>`, `extendedness_flag_<band>`, `refExtendedness`
- `cmodel_color_g_i`, `cmd_g_minus_i`, `cmd_i_mag`, plus `g-r`, `r-i`, `u-g`, `i-z`, `z-y` colors
- `base_clean`, `cmd_gi_cmodel_clean`, `cmd_gr_ri_cmodel_clean`, `pS_<band>_ready`, `extendedness_<band>_ready`, `all_bands_psf_cmodel_ready`

## Missing or Not Directly Present
- No `sizeExtendedness` family was found in the local DP2 ECDFS/COSMOS object exports.
- No `base_ClassificationExtendedness_value` column was found in these DP2 exports; the available Object-schema extendedness quantities are `refExtendedness` and `<band>_extendedness`.
- Magnitudes are not required as raw inputs. COSMOS contains some g/r/i CModel magnitude columns, but the analysis derives AB magnitudes from fluxes for both fields for consistency.
- Final science-grade quality cuts are not fixed here; the workflow stores task-specific masks and keeps the raw flags for advisor review.
