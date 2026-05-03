# Butler ECDFS extraction validation

## Input

- Butler output: `/Users/jinshuozhang/Library/CloudStorage/Dropbox/Rubin-Work/Rubin-star-galaxy-classification/star-galaxy-separation/outputs/dp1_ecdfs_object_butler_alltracts.parquet`
- File exists: `True`
- Rows: 494,850
- Columns: 75
- Target columns present: 75/75
- Missing target columns: none
- Extra columns beyond target list: none

## Coordinate validation

- RA column: `coord_ra`
- Dec column: `coord_dec`
- RA range: 52.350970 to 53.929610 deg
- Dec range: -28.758696 to -27.405178 deg
- Within requested ECDFS box: `True`
- Footprint consistent with ECDFS: `True`

## Flux/error finite-positive checks

| band | PSF flux finite | PSF flux >0 | PSF err finite | CModel flux finite | CModel flux >0 | CModel err finite |
|---|---:|---:|---:|---:|---:|---:|
| `u` | 457,027 | 385,984 | 457,027 | 452,486 | 382,835 | 451,866 |
| `g` | 487,591 | 465,135 | 487,591 | 481,999 | 459,903 | 481,728 |
| `r` | 480,079 | 457,990 | 480,079 | 474,374 | 452,880 | 474,065 |
| `i` | 484,862 | 458,440 | 484,862 | 478,909 | 453,618 | 478,513 |
| `z` | 480,615 | 441,540 | 480,615 | 474,424 | 437,100 | 473,899 |
| `y` | 444,993 | 356,241 | 444,993 | 439,376 | 354,789 | 438,251 |

## Basic flag counts

| flag column | false/0 | true/1 | null |
|---|---:|---:|---:|
| `detect_fromBlend` | 89,958 | 404,892 | 0 |
| `detect_isIsolated` | 404,892 | 89,958 | 0 |
| `u_psfFlux_flag` | 407,769 | 87,081 | 0 |
| `u_cModel_flag` | 452,486 | 42,364 | 0 |
| `u_extendedness_flag` | 98,536 | 396,314 | 0 |
| `u_blendedness_flag` | 57,254 | 437,596 | 0 |
| `u_pixelFlags_inexact_psfCenter` | 363,264 | 131,586 | 0 |
| `g_psfFlux_flag` | 434,069 | 60,781 | 0 |
| `g_cModel_flag` | 481,999 | 12,851 | 0 |
| `g_extendedness_flag` | 235,200 | 259,650 | 0 |
| `g_blendedness_flag` | 221,016 | 273,834 | 0 |
| `g_pixelFlags_inexact_psfCenter` | 185,341 | 309,509 | 0 |
| `r_psfFlux_flag` | 428,027 | 66,823 | 0 |
| `r_cModel_flag` | 474,374 | 20,476 | 0 |
| `r_extendedness_flag` | 265,894 | 228,956 | 0 |
| `r_blendedness_flag` | 249,619 | 245,231 | 0 |
| `r_pixelFlags_inexact_psfCenter` | 184,337 | 310,513 | 0 |
| `i_psfFlux_flag` | 432,235 | 62,615 | 0 |
| `i_cModel_flag` | 478,909 | 15,941 | 0 |
| `i_extendedness_flag` | 297,372 | 197,478 | 0 |
| `i_blendedness_flag` | 268,199 | 226,651 | 0 |
| `i_pixelFlags_inexact_psfCenter` | 214,659 | 280,191 | 0 |
| `z_psfFlux_flag` | 428,531 | 66,319 | 0 |
| `z_cModel_flag` | 474,424 | 20,426 | 0 |
| `z_extendedness_flag` | 197,733 | 297,117 | 0 |
| `z_blendedness_flag` | 159,272 | 335,578 | 0 |
| `z_pixelFlags_inexact_psfCenter` | 226,790 | 268,060 | 0 |
| `y_psfFlux_flag` | 398,352 | 96,498 | 0 |
| `y_cModel_flag` | 439,376 | 55,474 | 0 |
| `y_extendedness_flag` | 83,664 | 411,186 | 0 |
| `y_blendedness_flag` | 49,385 | 445,465 | 0 |
| `y_pixelFlags_inexact_psfCenter` | 379,067 | 115,783 | 0 |

## Derived analysis table

- Output: `outputs/dp1_ecdfs_analysis_table_butler.parquet`
- Rows: 494,850
- Columns: 150
- Magnitudes are flux-derived from nJy using AB zeropoint 3631 Jy; non-positive fluxes become `NaN`.
- CMD color: `cmodel_color_g_i = cmodel_mag_g - cmodel_mag_i`.
- PSF-CModel morphology quantity: `psf_minus_cmodel_mag_<band> = psf_mag_<band> - cmodel_mag_<band>`.

## Clean-mask counts

| mask | true count |
|---|---:|
| `base_clean` | 494,850 |
| `isolated_unblended_clean` | 89,958 |
| `cmd_gi_cmodel_clean` | 435,339 |
| `pS_extendedness_i_ready` | 111,968 |
| `pS_extendedness_r_ready` | 81,591 |
| `all_bands_psf_cmodel_ready` | 15,952 |
