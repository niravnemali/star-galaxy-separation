# TAP dp1.Object column validation

- Target columns checked: 75
- Columns present in TAP_SCHEMA: 75
- Columns missing from TAP_SCHEMA: 0
- Validation CSV: `/Users/jinshuozhang/Library/CloudStorage/Dropbox/Rubin-Work/Rubin-star-galaxy-classification/star-galaxy-separation/results/tap_dp1_object_column_validation.csv`

This validates TAP schema visibility for `dp1.Object`; it does not prove Butler dataset-column availability.

## Missing target columns

- None.

## Present target columns

`objectId`, `coord_ra`, `coord_dec`, `ebv`, `refBand`, `refExtendedness`, `detect_fromBlend`, `detect_isIsolated`, `parentObjectId`, `u_psfFlux`, `u_psfFluxErr`, `u_psfFlux_flag`, `u_cModelFlux`, `u_cModelFluxErr`, `u_cModel_flag`, `u_extendedness`, `u_extendedness_flag`, `u_blendedness`, `u_blendedness_flag`, `u_pixelFlags_inexact_psfCenter`, `g_psfFlux`, `g_psfFluxErr`, `g_psfFlux_flag`, `g_cModelFlux`, `g_cModelFluxErr`, `g_cModel_flag`, `g_extendedness`, `g_extendedness_flag`, `g_blendedness`, `g_blendedness_flag`, `g_pixelFlags_inexact_psfCenter`, `r_psfFlux`, `r_psfFluxErr`, `r_psfFlux_flag`, `r_cModelFlux`, `r_cModelFluxErr`, `r_cModel_flag`, `r_extendedness`, `r_extendedness_flag`, `r_blendedness`, `r_blendedness_flag`, `r_pixelFlags_inexact_psfCenter`, `i_psfFlux`, `i_psfFluxErr`, `i_psfFlux_flag`, `i_cModelFlux`, `i_cModelFluxErr`, `i_cModel_flag`, `i_extendedness`, `i_extendedness_flag`, `i_blendedness`, `i_blendedness_flag`, `i_pixelFlags_inexact_psfCenter`, `z_psfFlux`, `z_psfFluxErr`, `z_psfFlux_flag`, `z_cModelFlux`, `z_cModelFluxErr`, `z_cModel_flag`, `z_extendedness`, `z_extendedness_flag`, `z_blendedness`, `z_blendedness_flag`, `z_pixelFlags_inexact_psfCenter`, `y_psfFlux`, `y_psfFluxErr`, `y_psfFlux_flag`, `y_cModelFlux`, `y_cModelFluxErr`, `y_cModel_flag`, `y_extendedness`, `y_extendedness_flag`, `y_blendedness`, `y_blendedness_flag`, `y_pixelFlags_inexact_psfCenter`

## TAP_SCHEMA ADQL used

```sql
SELECT
    column_name,
    datatype,
    unit,
    description
FROM TAP_SCHEMA.columns
WHERE table_name = 'dp1.Object'
ORDER BY column_name
```
