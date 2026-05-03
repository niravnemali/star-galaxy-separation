# DP1 Butler column-list validation

## Verdict

The original `results/dp1_butler_column_list.txt` was **not yet a true Butler column list**. It contained verified current ECDFS FITS/export names such as `g_free_psfFlux` and `g_free_cModelFlux`. Those are valid for `data/ECDFS.fits`, but they should not be assumed to be direct DP1 Object/Butler column names.

The file has now been corrected to contain two explicit lists:

- `current_fits_column_names`: verified local `data/ECDFS.fits` columns.
- `target_butler_column_names`: schema-style DP1 Object target columns for a future direct Butler extraction.

The current status is **partially ready**: the target list is grounded in DP1 Object schema/tutorial concepts, but it still needs a live Rubin-environment test against the active DP1 collection.

## Columns likely directly usable in Butler/Object extraction

These names are supported by DP1 Object documentation/tutorial examples or are direct Object concepts:

- `objectId`
- `coord_ra`
- `coord_dec`
- `ebv`
- `refBand`
- `refExtendedness`
- `detect_fromBlend`
- `detect_isIsolated`
- `parentObjectId`
- `[u,g,r,i,z,y]_psfFlux`
- `[u,g,r,i,z,y]_psfFluxErr`
- `[u,g,r,i,z,y]_cModelFlux`
- `[u,g,r,i,z,y]_cModelFluxErr`
- `[u,g,r,i,z,y]_extendedness`
- `[u,g,r,i,z,y]_blendedness`
- `[u,g,r,i,z,y]_pixelFlags_inexact_psfCenter`

## Columns that were likely not directly usable as Butler/Object names

These are verified local FITS/export columns, but they carry `*_free_*` or project-export naming:

- `[u,g,r,i,z,y]_free_psfFlux`
- `[u,g,r,i,z,y]_free_psfFluxErr`
- `[u,g,r,i,z,y]_free_psfFlux_flag`
- `[u,g,r,i,z,y]_free_cModelFlux`
- `[u,g,r,i,z,y]_free_cModelFluxErr`
- `[u,g,r,i,z,y]_free_cModelFlux_flag`
- `[u,g,r,i,z,y]_free_cModelMag`
- `[u,g,r,i,z,y]_free_cModelMagErr`
- `[u,g,r,i,z,y]_free_psfMag`
- `[u,g,r,i,z,y]_free_psfMagErr`
- `[u,g,r,i,z,y]_sizeExtendedness`
- `[u,g,r,i,z,y]_sizeExtendedness_flag`

## Required renamings/replacements

| Current FITS/export name | Target Butler/Object name | Status |
|---|---|---|
| `[f]_free_psfFlux` | `[f]_psfFlux` | likely replacement |
| `[f]_free_psfFluxErr` | `[f]_psfFluxErr` | likely replacement |
| `[f]_free_psfFlux_flag` | `[f]_psfFlux_flag` or another PSF measurement flag | uncertain; verify live schema |
| `[f]_free_cModelFlux` | `[f]_cModelFlux` | likely replacement, but free-vs-fixed semantics differ |
| `[f]_free_cModelFluxErr` | `[f]_cModelFluxErr` | likely replacement |
| `[f]_free_cModelFlux_flag` | `[f]_cModel_flag` | likely replacement |
| `deblend_failed` | `deblend_failed` plus `detect_fromBlend`, `parentObjectId` | keep plus add schema blend fields |
| no current column | `refExtendedness`, `refBand` | add for reference-band morphology |
| no current column | `[f]_pixelFlags_inexact_psfCenter` | add for tutorial-inspired PSF cleaning |

## Missing schema fields that should be added

- `refExtendedness`
- `refBand`
- `detect_fromBlend`
- `parentObjectId`
- `[f]_pixelFlags_inexact_psfCenter`

## Fields in the target list that still need live-schema verification

- `[f]_psfFlux_flag`: likely measurement flag naming, but not explicitly proven in the local repo.
- `[f]_cModel_flag`: tutorial-supported for at least `i_cModel_flag`; all-band availability should be verified.
- `[f]_extendedness_flag` and `[f]_blendedness_flag`: current FITS has these, but active Butler Object schema should be checked before full extraction.

## Final verdict

**Partially ready.** The list is now honest and separated into local FITS columns versus target Butler/Object schema columns. It is ready for a small Rubin-environment test, not for an unverified full production Butler extraction.
