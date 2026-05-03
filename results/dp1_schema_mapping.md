# DP1 ECDFS FITS-to-Object-schema mapping

## Inspection summary

- Current completed workflow is local FITS based: `data/ECDFS.fits` is prepared by `scripts/prepare_dp1_ecdfs_catalog.py`.
- The local FITS table contains exported/transformed names such as `g_free_psfFlux` and `g_free_cModelFlux`.
- DP1 Object schema/tutorial examples use schema names such as `g_psfFlux`, `g_cModelFlux`, `refExtendedness`, `[f]_pixelFlags_*`, `detect_fromBlend`, and `parentObjectId`.
- Therefore, the current FITS names should be treated as verified local-export columns, not automatically as direct Butler query columns.

Official DP1 documentation checked:

- DP1 Object product page: `https://dp1.lsst.io/products/catalogs/object.html`
- DP1 Object table tutorial: `https://dp1.lsst.io/tutorials/notebook/201/notebook-201-1.html`
- DP1 ECDFS tutorial: `https://dp1.lsst.io/tutorials/notebook/301/notebook-301-4.html`

## Core columns

| Current ECDFS FITS column | Likely LSST Object schema equivalent | Quantity meaning | Band | Role in analysis | Confidence |
|---|---|---|---|---|---|
| `objectId` | `objectId` | Unique object identifier | none | joins, row identity | confirmed |
| `coord_ra` | `coord_ra` | Fiducial ICRS RA | none | footprint, matching, plots | confirmed |
| `coord_dec` | `coord_dec` | Fiducial ICRS Dec | none | footprint, matching, plots | confirmed |
| `ebv` | `ebv` | Milky Way dust E(B-V) at object coordinates | none | extinction/color correction | confirmed |
| `detect_isIsolated` | `detect_isIsolated` | detection/deblend isolation flag | none | possible cleaning/diagnostics | likely |
| `deblend_failed` | `deblend_failed` | deblender failure flag | none | base cleaning | confirmed |

## Per-band photometry and morphology mappings

Use `[f]` for each of `u`, `g`, `r`, `i`, `z`, and `y`.

| Current ECDFS FITS column | Likely LSST Object schema equivalent | Quantity meaning | Band | Role in analysis | Confidence |
|---|---|---|---|---|---|
| `[f]_free_psfFlux` | `[f]_psfFlux` | PSF flux in nJy | u/g/r/i/z/y | stellar photometry, PSF-CModel morphology, pS inputs | likely |
| `[f]_free_psfFluxErr` | `[f]_psfFluxErr` | PSF flux uncertainty in nJy | u/g/r/i/z/y | magnitude errors, S/N cuts | likely |
| `[f]_free_psfFlux_flag` | `[f]_psfFlux_flag` or related PSF measurement flag | PSF flux failure flag | u/g/r/i/z/y | cleaning | uncertain |
| `[f]_free_cModelFlux` | `[f]_cModelFlux` | CModel flux in nJy | u/g/r/i/z/y | galaxy photometry, CMD magnitudes, PSF-CModel morphology | likely |
| `[f]_free_cModelFluxErr` | `[f]_cModelFluxErr` | CModel flux uncertainty in nJy | u/g/r/i/z/y | magnitude errors, S/N cuts | likely |
| `[f]_free_cModelFlux_flag` | `[f]_cModel_flag` | CModel fit failure flag | u/g/r/i/z/y | cleaning | likely |
| `[f]_extendedness` | `[f]_extendedness` | PSF-vs-CModel flux-ratio morphology, 0 point-like and 1 extended | u/g/r/i/z/y | pS vs Rubin extendedness comparison | confirmed |
| `[f]_extendedness_flag` | `[f]_extendedness_flag` or related measurement flag | extendedness failure flag | u/g/r/i/z/y | cleaning before using extendedness | likely |
| `[f]_sizeExtendedness` | no clear direct Object tutorial equivalent found | size-based morphology quantity in local export | u/g/r/i/z/y | legacy morphology comparison only | uncertain |
| `[f]_sizeExtendedness_flag` | no clear direct Object tutorial equivalent found | sizeExtendedness failure flag | u/g/r/i/z/y | local cleaning if sizeExtendedness is used | uncertain |
| `[f]_blendedness` | `[f]_blendedness` | neighbor/blend contamination proxy | u/g/r/i/z/y | optional stricter photometry cleaning | confirmed |
| `[f]_blendedness_flag` | `[f]_blendedness_flag` or related blendedness flag | blendedness measurement flag | u/g/r/i/z/y | optional stricter cleaning | likely |
| `[f]_free_cModelMag`, `[f]_free_cModelMagErr` | `[f]_cModelMag`, `[f]_cModelMagErr` in TAP only; not expected from Butler Object columns | materialized AB magnitudes/errors | u/g/r/i/z/y | not used as source of truth here | likely |
| `[f]_psfMag`, `[f]_psfMagErr`, `[f]_free_psfMag`, `[f]_free_psfMagErr` | `[f]_psfMag`, `[f]_psfMagErr` in TAP only; not expected from Butler Object columns | materialized AB magnitudes/errors | u/g/r/i/z/y | not used as source of truth here | likely |

## Schema quantities absent from the current FITS export

| Schema/tutorial quantity | Present in current ECDFS FITS? | Notes |
|---|---:|---|
| `refExtendedness` | no | Official/tutorial Object quantity and recommended reference-band star/galaxy indicator; should be requested in Butler/TAP extraction. |
| `refBand` | no | Needed to interpret `refExtendedness`; should be requested in Butler/TAP extraction. |
| `parentObjectId` | no | Tutorial/schema deblend parent ID; useful for blend diagnostics. |
| `detect_fromBlend` | no | Tutorial/schema blend flag; should replace or supplement local `deblend_failed`. |
| `[f]_pixelFlags_inexact_psfCenter` | no | Tutorial-inspired PSF cleaning flag; should be requested if available. |
| `base_ClassificationExtendedness_value` | no | Older/Source-style name; not present in the current FITS export and not the target DP1 Object name for this workflow. |
| `sersic_no_data_flag`, `[f]_kronFlux_flag` | no | Mentioned in galaxy-photometry tutorial examples; not required for the current PSF/CModel-focused column list. |

## Important semantic caveat

`[f]_free_cModelFlux` is a schema-equivalent concept for CModel flux, but it is not guaranteed to be identical to `[f]_cModelFlux`. The DP1 tutorials describe `[f]_cModelFlux` as the fixed Object-table CModel flux and use it for galaxy work. The local FITS export appears to use free-parameter photometry (`*_free_*`). Downstream code should normalize either source into the analysis-table names (`[f]_cmodel_flux`, `[f]_psf_flux`, etc.) and record the source semantics.
