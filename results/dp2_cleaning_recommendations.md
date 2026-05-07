# DP2 Cleaning Recommendations
This file documents the DP2 cleaning logic now used for ECDFS and COSMOS. The main point is task-specific cleaning: footprint plots, CMD visualization, pS readiness, and final science samples should not all use one overly strict global cut.
## Confirmed / Implemented in Current DP2 Workflow
| Cut or mask | Logic | Reason | Confidence | Implemented where |
|---|---|---|---|---|
| `base_clean` | finite `coord_ra`/`coord_dec` after standardization to `ra`/`dec` | minimum coordinate validity for footprint/matching | confirmed | `src/dp2_external_validation.py`, `scripts/prepare_dp2_field_catalog.py` |
| `cmd_gi_cmodel_clean` | positive finite `g_cModelFlux` and `i_cModelFlux`; clean `g_cModel_flag` and `i_cModel_flag`; finite derived `cmodel_mag_g`, `cmodel_mag_i`, and `g-i` color | clean g-i vs i CModel CMD without requiring every band to be good | confirmed | `src/dp2_external_validation.py` |
| `cmd_gr_ri_cmodel_clean` | positive finite g/r/i CModel fluxes, clean g/r/i CModel flags, finite `g-r` and `r-i` colors | color-color validation and CMD-adjacent checks | confirmed | `src/dp2_external_validation.py` |
| `pS_<band>_ready` | positive finite PSF and CModel fluxes in the band; clean PSF/CModel flux flags; finite `<band>_diff` and CModel magnitude | required inputs for pS-style PSF-CModel morphology model | confirmed | `src/dp2_external_validation.py` |
| `extendedness_<band>_ready` | finite `<band>_extendedness` and clean `<band>_extendedness_flag` | valid Rubin native extendedness comparison | confirmed | `src/dp2_external_validation.py` |
| `all_bands_psf_cmodel_ready` | all six `pS_<band>_ready` masks true | strict all-band diagnostic only; not default CMD/sample cut | confirmed | `src/dp2_external_validation.py` |

## Likely / Schema-Inspired Flags to Keep Available
| Column | Suggested use | Reason | Confidence |
|---|---|---|---|
| `u_psfFlux_flag` | veto true/bad values when using PSF flux | PSF flux failure flag | likely |
| `u_cModel_flag` | veto true/bad values when using CModel flux | CModel flux failure flag | likely |
| `u_extendedness_flag` | veto true/bad values when using extendedness | extendedness measurement failure flag | likely |
| `u_pixelFlags_inexact_psfCenter` | consider vetoing true values for stricter PSF-centered morphology work | PSF-center quality issue can affect PSF flux/morphology | likely |
| `g_psfFlux_flag` | veto true/bad values when using PSF flux | PSF flux failure flag | likely |
| `g_cModel_flag` | veto true/bad values when using CModel flux | CModel flux failure flag | likely |
| `g_extendedness_flag` | veto true/bad values when using extendedness | extendedness measurement failure flag | likely |
| `g_pixelFlags_inexact_psfCenter` | consider vetoing true values for stricter PSF-centered morphology work | PSF-center quality issue can affect PSF flux/morphology | likely |
| `r_psfFlux_flag` | veto true/bad values when using PSF flux | PSF flux failure flag | likely |
| `r_cModel_flag` | veto true/bad values when using CModel flux | CModel flux failure flag | likely |
| `r_extendedness_flag` | veto true/bad values when using extendedness | extendedness measurement failure flag | likely |
| `r_pixelFlags_inexact_psfCenter` | consider vetoing true values for stricter PSF-centered morphology work | PSF-center quality issue can affect PSF flux/morphology | likely |
| `i_psfFlux_flag` | veto true/bad values when using PSF flux | PSF flux failure flag | likely |
| `i_cModel_flag` | veto true/bad values when using CModel flux | CModel flux failure flag | likely |
| `i_extendedness_flag` | veto true/bad values when using extendedness | extendedness measurement failure flag | likely |
| `i_pixelFlags_inexact_psfCenter` | consider vetoing true values for stricter PSF-centered morphology work | PSF-center quality issue can affect PSF flux/morphology | likely |
| `z_psfFlux_flag` | veto true/bad values when using PSF flux | PSF flux failure flag | likely |
| `z_cModel_flag` | veto true/bad values when using CModel flux | CModel flux failure flag | likely |
| `z_extendedness_flag` | veto true/bad values when using extendedness | extendedness measurement failure flag | likely |
| `z_pixelFlags_inexact_psfCenter` | consider vetoing true values for stricter PSF-centered morphology work | PSF-center quality issue can affect PSF flux/morphology | likely |
| `y_psfFlux_flag` | veto true/bad values when using PSF flux | PSF flux failure flag | likely |
| `y_cModel_flag` | veto true/bad values when using CModel flux | CModel flux failure flag | likely |
| `y_extendedness_flag` | veto true/bad values when using extendedness | extendedness measurement failure flag | likely |
| `y_pixelFlags_inexact_psfCenter` | consider vetoing true values for stricter PSF-centered morphology work | PSF-center quality issue can affect PSF flux/morphology | likely |

## Tentative Cuts That Need Advisor Review
| Column or idea | Possible logic | Why tentative |
|---|---|---|
| `<band>_blendedness` | apply a maximum blendedness threshold | threshold choice is science-dependent and was not imposed globally |
| `<band>_blendedness_flag` | veto true/bad values for stricter morphology work | kept available; not yet adopted as universal cut |
| `detect_fromBlend` | separate or veto deblended objects | can bias samples if imposed globally; needs review |
| `detect_isIsolated` | select isolated objects for calibration checks | useful for diagnostic subsets, not necessarily final catalog cut |
| `parentObjectId` | require parent/child relation checks for blends | matching and deblending treatment should be decided with advisor |
| signal-to-noise cuts | e.g. flux/fluxErr threshold by task and band | threshold depends on figure/science goal |

## Current Mask Counts
### ECDFS clean-mask counts

| Mask | Count | Fraction |
|---|---:|---:|
| `rows` | 732,402 | 1.0000 |
| `base_clean` | 732,402 | 1.0000 |
| `cmd_gi_cmodel_clean` | 637,589 | 0.8705 |
| `cmd_gr_ri_cmodel_clean` | 625,536 | 0.8541 |
| `all_bands_psf_cmodel_ready` | 366,627 | 0.5006 |
| `cmd_gi_zoom_display` | 621,892 | 0.8491 |

### COSMOS clean-mask counts

| Mask | Count | Fraction |
|---|---:|---:|
| `rows` | 865,287 | 1.0000 |
| `base_clean` | 865,287 | 1.0000 |
| `cmd_gi_cmodel_clean` | 782,942 | 0.9048 |
| `cmd_gr_ri_cmodel_clean` | 779,545 | 0.9009 |
| `all_bands_psf_cmodel_ready` | 604,371 | 0.6985 |
| `cmd_gi_zoom_display` | 765,826 | 0.8851 |

## Practical Recommendation
- Footprint plots: use the full coordinate-valid sample (`base_clean`).
- First-look full CMD: use a broad sanity-check sample; do not require all six bands.
- Clean/zoom CMD: use `cmd_gi_cmodel_clean` plus display limits, not `all_bands_psf_cmodel_ready`.
- pS vs extendedness diagnostics: use common samples requiring valid external label, pS, extendedness, and CMD coordinates.
- Final paper/science cuts: still need advisor review before being treated as final.
