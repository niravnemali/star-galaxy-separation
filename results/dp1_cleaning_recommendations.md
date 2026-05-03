# DP1 ECDFS cleaning recommendations

## Evidence searched locally

- Current project cleaning code: `src/validate_dp1_against_hst.py`
- Current notebook workflow: `notebooks/ECDFS-DP1.ipynb`
- Existing CMD/morphology plotting convention: `src/plotting.py:plot_compareCMDs`
- Existing morphology-star selection helper: `src/SG_separation.py`

The current repo has `notebooks/ECDFS-DP1.ipynb`, but no `tutorial-notebooks/DP1` directory is present in this checkout. DP1 tutorial/schema examples used here come from the documented DP1 Object/ECDFS tutorial material referenced in `results/dp1_schema_mapping.md`. They provide useful selection examples, but not a single universal Object-table cleaning recipe for this ECDFS preparation table. The recommendations below therefore separate project-confirmed cuts from tutorial-inspired cuts and tentative candidate cuts.

## Confirmed / reused from existing project code

- Column(s): `coord_ra, coord_dec, objectId`
  Keep/veto logic: keep finite coordinates and non-null object IDs
  Reason: required to identify the ECDFS footprint and preserve object-level joins
  Confidence: confirmed
  Local evidence: core assumption across `src/pipeline.py`, `src/match_hst_to_dp1.py`, and `notebooks/ECDFS-DP1.ipynb`

- Column(s): `deblend_failed`
  Keep/veto logic: keep rows where `deblend_failed == False`
  Reason: already used in `src/validate_dp1_against_hst.py` before morphology interpretation
  Confidence: confirmed
  Local evidence: project cleaning function `clean_morphology_sample(...)`

- Column(s): `*_free_psfFlux / *_free_cModelFlux`
  Keep/veto logic: for any band used, require finite and positive PSF/CModel fluxes
  Reason: needed for physically meaningful flux-to-mag conversion and PSF-CModel morphology diagnostics
  Confidence: confirmed
  Local evidence: existing repo feature construction and `clean_morphology_sample(...)`

- Column(s): `*_free_psfFlux_flag, *_free_cModelFlux_flag`
  Keep/veto logic: for any band used, keep rows where the corresponding flux flags are `False`
  Reason: already used in `src/validate_dp1_against_hst.py` for i-band morphology cleaning; should generalize band-by-band for CMD work
  Confidence: confirmed
  Local evidence: project cleaning function `clean_morphology_sample(...)`

- Column(s): `*_extendedness_flag, *_sizeExtendedness_flag`
  Keep/veto logic: when using `*_extendedness` or `*_sizeExtendedness`, keep rows where the corresponding flag is `False`
  Reason: the current notebook extendedness add-on explicitly excludes flagged values before plotting/comparison
  Confidence: confirmed
  Local evidence: `notebooks/ECDFS-DP1.ipynb` extendedness section

## Tutorial-inspired examples found locally

- Column(s): `refExtendedness`
  Keep/veto logic: tutorial star samples use `refExtendedness == 0`; tutorial galaxy samples use band extendedness or `refExtendedness` to select resolved objects
  Reason: DP1 object and stellar-color tutorials use this as the reference point-source selector
  Confidence: tutorial_inspired
  Local evidence: DP1 Object tutorial/schema examples; not present in the current ECDFS FITS extraction

- Column(s): `*_extendedness`
  Keep/veto logic: tutorial star/PSF examples use `*_extendedness == 0`; galaxy examples use `*_extendedness == 1`
  Reason: standard tutorial split between point-like and extended Object-table samples
  Confidence: tutorial_inspired
  Local evidence: DP1 PSF-star, galaxy-photometry, and ECDFS tutorial examples; present in the current ECDFS FITS extraction for all six bands

- Column(s): `*_pixelFlags_inexact_psfCenter`
  Keep/veto logic: for clean PSF-star examples, tutorial keeps rows where this flag is `0`
  Reason: rejects objects with problematic PSF-center placement in the PSF photometry demo
  Confidence: tutorial_inspired
  Local evidence: DP1 PSF-star tutorial example; not present in the current ECDFS FITS extraction

- Column(s): `*_cModelFlux / *_cModelFluxErr`
  Keep/veto logic: tutorial galaxy/ECDFS examples use CModel signal-to-noise cuts such as `r_cModelFlux/r_cModelFluxErr > 20` and `i_cModelFlux/i_cModelFluxErr > 20`
  Reason: ensures high-S/N CModel photometry for morphology/CMD examples
  Confidence: tutorial_inspired
  Local evidence: DP1 ECDFS and galaxy-photometry tutorial examples; equivalent `*_free_cModelFlux` and `*_free_cModelFluxErr` columns are present here

- Column(s): `i_cModel_flag, i_kronFlux_flag, sersic_no_data_flag`
  Keep/veto logic: tutorial galaxy examples keep rows where these flags are `0`
  Reason: rejects failed CModel/Kron/Sersic-related galaxy measurements in the DP1 galaxy photometry demo
  Confidence: tutorial_inspired
  Local evidence: DP1 galaxy-photometry tutorial example; these exact columns are not present in the current ECDFS FITS extraction

## Likely defaults consistent with the current repo workflow

- Column(s): `g/i CModel fluxes and flags`
  Keep/veto logic: for the first-look `g-i` vs `i` CMD, use rows with valid `g` and `i` CModel fluxes and clean CModel flux flags
  Reason: this is the minimum defensible band-specific clean subset for a coadd-style CMD using `g-i` color and `i` magnitude
  Confidence: likely
  Local evidence: consistent with the project-requested CMD plus existing local flux-based workflow

- Column(s): `r / g / i PSF+CModel fluxes`
  Keep/veto logic: for later pS-related morphology work, require valid PSF and CModel fluxes in the band(s) being modeled, at minimum `r`, or preferably `g/r/i` together for legacy `dm3`-style work
  Reason: the repo's star/galaxy morphology code uses `*_diff`, which is undefined unless both fluxes are usable
  Confidence: likely
  Local evidence: `src/psf_cmodel_fit.py`, `src/SG_separation.py`, and `src/build_ecdfs_probability_models.py`

## Tentative candidate cuts that still need project confirmation

- Column(s): `*_blendedness_flag`
  Keep/veto logic: consider keeping only rows where `*_blendedness_flag == False` for stricter photometric CMD samples
  Reason: the column family exists and is plausibly useful for conservative coadd photometry, but no current repo code uses it as a confirmed default cut
  Confidence: tentative
  Local evidence: present in the ECDFS FITS file; not yet adopted elsewhere in the local project

- Column(s): `*_blendedness`
  Keep/veto logic: do not impose a hard blendedness threshold yet; inspect band-dependent distributions first
  Reason: the quantity is present, but local code does not define a confirmed science threshold
  Confidence: tentative
  Local evidence: distribution available in the raw file; no local threshold recipe found

## Actual flag coverage in the current ECDFS FITS file

- `u_free_psfFlux_flag`: {'True': 394557, 'False': 100294}
- `u_free_cModelFlux_flag`: {'False': 387784, 'True': 107067}
- `u_extendedness_flag`: {'True': 396315, 'False': 98536}
- `u_sizeExtendedness_flag`: {'True': 437597, 'False': 57254}
- `u_blendedness_flag`: {'True': 437597, 'False': 57254}
- `g_free_psfFlux_flag`: {'True': 256105, 'False': 238746}
- `g_free_cModelFlux_flag`: {'False': 462742, 'True': 32109}
- `g_extendedness_flag`: {'True': 259651, 'False': 235200}
- `g_sizeExtendedness_flag`: {'True': 273835, 'False': 221016}
- `g_blendedness_flag`: {'True': 273835, 'False': 221016}
- `r_free_psfFlux_flag`: {'False': 269819, 'True': 225032}
- `r_free_cModelFlux_flag`: {'False': 454674, 'True': 40177}
- `r_extendedness_flag`: {'False': 265894, 'True': 228957}
- `r_sizeExtendedness_flag`: {'False': 249619, 'True': 245232}
- `r_blendedness_flag`: {'False': 249619, 'True': 245232}
- `i_free_psfFlux_flag`: {'False': 300792, 'True': 194059}
- `i_free_cModelFlux_flag`: {'False': 456452, 'True': 38399}
- `i_extendedness_flag`: {'False': 297372, 'True': 197479}
- `i_sizeExtendedness_flag`: {'False': 268199, 'True': 226652}
- `i_blendedness_flag`: {'False': 268199, 'True': 226652}
- `z_free_psfFlux_flag`: {'True': 294758, 'False': 200093}
- `z_free_cModelFlux_flag`: {'False': 439953, 'True': 54898}
- `z_extendedness_flag`: {'True': 297118, 'False': 197733}
- `z_sizeExtendedness_flag`: {'True': 335579, 'False': 159272}
- `z_blendedness_flag`: {'True': 335579, 'False': 159272}
- `y_free_psfFlux_flag`: {'True': 407817, 'False': 87034}
- `y_free_cModelFlux_flag`: {'False': 356930, 'True': 137921}
- `y_extendedness_flag`: {'True': 411187, 'False': 83664}
- `y_sizeExtendedness_flag`: {'True': 445466, 'False': 49385}
- `y_blendedness_flag`: {'True': 445466, 'False': 49385}
- `deblend_failed`: {False: 494851}

## What local code does *not* currently provide

- No local tutorial-derived single best-practice `Object` cleaning recipe was found; tutorial examples are sample-specific selections.
- `plot_compareCMDs` in `src/plotting.py` uses display-oriented magnitude windows and a morphology split (`r_diff <= 0.016`), but those are classifier/display choices rather than generic DP1 cleaning flags.
