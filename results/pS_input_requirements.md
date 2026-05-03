# pS input requirements for the current ECDFS workflow

## Existing pS workflow files inspected

- `src/pipeline.py`
- `src/magnitude_calc.py`
- `src/psf_cmodel_fit.py`
- `src/build_ecdfs_probability_models.py`
- `notebooks/ECDFS-DP1.ipynb`

## What the current pS workflow expects

- The raw FITS loader currently starts from `*_free_cModelFlux`, `*_free_psfFlux`, `*_free_cModelFluxErr`, and `*_free_psfFluxErr` via `pipeline.load_csv_dp1(...)`.
- `pipeline.magnitude_calculations(...)` then builds the repo's standard derived columns: `u/g/r/i/z/y_modelFlux_mag`, `u/g/r/i/z/y_psfFlux_mag`, `u/g/r/i/z/y_diff`, `u/g/r/i/z/y_psfRelErr`, and colors `ug`, `gr`, `ri`, `iz`, `zy`, `gi`.
- `psf_cmodel_fit.compute_pS(...)` operates on the per-band pair (`*_diff`, `*_modelFlux_mag`).
- `build_ecdfs_probability_models.compute_current_pSr(...)` explicitly recomputes `pS_r` from `r_diff` and `r_modelFlux_mag`.

## Prepared-table coverage

- `objectId / coord_ra / coord_dec`: satisfied. core row identity and sky coordinates.
- `u..y raw CModel fluxes`: satisfied. needed to rebuild notebook-style magnitudes/colors.
- `u..y raw PSF fluxes`: satisfied. needed to rebuild notebook-style PSF-CModel differences.
- `u..y raw CModel flux errors`: satisfied. useful for uncertainty propagation and later QC.
- `u..y raw PSF flux errors`: satisfied. useful for uncertainty propagation and later QC.
- `u..y derived CModel magnitudes`: satisfied. direct pS coordinate input via `*_modelFlux_mag`.
- `u..y derived PSF magnitudes`: satisfied. useful for diagnostics and cross-checks.
- `u..y derived PSF-CModel morphology`: satisfied. direct pS coordinate input via `*_diff`.
- `u..y relative PSF errors`: satisfied. already used in the current repo feature family.
- `Extinction/colors`: satisfied. existing notebook-style color diagnostics.
- `Extendedness family`: satisfied. later comparison to native Rubin morphology.

## Practical compatibility notes

- The new prepared table intentionally renames the raw identity/coordinate columns to `object_id`, `ra`, and `dec` for readability.
- The pS-specific derived columns (`*_modelFlux_mag`, `*_psfFlux_mag`, `*_diff`, `*_psfRelErr`, and the color indices) are kept in the existing repo naming convention so later pS work can reuse them directly.
- If an old notebook/script expects the original raw coordinate names (`objectId`, `coord_ra`, `coord_dec`), only a trivial three-column rename adapter is needed.
- Native Rubin `*_extendedness` columns are present for all six bands and are the most direct quantities for later pS-vs-extendedness comparisons.
