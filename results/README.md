# Star/Galaxy Diagnostics

This directory contains CMD-based diagnostics for the Rubin star/galaxy separation workflow for the Rubin star/galaxy separation workflow.

## Definitions

- `a`: true/reference star predicted as star
- `b`: true/reference galaxy predicted as galaxy
- `c`: true/reference star predicted as galaxy
- `d`: true/reference galaxy predicted as star

Metrics are defined as:

- star completeness = `a / (a + c)`
- star contamination = `d / (a + d)`
- galaxy completeness = `b / (b + d)`
- galaxy contamination = `c / (b + c)`

All comparisons in this Butler run use the same common sample before computing panels and metrics.

## CMD plots

Each `cmd_four_panel_*.png/.pdf` file shows a 2x2 i vs g-i CMD with full-sample background contours and one confusion subset overlaid.

Truth comparisons use:

1. correct stars
2. correct galaxies
3. misclassified stars
4. misclassified galaxies

Method-vs-method comparisons use agreement/disagreement labels instead of truth wording.

The y-axis is i magnitude and is inverted so brighter objects appear higher.

## Columns and conventions used

- truth: Truth column `hst_label` (string star/galaxy labels)
- pS: recomputed `pS_r` from Butler `r_diff` and `r_modelFlux_mag` using existing pS workflow; star if pS_r >= 0.5
- g_mag: cmodel_mag_g
- i_mag: cmodel_mag_i

## Butler-based run notes

- These diagnostics use the Butler-based ECDFS analysis table.
- Butler catalog: `outputs/dp1_ecdfs_analysis_table_butler.parquet`
- Truth labels come from: `/Users/jinshuozhang/Library/CloudStorage/Dropbox/Rubin-Work/Rubin-star-galaxy-classification/star-galaxy-separation/outputs/hst_dp1_matched_clean_ellipse.csv`
- Matched Butler/HST sample: `/Users/jinshuozhang/Library/CloudStorage/Dropbox/Rubin-Work/Rubin-star-galaxy-classification/star-galaxy-separation/outputs/hst_dp1_matched_butler_analysis.parquet`
- pS_r status: recomputed `pS_r` from Butler `r_diff` and `r_modelFlux_mag` using existing pS workflow
- pS_r threshold for star classification: `0.5`
- All requested Butler comparisons use the common sample with valid truth, pS_r, extendedness_i, g-i, i, and CMD-clean photometry.
- Remaining caveat: final clean cuts should be reviewed before final science interpretation before final science interpretation.

## Comparisons generated

| comparison | reference_column | predicted_column | valid_objects |
|---|---|---|---:|
| pS vs truth | hst_label | pS_r | 14787 |
| extendedness_i vs truth | hst_label | i_extendedness | 14787 |
| pS vs extendedness_i | i_extendedness | pS_r | 14787 |
| extendedness_i vs pS | pS_r | i_extendedness | 14787 |


## Run command

```bash
python scripts/star_galaxy_diagnostics.py --catalog outputs/dp1_ecdfs_analysis_table_butler.parquet --use-butler-table --truth-catalog /Users/jinshuozhang/Library/CloudStorage/Dropbox/Rubin-Work/Rubin-star-galaxy-classification/star-galaxy-separation/outputs/hst_dp1_matched_clean_ellipse.csv --matched-output /Users/jinshuozhang/Library/CloudStorage/Dropbox/Rubin-Work/Rubin-star-galaxy-classification/star-galaxy-separation/outputs/hst_dp1_matched_butler_analysis.parquet --output-dir results/star_galaxy_diagnostics_butler
```
