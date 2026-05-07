# DP2 Paper-Oriented External Validation Summary

## Fields Processed

| Field | Prepared DP2 table | Rows | Notes |
|---|---|---:|---|
| ECDFS | `outputs/dp2_ecdfs_analysis_table.parquet` | 732,402 | Rubin DP2 Object FITS export standardized with magnitudes, colors, morphology quantities, and clean masks. |
| COSMOS | `outputs/dp2_cosmos_analysis_table.parquet` | 865,287 | Same preparation code path as ECDFS. |

## External Validation Catalogs

| Field | External catalog | Matched output | Match radius | Final matches | Label caveat |
|---|---|---|---:|---:|---|
| ECDFS | `data/external/hst_truth_catalog.csv` | `outputs/dp2_ecdfs_hst_matched.parquet` | 0.3 arcsec | 7,730 | HST/GOODS-S labels used as external validation labels. |
| COSMOS | `data/external/cosmos2020_farmer_truth_catalog.csv` | `outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet` | 0.3 arcsec | 379,373 | COSMOS2020 FARMER labels are external validation labels, not perfect truth. |

## Matching Quality

| Field | External rows | Match fraction | Median separation | P95 separation | Main label counts in matched table |
|---|---:|---:|---:|---:|---|
| ECDFS | 50,507 | 0.153 | 0.230 arcsec | 0.293 arcsec | galaxy: 7,625; star: 105 |
| COSMOS | 580,691 | 0.653 | 0.079 arcsec | 0.247 arcsec | galaxy: 363,613; star: 15,760 |

## Clean Cuts

Cleaning remains task-specific:

- Footprints use coordinate-valid objects.
- CMD plots use g/i CModel-specific masks rather than requiring every band.
- pS and extendedness diagnostics use common samples with the required score, external label, CMD coordinates, and relevant clean masks.
- No final science-quality global cut has been imposed yet.

DP2 column and cleaning documentation is available in:

- `results/dp2_column_list.txt`
- `results/dp2_column_inventory.md`
- `results/dp2_cleaning_recommendations.md`

## Baseline External-Validation Results

The current completed quantitative baseline uses Rubin `i_extendedness` / `dp2_extendedness_i`.

| Field | External star/galaxy sample in baseline | Star completeness | Star purity | Star contamination | Galaxy completeness | Galaxy purity |
|---|---:|---:|---:|---:|---:|---:|
| ECDFS | 6,822 | 0.696 | 0.0669 | 0.933 | 0.853 | 0.995 |
| COSMOS | 335,159 | 0.670 | 0.168 | 0.832 | 0.841 | 0.981 |

These baseline values should not be treated as final science conclusions; they mainly verify that the DP2 matching and diagnostic machinery runs end-to-end.

## pS Model Version Comparison

All six DP2 notebooks now save standardized per-object pS outputs:

- `outputs/dp2_ecdfs_ps_v1.parquet` through `outputs/dp2_ecdfs_ps_v6.parquet`
- `outputs/dp2_cosmos_ps_v1.parquet` through `outputs/dp2_cosmos_ps_v6.parquet`

The full pS v1-v6 comparison has been generated:

- `results/dp2_ps_version_comparison/README.md`
- `results/dp2_ps_version_comparison/ecdfs/version_band_summary.csv`
- `results/dp2_ps_version_comparison/ecdfs/threshold_metrics.csv`
- `results/dp2_ps_version_comparison/ecdfs/metrics_vs_mag.csv`
- `results/dp2_ps_version_comparison/cosmos/version_band_summary.csv`
- `results/dp2_ps_version_comparison/cosmos/threshold_metrics.csv`
- `results/dp2_ps_version_comparison/cosmos/metrics_vs_mag.csv`

At the default `pS >= 0.5` threshold, the strongest star F1-like balance rows are:

| Field | Version | Band | Star completeness | Star purity | Star contamination | Note |
|---|---|---|---:|---:|---:|---|
| ECDFS | v2 | y | 0.596 | 0.868 | 0.132 | ECDFS star sample is small, so treat as noisy. |
| COSMOS | v2 | y | 0.401 | 0.985 | 0.015 | Best COSMOS star F1-like balance among default-threshold rows. |
| COSMOS | v1 | r | 0.455 | 0.752 | 0.248 | Best COSMOS r-band default-threshold row. |

## Key Figures Generated

- `results/dp2_ecdfs/footprint.png`
- `results/dp2_ecdfs/cmd_full.png`
- `results/dp2_ecdfs/cmd_clean_zoom.png`
- `results/dp2_ecdfs/color_color_gr_ri.png`
- `results/dp2_ecdfs/r_extendedness_sanity.png`
- `results/dp2_ecdfs_hst_validation/match_sep_hist.png`
- `results/dp2_ecdfs_hst_validation/baseline/cmd_four_panel_extendedness_i_vs_external.png`
- `results/dp2_ecdfs_hst_validation/baseline/metrics_vs_i_extendedness_i_vs_external.png`
- `results/dp2_cosmos/footprint.png`
- `results/dp2_cosmos/cmd_full.png`
- `results/dp2_cosmos/cmd_clean_zoom.png`
- `results/dp2_cosmos/color_color_gr_ri.png`
- `results/dp2_cosmos/r_extendedness_sanity.png`
- `results/dp2_cosmos_external_validation/match_sep_hist.png`
- `results/dp2_cosmos_external_validation/baseline/cmd_four_panel_extendedness_i_vs_external.png`
- `results/dp2_cosmos_external_validation/baseline/metrics_vs_i_extendedness_i_vs_external.png`
- `results/dp2_ps_version_comparison/ecdfs/purity_completeness_r.png`
- `results/dp2_ps_version_comparison/ecdfs/contamination_completeness_r.png`
- `results/dp2_ps_version_comparison/ecdfs/metrics_vs_mag_r.png`
- `results/dp2_ps_version_comparison/cosmos/purity_completeness_r.png`
- `results/dp2_ps_version_comparison/cosmos/contamination_completeness_r.png`
- `results/dp2_ps_version_comparison/cosmos/metrics_vs_mag_r.png`

## Remaining Caveats

- COSMOS2020 FARMER labels are external validation labels, not perfect truth.
- ECDFS HST external matching yields a relatively small star sample, so star purity estimates are noisy.
- `outputs/dp2_ecdfs_ps_v6.parquet` contains 2,047 duplicate `object_id` rows; validation scripts report this and keep the first row per object.
- The preferred pS version depends on the chosen purity/completeness tradeoff and should be reviewed with the advisor.
- Final paper-quality clean cuts still need review.

## Recommended Next Steps

1. Review `results/dp2_ps_version_comparison/README.md` and the overlay curves.
2. Choose a candidate pS version/band/threshold with the advisor based on the desired purity/completeness tradeoff.
3. Decide whether final paper figures should use default threshold 0.5 or a threshold selected from the threshold-scan curves.
