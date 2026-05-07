# DP2 ECDFS and COSMOS Output Parity

The DP2 port is organized so that the ECDFS workflow and COSMOS workflow use the same preparation, matching, and validation code paths wherever possible.

## Field Catalog Preparation

| Analysis product | ECDFS output | COSMOS output | Status |
|---|---|---|---|
| Prepared DP2 analysis table | `outputs/dp2_ecdfs_analysis_table.parquet` | `outputs/dp2_cosmos_analysis_table.parquet` | complete |
| RA/Dec footprint | `results/dp2_ecdfs/footprint.png` | `results/dp2_cosmos/footprint.png` | complete |
| Full first-look CMD | `results/dp2_ecdfs/cmd_full.png` | `results/dp2_cosmos/cmd_full.png` | complete |
| Clean/zoom CMD | `results/dp2_ecdfs/cmd_clean_zoom.png` | `results/dp2_cosmos/cmd_clean_zoom.png` | complete |
| g-r vs r-i color-color plot | `results/dp2_ecdfs/color_color_gr_ri.png` | `results/dp2_cosmos/color_color_gr_ri.png` | complete |
| r-band extendedness sanity plot | `results/dp2_ecdfs/r_extendedness_sanity.png` | `results/dp2_cosmos/r_extendedness_sanity.png` | complete |
| Clean-mask summary | `results/dp2_ecdfs/clean_mask_summary.md` | `results/dp2_cosmos/clean_mask_summary.md` | complete |

## External Validation Matching

| Analysis product | ECDFS output | COSMOS output | Status |
|---|---|---|---|
| External matched table | `outputs/dp2_ecdfs_hst_matched.parquet` | `outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet` | complete |
| Star/galaxy external validation subset | same matched table after star/galaxy filtering | `outputs/dp2_cosmos_external_truth_sample.parquet` | complete |
| Match separation histogram | `results/dp2_ecdfs_hst_validation/match_sep_hist.png` | `results/dp2_cosmos_external_validation/match_sep_hist.png` | complete |
| Matched footprint | `results/dp2_ecdfs_hst_validation/match_footprint.png` | `results/dp2_cosmos_external_validation/match_footprint.png` | complete |
| External label counts | `results/dp2_ecdfs_hst_validation/external_label_counts.png` | `results/dp2_cosmos_external_validation/external_label_counts.png` | complete |
| Match summary | `results/dp2_ecdfs_hst_validation/match_summary.md` | `results/dp2_cosmos_external_validation/match_summary.md` | complete |

## Baseline Star/Galaxy Diagnostics

The completed baseline diagnostics use Rubin `i`-band extendedness against the external validation labels.

| Analysis product | ECDFS output | COSMOS output | Status |
|---|---|---|---|
| Four-panel CMD, extendedness vs external labels | `results/dp2_ecdfs_hst_validation/baseline/cmd_four_panel_extendedness_i_vs_external.png` | `results/dp2_cosmos_external_validation/baseline/cmd_four_panel_extendedness_i_vs_external.png` | complete |
| Completeness/contamination vs i magnitude | `results/dp2_ecdfs_hst_validation/baseline/metrics_vs_i_extendedness_i_vs_external.png` | `results/dp2_cosmos_external_validation/baseline/metrics_vs_i_extendedness_i_vs_external.png` | complete |
| Metrics CSV | `results/dp2_ecdfs_hst_validation/baseline/summary.csv` | `results/dp2_cosmos_external_validation/baseline/summary.csv` | complete |

## pS Version Diagnostics

The shared notebook section now exists for all six DP2 notebooks, and it will write:

- ECDFS: `outputs/dp2_ecdfs_ps_v1.parquet` through `outputs/dp2_ecdfs_ps_v6.parquet`
- COSMOS: `outputs/dp2_cosmos_ps_v1.parquet` through `outputs/dp2_cosmos_ps_v6.parquet`

Full pS-vs-external and pS-version comparison plots have been generated under `results/dp2_ps_version_comparison/`.

## External Catalog Path

The GitHub-ready compact COSMOS2020 FARMER external-validation file is expected at:

`data/external/cosmos2020_farmer_truth_catalog.csv`

This file is used as an input data dependency and should not be interpreted as a repo-generated product.
