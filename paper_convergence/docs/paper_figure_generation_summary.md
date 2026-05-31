# Section 1 Paper Figure Generation Summary

This upload set contains only Section 1 COSMOS/COSMOS2020 paper-convergence material. Figure files are staged as PDFs only; PNG quick-look files are intentionally left unstaged.

## Sample Definition

- Primary field: COSMOS
- External label source: COSMOS2020
- Coordinate cut: configured in `paper_convergence/config/cosmos_paper_config.yaml`
- Main magnitude cut: `16 < uncorrected r-band CModel magnitude < 26`
- Color convention: CModel colors use the repository dust-correction convention when the needed columns are available
- Color-color axis ranges: fixed Week10 ranges recorded in the config file

## Section 1 Figures

- `paper_convergence/figures/section1_dataset/fig1_1_cosmos_dataset_overview.pdf`
- `paper_convergence/figures/section1_dataset/fig1_1b_cosmos_counts_vs_rmag_0p25mag.pdf`
- `paper_convergence/figures/diagnostics/cosmos_matching_radius_sweep.pdf`
- `paper_convergence/figures/section1_truth_labels/fig1_2_cosmos_truth_morphology_color_2x2.pdf`
- `paper_convergence/figures/section1_truth_labels/fig1_3_cosmos_truth_color_color_2x4.pdf`
- `paper_convergence/figures/section1_extendedness/fig1_4_cosmos_r_extendedness_color_color_2x4.pdf`
- `paper_convergence/figures/section1_extendedness/fig1_4_cosmos_refExtendedness_color_color_2x4.pdf`
- `paper_convergence/figures/section1_extendedness/fig1_5_cosmos_r_extendedness_confusion_cmd_4panel.pdf`
- `paper_convergence/figures/section1_extendedness/fig1_5_cosmos_refExtendedness_confusion_cmd_4panel.pdf`
- `paper_convergence/figures/section1_extendedness/fig1_6_cosmos_r_extendedness_roc_3bins.pdf`
- `paper_convergence/figures/section1_extendedness/fig1_6_cosmos_refExtendedness_roc_3bins.pdf`

## Section 1 Results And Notes

- `paper_convergence/results/section1_dataset/cosmos_paper_sample_definition.md`
- `paper_convergence/results/section1_dataset/cosmos_paper_sample_counts.csv`
- `paper_convergence/results/section1_dataset/fig1_1_cosmos_dataset_overview_counts.csv`
- `paper_convergence/results/section1_dataset/fig1_1b_cosmos_counts_vs_rmag_0p25mag.csv`
- `paper_convergence/results/diagnostics/cosmos_matching_radius_sweep.csv`
- `paper_convergence/results/section1_truth_labels/fig1_2_cosmos_truth_morphology_color_2x2_summary.csv`
- `paper_convergence/results/section1_truth_labels/fig1_3_cosmos_truth_color_color_2x4_summary.csv`
- `paper_convergence/results/section1_extendedness/extendedness_convention_check.csv`
- `paper_convergence/results/section1_extendedness/extendedness_vs_refExtendedness_comparison_summary.csv`
- `paper_convergence/results/section1_extendedness/extendedness_vs_refExtendedness_comparison_summary.md`
- `paper_convergence/results/section1_extendedness/fig1_4_cosmos_r_extendedness_color_color_2x4_summary.csv`
- `paper_convergence/results/section1_extendedness/fig1_4_cosmos_refExtendedness_color_color_2x4_summary.csv`
- `paper_convergence/results/section1_extendedness/fig1_5_cosmos_r_extendedness_confusion_cmd_4panel_summary.csv`
- `paper_convergence/results/section1_extendedness/fig1_5_cosmos_refExtendedness_confusion_cmd_4panel_summary.csv`
- `paper_convergence/results/section1_extendedness/fig1_6_cosmos_r_extendedness_roc_3bins_summary.csv`
- `paper_convergence/results/section1_extendedness/fig1_6_cosmos_refExtendedness_roc_3bins_summary.csv`

## Caveats

- The main paper sample is COSMOS/COSMOS2020; ECDFS/HST is not included in this Section 1 upload set.
- The COSMOS2020-only count curve uses the full local FARMER catalog HSC r magnitude when available. This is a coverage diagnostic and is not identical to DP2 r CModel magnitude.
- Extendedness and `refExtendedness` conventions are recorded in `extendedness_convention_check.csv`.
- PNG files, Section 2/5 outputs, private data, FITS files, parquet files, and large catalogs are intentionally excluded from the staged upload set.
