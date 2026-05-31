# Section 1 Figure Manifest

This manifest lists the Section 1 COSMOS/COSMOS2020 paper-convergence figures prepared for PDF-only upload. PNG versions may exist locally for quick viewing but are not part of this upload set.

## Section 1.1 Dataset Overview

- Figure: `fig1_1`
- File: `paper_convergence/figures/section1_dataset/fig1_1_cosmos_dataset_overview.pdf`
- Notebook/script: `paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb`; `paper_convergence/code/run_paper_phase1_cosmos.py`
- Caption draft: COSMOS DP2 and COSMOS2020 matched-sample overview, showing the matched sky distribution, number counts versus uncorrected r-band CModel magnitude, and r-band CModel flux uncertainty ratio.
- Sample definition: COSMOS coordinate cut, COSMOS2020 external labels, and `16 < uncorrected r CModel magnitude < 26`.
- Caveats: The COSMOS2020-only count curve uses the local full FARMER catalog HSC r magnitude when available; the matched/DP2 curves use DP2 r-band CModel magnitude.
- Status: main paper candidate.

## Section 1.1 Count Diagnostic

- Figure: `fig1_1b`
- File: `paper_convergence/figures/section1_dataset/fig1_1b_cosmos_counts_vs_rmag_0p25mag.pdf`
- Notebook/script: `paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb`; `paper_convergence/code/run_paper_phase1_cosmos.py`
- Caption draft: COSMOS number counts versus r-band magnitude using 0.25 mag bins, with DP2-only, matched COSMOS2020 stars, matched COSMOS2020 galaxies, and COSMOS2020-only objects shown separately.
- Sample definition: Same coordinate cut as Fig. 1.1; DP2 curves use uncorrected DP2 r CModel magnitude; COSMOS2020-only uses HSC r magnitude from the full local FARMER catalog when available.
- Caveats: The external-only magnitude is not identical to DP2 CModel magnitude and should be treated as a coverage diagnostic.
- Status: diagnostic only.

## Matching Radius Diagnostic

- Figure: `cosmos_matching_radius_sweep`
- File: `paper_convergence/figures/diagnostics/cosmos_matching_radius_sweep.pdf`
- Notebook/script: `paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb`; `paper_convergence/code/run_paper_phase1_cosmos.py`
- Caption draft: Nearest-neighbor COSMOS2020-to-DP2 match counts as a function of matching radius.
- Sample definition: COSMOS coordinate cut; radii from 0.1 to 1.0 arcsec.
- Caveats: The DP2 denominator uses the rectangular COSMOS coordinate cut.
- Status: diagnostic only.

## Section 1.2 Truth-Label Morphology And Color

- Figure: `fig1_2`
- File: `paper_convergence/figures/section1_truth_labels/fig1_2_cosmos_truth_morphology_color_2x2.pdf`
- Notebook/script: `paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb`; `paper_convergence/code/run_paper_phase1_cosmos.py`
- Caption draft: COSMOS2020-labeled stars and galaxies in morphology, color-magnitude, and color-color spaces.
- Sample definition: Matched COSMOS2020 labels with `16 < uncorrected r CModel magnitude < 26`.
- Caveats: Colors use the repository extinction-correction convention and fixed Week10 color-color axis ranges.
- Status: main paper candidate.

## Section 1.3 Truth-Label Color-Color Diagrams

- Figure: `fig1_3`
- File: `paper_convergence/figures/section1_truth_labels/fig1_3_cosmos_truth_color_color_2x4.pdf`
- Notebook/script: `paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb`; `paper_convergence/code/run_paper_phase1_cosmos.py`
- Caption draft: COSMOS2020-labeled star and galaxy color-color diagrams split into `16 < rmag < 25` and `25 < rmag < 26`.
- Sample definition: Matched COSMOS2020 labels; colors are dust-corrected CModel colors; rmag is uncorrected DP2 r CModel magnitude.
- Caveats: Each panel has its own finite-color sample count, reported in the panel title and summary CSV.
- Status: main paper candidate.

## Section 1.4 Extendedness Color-Color Diagrams

- Figure: `fig1_4_r_extendedness`
- File: `paper_convergence/figures/section1_extendedness/fig1_4_cosmos_r_extendedness_color_color_2x4.pdf`
- Notebook/script: `paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb`; `paper_convergence/code/run_paper_phase2_cosmos.py`
- Caption draft: COSMOS color-color diagrams split by r-band extendedness classification.
- Sample definition: COSMOS DP2 paper sample with valid r-band extendedness.
- Caveats: Extendedness convention is recorded in `paper_convergence/results/section1_extendedness/extendedness_convention_check.csv`.
- Status: main paper candidate.

- Figure: `fig1_4_refExtendedness`
- File: `paper_convergence/figures/section1_extendedness/fig1_4_cosmos_refExtendedness_color_color_2x4.pdf`
- Notebook/script: `paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb`; `paper_convergence/code/run_paper_phase2_cosmos.py`
- Caption draft: COSMOS color-color diagrams split by `refExtendedness` classification.
- Sample definition: COSMOS DP2 paper sample with valid `refExtendedness`.
- Caveats: `refExtendedness` convention is recorded in `paper_convergence/results/section1_extendedness/extendedness_convention_check.csv`.
- Status: appendix candidate.

## Section 1.5 Extendedness Confusion CMDs

- Figure: `fig1_5_r_extendedness`
- File: `paper_convergence/figures/section1_extendedness/fig1_5_cosmos_r_extendedness_confusion_cmd_4panel.pdf`
- Notebook/script: `paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb`; `paper_convergence/code/run_paper_phase2_cosmos.py`
- Caption draft: COSMOS2020 truth label versus r-band extendedness classification in dust-corrected g-i color and uncorrected r-band CModel magnitude.
- Sample definition: Matched COSMOS2020 objects with valid r-band extendedness.
- Caveats: y-axis uses uncorrected r CModel magnitude.
- Status: main paper candidate.

- Figure: `fig1_5_refExtendedness`
- File: `paper_convergence/figures/section1_extendedness/fig1_5_cosmos_refExtendedness_confusion_cmd_4panel.pdf`
- Notebook/script: `paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb`; `paper_convergence/code/run_paper_phase2_cosmos.py`
- Caption draft: COSMOS2020 truth label versus `refExtendedness` classification in dust-corrected g-i color and uncorrected r-band CModel magnitude.
- Sample definition: Matched COSMOS2020 objects with valid `refExtendedness`.
- Caveats: y-axis uses uncorrected r CModel magnitude.
- Status: appendix candidate.

## Section 1.6 Extendedness Performance

- Figure: `fig1_6_r_extendedness`
- File: `paper_convergence/figures/section1_extendedness/fig1_6_cosmos_r_extendedness_roc_3bins.pdf`
- Notebook/script: `paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb`; `paper_convergence/code/run_paper_phase2_cosmos.py`
- Caption draft: Binary r-band extendedness operating-point ROC diagnostics in three r-magnitude bins.
- Sample definition: Matched COSMOS2020 objects with valid r-band extendedness.
- Caveats: Extendedness is binary, so the figure shows step/operating-point ROC diagnostics rather than a smooth score-threshold ROC curve.
- Status: main paper candidate.

- Figure: `fig1_6_refExtendedness`
- File: `paper_convergence/figures/section1_extendedness/fig1_6_cosmos_refExtendedness_roc_3bins.pdf`
- Notebook/script: `paper_convergence/notebooks/paper_section1_dataset_and_extendedness.ipynb`; `paper_convergence/code/run_paper_phase2_cosmos.py`
- Caption draft: Binary `refExtendedness` operating-point ROC diagnostics in three r-magnitude bins.
- Sample definition: Matched COSMOS2020 objects with valid `refExtendedness`.
- Caveats: `refExtendedness` is binary, so the figure shows step/operating-point ROC diagnostics rather than a smooth score-threshold ROC curve.
- Status: appendix candidate.
