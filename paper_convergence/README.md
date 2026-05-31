# Paper Convergence

This folder contains the paper-convergence pipeline for the Rubin star-galaxy separation project.

- Primary paper sample: COSMOS DP2 matched to COSMOS2020 external labels.
- ECDFS/HST is secondary and can be rerun later as a comparison.
- Figures are in `figures/`.
- Numerical summaries are in `results/`.
- Notebooks are in `notebooks/`.
- Reusable paper-specific code is in `code/`.
- Documentation and manifests are in `docs/`.
- Logs are in `logs/`.
- Private data, FITS files, parquet catalogs, and large Rubin catalog files are not stored here.
- Main sample selection uses uncorrected r-band CModel magnitude.
- Colors are computed from CModel magnitudes corrected for dust extinction with the existing repo extinction convention.
- Shared plot style is centralized in `paper_convergence/code/paper_plot_style.py`.

## Reproducibility

From the repository root, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 paper_convergence/code/run_paper_phase1_cosmos.py
PYTHONDONTWRITEBYTECODE=1 python3 paper_convergence/code/run_paper_phase2_cosmos.py
PYTHONDONTWRITEBYTECODE=1 python3 paper_convergence/code/run_paper_phase3_cosmos.py
PYTHONDONTWRITEBYTECODE=1 python3 paper_convergence/code/run_paper_phase4_cosmos.py
```
