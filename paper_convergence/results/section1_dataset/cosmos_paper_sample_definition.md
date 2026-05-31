# COSMOS Paper Sample Definition

Generated: 2026-05-31T01:07:11

## Input Files

- DP2 analysis table: `outputs/dp2_cosmos_analysis_table.parquet`
- DP2/COSMOS2020 matched table: `outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet`
- COSMOS2020 truth catalog: `data/cosmos2020_farmer_truth_catalog_github.csv`

These input parquet/catalog files are local data dependencies and should not be committed.

## Cuts

- Field: COSMOS
- External labels: COSMOS2020 FARMER truth labels
- Coordinate cut: 149.50413 < RA < 150.9917, 1.4876 < Dec < 2.97521
- Magnitude cut: 16.0 < uncorrected r-band CModel magnitude < 26.0

## Verified Conventions

Truth-label convention from matched table:

```text
 truth_binary truth_label  count
            0      galaxy 363609
            1        star  15760
```

Dust/color convention:

CModel magnitude columns are treated as uncorrected; colors are corrected with magnitude_calc.extcoeff and Ar = ebv * 3.10 / 1.20. No pre-existing dust-corrected color columns were used. Existing cmodel_color columns were checked and match raw CModel magnitude differences, so the paper color columns are recomputed from magnitudes before applying the repo extinction correction.

Extinction scale: Ar = ebv * 2.58333333

COSMOS2020-only magnitude limitation:

data/cosmos2020_farmer_truth_catalog_github.csv contains id, ra, dec, label, truth_binary, acs_mu_class, flag_combined only; no external r magnitude directly comparable to DP2 r CModelMag is available.

## Counts

See `paper_convergence/results/section1_dataset/cosmos_paper_sample_counts.csv`.
