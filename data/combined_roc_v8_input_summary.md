# Combined-Band ROC v8 Input Summary

- pS version: `v8`.
- Reference magnitude column: `dp2_cmodel_mag_r`.
- Positive class: external-label star.
- pS columns used: `pS_u`, `pS_g`, `pS_r`, `pS_i`, `pS_z`, `pS_y`.
- Combined scores: simple mean and empirical histogram-density log-likelihood ratio.

## ECDFS

- External label source: `HST`
- pS file: `outputs/dp2_ecdfs_ps_v8.parquet`
- matched label file: `outputs/dp2_ecdfs_hst_matched.parquet`
- rows with external labels: 7,730
- rows with valid `dp2_cmodel_mag_r`: 7,672
- external stars with valid reference magnitude: 103
- external galaxies with valid reference magnitude: 7,569

## COSMOS

- External label source: `COSMOS2020`
- pS file: `outputs/dp2_cosmos_ps_v8.parquet`
- matched label file: `outputs/dp2_cosmos_cosmos2020_farmer_matched.parquet`
- rows with external labels: 379,369
- rows with valid `dp2_cmodel_mag_r`: 379,012
- external stars with valid reference magnitude: 15,726
- external galaxies with valid reference magnitude: 363,286
