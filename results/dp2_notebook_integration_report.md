# DP2 notebook integration report

Standardized external-validation cells were appended to each DP2 notebook. The pS algorithm cells were not changed.

| notebook | version | validation section added | old/conflicting cells removed | standardized pS outputs |
|---|---|---|---:|---|
| `ECDFS-DP2v1.ipynb` | `v1` | yes | 0 | `outputs/dp2_ecdfs_ps_v1.parquet`, `outputs/dp2_cosmos_ps_v1.parquet` |
| `ECDFS-DP2v2.ipynb` | `v2` | yes | 0 | `outputs/dp2_ecdfs_ps_v2.parquet`, `outputs/dp2_cosmos_ps_v2.parquet` |
| `ECDFS-DP2v3.ipynb` | `v3` | yes | 5 | `outputs/dp2_ecdfs_ps_v3.parquet`, `outputs/dp2_cosmos_ps_v3.parquet` |
| `ECDFS-DP2v4.ipynb` | `v4` | yes | 0 | `outputs/dp2_ecdfs_ps_v4.parquet`, `outputs/dp2_cosmos_ps_v4.parquet` |
| `ECDFS-DP2v5.ipynb` | `v5` | yes | 0 | `outputs/dp2_ecdfs_ps_v5.parquet`, `outputs/dp2_cosmos_ps_v5.parquet` |
| `ECDFS-DP2v6.ipynb` | `v6` | yes | 0 | `outputs/dp2_ecdfs_ps_v6.parquet`, `outputs/dp2_cosmos_ps_v6.parquet` |

The new cells print validation commands but keep `subprocess.run` commented by default, so opening the notebooks will not rerun heavy validation jobs automatically.

Current blocker: these pS output files are created only after each notebook version is executed through the pS-output cell.
