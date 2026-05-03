# FITS vs Butler diagnostic comparison

This is a short bookkeeping comparison only. It should not be treated as a final science interpretation.

| comparison | FITS valid objects | Butler valid objects | FITS star completeness | Butler star completeness | FITS star contamination | Butler star contamination |
|---|---:|---:|---:|---:|---:|---:|
| pS vs truth | 14,688 | 14,787 | 0.766 | 0.837 | 0.906 | 0.906 |
| extendedness_i vs truth | 14,688 | 14,787 | 0.766 | 0.762 | 0.896 | 0.897 |
| pS vs extendedness_i | 14,688 | 14,787 | 0.441 | 0.757 | 0.601 | 0.373 |

## Notes

- The old FITS diagnostics used the historical FITS/export photometry names and sample construction.
- The Butler diagnostics use `outputs/dp1_ecdfs_analysis_table_butler.parquet` joined by object ID to the HST matched truth table.
- Differences can come from schema photometry differences, common-sample filtering, and the newly recomputed Butler-based `pS_r`.
- Broad consistency should be judged with plots and reviewed sample cuts, not this table alone.
