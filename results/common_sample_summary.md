# Butler common sample summary

- Total Butler table rows: 494,850
- HST truth rows after object-ID deduplication: 17,954
- Butler rows matched to HST truth: 17,954
- Rows after finite CMD, truth, and CMD-clean cuts: 17,773
- Rows with finite/valid pS_r: 17,866 in matched table; 17,742 after base cuts
- Rows with valid extendedness_i: 14,800 after base cuts
- Final common sample rows: 14,787
- Truth counts in matched table: {'galaxy': 17590, 'star': 364}
- Truth counts after base cuts: {'star': 352, 'galaxy': 17421}

## pS_r status

- Status: recomputed `pS_r` from Butler `r_diff` and `r_modelFlux_mag` using existing pS workflow
- Finite `pS_r` values in matched table: 17,866
- Classification threshold: star if `pS_r >= 0.5`

## Required common-sample columns

- Truth label column: `hst_label`
- pS column: `pS_r`
- extendedness_i column: `i_extendedness`
- color column: `cmd_g_minus_i`
- i magnitude column: `cmd_i_mag`
- CMD clean mask: `cmd_gi_cmodel_clean` when present
