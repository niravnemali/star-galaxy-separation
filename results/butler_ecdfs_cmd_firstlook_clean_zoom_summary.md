# Butler ECDFS clean/zoom first-look CMD

- Input table: `outputs/dp1_ecdfs_analysis_table_butler.parquet`
- Total Butler table rows: 494,850
- Rows passing `cmd_gi_cmodel_clean`: 435,339
- Rows with finite `g-i` and `i` after CMD clean mask: 435,339
- Rows inside displayed zoom range: 404,384
- x-axis: `cmodel_mag_g - cmodel_mag_i`
- y-axis: `cmodel_mag_i`
- Display x limits: `-1 < g-i < 4`
- Display y limits: `18 < i < 29`, plotted with inverted magnitude axis as `ylim=(29, 18)`
- Cleaning note: this is a task-specific g/i CModel CMD clean mask for visualization. It does not require all six bands to be good and is not the final science sample.
