# Task-specific cleaning note

The Butler-based ECDFS products intentionally do not use one strict global clean cut for every task.

- Footprint plot: use the full Butler ECDFS sample, requiring only usable coordinates.
- Full first-look CMD: keep as a broad catalog sanity check.
- Clean/zoom CMD: use the g/i-specific `cmd_gi_cmodel_clean` mask and display limits `-1 < g-i < 4`, `18 < i < 29`. This requires clean CModel photometry only in the bands used by the CMD, not all six bands.
- Star/galaxy diagnostics: keep the existing common sample logic with valid truth, `pS_r`, `extendedness_i`, CMD quantities, and CMD-clean photometry.
- Final science cuts: still require project review before being treated as final analysis selections.

New clean/zoom CMD outputs:

- `results/butler_ecdfs_cmd_firstlook_clean_zoom.png`
- `results/butler_ecdfs_cmd_firstlook_clean_zoom.pdf`
- `results/butler_ecdfs_cmd_firstlook_clean_zoom_summary.md`
