# RSP TAP API run instructions

This workflow uses Rubin RSP TAP/API access with a token. It does not use the Butler and does not require local `lsst.daf.butler`.

## Set token

```bash
export RUBIN_TOKEN="<your-token>"
```

Security reminder: never commit the token, never print it, and never paste it into logs.

## Tract/patch discovery plus column validation

```bash
python scripts/rsp_tap_ecdfs_query.py
```

## Column validation only

```bash
python scripts/rsp_tap_ecdfs_query.py --skip-tract-query
```

## Small ECDFS Object sample extraction

```bash
python scripts/rsp_tap_ecdfs_query.py --extract-small-sample --limit 1000
```

## Expected outputs

- `results/tap_ecdfs_tract_patch_candidates.csv`
- `results/tap_ecdfs_tract_patch_summary.md`
- `results/tap_dp1_object_column_validation.csv`
- `results/tap_dp1_object_column_validation.md`
- `outputs/tap_ecdfs_object_test_sample.csv` when `--extract-small-sample` is used
- `outputs/tap_ecdfs_object_test_sample.parquet` if a parquet engine is installed

## Default endpoint

The default TAP URL is `https://data.lsst.cloud/api/tap`. Override it with `--tap-url` if Rubin changes the endpoint or if you use another RSP deployment.

TAP success does not imply Butler success. Butler extraction still requires a Rubin/RSP Science Pipelines environment with `lsst.daf.butler`.
