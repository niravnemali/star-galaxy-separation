# TAP ECDFS tract/patch candidates

This is TAP-based catalog selection, not a Butler extraction.

- Distinct tract/patch combinations: 83
- Tracts: 4848, 4849, 5063, 5064
- Patches: 0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 2, 20, 21, 22, 23, 24, 25, 26, 27, 28, 3, 30, 31, 32, 33, 34, 35, 36, 37, 38, 4, 41, 42, 43, 44, 45, 46, 47, 48, 5, 53, 54, 56, 57, 6, 60, 61, 66, 67, 68, 69, 7, 70, 71, 72, 73, 75, 76, 77, 78, 79, 8, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99
- Output CSV: `/Users/jinshuozhang/Library/CloudStorage/Dropbox/Rubin-Work/Rubin-star-galaxy-classification/star-galaxy-separation/results/tap_ecdfs_tract_patch_candidates.csv`

## ADQL used

```sql
SELECT DISTINCT
    tract,
    patch
FROM dp1.Object
WHERE CONTAINS(
    POINT('ICRS', coord_ra, coord_dec),
    POLYGON(
        'ICRS',
        52.35, -28.76,
        53.93, -28.76,
        53.93, -27.40,
        52.35, -27.40
    )
) = 1
ORDER BY tract, patch
```
