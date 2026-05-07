# COSMOS2020 FARMER Compact External-Validation Catalog

- Raw input: `/Users/jinshuozhang/Library/CloudStorage/Dropbox/Rubin-Work/Rubin-star-galaxy-classification/star-galaxy-separation/data/external/COSMOS2020_FARMER_R1_v2.2_p3.fits`
- Output: `/Users/jinshuozhang/Library/CloudStorage/Dropbox/Rubin-Work/Rubin-star-galaxy-classification/star-galaxy-separation/data/external/cosmos2020_farmer_truth_catalog.csv`
- Raw rows: 964,506
- Compact star/galaxy rows: 580,691
- Label counts: {'galaxy': 563218, 'star': 17473}
- FLAG_COMBINED counts in compact rows: {0: 536867, 1: 43824}
- Label rule: `ACS_MU_CLASS=1 -> galaxy`, `ACS_MU_CLASS=2 -> star`; fake/spurious and unlabeled rows excluded.
- Caveat: labels are external validation labels, not perfect truth.
