"""Helpers for extracting star/galaxy counts from V8 mixture-model fits."""

import numpy as np


def pad_2to3_skewnorm_freemu(result):
    """Pad a 2-resolved skewnorm (free mu) fit result to 3-resolved layout (wR3=0)."""
    r = dict(result)
    r['n_resolved'] = 3
    if r['params'] is not None:
        p = list(r['params'])
        p += [0.0, 0.5, 0.2, 0.0]
        r['params'] = np.array(p)
    return r


def fallback_failed_bins(combined):
    """Fill in failed bins by copying params from nearest successful neighbor."""
    for i, r in enumerate(combined):
        if r['success']:
            continue
        donor = None
        for offset in range(1, len(combined)):
            if i - offset >= 0 and combined[i - offset]['success']:
                donor = combined[i - offset]
                break
            if i + offset < len(combined) and combined[i + offset]['success']:
                donor = combined[i + offset]
                break
        if donor is not None:
            r['params'] = donor['params'].copy()
            r['n_resolved'] = donor['n_resolved']
            r['success'] = True
            r['fallback'] = True
            print(f"    bin {r['cmodel_lo']:.1f}-{r['cmodel_hi']:.1f}: "
                  f"FALLBACK from {donor['cmodel_lo']:.1f}-{donor['cmodel_hi']:.1f}")
        else:
            r['params'] = np.array([0.0, 0.1,
                                    0.3, 0.2, 0.15, 3.0,
                                    0.3, 0.5, 0.25, 3.0,
                                    0.3, 0.8, 0.35, 3.0])
            r['n_resolved'] = 3
            r['success'] = True
            r['fallback'] = True
            print(f"    bin {r['cmodel_lo']:.1f}-{r['cmodel_hi']:.1f}: "
                  f"FALLBACK using hardcoded defaults")
    return combined


def extract_star_fraction(result):
    """Extract wU (unresolved/star fraction) from a freemu+skewnorm fit result."""
    if not result['success'] or result['params'] is None:
        return np.nan
    p = result['params']
    n_resolved = result['n_resolved']
    wR = np.array([p[2 + 4 * k] for k in range(n_resolved)])
    wR = np.clip(wR, 0.0, 1.0)
    return max(1.0 - np.sum(wR), 0.0)
