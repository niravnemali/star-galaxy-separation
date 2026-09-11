#!/usr/bin/env python3
"""Paper sample: the narrow, high-confidence HST/Rubin disagreements.

Adds to the base selection (HST-unresolved, Rubin-resolved, quality flags):

  * dm = r_psfMag - r_cModelMag > 0.05   -- exclude marginal extendedness calls
  * 20 < r_cModelMag < 22                -- well-measured magnitude range

Sorted by dm descending, so the most strongly Rubin-extended objects lead.

Usage
-----
    python make_sample_paper.py
"""

from __future__ import annotations

from cutout_samples import Cut, build_sample, load_labeled_catalog

DM_MIN = 0.05
R_MIN, R_MAX = 20.0, 22.0


def main() -> None:
    pool = load_labeled_catalog()

    cuts = [
        Cut(f"dm > {DM_MIN}", lambda d: d["r_psf_minus_cModel"] > DM_MIN),
        Cut(f"{R_MIN:g} < r < {R_MAX:g}",
            lambda d: d["r_cModelMag"].between(R_MIN, R_MAX, inclusive="neither")),
    ]

    build_sample(
        pool,
        cuts,
        name="paper",
        description=f"dm > {DM_MIN}, {R_MIN:g} < r < {R_MAX:g}, sorted by dm desc",
    )


if __name__ == "__main__":
    main()
