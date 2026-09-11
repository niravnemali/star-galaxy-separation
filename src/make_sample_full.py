#!/usr/bin/env python3
"""Full sample: every HST-unresolved / Rubin-resolved object with r < 23.

The base selection plus only the original faint limit -- no dm cut. This is the
broad set for exploration; ``make_sample_paper.py`` is the narrow one.

Sorted by dm descending so it shares an ordering convention with the other
samples (obj_num is per-sample, so numbers do not carry across).

Usage
-----
    python make_sample_full.py
"""

from __future__ import annotations

from cutout_samples import Cut, build_sample, load_labeled_catalog

R_MAX = 23.0


def main() -> None:
    pool = load_labeled_catalog()

    cuts = [
        Cut(f"r < {R_MAX:g}", lambda d: d["r_cModelMag"] < R_MAX),
    ]

    build_sample(
        pool,
        cuts,
        name="full",
        description=f"r < {R_MAX:g}, no dm cut, sorted by dm desc",
    )


if __name__ == "__main__":
    main()
