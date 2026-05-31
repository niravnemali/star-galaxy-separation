"""Matching diagnostics for COSMOS paper figures."""

from __future__ import annotations

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from paper_sample_selection import coordinate_mask


def cosmos_matching_radius_sweep(
    dp2: pd.DataFrame,
    external: pd.DataFrame,
    radii_arcsec: tuple[float, ...] = (0.1, 0.2, 0.3, 0.5, 0.7, 1.0),
) -> pd.DataFrame:
    """One-way nearest-neighbor match from external labels to DP2."""

    dp2_use = dp2.loc[coordinate_mask(dp2, "ra", "dec"), ["object_id", "ra", "dec"]].dropna().copy()
    ext_use = external.loc[coordinate_mask(external, "ra", "dec"), ["id", "ra", "dec", "truth_binary"]].dropna().copy()

    dp2_coord = SkyCoord(
        ra=pd.to_numeric(dp2_use["ra"], errors="coerce").to_numpy() * u.deg,
        dec=pd.to_numeric(dp2_use["dec"], errors="coerce").to_numpy() * u.deg,
    )
    ext_coord = SkyCoord(
        ra=pd.to_numeric(ext_use["ra"], errors="coerce").to_numpy() * u.deg,
        dec=pd.to_numeric(ext_use["dec"], errors="coerce").to_numpy() * u.deg,
    )
    idx, sep2d, _ = ext_coord.match_to_catalog_sky(dp2_coord)
    sep_arcsec = sep2d.arcsec
    truth = pd.to_numeric(ext_use["truth_binary"], errors="coerce").to_numpy()

    rows = []
    for radius in radii_arcsec:
        ok = sep_arcsec <= radius
        unique_dp2 = np.unique(idx[ok]).size
        n_matches = int(ok.sum())
        n_star = int(((truth == 1) & ok).sum())
        n_gal = int(((truth == 0) & ok).sum())
        rows.append(
            {
                "radius_arcsec": radius,
                "N_matches": n_matches,
                "N_matched_star": n_star,
                "N_matched_galaxy": n_gal,
                "match_fraction_relative_to_external": n_matches / len(ext_use) if len(ext_use) else np.nan,
                "match_fraction_relative_to_DP2_inside_external_footprint": (
                    unique_dp2 / len(dp2_use) if len(dp2_use) else np.nan
                ),
                "N_external_inside_coordinate_cut": len(ext_use),
                "N_DP2_inside_coordinate_cut": len(dp2_use),
                "notes": "nearest DP2 object to each COSMOS2020 object; DP2 denominator uses rectangular COSMOS cut",
            }
        )
    return pd.DataFrame(rows)

