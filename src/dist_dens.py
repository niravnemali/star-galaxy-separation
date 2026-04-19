from astropy.table import Table
import matplotlib.pyplot as plt

import pandas as pd
from functools import partial
import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u

import sys
sys.path.append('../src')
import modelLocusTools as mlt 
from config import *

# ==========
# DISTANCE CALCULATION

def distance_calc(df, feh=False, dm=False, extra_cut=False, correctDP1=False):
    #compute Mr and metallicity
    # with an analytic method (later using PhotoD)
    ## analytic expressions for FeH(ug,gr) and Mr(gi, FeH) from the SDSSS MW Tomography paper series
    if feh:
        df['FeH'] = df['FeH']
    else:
        df['FeH'] = mlt.photoFeH(df['ug'], df['gr'], correctDP1)
    df['Mr'] = mlt.getMr(df['gi'], df['FeH'], correctDP1)

    if extra_cut:
        df = df.copy()
        df = df[(df['gr'] > 0.2) & (df['gr'] < grRedCut) & (df['Mr'] > MrBright) & (df['Mr'] < MrFaint)]
    # distance modulus and distance in kpc
    if dm:
        df['DM'] = df['DM']
    else:
        df['DM'] = df[f'r{model_flux_mag}'] - df['Mr']
    df['Dkpc'] = 0.01 * 10**(0.2*df['DM'])

    return df

# DENSITY CALCULATION
def coordinate_calc_galaxy(df, r_s=8):
    df = df.copy()
    coords = SkyCoord(
    ra  = df[ra_cord].values * u.deg,
    dec = df[dec_cord].values    * u.deg,
    frame='icrs'
    )
    df = df.copy()

    df['l'] = coords.galactic.l.deg
    df['b'] = coords.galactic.b.deg

    #----
    df['X'] = r_s - df['Dkpc']*np.cos(np.radians(df['l'])) * np.cos(np.radians(df['b']))
    df['Y'] = -df['Dkpc']*np.sin(np.radians(df['l'])) * np.cos(np.radians(df['b']))
    df['Z'] = df['Dkpc']*np.sin(np.radians(df['b']))

    df['r_gc'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)

    return df

def dm_binning(df, step=0.2):
    df = df.copy()
    dm_min = df['DM'].min()
    dm_max = df['DM'].max()
    bins = np.arange(
        np.floor(dm_min/step)*step,
        np.ceil(dm_max/step)*step + step,
        step
    )

    df['DM_bins'] = pd.cut(df['DM'], bins=bins, right=False)

    vc = df['DM_bins'].value_counts().sort_index()
    number_stars = vc.reset_index()
    number_stars.columns = ['DM_bins', 'number_stars']

    # get median value of r_gc and append it per bin
    median_r_bin = df.groupby('DM_bins')['r_gc'].median()
    number_stars['r_gc_median'] = number_stars['DM_bins'].map(median_r_bin)
    
    return df, number_stars


def omega_calc(df):
    ra_min, ra_max = df[ra_cord].min(), df[ra_cord].max()
    dec_min, dec_max = df[dec_cord].min(), df[dec_cord].max()

    ra_minr, ra_maxr = np.deg2rad(ra_min), np.deg2rad(ra_max)
    dec_minr, dec_maxr = np.deg2rad(dec_min), np.deg2rad(dec_max)

    omega = (ra_maxr - ra_minr) * (np.sin(dec_maxr) - np.sin(dec_minr))
    return omega

def density_calc(df, omega):
    for i in range(df.shape[0]):
        DM1 = df['DM_bins'][i].left
        DM2 = df['DM_bins'][i].right

        D1 = 10**((DM1 + 5) / 5)
        D2 = 10**((DM2 + 5) / 5)

        dV = (omega / 3) * (D2**3 - D1**3)
        dN = df['number_stars'][i]
        df.loc[i, 'rho'] = dN / dV
    return df

def error_density(df):
    df = df.copy()
    df['sigma'] = df['rho'] / np.sqrt(df['number_stars'])

    df['log_rho'] = np.where(
        df['rho'] > 0,
        np.log10(df['rho']),
        0.0
    )

    frac_err = df['sigma'] / df['rho']

    df['upper_error'] = np.where(
        frac_err >= 0,
        np.log10(1 + frac_err),
        0.0
    )

    df['lower_error'] = np.where(
        (frac_err >= 0) & (frac_err < 1),
        -np.log10(1 - frac_err),
        0.0
    )

    return df

def avg_d(df):
    for i in range(df.shape[0]):
        DM_center = (df['DM_bins'][i].left + df['DM_bins'][i].right) / 2
        df.loc[i, 'DM_center'] = DM_center
        df.loc[i, 'D(kpc)'] = 0.01 * 10**(0.2*df.loc[i, 'DM_center'])
    return df

def density_to_df(df, dens_stars):
    df = df.reset_index(drop=True)

    for i in range(df.shape[0]):
        for j in range(dens_stars.shape[0]):
            if df['DM_bins'][i] == dens_stars['DM_bins'][j]:
                df.loc[i, 'rho'] = dens_stars['rho'][j]
    
    return df
