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

# ------





def dm_binning(df, model_col, abs_col, step=0.2, dm='DM'):
    df = df.copy()
    df[dm] = df[f'r_{model_col}'] - df[abs_col]

    dm_min = df[dm].min()
    dm_max = df[dm].max()
    bins = np.arange(
        np.floor(dm_min/step)*step,
        np.ceil(dm_max/step)*step + step,
        step
    )

    df['DM_bins'] = pd.cut(df[dm], bins=bins, right=False)

    vc = df['DM_bins'].value_counts().sort_index()
    number_stars = vc.reset_index()
    number_stars.columns = ['DM_bins', 'number_stars']
    
    return df, number_stars


def density_calc(number_stars, dm_bins, num_stars, omega, step=0.2):
    for i in range(number_stars.shape[0]):

        centerDM = (number_stars[dm_bins][i].left + number_stars[dm_bins][i].right) / 2

        D_center = (10**((centerDM + 5)/5))

        rho = (number_stars[num_stars][i]) / (0.2 * np.log(10) * omega * step * D_center**3)

        number_stars.loc[i, 'rho'] = rho
    return number_stars

def density_to_df(df, dens_stars, dm_bins, rho_col):
    df = df.reset_index(drop=True)

    for i in range(df.shape[0]):
        for j in range(dens_stars.shape[0]):
            if df[dm_bins][i] == dens_stars[dm_bins][j]:
                df.loc[i, rho_col] = dens_stars[rho_col][j]
    
    return df

def avg_r_gc(df):
    for i in range(df.shape[0]):
        DM_center = (df['DM_bins'][i].left + df['DM_bins'][i].right) / 2
        df.loc[i, 'DM_center'] = DM_center
        df.loc[i, 'D(kpc)'] = 0.01 * 10**(0.2*df.loc[i, 'DM_center'])
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

def tri_dm_binning(df, step=0.2, dm='mu0'):
    df = df.copy()
    dm_min = df[dm].min()
    dm_max = df[dm].max()
    bins = np.arange(
        np.floor(dm_min/step)*step,
        np.ceil(dm_max/step)*step + step,
        step
    )

    df['DM_bins'] = pd.cut(df[dm], bins=bins, right=False)

    vc = df['DM_bins'].value_counts().sort_index()
    number_stars = vc.reset_index()
    number_stars.columns = ['DM_bins', 'number_stars']

    return df, number_stars

def avg_r_gc_trilegal(df):
    for i in range(df.shape[0]):
        bins = df['DM_bins'][i]
        DM_center = (bins.left + bins.right) / 2
        df.loc[i, 'DM_center'] = DM_center
        df.loc[i, 'D(kpc)'] = 0.01 * 10**(0.2*df.loc[i, 'DM_center'])
    return df
