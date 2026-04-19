from astropy.table import Table
import matplotlib.pyplot as plt
import os
import pandas as pd
from functools import partial
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

# importing plotting and locus tools: 
import sys
sys.path.append('../src')
import modelLocusTools as mlt 
import magnitude_calc as mc
import SG_separation as sg
import dist_dens as dd
from config import *
# ============


def load_csv_dp1(file_path):
    '''
    Load fits file of ComCam area and return a pandas DataFrame.
    '''

    df = Table.read(file_path, format='fits').to_pandas()

    pattern_map = {
    r'_free_cModelFlux$': model_flux,
    r'_free_psfFlux$':    psf_flux,
    r'_free_cModelFluxErr$': model_err,
    r'_free_psfFluxErr$': psf_err
    }

    new_cols = (
    df.columns
      .to_series()
      .replace(pattern_map, regex=True)
    )
    df.columns = new_cols.values

    return df

def load_csv_tri(file_path, c):
    df = Table.read(file_path, format='fits').to_pandas()
    radius = 0.6*u.deg

    pattern_map = {r'mag$': model_flux_mag}

    new_cols = (
    df.columns
      .to_series()
      .replace(pattern_map, regex=True)
    )
    df.columns = new_cols.values

    df = df.rename(columns={'ra': ra_cord,
                        'dec': dec_cord,
                        'm_h': 'FeH',
                        'mu0': 'DM'
                        })

    t_coo = SkyCoord(df[ra_cord], df[dec_cord], unit='deg', frame='icrs')
    separation_t = t_coo.separation(c)
    inside_t = separation_t <= radius

    df = df[inside_t]#.reset_index(drop=True)
    return df

def magnitude_calculations(df, Ar, tri=False):
    '''
    Performing psf and model mag differences and accounting for extinction for all the colors.
    '''
    if tri:
        df = mc.mag_extinction(df, Ar)
        return df
    else:
        df = mc.mag_calc(df)
        df = mc.mag_extinction(df, Ar)
        return df


def selection_stars(df, morph_sep=True, tri=False):
    '''
    Selection of stars based on the mag_diff morphology and the stellar locus.
    '''
    df = df.copy()
    # if TRUE then star is within the range
    df['r_band_cut'] = df[f'r{model_flux_mag}'] < rBandFaintLimit

    print('after r band cut:', df['r_band_cut'].sum(), ' stars')

    if morph_sep:
        df = sg.morph_separation(df, SGcut=morph_cut)

    if tri:
        df = sg.is_S_MS(df, '../output/RubinStellarLocus_MS.txt', Rcolors, kSigma=2, flagName='isMS', tri=True)
    else:
        df = sg.is_S_MS(df, '../output/RubinStellarLocus_MS.txt', Rcolors, kSigma=2, flagName='isMS')

    #if morph_sep:
        #df = sg.morph_separation(df, SGcut=morph_cut)
    
    #legacy_mask = df['r_band_cut'] & df['gi_cut'] & df['isMS'] & (df['gi'] < 1.0) & df['SGcut']
    #print('final star count:', legacy_mask.sum(), ' stars')
    return df


def distance_density(df, feh=False, dm=False, extra_cut=False, step=0.2, correctDP1=True):
    df = dd.distance_calc(df, feh, dm, extra_cut=extra_cut, correctDP1=correctDP1)
    df = dd.coordinate_calc_galaxy(df)
    df, df_bins = dd.dm_binning(df, step=step)
    omega = dd.omega_calc(df)
    df_bins = dd.density_calc(df_bins, omega)
    
    return df, df_bins
