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
# MAGNITUDE CALCULATIONS
# ============

# def nJy_to_mag(nJy):
#     '''
#     nJy to magnitude converter
#     '''
#     if np.isnan(nJy):
#         return np.nan
#     else:
#         return -2.5 * np.log10((nJy * 1e-9) / 3631)

def nJy_to_mag(nJy):
    '''
    nJy to magnitude converter
    '''
    return np.where(np.isnan(nJy), np.nan, -2.5 * np.log10((nJy * 1e-9) / 3631))

def mag_diff(cmodel, psf):
    '''
    Calculate magnitude difference between a model and PSF fluxes
    '''
    if not np.isfinite(cmodel) or not np.isfinite(psf):
        return np.nan
    if cmodel <= 0 or psf <= 0:
        return np.nan
    else:
        return 2.5 * np.log10(cmodel / psf)

def mag_diff(cmodel, psf):
    """
    Calculate the magnitude difference between cmodel and PSF fluxes.
    Works with scalars or array-like inputs (e.g., Pandas Series or NumPy arrays).

    Parameters
    ----------
    cmodel : float or array-like
        Model flux (e.g., cModel flux).
    psf : float or array-like
        PSF flux.

    Returns
    -------
    float or array-like
        Magnitude difference, or np.nan where input is invalid.
    """
    cmodel = np.asarray(cmodel)
    psf = np.asarray(psf)

    valid = (np.isfinite(cmodel) & np.isfinite(psf) & (cmodel > 0) & (psf > 0))
    result = np.full_like(cmodel, np.nan, dtype=np.float64)
    result[valid] = 2.5 * np.log10(cmodel[valid] / psf[valid])
    
    return result if isinstance(cmodel, (np.ndarray, pd.Series)) else result.item()


# def rel_err(psf_flux, psf_flux_err):
#     '''
#     Calculate the relative error of PSF flux
#     '''
#     if np.isnan(psf_flux) or np.isnan(psf_flux_err) or psf_flux == 0:
#         return np.nan
#     else:
#         return 1.09*(psf_flux_err / psf_flux)

def rel_err(psf_flux, psf_flux_err):
    '''
    Calculate the relative error of PSF flux
    '''
    # Use np.where to handle conditions element-wise
    result = np.where(
        np.isnan(psf_flux) | np.isnan(psf_flux_err) | (psf_flux == 0),
        np.nan,
        1.09 * (psf_flux_err / psf_flux)
    )
    return result

# def mag_calc(df):
#     '''
#     Calculate magnitudes and magnitude differences for given filters. LP: Slow, check it out.
#     '''
#     for col in df.columns:
#             if col.endswith((model_flux, psf_flux)):
#                 df[col + '_mag'] = df[col].apply(nJy_to_mag_v2)

#     for f in filters:
#         df[f'{f}{model_flux_diff}'] = df.apply(lambda row: mag_diff(row[f'{f}{model_flux}'], row[f'{f}{psf_flux}']), axis=1)
#         df[f'{f}{rel_error}'] = df.apply(lambda row: rel_err(row[f'{f}{psf_flux}'], row[f'{f}{psf_err}']), axis=1)
        
#     return df

def mag_calc(df):
    '''
    Calculate magnitudes and magnitude differences for given filters.
    '''
    # Precompute column names for magnitude conversion
    mag_cols = [col for col in df.columns if col.endswith((model_flux, psf_flux))]

    # Vectorized operation for magnitude conversion
    for col in mag_cols:
        df[col + '_mag'] = nJy_to_mag(df[col])

    # Precompute column names for magnitude differences and relative errors
    for f in filters:
        model_col = f'{f}{model_flux}'
        psf_col = f'{f}{psf_flux}'
        psf_err_col = f'{f}{psf_err}'

        # Vectorized operations for magnitude differences and relative errors
        df[f'{f}{model_flux_diff}'] = mag_diff(df[model_col], df[psf_col])
        df[f'{f}{rel_error}'] = rel_err(df[psf_col], df[psf_err_col])

    return df

def extcoeff():
        ## coefficients to correct for ISM dust (for S82 from Berry+2012, Table 1)
        ## extcoeff(band) = A_band / A_r 
        extcoeff = {}
        extcoeff['u'] = 1.810
        extcoeff['g'] = 1.400
        extcoeff['r'] = 1.000  # by definition
        extcoeff['i'] = 0.759 
        extcoeff['z'] = 0.561 
        ### adjust for Rubin
        extcoeff['y'] = 0.400  # educated guess (but ther is a reference!)
        ## note that Ar = (3.10/1.20)*ebv 
        return extcoeff 

# Ar = df['ebv']*3.10/1.20 - RUBIN
# Ar = df['av']/1.20

def mag_extinction(df, Ar):
     # colors uncorrected for interstellar extinction
    ug0 = df[f'u{model_flux_mag}'] - df[f'g{model_flux_mag}']
    gr0 = df[f'g{model_flux_mag}'] - df[f'r{model_flux_mag}']
    ri0 = df[f'r{model_flux_mag}'] - df[f'i{model_flux_mag}']
    iz0 = df[f'i{model_flux_mag}'] - df[f'z{model_flux_mag}']
    zy0 = df[f'z{model_flux_mag}'] - df[f'y{model_flux_mag}']
    # now correct for interstellar extinction (deredden)
    Cext = extcoeff()

    df.loc[:, 'ug'] = ug0 - (Cext['u'] - Cext['g'])*Ar
    df.loc[:, 'gr'] = gr0 - (Cext['g'] - Cext['r'])*Ar
    df.loc[:, 'ri'] = ri0 - (Cext['r'] - Cext['i'])*Ar
    df.loc[:, 'iz'] = iz0 - (Cext['i'] - Cext['z'])*Ar
    df.loc[:, 'zy'] = zy0 - (Cext['z'] - Cext['y'])*Ar
    df.loc[:, 'gi'] = df['gr'] + df['ri']
    df.loc[:, 'Ar'] = Ar

    return df