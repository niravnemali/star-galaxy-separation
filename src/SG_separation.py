from astropy.table import Table
import matplotlib.pyplot as plt
import os
import pandas as pd
from functools import partial
import numpy as np

# importing plotting and locus tools: 
import sys
sys.path.append('../src')
import modelLocusTools as mlt 
from config import *

# ============

# STAR GALAXY SEPARATION

def morph_separation(df, SGcut=0.04):
    #dfS = df[df['dm3']<SGcut]
    #dfG = df[df['dm3']>SGcut]
    #df['SGcut'] = df['dm3'] < SGcut
    #dfS = df[df['SGcut']].copy()
    # ------

    df.loc[:, 'dm3'] = (df[f'g{model_flux_diff}']+df[f'r{model_flux_diff}']+df[f'i{model_flux_diff}'])/3.0 

    df['SGcut'] = False
    #cond = df['blue_MS'] & (df['dm3'] < SGcut)
    cond = df['r_band_cut'] & (df['dm3'] < SGcut)
    df.loc[cond, 'SGcut'] = True

    #print('from:', df['blue_MS'].sum(),' selected:', df['SGcut'].sum(), ' stars')
    print('from:', df['r_band_cut'].sum(),' selected:', df['SGcut'].sum(), ' stars')

    return df

def is_S_MS(df, locus_file, Rcolors, kSigma=2, flagName='isMS', tri=False):
    Rlocus = mlt.readRubinLocus(locus_file)

    giMinL = np.min(Rlocus['gi'])
    giMaxL = np.max(Rlocus['gi'])


    # incorporating the r and gi cuts into one mask
    if tri:
        df['gi_cut'] = (df['gi'] > giMinL) & (df['gi'] < giMaxL) & (df['r_band_cut']==True)
    else:
        df['gi_cut'] = (df['gi'] > giMinL) & (df['gi'] < giMaxL) & (df['SGcut']==True)
    print('after gi cut:', df['gi_cut'].sum(), ' stars')


    # choosing if stars are in the locus
    df = mlt.setInLocusFlag(df, flagName, Rcolors, Rlocus, kSigma) 
    print('after locus cut:', df[flagName].sum(), ' stars')

    # as only stars with both r and gi cuts were selected for isMS, now we only select blue stars
    df['blue_MS'] = df[flagName] & (df['gi'] < 1.0)
    print('after blue MS cut:', df['blue_MS'].sum(), ' stars')

    return df