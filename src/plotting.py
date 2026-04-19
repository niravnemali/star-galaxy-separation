from astropy.table import Table
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font = FontProperties()


import pandas as pd
from functools import partial
import numpy as np
np.random.seed(42)

import sys
sys.path.insert(0,'../src/')
np.random.seed(42)

import data_processing as dp
from config import *

def plot_mag_diff(df, name, extent=[-0.1, 1.5, 16, 25], cmap='cividis', file_path='../plots/', filters=['g', 'r', 'i']):
    fig, ax = plt.subplots(2,2, figsize=(11,9))
    ax = ax.flatten()

    for i,j in enumerate(filters):
        h = ax[i].hexbin(df[f'{j}{model_flux_diff}'], df[f'{j}{model_flux_mag}'], mincnt=1, gridsize=300, 
                        bins='log',extent=extent,
                        vmin=1, vmax=100, cmap=cmap
                        )
        cbar = fig.colorbar(h, ax=ax[i], cmap=cmap)
        ax[i].set_title(j+' band', fontsize=18)
        ax[i].set_xlabel(r"PSF - CModel mag", fontsize=16)
        ax[i].set_ylabel(r"CModel mag", fontsize=16)
        ax[i].grid(True)
        ax[i].set_xlim(extent[0], extent[1])
        ax[i].set_ylim(extent[2], extent[3])
        ax[i].invert_yaxis()
        cbar.set_label('stars per bin', fontsize=12)

    band3 = (df[f'g{model_flux_diff}'] + df[f'r{model_flux_diff}'] + df[f'i{model_flux_diff}']) / 3
    mag3 = (df[f'g{model_flux_mag}'] + df[f'r{model_flux_mag}'] + df[f'i{model_flux_mag}']) / 3

    ax[3].hexbin(band3, df[f'r{model_flux_mag}'], mincnt=1, gridsize=300, 
                bins='log',extent=extent,
                vmin=1, vmax=100, cmap=cmap
                )

    cbar = fig.colorbar(h, ax=ax[3], cmap=cmap)
    ax[3].set_title('3 band (gri) average', fontsize=18)
    ax[3].set_xlabel(r"PSF - CModel mag", fontsize=16)
    ax[3].set_ylabel(r"r CModel mag", fontsize=16)
    ax[3].grid(True)
    ax[3].set_xlim(extent[0], extent[1])
    ax[3].set_ylim(extent[2], extent[3])
    ax[3].invert_yaxis()
    cbar.set_label('stars per bin', fontsize=12)

    plt.tight_layout()
    plt.savefig(file_path+name, dpi=300)
    plt.show()


## minor changes to plot_mag_diff to make the zoomed in plot (it could be generalized but ZI was lazy) 
def plot_mag_diff2(df, name, extent=[-0.1, 0.15, 14, 25], cmap='cividis', file_path='../plots/', filters=['g', 'r', 'i']):
    fig, ax = plt.subplots(2,2, figsize=(11,9))
    ax = ax.flatten()

    for i,j in enumerate(filters):
        h = ax[i].hexbin(df[f'{j}{model_flux_diff}'], df[f'{j}{model_flux_mag}'], mincnt=1, gridsize=300, 
                        bins='log',extent=extent,
                        vmin=1, vmax=10, cmap=cmap
                        )
        cbar = fig.colorbar(h, ax=ax[i], cmap=cmap)
        ax[i].set_title(j+' band', fontsize=18)
        ax[i].set_xlabel(r"PSF - CModel mag", fontsize=16)
        ax[i].set_ylabel(r"CModel mag", fontsize=16)
        ax[i].grid(True)
        ax[i].set_xlim(-0.1, extent[1])
        ax[i].set_ylim(16, 25)
        ax[i].invert_yaxis()
        cbar.set_label('stars per bin', fontsize=12)

    band3 = (df[f'g{model_flux_diff}'] + df[f'r{model_flux_diff}'] + df[f'i{model_flux_diff}']) / 3
    mag3 = (df[f'g{model_flux_mag}'] + df[f'r{model_flux_mag}'] + df[f'i{model_flux_mag}']) / 3

    ax[3].hexbin(band3, df[f'r{model_flux_mag}'], mincnt=1, gridsize=300, 
                bins='log',extent=extent,
                vmin=1, vmax=10, cmap=cmap
                )

    cbar = fig.colorbar(h, ax=ax[3], cmap=cmap)
    ax[3].set_title('3 band (gri) average', fontsize=18)
    ax[3].set_xlabel(r"PSF - CModel mag", fontsize=16)
    ax[3].set_ylabel(r"r CModel mag", fontsize=16)
    ax[3].grid(True)
    ax[3].set_xlim(-0.1, extent[1])
    ax[3].set_ylim(16, 25)
    ax[3].invert_yaxis()
    ax[3].plot([0.04, 0.04], [16.5, 24.8], c='red', lw=2)
    ax[3].plot([0.016, 0.016], [16.5, 24.8], c='red', ls='--', lw=2)
    cbar.set_label('stars per bin', fontsize=12)

    plt.tight_layout()
    plt.savefig(file_path+name, dpi=300)
    plt.show()

# plot for Leanne's DP1 paper 
def plot_mag_diffDP1(df, name, extent=[-0.1, 1.5, 14, 25], cmap='cividis', file_path='../plots/', filters=['g', 'r']):
    fig, ax = plt.subplots(2,2, figsize=(11,9))
    ax = ax.flatten()

    for i,j in enumerate(filters):
        h = ax[i].hexbin(df[f'{j}{model_flux_diff}'], df[f'{j}{model_flux_mag}'], mincnt=1, gridsize=300, 
                        bins='log',extent=extent,
                        vmin=1, vmax=100, cmap=cmap
                        )
        cbar = fig.colorbar(h, ax=ax[i], cmap=cmap)
        ax[i].set_title(j+' band', fontsize=18)
        ax[i].set_xlabel(r"PSF mag - CModel mag", fontsize=16)
        ax[i].set_ylabel(r"CModel mag", fontsize=16)
        ax[i].grid(True)
        ax[i].set_xlim(-0.1, 1.5)
        ax[i].set_ylim(16, 25)
        ax[i].invert_yaxis()
        ax[i].plot([0.016, 0.016], [16.5, 24.8], c='red', ls='--', lw=2)
        cbar.set_label('star count', fontsize=12)


    df['rmag'] = df[f'r{model_flux_mag}']
    df['gr'] = df[f'g{model_flux_mag}'] - df[f'r{model_flux_mag}']
    dmag = df[f'r{model_flux_diff}']
    dfS = df[dmag<=0.016]
    dfG = df[dmag>0.016]
    print('all:', len(df), ' S:', len(dfS), ' G:', len(dfG))
    
    ax[2].hexbin(dfS['gr'], dfS['rmag'], mincnt=1, gridsize=300, 
                bins='log',extent=extent,
                vmin=1, vmax=100, cmap=cmap
                )
    cbar = fig.colorbar(h, ax=ax[2], cmap=cmap)
    ax[2].set_title('unresolved sources', fontsize=18)
    ax[2].set_xlabel(r"g-r (CModel mags)", fontsize=16)
    ax[2].set_ylabel(r"r CModel mag", fontsize=16)
    ax[2].grid(True)
    ax[2].set_xlim(-0.1, 1.5)
    ax[2].set_ylim(16, 25)
    ax[2].invert_yaxis()
    cbar.set_label('star count', fontsize=12)

    ax[3].hexbin(dfG['gr'], dfG['rmag'], mincnt=1, gridsize=300, 
                bins='log',extent=extent,
                vmin=1, vmax=100, cmap=cmap
                )
    cbar = fig.colorbar(h, ax=ax[3], cmap=cmap)
    ax[3].set_title('resolved sources', fontsize=18)
    ax[3].set_xlabel(r"g-r (CModel mags)", fontsize=16)
    ax[3].set_ylabel(r"r CModel mag", fontsize=16)
    ax[3].grid(True)
    ax[3].set_xlim(-0.1, 1.5)
    ax[3].set_ylim(16, 25)
    ax[3].invert_yaxis()
    cbar.set_label('star count', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(file_path+name, dpi=300)
    plt.show()



## compare CMDs for unresolved and resolved objects in the 3 DP1 fields
def plot_compareCMDs(data, name, fields=[0, 2], extent=[-0.1, 1.5, 14, 25], cmap='cividis', file_path='../plots/', field_names=['ECDFS', 'EDFS', 'SV 95 -25']):
    fig, ax = plt.subplots(2,2, figsize=(11,9))
    ax = ax.flatten()        

    jrow = 0 
    for j in fields:
        dfA = data[j]
        df = dfA[dfA[f'r{model_flux_mag}']<25.0]
        df['rmag'] = df[f'r{model_flux_mag}']
        df['gr'] = df[f'g{model_flux_mag}'] - df[f'r{model_flux_mag}']
        dmag = df[f'r{model_flux_diff}']
        dfS = df[dmag<=0.016]
        dfG = df[dmag>0.016]
        print('row=', jrow, 'all:', len(df), ' S:', len(dfS), ' G:', len(dfG))
        print(' for r<24:')
        dfB = dfA[dfA[f'r{model_flux_mag}']<24.0]
        dfB['rmag'] = dfB[f'r{model_flux_mag}']
        dfB['gr'] = dfB[f'g{model_flux_mag}'] - dfB[f'r{model_flux_mag}']
        dmagB = dfB[f'r{model_flux_diff}']
        dfSB = dfB[dmag<=0.016]
        dfGB = dfB[dmag>0.016]
        print('       all:', len(dfB), ' S:', len(dfSB), ' G:', len(dfGB))
        
        jp = jrow
        h = ax[jp].hexbin(dfS['gr'], dfS['rmag'], mincnt=1, gridsize=300, 
                bins='log',extent=extent,
                vmin=1, vmax=100, cmap=cmap
                )
        cbar = fig.colorbar(h, ax=ax[jp], cmap=cmap)
        ax[jp].set_xlabel(r"g-r (CModel mags)", fontsize=16)
        ax[jp].set_ylabel(r"r CModel mag", fontsize=16)
        ax[jp].grid(True)
        ax[jp].set_xlim(-0.1, 1.5)
        ax[jp].set_ylim(16, 25)
        ax[jp].invert_yaxis()
        cbar.set_label('stars per bin', fontsize=12)
        if (jrow==0) or (jrow==1): ax[jp].set_title(f'{field_names[j]} \nunresolved sources', fontsize=18)

        jp = jrow + 2 
        h = ax[jp].hexbin(dfG['gr'], dfG['rmag'], mincnt=1, gridsize=300, 
                bins='log',extent=extent,
                vmin=1, vmax=100, cmap=cmap
                )
        cbar = fig.colorbar(h, ax=ax[jp], cmap=cmap)
        ax[jp].set_xlabel(r"g-r (CModel mags)", fontsize=16)
        ax[jp].set_ylabel(r"r CModel mag", fontsize=16)
        ax[jp].grid(True)
        ax[jp].set_xlim(-0.1, 1.5)
        ax[jp].set_ylim(16, 25)
        ax[jp].invert_yaxis()
        cbar.set_label('stars per bin', fontsize=12)
        if (jrow==0) or (jrow==1): ax[jp].set_title('resolved sources', fontsize=18)
        jrow = jrow + 1 
    
    plt.tight_layout()
    plt.savefig(file_path+name, dpi=300)
    plt.show()


    
    
def hist_mag_diff(df, diff_col, model_col, name, file_path='../plots/'):
    fig, ax = plt.subplots(3,2, figsize=(11,15))
    ax = ax.flatten()
    # < 21
    df['3band_diff'] = (df[f'g_{diff_col}'] + df[f'r_{diff_col}'] + df[f'i_{diff_col}']) / 3
    df['3band_mag'] = (df[f'g_{model_col}'] + df[f'r_{model_col}'] + df[f'i_{model_col}']) / 3

    df_21 = df[df['3band_mag'] < 21]

    diff21 = df_21['3band_diff']

    diff21 = diff21[np.isfinite(diff21)]

    ax[0].hist(diff21, bins=500, color='blue', histtype='step', density=True)
    ax[0].set_title('Slice: <21')
    ax[0].set_xlabel(r"PSF - Model mag")
    ax[0].set_ylabel('Density')

    # 21-22
    df_22 = df[(df['3band_mag'] >= 21) & (df['3band_mag'] < 22)]

    diff22 = df_22['3band_diff']

    diff22 = diff22[np.isfinite(diff22)]

    ax[1].hist(diff22, bins=500, color='blue', histtype='step', density=True)
    ax[1].set_xlim(-0.1, 0.2)
    ax[1].set_title('Slice: 21-22')
    ax[1].set_xlabel(r"PSF - Model mag")
    ax[1].set_ylabel('Density') 

    # 22-23
    df_22_23 = df[(df['3band_mag'] >= 22) & (df['3band_mag'] < 23)]
    diff_22_23 = df_22_23['3band_diff']
    diff_22_23 = diff_22_23[np.isfinite(diff_22_23)]

    ax[2].hist(diff_22_23, bins=500, color='blue', histtype='step', density=True)
    ax[2].set_xlim(-0.1, 0.2)
    ax[2].set_title('Slice: 22-23')
    ax[2].set_xlabel(r"PSF - Model mag")
    ax[2].set_ylabel('Density') 

    # 23-23.5
    df_23_23_5 = df[(df['3band_mag'] >= 23) & (df['3band_mag'] < 23.5)]
    diff_23_23_5 = df_23_23_5['3band_diff']
    diff_23_23_5 = diff_23_23_5[np.isfinite(diff_23_23_5)]

    ax[3].hist(diff_23_23_5, bins=500, color='blue', histtype='step', density=True)
    ax[3].set_xlim(-0.1, 0.2)
    ax[3].set_title('Slice: 23-23.5')
    ax[3].set_xlabel(r"PSF - Model mag")
    ax[3].set_ylabel('Density') 

    # 23.5-24
    df_23_5_24 = df[(df['3band_mag'] >= 23.5) & (df['3band_mag'] < 24)]
    diff_23_5_24 = df_23_5_24['3band_diff']
    diff_23_5_24 = diff_23_5_24[np.isfinite(diff_23_5_24)]

    ax[4].hist(diff_23_5_24, bins=500, color='blue', histtype='step', density=True)
    ax[4].set_xlim(-0.1, 0.2)
    ax[4].set_title('Slice: 23.5-24')
    ax[4].set_xlabel(r"PSF - Model mag")
    ax[4].set_ylabel('Density') 

    # 24-25
    df_24_25 = df[(df['3band_mag'] >= 24) & (df['3band_mag'] < 25)]
    diff_24_25 = df_24_25['3band_diff']
    diff_24_25 = diff_24_25[np.isfinite(diff_24_25)]

    ax[5].hist(diff_24_25, bins=500, color='blue', histtype='step', density=True)
    ax[5].set_xlim(-0.1, 0.2)
    ax[5].set_title('Slice: 24-25')
    ax[5].set_xlabel(r"PSF - Model mag")
    ax[5].set_ylabel('Density') 

    plt.tight_layout()
    plt.savefig(file_path+name, dpi=300)
    plt.show()

def power_law(df, q, n, rho0, ref_id):
    #ref_id = df.index[df['rho'] == rho0].tolist()[0]

    S0 = ((df.loc[ref_id, 'X']**2
         + df.loc[ref_id, 'Y']**2)
        + (df.loc[ref_id, 'Z']/q)**2)**(n/2)
    
    start = rho0/S0

    log_start = np.log10(start)

    S = (df['X']**2 + df['Y']**2) + (df['Z']/q)**2
    return log_start + (n/2) * np.log10(S)

def feH_distance(df, ax, fig, i, field, g11, g12, g21, g22, tri=False, sigmax=0.2, sigmay=0.05, cmap='cividis'):
    if tri:
        df = df[(df['FeH'] < 0) & (df['FeH'] > -2.10)].copy()
        extent = [5, 40, -2.1, 0]
    else:
        lowFeH = -3.0
        df = df[(df['FeH'] < 0) & (df['FeH'] > lowFeH)].copy()
        extent = [5, 40, lowFeH, 0]
    if tri:
        noise_x = np.random.normal(loc=0.0, scale=sigmax, size=len(df))
        noise_y = np.random.normal(loc=0.0, scale=sigmay, size=len(df))

        df['r_gc'] += noise_x
        df['FeH'] += noise_y

    ax[i].set_title(f'Field: {field}', fontsize=20)
    ax[i].grid()
    h = ax[i].hexbin(df['r_gc'], df[f'FeH'], mincnt=1, gridsize=200, 
                        bins='log',extent=extent,
                        vmin=1, vmax=10, cmap=cmap
                        )
    cbar = fig.colorbar(h, ax=ax[i], cmap=cmap)
    #ax[i].scatter(df['Dkpc'], df['FeH'], s=1.0, color='blue', alpha=0.8)
    #ax[i].plot([g11, g11], [-3.5, 0], color='red', lw=2.0, ls='--', label=f'{g11} kpc')
    #ax[i].plot([g21, g21], [-3.5, 0], color='orange', lw=2.0, ls='--', label=f'{g21} kpc')
    ax[i].set_xlim(5, 40)
    if tri:
        ax[i].set_ylim(-2.1, 0)
    ax[i].set_xlabel('Galactocentric distance (kpc)', fontsize=18)
    ax[i].set_ylabel('[Fe/H]', fontsize=18)

    ax[i+1].grid()
    group1 = df[(df['r_gc'] < g12) & (df['r_gc'] > g11)]
    group2 = df[(df['r_gc'] < g22) & (df['r_gc'] > g21)]
    ax[i+1].hist(group1['FeH'], bins='auto', color='#002554', alpha=1.0, label=f'{g11}-{g12} kpc', histtype='step', lw=1.3, density=True)
    ax[i+1].hist(group2['FeH'], bins='auto', color='#daad38', alpha=1.0, label=f'{g21}-{g22} kpc', histtype='step', lw=1.3, density=True)
    if tri:
        ax[i+1].set_xlim(-2.5, 0)
    else:
        ax[i+1].set_xlim(lowFeH, 0)
    cbar.set_label('stars per bin', fontsize=14)
    ax[i+1].set_xlabel('[Fe/H]', fontsize=18)
    ax[i+1].set_ylabel('Count', fontsize=18)
    ax[i+1].legend(fontsize=14)

def plot_hist_density_log(df, bins, ax, i, rho0, index, q, n1, n2, limit_down, limit_upp, field, xlim1=20, xlim2=55):
    ax[i].set_title(f'Field: {field}', fontsize=16)
    ax[i].grid()
    ax[i].hist(df['DM'], bins='auto', range=(0,300), histtype='step', lw=1, color='#002554')
    ax[i].set_xlim(8, xlim1)
    ax[i].set_xlabel('Distance modulus (mag)', fontsize=14)
    ax[i].set_ylabel('# of stars', fontsize=16)

    ax[i+1].grid()

    bins_r_gc = bins.loc[pd.to_numeric(bins['r_gc_median'], errors='coerce') > 10].copy()

    ax[i+1].errorbar(bins_r_gc['r_gc_median'], np.log10(bins_r_gc['rho']), yerr=[(bins_r_gc['lower_error']), bins_r_gc['upper_error']] ,fmt='.', capsize=3, label='Density of stars', color='#595f6e')
    ## models 

    
    # "best-fit" model 
    x = df['r_gc']
    y = power_law(df, q, n1, rho0, index)
    xOK = x[(x>=limit_down)&(x<limit_upp)]
    yOK = y[(x>=limit_down)&(x<limit_upp)]
    ax[i+1].plot(xOK, yOK, c='#da4138', lw=1.2, label=f'q={q}, n={n1}')
    y = power_law(df, 1.0, n2, rho0, index)
    xOK = x[(x>=limit_down)&(x<limit_upp)]
    yOK = y[(x>=limit_down)&(x<limit_upp)]
    ax[i+1].plot(xOK, yOK, c='#da4138', lw=1.2, ls='--', label=f'q=1.0, n={n2}')
    yMax = np.max(yOK) 


    # TRILEGAL model
    x = df['r_gc']
    y = power_law(df, 0.62, -2.75, rho0, index)
    xOK = x[(x>=limit_down)&(x<limit_upp)]
    yOK = y[(x>=limit_down)&(x<limit_upp)]
    yOK = yMax / np.max(yOK) * yOK
    ax[i+1].plot(xOK, yOK, c='#e1ca54', lw=1.2,label=f'q=0.62, n=-2.75') 

    
    ax[i+1].legend(loc='upper right', fontsize=12)
    ax[i+1].set_xlim(0, xlim2)
    ax[i+1].set_ylim(-10, -3.5)
    ax[i+1].set_xlabel('Galactocentric distance (kpc)', fontsize=14)
    ax[i+1].set_ylabel('$\log_{10}[\\rho$ (stars pc$^{-3}$)]', fontsize=16)



# lazy ZI added a specialized version for TRILEGAL     
def plot_hist_density_TRI(df, bins, ax, i, rho0, index, q, n1, n2, limit_down, limit_upp, field, xlim1=20, xlim2=55):
    ax[i].set_title(f'Field: {field}', fontsize=16)
    ax[i].grid()
    ax[i].hist(df['DM'], bins='auto', range=(0,300), histtype='step', lw=1, color='#002554')
    ax[i].set_xlim(8, xlim1)
    ax[i].set_xlabel('Distance modulus (mag)', fontsize=14)
    ax[i].set_ylabel('# of stars', fontsize=16)

    ax[i+1].grid()

    bins_r_gc = bins.loc[pd.to_numeric(bins['r_gc_median'], errors='coerce') > 10].copy()

    ax[i+1].errorbar(bins_r_gc['r_gc_median'], np.log10(bins_r_gc['rho']), yerr=[(bins_r_gc['lower_error']), bins_r_gc['upper_error']] ,fmt='.', capsize=3, label='Density of stars', color='#595f6e')
    ## models 
   
    # "best-fit" model 
    x = df['r_gc']

    y = power_law(df, q, -n1, rho0, index)
    xOK = x[(x>=limit_down)&(x<limit_upp)]
    yOK = y[(x>=limit_down)&(x<limit_upp)]
    ax[i+1].plot(xOK, yOK, c='#e1ca54', lw=1.2, label=f'q={q}, n={n1}')
    yMax = np.max(yOK) 
    
    y = power_law(df, 1.0, -n2, rho0, index)
    xOK = x[(x>=limit_down)&(x<limit_upp)]
    yOK = y[(x>=limit_down)&(x<limit_upp)]
    yOK = yMax / np.max(yOK) * yOK
    ax[i+1].plot(xOK, yOK, c='#daad38', lw=1.2, ls='--', label=f'q=1.0, n={n2}')

    ax[i+1].legend(loc='upper right', fontsize=12)
    ax[i+1].set_xlim(0, xlim2)
    ax[i+1].set_ylim(-10, -3.5)
    ax[i+1].set_xlabel('Galactocentric distance (kpc)', fontsize=14)
    ax[i+1].set_ylabel('$\log_{10}[\\rho$ (stars pc$^{-3}$)]', fontsize=16)
