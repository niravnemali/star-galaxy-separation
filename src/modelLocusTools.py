import numpy as np
from astropy.table import Table, vstack
from scipy.stats import gaussian_kde
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy.interpolate import griddata
import pandas as pd

# taken from LocusTools.py
def readSDSSDSEDlocus(datafile, fixForStripe82=False): 
        ## Mr, as function of [Fe/H], along the SDSS/LSST stellar 
        ## for more details see the file header
        colnames = ['tLoc', 'Mr', 'FeH', 'ug', 'gr', 'ri', 'iz']
        LSSTlocus = Table.read(datafile, format='ascii', names=colnames)
        LSSTlocus['gi'] = LSSTlocus['gr'] + LSSTlocus['ri']
        if (fixForStripe82):
            print('Fixing input Mr-FeH-colors grid to agree with the SDSS v4.2 catalog')
            # for SDSS v4.2 catalog, see: http://faculty.washington.edu/ivezic/sdss/catalogs/stripe82.html
            # implement empirical corrections for u-g and i-z colors to make it better agree with the SDSS v4.2 catalog
            # fix u-g: slightly redder for progressively redder stars and fixed for gi>giMax
            ugFix = LSSTlocus['ug']+0.02*(2+LSSTlocus['FeH'])*LSSTlocus['gi']
            giMax = 1.8
            ugMax = 2.53 + 0.13*(1+LSSTlocus['FeH'])
            LSSTlocus['ug'] = np.where(LSSTlocus['gi']>giMax, ugMax , ugFix)
            # fix i-z color: small offsets as functions of r-i and [Fe/H]
            off0 = 0.08
            off2 = -0.09
            off5 = 0.008
            offZ = 0.01
            Z0 = 2.5
            LSSTlocus['iz'] += off0*LSSTlocus['ri']+off2*LSSTlocus['ri']**2+off5*LSSTlocus['ri']**5
            LSSTlocus['iz'] += offZ*(Z0+LSSTlocus['FeH'])
        return LSSTlocus

    
def readRubinLocus(datafile): 
        ## for MS locus (e.g. RubinStellarLocus_MS.txt)
        colnames = ['gi', 'Npts', 'ugL', 'ugS', 'grL', 'grS', 'riL', 'riS', 'izL', 'izS', 'zyL', 'zyS']
        locus = Table.read(datafile, format='ascii', names=colnames)
        return locus
   
### transformation from (r, g-i) by SDSS to Gaia's G and BpRp
### uses 3-rd order polynomial, rms = 0.02 mag, peak deviation < 0.05 mag 
def SDSS2Gaia(r, gi):
    # transform from SDSS (r, gi) to Gaia (G, BpRp)
    thetaG = np.array([-0.04500009,  0.12475687,  0.02505529, -0.05476676])
    G = r + polynomial_fit(thetaG, gi)
    thetaBpRp = np.array([ 0.48681787,  0.84121373, -0.19072356,  0.06582643])
    BpRp = polynomial_fit(thetaBpRp, gi)
    return G, BpRp

# this function computes polynomial models given some data x
# and parameters theta
def polynomial_fit(theta, x):
    """Polynomial model of degree (len(theta) - 1)"""
    return sum(t * x ** n for (n, t) in enumerate(theta))


def interpolateLocus(gi, locus, color):
    interpFunc = interpolate.interp1d(locus['gi'], locus[color+'L'], kind='nearest-up')
    locusColor = interpFunc(gi)
    interpFunc = interpolate.interp1d(locus['gi'], locus[color+'S'], kind='nearest-up')
    locusWidth = interpFunc(gi)
    return locusColor, locusWidth

def setInLocusFlag(df, flagName, colors, locus, kSigma=2.0, allow_nan=True):

    gi_mask = df['gi_cut'].to_numpy()
    gi_idx  = df.index[gi_mask]

    # init: False everywhere, True on gi rows
    df[flagName] = False
    df.loc[gi_idx, flagName] = True

    gi_vals = df.loc[gi_idx, 'gi'].to_numpy()

    for color in colors:
        if color not in df.columns:
            continue  # silently skip missing color column

        obs = df.loc[gi_idx, color].to_numpy()
        loc_c, loc_w = interpolateLocus(gi_vals, locus, color)

        cdev = np.abs((obs - loc_c) / loc_w)

        if allow_nan:
            fail = (cdev > kSigma) & ~np.isnan(cdev)   # NaNs pass (legacy)
        else:
            fail = (cdev > kSigma) |  np.isnan(cdev)   # strict

        if np.any(fail):
            df.loc[gi_idx[fail], flagName] = False

    return df


def photoFeH(ug, gr, correctDP1=True): 
        x = np.array(ug)
        y = np.array(gr)
        ## correct for Rubin DP1 vs. SDSS color terms
        if correctDP1:
                meanRI = 0.2 
                y = 1.058 * y + 0.058*meanRI - 0.002
                print('photoFeH: applied Rubin DP1 vs. SDSS color term')
                # NB no correction for the u-g color yet...
                # x = x + 0.08  # brings the halo FeH distribution means to about -1.5
                # x = x - 0.10  # this is definitely bad as it shifts FeH to below -2.5
                x = x + 0.15    # prediction by Lynne Jones based on Kurucuz models, it overcorrets
        ## photometric metallicity introduced in Ivezic et al. (2008), Tomography II
        ## and revised in Bond et al. (2012), Tomography III (see Appendix A.1) 
        ## valid for SDSS bands and F/G stars defined by 
        ## 0.2 < g−r < 0.6 and −0.25 + 0.5*(u−g) < g−r < 0.05 + 0.5*(u−g)
        A, B, C, D, E, F, G, H, I, J = (-13.13, 14.09, 28.04, -5.51, -5.90, -58.68, 9.14, -20.61, 0.0, 58.20)
        return A + B*x + C*y + D*x*y + E*x**2 + F*y**2 + G*x**2*y + H*x*y**2 + I*x**3 + J*y**3 

def getMr(giIN, FeH, correctDP1=True):

        ## correct for Rubin DP1 vs. SDSS color terms
        if correctDP1:
                gi = 1.054 * giIN + 0.014 
                print('getMr: applied Rubin DP1 vs. SDSS color term')
        else:
                gi = giIN 
                
        ## Mr(g-i, FeH) introduced in Ivezic et al. (2008), Tomography II
        MrFit = -5.06 + 14.32*gi -12.97*gi**2 + 6.127*gi**3 -1.267*gi**4 + 0.0967*gi**5
        ## based on Gaia parallax sample, the following correction should be added
        if (False): 
             MrCorr = 12.5*(0.4-gi)**2 - 0.03
             MrCorr = np.where(gi>0.4, 0.0, MrCorr)
             MrCorr = np.where(gi<0.5, MrCorr-0.1, MrCorr+0.01)
             MrFit = MrFit + MrCorr
        ## offset for metallicity, valid for -2.5 < FeH < 0.2
        FeHoffset = 4.50 -1.11*FeH -0.18*FeH**2
        return MrFit + FeHoffset
