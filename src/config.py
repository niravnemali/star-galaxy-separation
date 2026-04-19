filters = ['u', 'g', 'r', 'i', 'z', 'y']
fields = ['ECDFS', 'EDFS', 'Rubin_SV_95_25']

model_flux = '_modelFlux'
psf_flux = '_psfFlux'
model_err = '_modelErr'
psf_err = '_psfErr'

model_flux_mag = f'{model_flux}_mag'
psf_flux_mag = f'{psf_flux}_mag'
model_flux_diff = '_diff'
rel_error = '_psfRelErr'

ra_cord = 'coord_ra'
dec_cord = 'coord_dec'

rBandFaintLimit = 24.0 
Rcolors = ['ug', 'gr', 'ri', 'iz', 'zy']

morph_cut = 0.04

# parameter for color selection around the main stellar locus
kSigma = 2.0
# Mr cuts for blue stars; incompleteness starts at DMinc = rBandFaintLimit - MrFaint
# and no stars further than DMmax = rBandFaintLimit - MrBright are selected
# grRedCut is added to ensure FeH estimates are reliable
MrBright = 4.0
MrFaint = 5.5
grRedCut = 0.60
