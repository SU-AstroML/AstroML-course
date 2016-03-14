import numpy as np
import scipy as sp

import scipy.interpolate as spi
import matplotlib.pyplot as plt


from astroML.datasets import fetch_sdss_specgals
from astroML.datasets import fetch_sdss_spectrum

from sklearn.manifold import Isomap

d_galaxies= fetch_sdss_specgals()

Ngals = 1000

k = 20
n = 2

d_galaxies = d_galaxies[0:Ngals]

#print d_galaxies
#print d_galaxies.shape 

mjd_galaxies = d_galaxies['mjd']
plate_galaxies = d_galaxies['plate']
fiberID_galaxies = d_galaxies['fiberID']

h_alpha_flux_galaxies = d_galaxies['h_alpha_flux']
extinction_r_galaxies = d_galaxies['extinction_r']
velDisp_galaxies = d_galaxies['velDisp']
z_galaxies = d_galaxies['z']
modelMag_u_galaxies = d_galaxies['modelMag_u']
modelMag_i_galaxies = d_galaxies['modelMag_i']
modelMag_r_galaxies = d_galaxies['modelMag_r']
modelMag_g_galaxies = d_galaxies['modelMag_g']

properties = [h_alpha_flux_galaxies,extinction_r_galaxies,velDisp_galaxies,z_galaxies,\
        modelMag_u_galaxies-modelMag_g_galaxies,modelMag_u_galaxies-modelMag_i_galaxies]
nproperites = ["H_alpha","extinction_r","velocity Dispersion","redshift","u-g","u-i"]

#print mjd_galaxies

spectra = []

for mjd,plate,fiberID in zip(mjd_galaxies,plate_galaxies,fiberID_galaxies):
    spectra.append(fetch_sdss_spectrum(plate,mjd,fiberID))

#print spectra[0].wavelength()
#print spectra[1].wavelength()

#nwav = len(spectra[0].wavelength())

wavs0 = spectra[0].wavelength()
wavmin = wavs0.min()
wavmax = wavs0.max()

nwav = 5000

wavs = np.linspace(wavmin,wavmax,nwav)

data = np.zeros((Ngals,nwav))

for i,s in enumerate(spectra):
    spec= s.spectrum
    wavsi = s.wavelength()
    intpol = spi.interp1d(wavsi,spec,bounds_error=False,fill_value = 0.)
    spec = intpol(wavs)
    spec/=spec.max()
    data[i]= spec

#print data

iso = Isomap(k,n)

#iso.fit(data)

print "projecting and fitting: "
proj = iso.fit_transform(data)

print "proj.shape"
print proj.shape

fig,axes = plt.subplots(2,3)


print proj[:,0]
print proj[:,1]
print proj

for prop,nprop,ax in zip(properties,nproperites,axes.flatten()):
    ax.set_title(nprop)
    ax.scatter(proj[:,0],proj[:,1],c=prop)

for i in range(2):
    axes[i,0].set_ylabel("ISO proj. ax 1")

for i in range(3):
    axes[-1,i].set_xlabel("ISO proj. ax 0")


#ax.scatter(proj[:,0],proj[:,1],color='k')

plt.show()









