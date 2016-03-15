import numpy as np
import pylab as pl
from sklearn.manifold import LocallyLinearEmbedding
from astroML.datasets import fetch_sdss_specgals
from astroML.datasets import fetch_sdss_spectrum

data = fetch_sdss_specgals()
print data.dtype.names
ngals = 326
nwavel = 3855
plates = data['plate'][:ngals]
mjds = data['mjd'][:ngals]
fiberIDs = data['fiberID'][:ngals]
h_alpha = data['h_alpha_flux'][:ngals]
bptclass = data['bptclass'][:ngals]
specdata = np.zeros((ngals, nwavel))

i = 0
for plate, mjd, fiberID in zip(plates, mjds, fiberIDs):
    tempdata = fetch_sdss_spectrum(plate, mjd, fiberID)
    specdata[i, :] = tempdata.spectrum/tempdata.spectrum.mean()
    i += 1

# Apply LLE
k = 7
for fignum, n in enumerate([2, 3]):
    lle = LocallyLinearEmbedding(k, n)
    lle.fit(specdata)
    proj = lle.transform(specdata)
    pl.subplot(2, 1, fignum+1)
    pl.scatter(proj[:,0], proj[:,1], c=bptclass, s=50)
pl.colorbar()
pl.show()