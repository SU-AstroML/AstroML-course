import numpy as np
import pylab as pl
from get_features import read_files, train_test, plot_prediction, add_asymmetry, read_files_nan
from fit_method import *


features, labels, nan = read_files_nan()

pl.figure('Features')
pl.subplot(231)
pl.title('k2v')
pl.imshow(np.log(features[:,0]).reshape(252,252))
pl.colorbar(label='Log Intensity')
pl.subplot(232)
pl.title('k3')
pl.imshow(np.log(features[:,1]).reshape(252,252))
pl.colorbar(label='Log Intensity')
pl.subplot(233)
pl.title('k2r')
pl.imshow(np.log(features[:,2]).reshape(252,252))
pl.colorbar(label='Log Intensity')
pl.subplot(234)
pl.title('k2v')
pl.imshow(features[:,6].reshape(252,252))
pl.colorbar(label='Doppler shift')
pl.subplot(235)
pl.title('k3')
pl.imshow(features[:,7].reshape(252,252))
pl.colorbar(label='Doppler shift')
pl.subplot(236)
pl.title('k2r')
pl.imshow(features[:,8].reshape(252,252))
pl.colorbar(label='Doppler shift')

pl.show()
