import numpy as np
import pylab as pl
from get_features import read_files, train_test, plot_prediction, add_asymmetry, read_files_nan
from fit_method import *

features, labels = read_files()
features = add_asymmetry(features, np.arange(3), np.arange(3))
features = add_asymmetry(features, 3+np.arange(3), 3+np.arange(3))
features = add_asymmetry(features, 6+np.arange(3), 6+np.arange(3))
features = add_asymmetry(features, 9+np.arange(3), 9+np.arange(3))
features_train, features_test, labels_train, labels_test = train_test(features, labels)
labels_train = np.squeeze(labels_train)
labels_test  = np.squeeze(labels_test)
labels = np.squeeze(labels)


features1, labels1, nan1 = read_files_nan()
x_low  = 0
y_low  = 0
x_high = 100
y_high = 100

features2 = np.zeros(features1.shape)
labels_true = np.zeros(labels1.shape)
mask = np.zeros((252,252))
mask[x_low:x_high, y_low:y_high] = 1
mask = mask.reshape(-1,1)
for i in xrange(features1.shape[0]):
	labels_true[i] = labels1[i]*(1. - mask[i])
	for j in xrange(features1.shape[1]):
		features2[i,j] = features1[i,j]*mask[i]
	
features3 = features2.copy()
features3 = add_asymmetry(features2, np.arange(3), np.arange(3))
features3 = add_asymmetry(features3, 3+np.arange(3), 3+np.arange(3))
features3 = add_asymmetry(features3, 6+np.arange(3), 6+np.arange(3))
features3 = add_asymmetry(features3, 9+np.arange(3), 9+np.arange(3))
features3[np.isnan(features3)] = 0

def make_plot(labels_pred, labels_true, method):
	logP = np.log(labels_pred.reshape(252,252))
	logT = np.log(labels_true.reshape(252,252))
	pl.title(method)
	pl.imshow(logP)
	pl.imshow(logT)
	pl.colorbar(label='log(T)')
	print method+" plotted."

pl.figure()
labels_pred = fit_NadarayaWatson(features_train, labels_train, features3, kernel='gaussian', alpha=0.01)
pl.subplot(223)
make_plot(labels_pred, labels_true, 'Nadaraya-Watson')

labels_pred = fit_Linear(features_train, labels_train, features3)
pl.subplot(221)
make_plot(labels_pred, labels_true, 'Linear')

labels_pred = fit_DecisionTree(features_train, labels_train, features3)
pl.subplot(224)
make_plot(labels_pred, labels_true, 'Decision Tree')

labels_pred = fit_KNeighbors(features_train, labels_train, features3, n_neighbors=15)
pl.subplot(222)
make_plot(labels_pred, labels_true, 'KNeighbors, k=15')



pl.savefig('./plots/cross-map.png')
pl.show()
