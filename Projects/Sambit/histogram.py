import numpy as np
import pylab as pl
from get_features import read_files, train_test, plot_prediction, add_asymmetry
from fit_method import *
import matplotlib

matplotlib.rcParams.update({'font.size': 10})

features, labels = read_files()
features = add_asymmetry(features, np.arange(3), np.arange(3))
features = add_asymmetry(features, 3+np.arange(3), 3+np.arange(3))
features = add_asymmetry(features, 6+np.arange(3), 6+np.arange(3))
features = add_asymmetry(features, 9+np.arange(3), 9+np.arange(3))
features_train, features_test, labels_train, labels_test = train_test(features, labels)
labels_train = np.squeeze(labels_train)
labels_test  = np.squeeze(labels_test)

pl.figure('2D Histogram')
pl.suptitle('2D Histogram')

pl.subplot(331)
pl.title('Linear')
print "Linear Regression"
labels_pred = fit_Linear(features_train, labels_train, features_test)
H, xedges, yedges = plot_prediction(labels_pred, labels_test, method='Linear', xrang=[3000,13000], yrang=[3000,13000])
pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
pl.xlabel('Predicted')
pl.ylabel('True')
pl.colorbar()


pl.subplot(332)
pl.title('DecisionTree')
print "DecisionTree Regression"
labels_pred = fit_DecisionTree(features_train, labels_train, features_test)
H, xedges, yedges = plot_prediction(labels_pred, labels_test, method='DecisionTree', xrang=[3000,13000], yrang=[3000,13000])
pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
pl.xlabel('Predicted')
pl.ylabel('True')
pl.colorbar()

"""
pl.subplot(333)
pl.title('Polynomial')
print "Polynomial Regression"
labels_pred = fit_Polynomial(features_train, labels_train, features_test, order=3)
H, xedges, yedges = plot_prediction(labels_pred, labels_test, method='Polynomial', xrang=[3000,13000], yrang=[3000,13000])
pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
pl.xlabel('Predicted')
pl.ylabel('True')


pl.subplot(333)
pl.title('BasisFunc')
print "Basis Function Regression"
labels_pred = fit_BasisFunction(features_train, labels_train, features_test, kernel='gaussian')
H, xedges, yedges = plot_prediction(labels_pred, labels_test, method='BasisFunc', xrang=[3000,13000], yrang=[3000,13000])
pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
pl.xlabel('Predicted')
pl.ylabel('True')
"""
pl.subplot(333)
pl.title('Lasso')
print "Lasso Regression"
labels_pred = fit_Lasso(features_train, labels_train, features_test)
H, xedges, yedges = plot_prediction(labels_pred, labels_test, method='Lasso', xrang=[3000,13000], yrang=[3000,13000])
pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
pl.xlabel('Predicted')
pl.ylabel('True')
pl.colorbar()


pl.subplot(334)
pl.title('Ridge')
print "Ridge Regression"
alphas = 10**np.linspace(-10, 10, 100)
labels_pred = fit_Ridge(features_train, labels_train, features_test, alphas=alphas)
H, xedges, yedges = plot_prediction(labels_pred, labels_test, method='Ridge', xrang=[3000,13000], yrang=[3000,13000])
pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
pl.xlabel('Predicted')
pl.ylabel('True')
pl.colorbar()


pl.subplot(335)
pl.title('NadarayaWatson')
print "Nadaraya-Watson Regression"
labels_pred = fit_NadarayaWatson(features_train, labels_train, features_test, kernel='gaussian', alpha=0.01)
H, xedges, yedges = plot_prediction(labels_pred, labels_test, method='NadarayaWatson', xrang=[3000,13000], yrang=[3000,13000])
pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
pl.xlabel('Predicted')
pl.ylabel('True')
pl.colorbar()


pl.subplot(336)
pl.title('KNeighbors')
print "K-Neighbors Regression"
labels_pred = fit_KNeighbors(features_train, labels_train, features_test, n_neighbors=15)
H, xedges, yedges = plot_prediction(labels_pred, labels_test, method='KNeighbors', xrang=[3000,13000], yrang=[3000,13000])
pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
pl.xlabel('Predicted')
pl.ylabel('True')
pl.colorbar()


pl.subplot(337)
pl.title('RANSAC')
print "RANSAC Regression"
labels_pred = fit_RANSAC(features_train, labels_train, features_test)
H, xedges, yedges = plot_prediction(labels_pred, labels_test, method='RANSAC', xrang=[3000,13000], yrang=[3000,13000])
pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
pl.xlabel('Predicted')
pl.ylabel('True')
pl.colorbar()

pl.subplot(338)
pl.title('TheilSen')
print "TheilSen Regression"
labels_pred = fit_TheilSen(features_train, labels_train, features_test)
H, xedges, yedges = plot_prediction(labels_pred, labels_test, method='TheilSen', xrang=[3000,13000], yrang=[3000,13000])
pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
pl.xlabel('Predicted')
pl.ylabel('True')
pl.colorbar()

"""
pl.subplot(339)
pl.title('SVR')
print "SVR Regression"
labels_pred = fit_SVR(features_train, labels_train, features_test)
H, xedges, yedges = plot_prediction(labels_pred, labels_test, method='SVR', xrang=[3000,13000], yrang=[3000,13000])
pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
pl.xlabel('Predicted')
pl.ylabel('True')
pl.colorbar()

"""
pl.subplot(339)
pl.title('Polynomial')
print "Polynomial Regression"
labels_pred = fit_Polynomial(features_train, labels_train, features_test, order=3)
H, xedges, yedges = plot_prediction(labels_pred, labels_test, method='Polynomial', xrang=[3000,13000], yrang=[3000,13000])
pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
pl.xlabel('Predicted')
pl.ylabel('True')
pl.colorbar()


#pl.legend(loc=0)
pl.show()
