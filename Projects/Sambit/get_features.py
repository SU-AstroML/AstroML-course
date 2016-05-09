import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.gaussian_process import GaussianProcess
from sklearn.cross_validation import train_test_split
import pylab as pl
from helper_functions import get_features_matrix, get_values, remove_nan, fit_Ridge, fit_Lasso

# Read the .npy files
def read_files():
	k_lam0, h_lam0 = 279.63509493e0, 280.35297192e0 # nm
	
	intensity_files_k = np.array(['k2r_i', 'k3_i', 'k2v_i'])
	intensity_files_h = np.array(['h2r_i', 'h3_i', 'h2v_i'])
	freqshift_files_k = np.array(['k2r_dv', 'k3_dv', 'k2v_dv'])
	freqshift_files_h = np.array(['h2r_dv', 'h3_dv', 'h2v_dv'])
	
	intensity = np.hstack((get_values(intensity_files_k),get_values(intensity_files_h)))
	freqshift = np.hstack((get_values(freqshift_files_k, orig=k_lam0),get_values(freqshift_files_h, 	orig=h_lam0)))
	features_train_ = np.hstack((intensity, freqshift))
	labels_train_   = np.load('avg_T.npy')
	labels_train    = labels_train_.reshape(-1,1)
	
	nan_map = labels_train.copy()
	
	for i in xrange(features_train_.shape[1]):
		nan_map = nan_map + features_train_[:,i].reshape(nan_map.shape[0],1)
	
	labels_train   = labels_train[~np.isnan(nan_map)]
	for i in xrange(features_train_.shape[1]):
		cln = features_train_[:,i].reshape(-1,1)
		if i == 0: features_train = cln[~np.isnan(nan_map)].reshape(-1,1)
		else: features_train = np.hstack((features_train,cln[~np.isnan(nan_map)].reshape(-1,1)))

	return features_train, labels_train

def read_files_nan():
	k_lam0, h_lam0 = 279.63509493e0, 280.35297192e0 # nm
	
	intensity_files_k = np.array(['k2r_i', 'k3_i', 'k2v_i'])
	intensity_files_h = np.array(['h2r_i', 'h3_i', 'h2v_i'])
	freqshift_files_k = np.array(['k2r_dv', 'k3_dv', 'k2v_dv'])
	freqshift_files_h = np.array(['h2r_dv', 'h3_dv', 'h2v_dv'])
	
	intensity = np.hstack((get_values(intensity_files_k),get_values(intensity_files_h)))
	freqshift = np.hstack((get_values(freqshift_files_k, orig=k_lam0),get_values(freqshift_files_h, 	orig=h_lam0)))
	features_train_ = np.hstack((intensity, freqshift))
	labels_train_   = np.load('avg_T.npy')
	labels_train    = labels_train_.reshape(-1,1)
	
	nan_map = labels_train.copy()
	
	for i in xrange(features_train_.shape[1]):
		nan_map = nan_map + features_train_[:,i].reshape(nan_map.shape[0],1)
	
	features_train = features_train_.copy()

	return features_train, labels_train, nan_map

# Get training and test set
def train_test(features, labels):
	trainer = np.hstack((features, labels[:, np.newaxis]))
	np.random.shuffle(trainer)
	train, test = train_test_split(trainer)
	features_train = train[:,:-1]
	features_test  = test[:,:-1]
	labels_train = train[:,-1][:,np.newaxis]
	labels_test  = test[:,-1][:,np.newaxis]
	return features_train, features_test, labels_train, labels_test

# 2D Histogram of the prediction and original
def plot_prediction(labels_pred, labels_test, method='Regression', xrang=[0,20000], yrang=[0,20000]):
	labels_pred = np.squeeze(labels_pred)
	labels_test = np.squeeze(labels_test)
	H, xedges, yedges = np.histogram2d(labels_pred, labels_test, bins=(200, 200), range=(xrang,yrang))
	return H, xedges, yedges

# Add the asymmetry features
def add_asymmetry(features, xx, yy):
	for i in xx:
		for j in yy:
			new = (features[:, i] - features[:, j])/(features[:, i] + features[:, j])
			new = new[:,np.newaxis]
			features = np.hstack((features, new))
	return features


