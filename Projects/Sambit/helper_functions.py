import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.gaussian_process import GaussianProcess
import pylab as pl

k_lam0, h_lam0 = 279.63509493e0, 280.35297192e0 # nm

def make_hist2d(X, Y, xyrange=([0,20000],[0,20000])):
	H, xedges, yedges = np.histogram2d(X, Y, bins=(200, 200), range=xyrange)
	pl.figure('2D Histogram')
	pl.imshow(H, interpolation='nearest', origin='low', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]])
	pl.show()
	return H, xedges, yedges

def get_features_matrix(features):
    for i in range(13):
        for j in range(12):
            interactions = (features[:,i]*features[:,j]).T
            interactions = interactions[:, np.newaxis]
            features = np.hstack([features, interactions])
    return features

def get_values(filenames, orig=0):
	out = np.array([])
	for fl in filenames:
		fl  = fl+'.npy'
		flv = np.load(fl)
		flv += orig
		if len(out) == 0: out = flv.reshape(flv.shape[0]*flv.shape[1],1)
		else:		  out = np.hstack((out,flv.reshape(flv.shape[0]*flv.shape[1],1)))
	return out

def remove_nan(arr, arr1=-1):
	if arr1 == -1: arr1 = arr.copy()
	return arr[~np.isnan(arr1)] 

def get_z_values(ztk, k3_dv, k_lam, k_lam0):
	k_dv = 3e+8 * (k_lam / k_lam0 -1e0)*1e-3
	k_z  = np.zeros(k3_dv.shape)

	for i in xrange(k3_dv.shape[0]):
		for j in xrange(k3_dv.shape[1]):
			kdv = k3_dv[i,j]
			if kdv!=kdv:
				k_z[i,j] = kdv
			else:
				dv_low  = k_dv[k_dv <= kdv].max()
				dv_high = k_dv[k_dv > kdv].min()
				dv_low_loc  = np.argwhere(k_dv == dv_low)
				dv_high_loc = np.argwhere(k_dv == dv_high)
				z_low  = ztk[i,j,dv_low_loc] 
				z_high = ztk[i,j,dv_high_loc] 
				k_z[i,j] = z_low + (kdv-dv_low)*(z_high-z_low)/(dv_high-dv_low)
	return k_z


def get_T_values(T_cube, z_geo, k3_z):
	ni, nj = k3_z.shape
	T = np.zeros((ni,nj))
	for i in xrange(ni):
		for j in xrange(nj):
			z = k3_z[i,j]
			if z!=z:
				T[i,j] = z
			else:
				z_low  = z_geo[z_geo <= z].max()
				z_high = z_geo[z_geo > z].min()
				z_low_loc  = np.argwhere(z_geo == z_low)
				z_high_loc = np.argwhere(z_geo == z_high)
				T_low  = T_cube[i,j,z_low_loc]
				T_high = T_cube[i,j,z_high_loc]
				T[i,j] = T_low + (z-z_low)*(T_high-T_low)/(z_high-z_low)
	return T

def get_vz_values(vz_cube, z_geo, k3_z):
	ni, nj = k3_z.shape
	vz = np.zeros((ni,nj))
	for i in xrange(ni):
		for j in xrange(nj):
			z = k3_z[i,j]
			if z!=z:
				vz[i,j] = z
			else:
				z_low  = z_geo[z_geo <= z].max()
				z_high = z_geo[z_geo > z].min()
				z_low_loc  = np.argwhere(z_geo == z_low)
				z_high_loc = np.argwhere(z_geo == z_high)
				vz_low  = vz_cube[i,j,z_low_loc]
				vz_high = vz_cube[i,j,z_high_loc]
				vz[i,j] = vz_low + (z-z_low)*(vz_high-vz_low)/(z_high-z_low)
	return vz


def fit_Ridge(features_train, labels_train, features_pred, alphas=(0.1, 1.0, 10.0)):
	model = RidgeCV(normalize=True, store_cv_values=True, alphas=alphas)
	model.fit(features_train, labels_train)
	#print model.coef_
	#print model.alpha_
	cv_errors = np.mean(model.cv_values_, axis=0)
	print "CV error min: ", np.min(cv_errors)
		
	# Test the model
	labels_pred = model.predict(features_pred)
	return labels_pred

def fit_Lasso(features_train, labels_train, features_pred):
	model = LassoCV()
	model.fit(features_train, labels_train)
	mse = model.mse_path_
	print "Mean square error: ", mse.shape
		
	# Test the model
	labels_pred = model.predict(features_pred)

	return labels_pred

def fit_GaussianProcess(features_train, labels_train, features_pred):
	gp = GaussianProcess()
	gp.fit(features_train, labels_train)
	labels_pred, mse = gp.predict(features_pred, eval_MSE=True)
	print mse.shape
	return labels_pred



