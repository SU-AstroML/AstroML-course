import numpy as np
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression, RANSACRegressor, SGDRegressor, TheilSenRegressor
from sklearn.gaussian_process import GaussianProcess
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import astroML
from astroML.linear_model import PolynomialRegression, BasisFunctionRegression, NadarayaWatson

mu0 = np.linspace(0,1,10)[:, np.newaxis]

# Linear Regression
def fit_Linear(features_train, labels_train, features_pred, package='astroML', fit_intercept=True,\
										 normalize=False, n_jobs=1):
	if package=='astroML':
		model = astroML.linear_model.LinearRegression()
		model.fit(features_train, labels_train)
	elif package=='sklearn':
		model = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, n_jobs=n_jobs)
		model.fit(features_train, labels_train)
		score = model.score(features_train, labels_train)
		print "Linear - coefficient of determination R^2 of the prediction: ", score
	else:	print "Unknown package"
	labels_pred = model.predict(features_pred)
	return labels_pred

# Polynomial Regression
def fit_Polynomial(features_train, labels_train, features_pred, order=3):
	model = PolynomialRegression(order)
	model.fit(features_train, labels_train)
	labels_pred = model.predict(features_pred)
	return labels_pred

# Basis Function Regression
def fit_BasisFunction(features_train, labels_train, features_pred, kernel='gaussian', mu=mu0, sigma=0.1):
	model = BasisFunctionRegression(kernel, mu=mu, sigma=sigma)
	model.fit(features_train, labels_train)
	labels_pred = model.predict(features_pred)
	return labels_pred

	
# Ridge Regression
def fit_Ridge(features_train, labels_train, features_pred, alphas=(0.1, 1.0, 10.0)):
	model = RidgeCV(normalize=True, store_cv_values=True, alphas=alphas)
	model.fit(features_train, labels_train)
	cv_errors = np.mean(model.cv_values_, axis=0)
	print "RIDGE - CV error min: ", np.min(cv_errors)	
	# Test the model
	labels_pred = model.predict(features_pred)
	return labels_pred

# Lasso Regression
def fit_Lasso(features_train, labels_train, features_pred):
	model = LassoCV()
	model.fit(features_train, labels_train)
	mse = model.mse_path_
	print "LASSO - Mean square error: ", mse.shape
	# Test the model
	labels_pred = model.predict(features_pred)
	return labels_pred

# Nadaraya-Watson Regression
def fit_NadarayaWatson(features_train, labels_train, features_pred, kernel='gaussian', alpha=0.05):
	model = NadarayaWatson(kernel, alpha)
	model.fit(features_train, labels_train)
	labels_pred = model.predict(features_pred)
	return labels_pred

# Gaussian Process
def fit_GaussianProcess(features_train, labels_train, features_pred, corr='squared_exponential'):
	gp = GaussianProcess(corr=corr)
	gp.fit(features_train, labels_train)
	labels_pred = gp.predict(features_pred)
	return labels_pred

# K-Neighbors Regression
def fit_KNeighbors(features_train, labels_train, features_pred, n_neighbors=5):
	model = KNeighborsRegressor(n_neighbors=n_neighbors)
	model.fit(features_train, labels_train)
	labels_pred = model.predict(features_pred)
	score = model.score(features_train, labels_train)
	print "KNeighbors - coefficient of determination R^2 of the prediction: ", score
	return labels_pred

# RANSAC Regression
def fit_RANSAC(features_train, labels_train, features_pred):
	model = RANSACRegressor()
	model.fit(features_train, labels_train)
	labels_pred = model.predict(features_pred)
	print "RANSAC - coefficient of determination R^2 of the prediction: ", model.score(features_train, labels_train)
	return labels_pred

# SGD Regression
def fit_SGD(features_train, labels_train, features_pred):
	model = SGDRegressor()
	model.fit(features_train, labels_train)
	labels_pred = model.predict(features_pred)
	print "SGD - coefficient of determination R^2 of the prediction: ", model.score(features_train, labels_train)
	return labels_pred

# Theil-Sen Regression
def fit_TheilSen(features_train, labels_train, features_pred):
	model = TheilSenRegressor()
	model.fit(features_train, labels_train)
	labels_pred = model.predict(features_pred)
	print "TheilSen - coefficient of determination R^2 of the prediction: ", model.score(features_train, labels_train)
	return labels_pred

# Epsilon-Support Vector Regression
def fit_SVR(features_train, labels_train, features_pred):
	model = SVR()
	model.fit(features_train, labels_train)
	labels_pred = model.predict(features_pred)
	print "SVR - coefficient of determination R^2 of the prediction: ", model.score(features_train, labels_train)
	return labels_pred

# Decision Tree Regression
def fit_DecisionTree(features_train, labels_train, features_pred):
	model = DecisionTreeRegressor()
	model.fit(features_train, labels_train)
	labels_pred = model.predict(features_pred)
	print "DecisionTree - coefficient of determination R^2 of the prediction: ", model.score(features_train, labels_train)
	return labels_pred

