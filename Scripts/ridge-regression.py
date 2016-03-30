import numpy as np
from sklearn.linear_model import RidgeCV
import pylab as pl

def get_features_matrix(features):
    for i in range(13):
        for j in range(12):
            interactions = (features[:,i]*features[:,j]).T
            interactions = interactions[:, np.newaxis]
            features = np.hstack([features, interactions])
    return features

names =["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS",\
    "RAD","TAX","PTRATIO","B","STAT"]

features = get_features_matrix(np.loadtxt('../housing_train.dat'))

labels = np.loadtxt('../housing_prices_train.dat')
alphas = 10**np.linspace(-5, -1, 100)

model = RidgeCV(normalize=True, store_cv_values=True, alphas=alphas)
model.fit(features, labels)
print model.coef_
print model.alpha_
cv_errors = np.mean(model.cv_values_, axis=0)

# Test the model
features_test = get_features_matrix(np.loadtxt('../housing_test.dat'))
prices_test = np.loadtxt('../housing_prices_test.dat')
prices_pred = model.predict(features_test)
score = np.mean((prices_test-prices_pred)**2)
print 'Score', score

pl.semilogx(alphas, cv_errors)

pl.figure()
ax = pl.subplot(111)
pl.plot(range(len(model.coef_)), model.coef_, 'o')
#ax.set_xticklabels(names)
#ax.set_xticks(range(len(model.coef_)))
pl.show()
