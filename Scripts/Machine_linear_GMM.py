
import numpy as np
from matplotlib import pyplot as plt
from astroML.classification import GMMBayes
from sklearn.cross_validation import train_test_split
from astroML.decorators import pickle_results
from astroML.datasets import fetch_LINEAR_geneva
from sklearn.metrics import roc_curve,auc


data = fetch_LINEAR_geneva()


attributes = ['gi', 'logP', 'ug', 'iK', 'JK', 'amp', 'skew']
#datalabels = ['$u-g$', '$g-i$', '$i-K$', '$J-K$', r'$\log(P)$', 'amplitude', 'skew']

cls = 'LCtype'


X = []
y = []

X=np.vstack([data[a] for a in attributes]).T
LCtype = data[cls].copy()

#LCtype[LCtype == 6] = 3
y=LCtype

print X.shape
print y[0]  



i = np.arange(len(data))
i_train, i_test = train_test_split(i, random_state=0, train_size=2000)

Xtrain = X[i][i_train]
Xtest = X[i][i_test]
ytrain = y[i][i_train]
ytest = y[i][i_test]


print "Xtrain",Xtrain.shape
print "ytrain",ytrain.shape
print Xtrain
print ytrain
print np.isnan(Xtrain).sum()
print np.isnan(ytrain).sum()

n_componentss = np.arange(5,7)
scores = np.zeros(len(n_componentss))

for j,n_components in enumerate(n_componentss):
        clf = GMMBayes(n_components, min_covar=1E-5, covariance_type='full',
                       random_state=0)
        clf.fit(Xtrain, ytrain)
        y_pred = clf.predict(Xtest)
        print y_pred
        print "score, ", (y_pred==ytest).sum()
        #fpr, tpr, thresholds = roc_curve(ytest, y_prob)
        #aucs[j]= auc(fpr,tpr)
        scores[j]=1.*(y_pred==ytest).sum()

imax = np.argmax(scores)
print "optimal N is ",n_componentss[imax]

N_comp = n_componentss[imax]
clf = GMMBayes(N_comp, min_covar=1E-5, covariance_type='full',
                       random_state=0)
clf.fit(Xtrain, ytrain)
y_pred = clf.predict(Xtest)
y_proba = clf.predict_proba(Xtest)

print y_proba
print y_proba.shape

fig,axes = plt.subplots(3,2)

labels = set(ytest)
print labels

for i,(ax,label) in enumerate(zip(axes.flatten(),labels)):
    y_prob = y_proba[:,i]
    fpr, tpr, thresholds = roc_curve(ytest, y_prob,pos_label=label)
    ax.plot(fpr,tpr,label='ROC feature #%i vs rest'%label)
    ax.legend(loc='best')

ix = 0
iy = 1

axm1 = axes[-1][-1]
axm1.scatter(Xtest[:,ix],Xtest[:,iy],c=y_pred,alpha=0.1,linewidth=0,s=3)
t = 0.1
xmin = np.percentile(Xtrain[:,ix],t)
xmax = np.percentile(Xtrain[:,ix],100-t)
ymin = np.percentile(Xtrain[:,iy],t)
ymax = np.percentile(Xtrain[:,iy],100-t)
#ax2.plot([tx,tx],[ymin,ymax],color='black')
#ax2.plot([xmin,xmax],[ty,ty],color='black')
#
axm1.set_xlim([xmin,xmax])
axm1.set_ylim([ymin,ymax])
axm1.set_xlabel(attributes[ix])
axm1.set_ylabel(attributes[iy])

plt.savefig("ROCs.pdf")
fig2,ax2 = plt.subplots(1,1)
ecolors = plt.cm.jet(ytest)
fitcolors=plt.cm.jet(y_pred)
ax2.scatter(Xtest[:,ix],Xtest[:,iy],c=y_pred,alpha=0.4,linewidth=0,s=6)
ax2.scatter(Xtest[:,ix],Xtest[:,iy],c=ytest,marker='+',alpha=0.5,linewidth=1,s=5)
t = 0.1
xmin = np.percentile(Xtrain[:,ix],t)
xmax = np.percentile(Xtrain[:,ix],100-t)
ymin = np.percentile(Xtrain[:,iy],t)
ymax = np.percentile(Xtrain[:,iy],100-t)
#ax2.plot([tx,tx],[ymin,ymax],color='black')
#ax2.plot([xmin,xmax],[ty,ty],color='black')
#
ax2.set_xlim([xmin,xmax])
ax2.set_ylim([ymin,ymax])
ax2.set_xlabel(attributes[ix])
ax2.set_ylabel(attributes[iy])

fig2.savefig("ROCs2.pdf")






