import numpy as np
import scipy as sp
import matplotlib as mpl
import scipy as sp
import scipy.stats as sps
import matplotlib.pyplot as plt
from sklearn.mixture import GMM

xmin =-2
xmax = 2
ymin =-2
ymax =2

#Want Smileyface PDF in 2D: 

pe = 0.2  #prob of eye 1
ps = 1.-pe #prob of smile

sigma = 0.1

eyes = [[-1,1],[1,1]]

print len(eyes)
randeyes = sps.randint(0,len(eyes))
print randeyes
print randeyes.rvs()

#smile the

#smile is a half circle

N = 10000
xs = np.zeros(N)
ys = np.zeros(N)
rs = sps.uniform().rvs(N)
ts = sps.uniform().rvs(N)*np.pi
cs = np.cos(ts)
ss = np.sin(ts)
dxs = sps.norm(0,sigma).rvs(N)
dys = sps.norm(0,sigma).rvs(N)

for n in range(N):
    a = rs[n]
    x = 0.
    y = 0.
    if a<pe:
        b = randeyes.rvs()
        x = eyes[b][0]
        y = eyes[b][1]
    else:
        x = cs[n]
        y = -1.*ss[n]

    xs[n]=x
    ys[n]=y

xs = xs+dxs
ys = ys+dys

X = np.transpose([xs,ys])

Nc = 30

mixmod = GMM(Nc,covariance_type='diag')
mixmod.fit(X)

pxs = np.linspace(xmin,xmax,1000)
pys = np.linspace(ymin,ymax,1000)
XX,YY = np.meshgrid(pxs,pys)
XXX = np.array([XX.ravel(),YY.ravel()]).T
Z = np.exp(mixmod.score_samples(XXX)[0])
Z = Z.reshape(XX.shape)
CS = plt.contourf(XX,YY,Z,1000,cmap=mpl.cm.afmhot)
CB = plt.colorbar(CS)
print "mixmod.aic:"
print mixmod.aic(X)
print "mixmod.aic:"
print mixmod.bic(X)
plt.show()
