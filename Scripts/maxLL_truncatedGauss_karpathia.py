import numpy as np
import scipy.stats as sps
import scipy.optimize as spo
import matplotlib.pyplot as plt

data = np.array([120,120,130])
xmin = 120
xmax = 1e9

def F(x,data,xmin,xmax):
    mu = x[0]
    s  = 15
    if (s<=0.):
        return 1e9
    n = sps.norm(mu,s)

    C = n.cdf(xmax)-n.cdf(xmin) #as in book
    C=1./C
    #ret = -1.*(((data-mu)**2)/(2*s**2)).sum()
    #ret += len(data)*np.log(C)
    ret = 1.*np.log(n.pdf(data)).sum() +len(data)*np.log(C)
    #return -2*likelihood
    return -2.*ret

#print "fmin, data=[127]"
#print spo.minimize(F,[120],args=([127],xmin,xmax),method='Nelder-Mead')
#print "fmin, data=[130]"
#print spo.minimize(F,[120],args=([130],xmin,xmax),method='Nelder-Mead')
#print "fmin, data=[140]"
#print spo.minimize(F,[120],args=([140],xmin,xmax),method='Nelder-Mead')
#print "fmin, data=[120,140]"
#print spo.minimize(F,[120],args=([120,140],xmin,xmax),method='Nelder-Mead')

fig,axes = plt.subplots(1,2)
x = np.linspace(80,150,1000)
y = np.zeros(1000)
for i in range(1000):
    y[i]= F([x[i]],[120,140],xmin,xmax)

axes[0].plot(x,y)

print 

iqs = np.linspace(120,200,100)
muh = np.zeros(len(iqs))

for i,iq in enumerate(iqs):
    ret = spo.minimize(F,[120],args=([iq],xmin,xmax),method='Nelder-Mead')
    muh[i]= ret['x'][0]

axes[0].set_xlabel('iq')
axes[0].set_ylabel('$-2\ln(L([120,140]|iq,15))$')
axes[1].set_xlabel('iq')
axes[1].set_ylabel('best fit $\mu_{iq}$ given observed iq')

#print muh

axes[1].plot(iqs,muh)

#N = 100
#a = np.zeros(N)
#b = sps.norm(80,15)
#for i in range(100):
#    k=b.rvs()
#    while (k<120):
#        k=b.rvs()
#    a[i]=k
#    print i





plt.show()
