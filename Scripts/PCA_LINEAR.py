from astroML.datasets import fetch_LINEAR_geneva
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

a = fetch_LINEAR_geneva()
attributes= a.dtype.names 
#attributes = ['gi', 'logP',\
#              'ug', 'iK', 'JK', 'amp', 'skew']

ndata = len(a['gi'])
oids = a[attributes[-1]]
attributes = attributes[0:-1]

nattributes = len(attributes)

data = np.zeros([ndata,nattributes])

for i,att in enumerate(attributes):
    col = a[att]
    data[:,i]=col

# Mask object ID: 

#print data.shape

odata = data
pca = PCA(copy=True,whiten=False)

pca.fit(data)

comp = pca.transform(data)

#print "comp"

#fig,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2)



x=(comp[:,0])
y=(comp[:,1])

evals = pca.explained_variance_ratio_
evects = pca.components_
#print "evects.shape"
#print evects.shape
#ax0.scatter(x,y)
#ax0.scatter(x,y,color='k',alpha=0.1)




#ax3.scatter(a['ug'],a['gi'])

#mdata = np.matrix(odata)
#me0 = np.matrix(evects[2])
#me1 = np.matrix(evects[3])

#print mdata.shape
#print me0.shape
#print me0
#print me1
#proj0 = mdata*me0.T
#proj1 = mdata*me1.T
#print proj0.shape
#ax3.scatter(proj0,proj1,color='k',alpha=0.1)
#print mdata[:,2]

fig,axes = plt.subplots(4,3)
print axes.shape

ax0 = axes[0,0]
ax1 = axes[0,1]
ax2 = axes[0,2]



ax0.set_title("LINEAR object coordinates")
ax0.set_xlabel(attributes[0])
ax0.set_ylabel(attributes[1])
ax0.scatter(data[:,0],data[:,1],marker="o",color='k',alpha=0.05)

ax1.set_title("PCA eigenvalues")
ax1.plot(evals,"o",color='blue')
ax1.set_yscale('log')
ax1.set_xlabel("Eigenvalue rank")
ax1.set_ylabel("Eigenvalue")

ax2.set_title("PCA 3 largest eigenvectors")
ax2.set_xticks(range(len(evects[:,0])))
ax2.set_xticklabels(attributes)
colors = plt.cm.rainbow(np.linspace(0,1,3))
for i in range(3):
    ax2.plot(range(len(evects[i])),evects[i],"o",color=colors[i],label="Eigenvector %i"%i)
ax2.legend(loc='best',framealpha=0.7,fontsize=5)

for i in range(3):
    for j in range(3):
        if i!=j:
            axes[i+1,j].scatter(comp[:,j],comp[:,i],c=oids,marker='.',alpha=0.05)
        else:
            axes[i+1,j].hist(comp[:,i],bins=100,linewidth=0,color="black",alpha=0.5)

for i in range(3):
    axes[i+1,0].set_ylabel("PCA comp %i"%i)
    axes[-1,i].set_xlabel("PCA comp %i"%i)


plt.savefig("machinePCA_LINEAR.pdf")
