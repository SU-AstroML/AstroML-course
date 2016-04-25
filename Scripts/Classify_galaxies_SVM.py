from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot as pydot 
import numpy as np
from astroML.classification import GMMBayes
from sklearn.metrics import roc_curve, auc
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

ellipticity_threshold=0.5
data = np.loadtxt("Galaxies_hands_on_Chap9.txt",skiprows=1)

replacewithmean=True
precondition = True #replace values with (value-mean_value)/stdev for each column

datanames=["cntr_01","name","RA","Dec","elliptical","spiral","photoz","photozErr","FUV","NUV","U","G","R","I","Z","J","H","K","W1","W2","W3","W4"]



#replace -9999 with nan: 

data[data==-9999]=np.nan

#data = data[~np.isnan(data).any(axis=1)]

n_not_color=8
n_color = data.shape[1]-n_not_color
#print n_color

#shuffle the data to avoid bias on training/test data: 

np.random.shuffle(data)

#labels are 1 if elliptical<threshold (i.e. 1=spirals)

labels = (data[:,4]<=ellipticity_threshold)*1.

#now to compute the features. There are sum_1^n_color 1 color numbers, 
#as well as photometric redshift and the attendant error

tdata = data[:,n_not_color-2:n_not_color]

tdatanames = datanames[n_not_color-2:n_not_color] 

print "start colors"
n_nonan=0
for i in range(n_color):
    ii=i+n_not_color
    for j in range(i+1,n_color):
        jj=j+n_not_color
        color = data[:,ii]-data[:,jj]
        if (np.isnan(color).any(axis=0)):
            if replacewithmean:
                c1 = np.nanmean(data[:,ii])
                c2 = np.nanmean(data[:,jj])
                c = c1-c2
                color[np.isnan(color)]=c
            else:
                color = np.zeros(len(color))
        if precondition:
            color = (color-np.nanmean(color))/np.nanstd(color)

        else:
            n_nonan+=1
        
        tdatanames.append(datanames[ii]+"-"+datanames[jj])
        tdata = np.hstack([tdata,color.reshape(-1,1)])

print "number of nonan colors: ",n_nonan



#print tdata[0]
data = tdata #avoid to keep two arrays
datanames=tdatanames
data[np.isnan(data)]=-9999.
#print data.shape

#find PCA directions: 
ifeature0 = datanames.index('FUV-NUV')
ifeature1 = datanames.index('W1-W3')

#split data in training and test sample, 80-20
n_samples = data.shape[0]
data_train = data[:n_samples*0.8, :]
data_test = data[n_samples*0.8:, :]
labels_train = labels[:n_samples*0.8]
labels_test = labels[n_samples*0.8:]





#print data.shape
#print data_train.shape
#print data_test.shape
#print data_test_pca.shape
#print labels.shape
#print labels_test.shape
#print labels_train.shape


#perform BDT between 2 and 15 levels:


ws = np.logspace(0,2,30)
scores = []
fprs = []
tprs = []
aucs = []
for w in ws:
    print w
    wgt = {0:1,1:w}
    model = SVC(class_weight=wgt)
    model.fit(data_train,labels_train)
    scores.append(model.score(data_test,labels_test))
    labels_pred = model.predict(data_test)
    ntp = 0
    nfp = 0
    nt = len(labels_test[labels_test==1]);   
    nf = len(labels_test[labels_test==0]);   
    for lp,lt in zip(labels_pred,labels_test):
        if (lp==1 and lt ==1):
            ntp+=1
        if (lp==1 and lt ==0):
            nfp+=1
    tprs.append( (1.*ntp)/(1.*nt))
    fprs.append( (1.*nfp)/(1.*nf))
    decisions = model.decision_function(data_test)
    fpr,tpr,thr = roc_curve(labels_test,decisions)
    a = auc(fpr,tpr)
    aucs.append(a)
    print a







fig,((ax0,ax1),(ax2,ax3))=plt.subplots(2,2)
#
#
imax = np.argmax(aucs)
print "Optimal weight is ",ws[imax]
print "peform SVM"
wgt = {0:1,1:ws[imax]}
model = SVC(class_weight=wgt)
model.fit(data_train,labels_train)
labels_pred=model.predict(data_test)

print (labels_pred==labels_test).mean()

decisions = model.decision_function(data_test)
print decisions
fpr,tpr,thr = roc_curve(labels_test,decisions)

print "SVM fit done"

#print "model_weights"
#print model_weights
##print "importance range"
#print importances
#print datanames
#print range(len(datanames))
#print "model_weights"
#print model_weights
#print model_weights.shape
#print  [x for (y,x) in sorted(zip(np.abs(model_weights),datanames))]
##
#
#clf = tree.DecisionTreeClassifier(max_depth=Ns[imax])
#clf = clf.fit(data_train, labels_train)
#decisions = clf.predict_proba(data_test)[:,1]
#decisions_train = clf.predict_proba(data_train)[:,1]
#fpr,tpr, thresholds = roc_curve(labels_test,decisions)
ax0.plot(fpr,tpr,label="ROC curve, optimal weight=%.2f"%ws[imax])
#ax0.plot(fpr,tpr,label="ROC curve, optimat weight")
ax0.set_xlabel("False positive Rate")
ax0.set_ylabel("True positive Rate")
ax0.legend(loc='best',fontsize=7)
#
ax1.plot(ws,aucs,label="Score given signal wgt")
ax1.legend(loc='best',fontsize=7)
ax1.set_xlabel("weight")
ax1.set_xscale('log')
ax1.set_ylabel("Score")
#
#
#
#dtree = clf.tree_
#ix = dtree.feature[0]
#tx = dtree.threshold[0]

if True:
    #model_weights = (np.absolute(model.coef_)).flatten()
    #importances=  [x for (y,x) in sorted(zip(np.abs(model_weights),range(len(datanames))))]
    #ix = importances[-1]
    #iy = importances[-2]
    ix = ifeature0
    iy = ifeature1
    supvec = model.support_
    #
    ##choose between 2 & 3
    #
    #
    ecolors = plt.cm.viridis(labels_train)
    ax2.scatter(data_train[:,ix],data_train[:,iy],c=ecolors,marker='.',linewidth=0,alpha=0.1,cmap=plt.cm.viridis)
    ax2.scatter(data_train[supvec,ix],data_train[supvec,iy],c=labels_train[supvec],marker='o',linewidth=0,alpha=0.2,cmap=plt.cm.viridis)
    #ax2.scatter(data_test[:,ix],data_test[:,iy],c=labels_test,edgecolors=ecolors,marker='o',linewidth=0.7,alpha=0.5,cmap=plt.cm.viridis)
    #
    t = 0.1
    xmin = np.percentile(data_train[:,ix],t)
    xmax = np.percentile(data_train[:,ix],100-t)
    ymin = np.percentile(data_train[:,iy],t)
    ymax = np.percentile(data_train[:,iy],100-t)
    #ax2.plot([tx,tx],[ymin,ymax],color='black')
    #ax2.plot([xmin,xmax],[ty,ty],color='black')
    #
    ax2.set_xlim([xmin,xmax])
    ax2.set_ylim([ymin,ymax])
    ax2.set_xlabel(datanames[ix])
    ax2.set_ylabel(datanames[iy])
#
#
##ax2.scatter(x_test[:,0data,np.log(-1.*x_test[:,5]),facecolors='none',edgecolors=ecolors,marker='o',linewidth=1,alpha=0.5,cmap=plt.cm.viridis)
##x_test_s = x_test[y_test==1]
##decisions_s=decisions[y_test==1]
###ax2.scatter(x_test_s[:,0],np.log(-1.*x_test_s[:,5]),color='black',marker='o',linewidth=0,s=3,alpha=0.5,cmap=plt.cm.viridis)
r=ax3.hist(decisions[labels_test==0],color='blue',alpha=0.5,bins=50)
##print r[1]
ax3.hist(decisions[labels_test==1],bins=r[1],color='yellow',alpha=0.5)
plt.savefig("Classify_SVM.pdf")
#
ax3.set_xlabel("Distance from Margin")
##
##
#with open("iris.dot", 'w') as f:
#    f = tree.export_graphviz(clf, out_file=f,filled=True,rounded=True)
#dot_data = StringIO() 
#tree.export_graphviz(clf,
#        out_file=dot_data,filled=True,rounded=True,feature_names=datanames,class_names=["Elliptical","Spiral"]) 
#graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
#graph.write_pdf("iris.pdf") 
