from __future__ import division
import numpy as np
from matplotlib import pyplot as plt
from astroML.datasets import fetch_imaging_sample
from astropy.table import Table, join, Row, Column
import astropy.coordinates as coord
import astropy.units as u
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot as pydot 
from astroML.classification import GMMBayes
from sklearn.metrics import roc_curve, auc
import pylab as pl
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from astroML.utils import completeness_contamination
from sklearn.tree import DecisionTreeRegressor



############################### DATA ###############################

ts = Table.read('MLprojectdata_Laura.txt', format='ascii')
# only want objects with all the data in this situation:
ts = ts[((ts['ra']!=-9999)&(ts['dec']!=-9999)&(ts['run']!=-9999)&(ts['rExtSFD']!=-9999)&(ts['uRaw']!=-9999)&(ts['gRaw']!=-9999)&(ts['rRaw']!=-9999)&(ts['iRaw']!=-9999)&(ts['zRaw']!=-9999)&(ts['uErr']!=-9999)&(ts['gErr']!=-9999)&(ts['rErr']!=-9999)&(ts['iErr']!=-9999)&(ts['zErr']!=-9999)&(ts['uRawPSF']!=-9999)&(ts['gRawPSF']!=-9999)&(ts['rRawPSF']!=-9999)&(ts['iRawPSF']!=-9999)&(ts['zRawPSF']!=-9999)&(ts['upsfErr']!=-9999)&(ts['gpsfErr']!=-9999)&(ts['rpsfErr']!=-9999)&(ts['ipsfErr']!=-9999)&(ts['zpsfErr']!=-9999)&(ts['type']!=-9999)&(ts['ISOLATED']!=-9999)&(ts['No']!=-9999)&(ts['objID']!=-9999)&(ts['photoz']!=-9999)&(ts['photozErr']!=-9999)&(ts['petroRad_u']!=-9999)&(ts['petroRad_g']!=-9999)&(ts['petroRad_r']!=-9999)&(ts['petroRad_i']!=-9999)&(ts['petroRad_z']!=-9999)&(ts['petroRadErr_u']!=-9999)&(ts['petroRadErr_g']!=-9999)&(ts['petroRadErr_r']!=-9999)&(ts['petroRadErr_i']!=-9999)&(ts['petroRadErr_z']!=-9999)&(ts['petroR90_u']!=-9999)&(ts['petroR90_g']!=-9999)&(ts['petroR90_r']!=-9999)&(ts['petroR90_i']!=-9999)&(ts['petroR90_z']!=-9999)&(ts['petroR90Err_u']!=-9999)&(ts['petroR90Err_g']!=-9999)&(ts['petroR90Err_r']!=-9999)&(ts['petroR90Err_i']!=-9999)&(ts['petroR90Err_z']!=-9999)&(ts['psfMag_u']!=-9999)&(ts['psfMag_g']!=-9999)&(ts['psfMag_r']!=-9999)&(ts['psfMag_i']!=-9999)&(ts['psfMag_z']!=-9999)&(ts['psfMagErr_u']!=-9999)&(ts['psfMagErr_g']!=-9999)&(ts['psfMagErr_r']!=-9999)&(ts['psfMagErr_i']!=-9999)&(ts['psfMagErr_z']!=-9999)&(ts['petroMag_u']!=-9999)&(ts['petroMag_g']!=-9999)&(ts['petroMag_r']!=-9999)&(ts['petroMag_i']!=-9999)&(ts['petroMag_z']!=-9999)&(ts['petroMagErr_u']!=-9999)&(ts['petroMagErr_g']!=-9999)&(ts['petroMagErr_r']!=-9999)&(ts['petroMagErr_i']!=-9999)&(ts['petroMagErr_z']!=-9999)&(ts['petroFlux_u']!=-9999)&(ts['petroFlux_g']!=-9999)&(ts['petroFlux_r']!=-9999)&(ts['petroFlux_i']!=-9999)&(ts['petroFlux_z']!=-9999)&(ts['petroFluxIvar_u']!=-9999)&(ts['petroFluxIvar_g']!=-9999)&(ts['petroFluxIvar_r']!=-9999)&(ts['petroFluxIvar_i']!=-9999)&(ts['petroFluxIvar_z']!=-9999)&(ts['clean']!=-9999)&(ts['classification']!=-9999))]

# testing only right classifications:
#ts = ts[(ts['classification']=='correct')]

print ts

# normalisation of the features:
redshift = (ts['photoz']-np.mean(ts['photoz']))/np.std(ts['photoz'])
ts.add_column(Column(data=redshift, name='redshift'))

clean = ts['clean']
clean = (ts['clean']-np.mean(ts['clean']))/np.std(ts['clean'])
ts.add_column(Column(data=clean, name='cleanflag'))

radius = (ts['petroRad_u']+ts['petroRad_g']+ts['petroRad_r']+ts['petroRad_i']+ts['petroRad_z'])/5
radius = (radius-np.mean(radius))/np.std(radius)
ts.add_column(Column(data=radius, name='radius'))

u = (ts['petroMag_u']-np.mean(ts['petroMag_u']))/np.std(ts['petroMag_u'])
g = (ts['petroMag_g']-np.mean(ts['petroMag_g']))/np.std(ts['petroMag_g'])
r = (ts['petroMag_r']-np.mean(ts['petroMag_r']))/np.std(ts['petroMag_r'])
i = (ts['petroMag_i']-np.mean(ts['petroMag_i']))/np.std(ts['petroMag_i'])
z = (ts['petroMag_z']-np.mean(ts['petroMag_z']))/np.std(ts['petroMag_z'])
ts.add_column(Column(data=u, name='u'))
ts.add_column(Column(data=g, name='g'))
ts.add_column(Column(data=r, name='r'))
ts.add_column(Column(data=i, name='i'))
ts.add_column(Column(data=z, name='z'))

ug = ts['petroMag_u']-ts['petroMag_g']
ur = ts['petroMag_u']-ts['petroMag_r']
ui = ts['petroMag_u']-ts['petroMag_i']
uz = ts['petroMag_u']-ts['petroMag_z']
gr = ts['petroMag_g']-ts['petroMag_r']
gi = ts['petroMag_g']-ts['petroMag_i']
gz = ts['petroMag_g']-ts['petroMag_z']
ri = ts['petroMag_r']-ts['petroMag_i']
rz = ts['petroMag_r']-ts['petroMag_z']
iz = ts['petroMag_i']-ts['petroMag_z']
ug = (ug-np.mean(ug))/np.std(ug)
ur = (ur-np.mean(ur))/np.std(ur)
ui = (ui-np.mean(ui))/np.std(ui)
uz = (uz-np.mean(uz))/np.std(uz)
gr = (gr-np.mean(gr))/np.std(gr)
gi = (gi-np.mean(gi))/np.std(gi)
gz = (gz-np.mean(gz))/np.std(gz)
ri = (ri-np.mean(ri))/np.std(ri)
rz = (rz-np.mean(rz))/np.std(rz)
iz = (iz-np.mean(iz))/np.std(iz)
ts.add_column(Column(data=ug, name='ug'))
ts.add_column(Column(data=ur, name='ur'))
ts.add_column(Column(data=ui, name='ui'))
ts.add_column(Column(data=uz, name='uz'))
ts.add_column(Column(data=gr, name='gr'))
ts.add_column(Column(data=gi, name='gi'))
ts.add_column(Column(data=gz, name='gz'))
ts.add_column(Column(data=ri, name='ri'))
ts.add_column(Column(data=rz, name='rz'))
ts.add_column(Column(data=iz, name='iz'))

SDSSthreshold_u = ts['uRawPSF']-ts['uRaw']
SDSSthreshold_g = ts['gRawPSF']-ts['gRaw']
SDSSthreshold_r = ts['rRawPSF']-ts['rRaw']
SDSSthreshold_i = ts['iRawPSF']-ts['iRaw']
SDSSthreshold_z = ts['zRawPSF']-ts['zRaw']
ts.add_column(Column(data=SDSSthreshold_u, name='SDSSthreshold_u'))
ts.add_column(Column(data=SDSSthreshold_g, name='SDSSthreshold_g'))
ts.add_column(Column(data=SDSSthreshold_r, name='SDSSthreshold_r'))
ts.add_column(Column(data=SDSSthreshold_i, name='SDSSthreshold_i'))
ts.add_column(Column(data=SDSSthreshold_z, name='SDSSthreshold_z'))

SDSSthreshold_mean = (SDSSthreshold_u+SDSSthreshold_g+SDSSthreshold_r+SDSSthreshold_i+SDSSthreshold_z)/5
ts.add_column(Column(data=SDSSthreshold_mean, name='SDSSthreshold_mean'))

aa = ts['SDSSthreshold_mean']
A = np.ma.zeros((len(aa)))
for i in range(len(aa)):
    if aa[i]> 0.145:
        o = 3
    else:
        o = 6
    A[i]=o
ts.add_column(Column(data=A, name='SDSSthreshold_result'))



# checking the criterion:
#features = ['SDSSthreshold_u','SDSSthreshold_g','SDSSthreshold_r','SDSSthreshold_i','SDSSthreshold_z','SDSSthreshold_mean'] 
#datanames = ['SDSSthreshold_u','SDSSthreshold_g','SDSSthreshold_r','SDSSthreshold_i','SDSSthreshold_z','SDSSthreshold_mean']

# adding features:
features = ['SDSSthreshold_u','SDSSthreshold_g','SDSSthreshold_r','SDSSthreshold_i','SDSSthreshold_z','SDSSthreshold_mean','cleanflag','ug','ur','ui','uz','gr','gi','gz','ri','rz','iz','radius','SDSSthreshold_mean','radius','u','g','r','i','z'] #redshift
datanames = ['SDSSthreshold_u','SDSSthreshold_g','SDSSthreshold_r','SDSSthreshold_i','SDSSthreshold_z','SDSSthreshold_mean','cleanflag','ug','ur','ui','uz','gr','gi','gz','ri','rz','iz','radius','SDSSthreshold_mean','radius','u','g','r','i','z']

# only other features:
#features = ['cleanflag','ug','ur','ui','uz','gr','gi','gz','ri','rz','iz','radius','u','g','r','i','z'] #redshift
#datanames = ['cleanflag','ug','ur','ui','uz','gr','gi','gz','ri','rz','iz','radius','u','g','r','i','z']



# shuffle the data for no biais:
np.random.shuffle(ts)

aa = ts['classification']
b = ts['type']
A = np.ma.zeros((len(aa)))
A = A.tolist()
for i in range(len(aa)):
    if aa[i]=='correct':
        o = b[i]
    else:
        o = 6
    A[i]=o
ts.add_column(Column(data=A, name='truetype'))


############################### RANDOM FOREST ###############################

#aa = ts['type']
aa = ts['truetype']


A = np.ma.zeros((len(aa)))
A = A.tolist()
for i in range(len(aa)):
    if aa[i]==3:
        o = "galaxy"
    else:
        o = "star"
    A[i]=o
ts.add_column(Column(data=A, name='typewhat'))


nbgal = sum(ts['typewhat']=='galaxy')
nbstar = sum(ts['typewhat']=='star')
print "There are {:d} galaxies, and {:d} stars.".format(nbgal,nbstar)

X = np.empty((len(ts), len(features)))
for featnum, feat in enumerate(features):
    X[:,featnum] = ts[feat]
print X

y = np.array(ts['typewhat'])

from sklearn.ensemble import RandomForestClassifier
RFmod = RandomForestClassifier(n_estimators = 25, oob_score = True) #n_estimators = the nb of trees in the forest
RFmod.fit(X,y)
accuracy = RFmod.oob_score_
oob_error = 1 - accuracy
print 'The out-of-bag error is {:.1f}%%'.format(100*oob_error)

from sklearn import cross_validation
cv_accuracy = cross_validation.cross_val_score(RFmod,X,y,cv=10)
cv_error = 1-cv_accuracy
print 'The cross-validation error is {:.1f}%%'.format(100*cv_error[0])

y_cv_preds = cross_validation.cross_val_predict(RFmod,X,y)
ts.add_column(Column(data=y_cv_preds, name='y_cv_preds'))

gal_acc = sum((y_cv_preds=='galaxy')&(y=='galaxy'))/sum(y=='galaxy')
star_acc = sum((y_cv_preds=='star')&(y=='star'))/sum(y=='star')

classes=['galaxy','star']

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_cv_preds, classes)
print 'confusion matrice : ', cm

normalized_cm = cm.astype('float')/cm.sum(axis = 1)[:,np.newaxis]
print 'normalized : ', normalized_cm

f=plt.figure(figsize = (8,8)) #size
ad=f.add_subplot(1,1,1)
plt.imshow(normalized_cm, interpolation = 'nearest', cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(len(classes))
ad.set_xticks(tick_marks)
ad.set_yticks(tick_marks)
ad.set_xticklabels(classes, rotation=45)
ad.set_yticklabels(classes)
plt.ylabel('$True$ $type$', fontsize=15)
plt.xlabel('$Predicted$ $type$', fontsize=15)
plt.tight_layout()
plt.title('Random Forest Classification on Testing set')
kwargs = dict(size=12, color='black', fontweight='bold')
ad.text(0, 0, "{0:.1f} % ".format(normalized_cm[0,0]*100), ha='center', **kwargs)
ad.text(1, 0, "{0:.1f} %".format(normalized_cm[0,1]*100), ha='center', **kwargs)
ad.text(0, 1, "{0:.1f} %".format(normalized_cm[1,0]*100), ha='center', **kwargs)
ad.text(1, 1, "{0:.1f} %".format(normalized_cm[1,1]*100), ha='center', **kwargs)
plt.show()

print 'normalized test test : '
print normalized_cm
print normalized_cm[0,0]*100
print normalized_cm[0,1]*100
print normalized_cm[1,0]*100
print normalized_cm[1,1]*100

print("{0:.1f}".format(normalized_cm[0,0]*100))

important = RFmod.feature_importances_  #the higher, the more important the feature
print 'Importance of features :'
print important
print 'max'
print 'Old features:'
print 'SDSSthreshold_u, SDSSthreshold_g, SDSSthreshold_r, SDSSthreshold_i, SDSSthreshold_z, SDSSthreshold_mean', max(0.02604229, 0.1081969,0.3167943, 0.23972689, 0.0746148, 0.23462482) #SDSSthreshold_r

print 'Only new features:'
print 'cleanflag,ug,ur,ui,uz,gr,gi,gz,ri,rz,iz,radius,u,g,r,i,z', max(0.00704505, 0.05807835, 0.05069153, 0.05583354, 0.05878824, 0.05773646, 0.05944974, 0.05271883, 0.06568582, 0.060497, 0.06023474, 0.07577322, 0.06396399, 0.06861453, 0.07035224, 0.07517767, 0.05935905) #radius

print 'Adding the new features:'
print 'SDSSthreshold_u,SDSSthreshold_g,SDSSthreshold_r,SDSSthreshold_i,SDSSthreshold_z,SDSSthreshold_mean,cleanflag,ug,ur,ui,uz,gr,gi,gz,ri,rz,iz,radius,SDSSthreshold_mean,radius,u,g,r,i,z', max(0.01457784, 0.06971733, 0.23183318, 0.20967692, 0.05228376, 0.11003777, 0.00125201, 0.01090401, 0.00929467, 0.00932286, 0.00950221, 0.00789431, 0.00803893, 0.00822022, 0.01208857, 0.00889459, 0.0085945, 0.01107124, 0.11407386, 0.01455566, 0.00935659, 0.01846253, 0.01672801, 0.01535001, 0.01826843) #SDSSthreshold_r


'''nbins = 30
gal_weights = np.ones(nbgal)/nbgal
star_weights = np.ones(nbstar)/nbstar
fig=plt.figure(figsize = (8,8)) #size
figg=fig.add_subplot(1,1,1)
plt.hist(ts['SDSSthreshold_mean'][ts['typewhat']=='galaxy'],nbins, weights = gal_weights, histtype='step', color='turquoise', label='Galaxies',  linewidth=3)
plt.hist(ts['SDSSthreshold_mean'][ts['typewhat']=='star'],nbins, weights = star_weights, histtype='step', color='deeppink', label='Stars',  linewidth=3)
figg.set_xlabel("SDSSthreshold_mean", fontsize=20)
plt.legend(fancybox = 'True')
plt.show()

gal_weights = np.ones(nbgal)/nbgal
star_weights = np.ones(nbstar)/nbstar
fig=plt.figure(figsize = (8,8)) #size
figg=fig.add_subplot(1,1,1)
plt.hist(ts['SDSSthreshold_r'][ts['typewhat']=='galaxy'],nbins, weights = gal_weights, histtype='step', color='turquoise', label='Galaxies',  linewidth=3)
plt.hist(ts['SDSSthreshold_r'][ts['typewhat']=='star'],nbins, weights = star_weights, histtype='step', color='deeppink', label='Stars',  linewidth=3)
figg.set_xlabel("SDSSthreshold_r", fontsize=20)
plt.legend(fancybox = 'True')
plt.show()'''




#y = ts['type']
y = ts['truetype']


for k in [ts]:
    
    '''# split the data into 80 percent training set and 20 percent testing set:
    data_train = X[:4799*0.8, :]
    labels_train = y[:4799*0.8]    
    data_test = X[4799*0.8:, :]
    labels_test = y[4799*0.8:]'''

    # split the data into 80 percent training set and 20 percent testing set:
    data_train = X[:6561*0.8, :]
    labels_train = y[:6561*0.8]    
    data_test = X[6561*0.8:, :]
    labels_test = y[6561*0.8:]


    trees = np.arange(1, 30)
    rms_test = np.zeros(len(trees))
    rms_train = np.zeros(len(trees))
    i_best = 0
    label_fit_best = None

    completeness = []
    contamination = []
    for i, t in enumerate(trees):
        clf = RandomForestClassifier(t)
        clf.fit(data_train, labels_train)

        label_fit_train = clf.predict(data_train)
        label_fit = clf.predict(data_test)
        rms_train[i] = np.mean(np.sqrt((label_fit_train - labels_train) ** 2))
        rms_test[i] = np.mean(np.sqrt((label_fit - labels_test) ** 2))

        tmp_completeness, tmp_contamination = completeness_contamination(label_fit, labels_test)
        contamination.append(tmp_contamination)
        completeness.append(tmp_completeness)

        if rms_test[i] <= rms_test[i_best]:
            i_best = i
            label_fit_best = label_fit

    best_tree = trees[i_best]
    print "Depth of tree",  best_tree
    print "Fraction of stars: ", sum(labels_train/len(labels_train))


    plt.figure(figsize = (7,7))
    plt.title('Random Forest', fontsize=15)
    plt.plot(trees, completeness, color='teal', label='Completeness', linewidth=4)
    plt.plot(trees, contamination, color='darkviolet', label='Contamination', linewidth=4)
    plt.legend(loc='best',prop={'size': 15})
    plt.xlabel('Number of trees in the Forest', fontsize=15)
    plt.show()
    
    plt.figure(figsize = (7,7))
    plt.title('Random Forest', fontsize=15)
    plt.plot(trees, rms_train, color='gold', label='Training set', linewidth=4)
    plt.plot(trees, rms_test, color='limegreen', label='Testing set', linewidth=4)
    plt.legend(loc='best',prop={'size': 15})
    plt.xlabel('Number of trees in the Forest', fontsize=15)
    plt.ylabel('RMS error', fontsize=15)
    plt.show()
    
    
    decisions = clf.predict_proba(data_test)[:,1]
    fpr_RF,tpr_RF, thresholds_RF = roc_curve(labels_test,decisions,pos_label=6)
    
        
    
    
    ############################### DECISION TREE ###############################
    
    depth = np.arange(1, 40)
    rms_test = np.zeros(len(depth))
    rms_train = np.zeros(len(depth))
    i_best = 0
    label_fit_best = None

    completeness = []
    contamination = []
    for i, d in enumerate(depth):
        clf = DecisionTreeRegressor(max_depth=d, random_state=0)
        clf.fit(data_train, labels_train)

        label_fit_train = clf.predict(data_train)
        label_fit = clf.predict(data_test)
        rms_train[i] = np.mean(np.sqrt((label_fit_train - labels_train) ** 2))
        rms_test[i] = np.mean(np.sqrt((label_fit - labels_test) ** 2))

        tmp_completeness, tmp_contamination = completeness_contamination(label_fit, labels_test)
        contamination.append(tmp_contamination)
        completeness.append(tmp_completeness)

        if rms_test[i] <= rms_test[i_best]:
            i_best = i
            label_fit_best = label_fit

    best_depth = depth[i_best]
    print "Depth of tree",  best_depth
    print "Fraction of ellipticals: ", sum(labels_train/len(labels_train))


    plt.figure(figsize = (7,7))
    plt.title('Decision Tree', fontsize=15)
    plt.plot(depth, completeness, color='teal', label='Completeness', linewidth=4)
    plt.plot(depth, contamination, color='darkviolet', label='Contamination', linewidth=4)
    plt.legend(loc='best',prop={'size': 15})
    plt.xlabel('Depth of the Tree', fontsize=15)
    plt.show()

    plt.figure(figsize = (7,7))
    plt.title('Decision Tree', fontsize=15)
    plt.plot(depth, rms_train, color='gold', label='Training set', linewidth=4)
    plt.plot(depth, rms_test, color='limegreen', label='Testing set', linewidth=4)
    plt.legend(loc='best',prop={'size': 15})
    plt.xlabel('Depth of the Tree', fontsize=15)
    plt.ylabel('RMS error', fontsize=15)
    plt.show()
    
    clf = DecisionTreeRegressor(max_depth=10, random_state=0)
    clf.fit(data_train, labels_train)
    decisions = clf.predict(data_test)
    fpr_DT,tpr_DT, thresholds_DT = roc_curve(labels_test,decisions,pos_label=6)
    
    
    
    ############################### BOOSTING CLASSIFICATION ###############################

    clf = GradientBoostingClassifier() 
    clf.fit(data_train,labels_train)
    y_pred = clf.predict(data_train)

    decisions = clf.predict_proba(data_test)[:,1]
    decisions_train = clf.predict_proba(data_train)[:,1]
    fpr_boo,tpr_boo, thresholds_boo = roc_curve(labels_test,decisions,pos_label=6)

    
    
    ############################### SUPPORT VECTOR MACHINE ###############################
    
    
    aa = labels_train
    print labels_train
    A = np.ma.zeros((len(aa)))
    #A = A.tolist()
    for i in range(len(aa)):
        if aa[i]==3:
            o = 0
        else:
            o = 1
        A[i]=o
    labels_train = A
    print labels_train
    print labels_train.shape
    
    
    aa = labels_test
    B = np.ma.zeros((len(aa)))
    #B = B.tolist()
    for i in range(len(aa)):
        if aa[i]==3:
            o = 0
        else:
            o = 1
        B[i]=o
    labels_test = B
    
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

    imax = np.argmax(aucs)
    print "Optimal weight is ",ws[imax]
    print "peform SVM"
    wgt = {0:1,1:ws[imax]}
    model = SVC(class_weight=wgt)
    model.fit(data_train,labels_train)
    labels_pred=model.predict(data_test)

    print (labels_pred==labels_test).mean()

    decisions = model.decision_function(data_test)
    fpr_SVM,tpr_SVM,thr_SVM = roc_curve(labels_test,decisions)

    
    
    
    ############################### RESULT ROC CURVES ###############################
    
    fig = plt.figure(figsize = (7,7))
    fig1=fig.add_subplot(1,1,1)
    fig1.plot(fpr_DT,tpr_DT,label="ROC curve Decision Tree", color='springgreen',linewidth=4)
    fig1.plot(fpr_RF,tpr_RF,label="ROC curve Random Forest", color='hotpink',linewidth=4)
    fig1.plot(fpr_boo,tpr_boo,label="ROC curve Boosting classification", color='orange',linewidth=4)
    fig1.plot(fpr_SVM,tpr_SVM,label="ROC curve Support Vector Machine", color='deepskyblue',linewidth=4)
    fig1.set_xlabel("False positive Rate", fontsize=15) 
    fig1.set_ylabel("True positive Rate", fontsize=15)
    fig1.legend(loc='best',prop={'size': 15})
    plt.show()
    








