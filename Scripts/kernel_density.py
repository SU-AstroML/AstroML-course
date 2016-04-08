import numpy as np
from astroML.classification import GMMBayes
from sklearn.metrics import roc_curve
import pylab as pl

training_data0 = np.load('ClassSample_training_0.npy')
training_data1 = np.load('ClassSample_training_1.npy')
training_data_full = np.concatenate((training_data0, training_data1))


training_data_full = training_data_full[~np.isnan(training_data_full).any(axis=1)]
training_data0 = training_data0[~np.isnan(training_data0).any(axis=1)]
training_data1 = training_data1[~np.isnan(training_data1).any(axis=1)]


pg = (1.*len(training_data1))/(1.*len(training_data_full))
pp = (1.*len(training_data0))/(1.*len(training_data_full))


np.random.shuffle(training_data_full)

n_samples = training_data_full.shape[0]
training_set = training_data_full[:n_samples*0.8, :]
test_set = training_data_full[n_samples*0.8:, :]

features = training_set[:, :-1]
labels = training_set[:, -1]

# Fit and plot
pl.figure()
ax1 = pl.subplot(121)
ax2 = pl.subplot(122)
for n_clusters in [2]:
    print n_clusters
    gmmb = GMMBayes(n_clusters)
    gmmb.fit(features, labels)
    scores = gmmb.predict_proba(features)

    fpr, tpr, thresholds = roc_curve(labels, scores[:,1])

    ax1.plot(fpr, tpr, label='%d clusters'%n_clusters)
    if n_clusters == 15:
        ax2.hist(scores[labels==0, 1], bins=100, normed=True, 
            alpha=0.5, label='Background')
        ax2.hist(scores[labels==1, 1], bins=100, normed=True, 
            alpha=0.5, label='Signal')
        ax2.legend(loc='best')

#np.save('fpr_tpr_GMMBayes_15clusters.npy', np.vstack([fpr, tpr]))

ax1.set_xlabel('False positive rate')
ax1.set_ylabel('True positive rate')
ax1.legend(loc='best')

# Predict on test set
predictions = gmmb.predict_proba(test_set[:, :-1])
print predictions.shape
predictions = predictions[:,0]
true_labels = test_set[:, -1]

#Find 50% point for background: 
print "pp"  ,pp
cut50 = np.percentile(predictions[true_labels==0],100.*(1.-0.5/pp))

print cut50

print "false positive probability given 0"
#print cut50<predictions[true_labels==0.]
print (1.*(cut50<predictions[true_labels==0.]).sum())/(1.*len(predictions[true_labels==0]))
print "false positive probability for all events"
#print cut50<predictions[true_labels==0.]
print (1.*(cut50<predictions[true_labels==0.]).sum())/(1.*len(predictions))

#print predictions

#testfpt= 1.*(predictions==true_labels and true_labels==1).sum()/(1.*len(predictions))
print "true positive fraction:" 

#pl.show()
