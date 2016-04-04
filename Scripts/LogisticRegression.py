import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from astroML.utils import split_samples
from astroML.utils import completeness_contamination
from sklearn.metrics import roc_curve

signal = np.load('ClassSample_training_1.npy')
background = np.load('ClassSample_training_0.npy')
data = np.concatenate([signal, background])
data = data[~np.isnan(data).any(axis=1)]

(features_train, features_test), (labels_train, labels_test) = split_samples(data[:,:8], data[:,8], fractions=[0.75,0.25])

fig_roc = plt.figure()
ax_roc = fig_roc.add_subplot(111)
featureIDs = np.arange(2, data.shape[1]+1)
classification = []
predictions = []
scores = []
for i in featureIDs:
    logr = LogisticRegression()
    logr.fit(features_train[:,:i], labels_train)
    labels_pred = logr.predict_proba(features_test[:,:i])[:,1]
    classification.append(logr)
    predictions.append(labels_pred)
    fpr, tpr, thresholds = roc_curve(labels_test, labels_pred)
    ax_roc.plot(fpr, tpr, label="%d features"%(i))
#    print fpr, tpr
ax_roc.set_xlabel("False positive rate")
ax_roc.set_ylabel("True positive rate")
plt.legend(loc="lower right")
plt.show()
