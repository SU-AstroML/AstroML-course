import numpy as np
from astroML.datasets import fetch_LINEAR_geneva
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
import pylab as plt
from astroML.utils import completeness_contamination

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues,
                            n_classes=5):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.tick_marks = np.arange(n_classes)
    #plt.xticks(tick_marks, iris.target_names, rotation=45)
    #plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

data = fetch_LINEAR_geneva()

attributes = ['gi', 'logP', 'ug', 'iK', 'JK', 'amp', 'skew']
cls = 'LCtype'
Ntrain = 3000

#------------------------------------------------------------
# Create attribute arrays
X = []
y = []

X.append(np.vstack([data[a] for a in attributes]).T)
X = np.array(X).squeeze()
y = data[cls].copy()

# Create training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                test_size=0.2, random_state=42)

# Train the classifier
C_vals = 10**np.linspace(-2, 2, 15)
completeness = np.zeros((len(C_vals), 5))
contamination = np.zeros_like(completeness)

for i, C in enumerate(C_vals):
    clf = SVC(kernel='linear', class_weight=None, C=C)
    clf.fit(X_train, y_train)

    # Predict unknown values
    y_pred = clf.predict(X_test)

    # Compute confusion matrix
    #cm = confusion_matrix(y_test, y_pred)
    #print 'Confusion matrix:'
    #print cm

    # Completeness/contamination
    for j, c in enumerate([1, 2, 4, 5, 6]):
        #print 'Class:', c
        compl_i, cont_i = completeness_contamination(y_pred==c, 
            y_test==c)
        completeness[i, j] = compl_i
        contamination[i, j] = cont_i
        #print 'Completeness:', completeness
        #print 'Contamination:', contamination

# Plot matrix
#plot_confusion_matrix(cm)
#plt.show()


# Plot contamination, completeness
plt.subplot(211)
plt.semilogx(C_vals, completeness)
plt.xlabel('C')
plt.ylabel('Completeness')
plt.subplot(212)
plt.semilogx(C_vals, contamination)
plt.xlabel('C')
plt.ylabel('Contamination')
plt.show()