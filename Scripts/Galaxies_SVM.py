import Classify_Galaxies_Parser
import numpy as np
import pylab as pl
from sklearn.svm import LinearSVC
from astroML.utils import completeness_contamination

np.random.seed(1)

# Load data, print some info
galaxy_parser = Classify_Galaxies_Parser.Galaxy_Parser('Galaxies_hands_on_Chap9_larger.txt', 
    precondition=True, replaceMean=True, trainfrac=0.8)

print galaxy_parser.data_test.shape
print galaxy_parser.data_train.shape
print galaxy_parser.datanames
print 'Num ellipticals: ', np.sum(galaxy_parser.labels_test)
print 'Num non-ellipticals: ', np.sum(1.-galaxy_parser.labels_test)

# Create and fit SVM classifier for different c
cs = 10**np.linspace(0., 1, 20)
contaminations = np.zeros_like(cs)
completenesses = np.zeros_like(cs)
for i, C in enumerate(cs):
    svm = LinearSVC(loss='squared_hinge', C=C)
    svm.fit(galaxy_parser.data_train, galaxy_parser.labels_train)

    # Evaluate SVM classifier
    predicted_labels = svm.predict(galaxy_parser.data_test)
    completeness, contamination = completeness_contamination(predicted_labels,
         galaxy_parser.labels_test)
    contaminations[i] = contamination
    completenesses[i] = completeness

pl.semilogx(cs, contaminations, '*-', label='contamination')
pl.semilogx(cs, completenesses, '*-', label='completeness')
pl.legend(loc='best')
pl.xlabel('C')


pl.show()