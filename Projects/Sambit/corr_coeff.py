import numpy as np
import pylab as pl
from scipy import stats
from get_features import read_files, train_test, plot_prediction, add_asymmetry


features, labels = read_files()
features = add_asymmetry(features, np.arange(3), np.arange(3))
features = add_asymmetry(features, 3+np.arange(3), 3+np.arange(3))
features = add_asymmetry(features, 6+np.arange(3), 6+np.arange(3))
features = add_asymmetry(features, 9+np.arange(3), 9+np.arange(3))
ticks = np.array(['k2r_i', 'k3_i', 'k2v_i', 'h2r_i', 'h3_i', 'h2v_i', 'k2r_dv', 'k3_dv', 'k2v_dv', 'h2r_dv', 'h3_dv', 'h2v_dv'])

corr   = np.zeros((12,12))
pvalue = np.zeros((12,12))

for i in xrange(12):
	for j in xrange(12):
		corr[i,j], pvalue[i,j] = stats.spearmanr(features[:,i], features[:,j])

pl.figure('Spearman coefficient')
pl.title('Spearman coefficient')
pl.imshow(corr, interpolation='nearest', origin='low')
pl.colorbar()
pl.xticks(range(len(ticks)), ticks, size='small')
pl.yticks(range(len(ticks)), ticks, size='small')


corr1   = np.zeros((features.shape[1]))
pvalue1 = np.zeros((features.shape[1]))
for k in xrange(features.shape[1]):
	corr1[k], pvalue1[k] = stats.spearmanr(features[:,k], labels)

corr1[np.isnan(corr1)] = 0

pl.figure('Spearman coefficient, Input-Output')
pl.title('Spearman coefficient, Input-Output')
pl.plot(corr1)
pl.scatter(np.arange(len(corr1)), corr1)
pl.plot(np.zeros(corr1.shape), '--')
#pl.xticks(range(len(ticks)), ticks, size='small')

pl.show()


