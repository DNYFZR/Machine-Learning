# Outlier Detection 
import numpy as np, matplotlib.pyplot as plt, sklearn 
from sklearn.neighbors import LocalOutlierFactor as LOF
np.random.seed(88)

# Input data
xi =  0.333 * np.random.randn(100,2)
xi = np.r_[xi+2, xi-2]

#  Add outliers
xo = np.random.uniform(low = -4, high = 4, size = (20,2))
x = np.r_[xi, xo]

n_out = len(xo)
g_truth = np.ones(len(x), dtype = int)
g_truth[-n_out:] = -1

# Build model 
clf = LOF(n_neighbors = 16, contamination = 0.1, leaf_size = 32)
y_pred = clf.fit_predict(x)
n_err = (y_pred != g_truth).sum()
x_score = clf.negative_outlier_factor_

# Plot
plt.title('Local Outlier Factor')  
plt.scatter(x[:,0], x[:,1], color = 'k', s = 3, label = 'Data points')

#  Plot radius proportional to outlier score
radius = (x_score.max() - x_score) / (x_score.max() - x_score.min())
plt.scatter(x[:,0], x[:,1], s = 1000 * radius, edgecolors = 'r', facecolors = 'none', label = 'Outlier score')

plt.axis('tight')
plt.xlim(-5,5)
plt.ylim(-5,5)
plt.xlabel('Prediction errors: %d' % (n_err))

legend = plt.legend(loc = 'upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()
