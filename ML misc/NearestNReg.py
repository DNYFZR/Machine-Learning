#Nearest Neighbours Regression
#Link: https://scikit-learn.org/stable/auto_examples/neighbors/plot_regression.html#sphx-glr-auto-examples-neighbors-plot-regression-py

import numpy as np, matplotlib.pyplot as plt, sklearn
from sklearn import neighbors as nb 

np.random.seed(0)
x = np.sort(5 * np.random.rand(40,1), axis = 0)
t = np.linspace(0, 5, 1000000)[:, np.newaxis]
y = np.sin(x).ravel()

#Add noise
y[::5] += 1 * (0.5 - np.random.rand(8))

#Fit regression model
n_nb = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = nb.KNeighborsRegressor(n_nb, weights = weights)
    y_ = knn.fit(x,y).predict(t)

    plt.subplot(2, 1, i+1)
    plt.scatter(x, y, color = 'darkorange', label = 'Data')
    plt.plot(t, y_, color = 'navy', label = 'Prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("K-Nearest Neighbours Regression (k = %i, weights = '%s')" % (n_nb, weights))

plt.tight_layout()
plt.show()