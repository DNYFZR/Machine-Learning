# k-Nearest Neighbours Regression
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from numpy.lib.financial import _ipmt_dispatcher
from sklearn.neighbors import KNeighborsRegressor

sns.set()
np.random.seed(88)

# Input data
x = np.sort(8 * np.random.rand(4000,1), axis = 0)
t = np.linspace(0, 8, 1000)[:, np.newaxis]
y = np.cos(x).ravel()

#   Add noise every 5 entries
y[::2] += 1 * (0.8 - np.random.rand(2000))

# Build kNN regression model
neighbors_n = 10
scores = {}

figure = plt.subplots()
for n in range(2, neighbors_n + 1):
    knn = KNeighborsRegressor(n_neighbors = n, weights = 'uniform', leaf_size = 8, metric = 'minkowski')
    knn = knn.fit(x,y)
    y_pred = knn.predict(t)
    scores[n] =  knn.score(x, y)

    plt.figure()
    plt.scatter(x, y, color = 'navy', label = 'Data')
    plt.plot(t, y_pred, color = 'gray', label = 'Prediction')
    plt.axis('tight')
    plt.legend()
    plt.title(f'kNN Regression \n {n}-neighbors')

plt.tight_layout()
plt.show()

plt.plot(scores.values())