#Anomaly Detection
#Link: https://www.johnwittenauer.net/machine-learning-exercises-in-python-part-8/
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sb 
from scipy.io import loadmat

data = loadmat('data/ex8data1.mat')
X = data['X']
print('Data shape:',X.shape)

#Tight cluster in the center with several that could be considered anomalies
fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(X[:,0], X[:,1]);

#Estimating a Gaussian distribution for each feature in the data
def estimate_Gaussian(X):
    mu = X.mean(axis = 0)
    sigma = X.var(axis = 0)
    return mu, sigma

mu, sigma = estimate_Gaussian(X)
print('Mean:', mu.round(3))
print('Variance:', sigma.round(3))

#Determine probability threshold which indicates that an example should be considered an anomaly.
# Use labeled validation data (true anomalies marked) and test model perf at identifying anomalies 
Xval = data['Xval']
yval = data['yval']

from scipy import stats
#computing how far each instance is from the mean and how that compares to the "typical" distance from the mean
dist = stats.norm(mu[0], sigma[0])

p = np.zeros((X.shape[0], X.shape[1]))
p[:,0] = stats.norm(mu[0], sigma[0]).pdf(X[:,0])
p[:,1] = stats.norm(mu[1], sigma[1]).pdf(X[:,1])

#determine the optimal probability threshold to assign data points as anomalies
pval = np.zeros((Xval.shape[0], Xval.shape[1]))
pval[:,0] = stats.norm(mu[0], sigma[0]).pdf(Xval[:,0])
pval[:,1] = stats.norm(mu[1], sigma[1]).pdf(Xval[:,1])

#Function that finds the best threshold value given the probability density values and true labels
# F1 is a function of the number of true positives, false positives, and false negatives
def select_threshold(pval, yval):
    best_epsilon = 0
    best_f1 = 0
    f1 = 0
    step = (pval.max() - pval.min()) / 1000

    for epsilon in np.arange(pval.min(), pval.max(), step):
        preds = pval < epsilon

        tp = np.sum(np.logical_and(preds == 1, yval == 1)).astype(float)
        fp = np.sum(np.logical_and(preds == 1, yval == 0)).astype(float)
        fn = np.sum(np.logical_and(preds == 0, yval == 1)).astype(float)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1> best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1

epsilon, f1 = select_threshold(pval, yval)
print('Epsilon:', epsilon)
print('F1:', f1)

#Apply to the data
outliers = np.where(p < epsilon)

fig, ax = plt.subplots(figsize = (8,4))
ax.scatter(X[:,0], X[:,1])
ax.scatter(X[outliers[0],0], X[outliers[0],1], s = 50, color = 'r', marker = 'o')
