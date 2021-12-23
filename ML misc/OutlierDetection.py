#Outlier Detection
#Link: https://scikit-learn.org/stable/auto_examples/applications/plot_outlier_detection_wine.html#sphx-glr-auto-examples-applications-plot-outlier-detection-wine-py
print(__doc__)

#Example 1 - Main Covariance Determinant on clusters
import numpy as np, matplotlib.pyplot as plt, matplotlib.font_manager as plt_fm, sklearn
from sklearn.covariance import EllipticEnvelope as EE  
from sklearn.svm import OneClassSVM
from sklearn.datasets import load_wine as sk_data

#1. Define classifiers
classifiers = {'Emperical Covariance':EE(support_fraction = 1, contamination = 0.25), 
                'Robust Covariance (MCD)':EE(contamination = 0.25), 
                'One Class Support Vector Machine':OneClassSVM(nu = 0.25, gamma = 0.35)}
colors = ['m', 'g', 'b']
legend1 = {}
legend2 = {}

#2. Get data
x1 = sk_data()['data'][:,[1,2]] #Two clusters

#3. Outlier frontier detection
xx1, yy1 = np.meshgrid(np.linspace(0,6,500), np.linspace(1,4.5,500))

for i, (clf_name, clf) in enumerate(classifiers.items()):
    plt.figure(1)
    clf.fit(x1)
    z1 = clf.decision_function(np.c_[xx1.ravel(), yy1.ravel()])
    z1 = z1.reshape(xx1.shape)
    legend1[clf_name] = plt.contour(xx1, yy1, z1, levels = [0], linewidths = 2, colors = colors[i])

legend1_val_list = list(legend1.values())
legend1_keys = list(legend1.keys())

#4. Plot resutls
plt.figure(1)
plt.title('Outlier Detection \n Cluster Analysis')
plt.scatter(x1[:,0], x1[:,1], color = 'black')

bbox_args = dict(boxstyle = 'round', fc = '0.8')
arrow_args = dict(arrowstyle = '->')

plt.annotate('Outlying Points', xy=(4,2), xycoords = 'data', textcoords = 'data', xytext = (3,1.25), bbox = bbox_args, arrowprops = arrow_args)
plt.xlim((xx1.min(), xx1.max()))
plt.ylim((yy1.min(), yy1.max()))
plt.legend((legend1_val_list[0].collections[0],
            legend1_val_list[1].collections[0],
            legend1_val_list[2].collections[0]),
            (legend1_keys[0], legend1_keys[1],legend1_keys[2]),
            loc = 'upper center',
            prop = plt_fm.FontProperties(size = 11))
plt.ylabel('Ash')
plt.xlabel('Malic Acid')

#plt.show()

#Example 2 - Main Covarniance Determinant on data distribution

#1. Get Data
x2 = sk_data()['data'][:,[6,9]]

#2. Outlier Detection Frontier
xx2, yy2 = np.meshgrid(np.linspace(-1, 5.5, 500), np.linspace(-2.5, 19, 500)) 
for i, (clf_name, clf) in enumerate(classifiers.items()):
    plt.figure(2)
    clf.fit(x2)
    z2 = clf.decision_function(np.c_[xx2.ravel(), yy2.ravel()])
    z2 = z2.reshape(xx2.shape)
    legend2[clf_name] = plt.contour(xx2, yy2, z2, levels=[0], linewidths = 2, colors = colors[i])

legend2_vals = list(legend2.values())
legend2_keys = list(legend2.keys())

#Plot Results
plt.figure(2)
plt.title('Outlier Detection \n Data Distribution')
plt.scatter(x2[:,0], x2[:,1], color = 'black')

plt.xlim((xx2.min(), xx2.max()))
plt.ylim((yy2.min(), yy2.max()))
plt.legend((legend2_vals[0].collections[0],
            legend2_vals[1].collections[0],
            legend2_vals[2].collections[0]),
            (legend2_keys[0], legend2_keys[1],legend2_keys[2]),
            loc = 'upper center',
            prop = plt_fm.FontProperties(size = 11))
plt.ylabel('Color Intensity')
plt.xlabel('Flavanoids')


fig,ax = plt.subplots(2)
ax[0].scatter(x1[:,0], x1[:,1])
ax[1].scatter(x2[:,0], x2[:,1])

plt.show()