#Decision Tree Regression
#Link: https://scikit-learn.org/stable/auto_examples/tree/plot_tree_regression.html#sphx-glr-auto-examples-tree-plot-tree-regression-py

print(__doc__)
import sklearn, numpy as np, matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor as dtr 

#Create random data
data = np.random.RandomState(1)
x = np.sort(5* data.rand(80, 1), axis = 0)
y = np.sin(x).ravel()
y[::5] += 3*(0.5 - data.rand(16))

#Fit regression model
reg1 = dtr(max_depth = 100)
reg2 = dtr(max_depth = 1000)
reg1.fit(x,y)
reg2.fit(x,y)

#Predict
x_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y1 = reg1.predict(x_test)
y2 = reg2.predict(x_test)

#Plot results
plt.figure()
plt.scatter(x, y, s=20, edgecolors='black', c='darkorange', label = 'data')

plt.plot(x_test, y1, color = 'cornflowerblue', label = 'Max-depth = 100', linewidth = 2)
plt.plot(x_test, y2, color = 'yellowgreen', label = 'Max-depth = 1000', linewidth = 2)

plt.xlabel('Data')
plt.ylabel('Target')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()