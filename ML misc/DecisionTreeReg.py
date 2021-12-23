#Decision Tree Regression
#   1D regression to fit a curve with noise, using local linear reg to approximate the curve. 

#Setup
import numpy as np, matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor as dtr 

#Load Data
rng = np.random.RandomState(1)
x = np.sort(5 * rng.rand(80, 1), axis = 0)
y = np.sin(x).ravel()
y[::5] += 3*(0.5 * rng.rand(16))

#Fit regression model
r1 = dtr(max_depth = 3.375)
r2 = dtr(max_depth = 5)

r1.fit(x, y)
r2.fit(x, y)

#Test model
x_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y1 = r1.predict(x_test)
y2 = r2.predict(x_test)

#Visualise
plt.figure()
plt.scatter(x, y, s = 20, edgecolor = 'black', c = 'darkorange', label = 'data')
plt.plot(x_test, y1, color = 'cornflowerblue', label = 'max depth = 3', linewidth = 2)
plt.plot(x_test, y2, color = 'green', label = 'max depth = 5', linewidth = 2)
plt.xlabel('Date')
plt.ylabel('Target')
plt.title('Decision Tree Regression')
plt.legend()
plt.show()