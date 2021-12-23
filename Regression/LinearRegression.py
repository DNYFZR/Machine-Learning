""" Linear Regression """
import numpy as np, matplotlib.pyplot as plt, seaborn as sns; sns.set()
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Set up the base data
x = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
y = np.array([3.1, 5.2, 6.8, 9.1, 10.9, 12.8, 15.4, 17.2, 19.01, 21.27])

# Build the model
model = LinearRegression(fit_intercept = True)
model.fit(x.reshape(-1,1), y)
y_pred = model.predict(x.reshape(-1,1))

xfit = np.linspace(0,x.max(),1000)
yfit = model.predict(xfit.reshape(-1,1))

# Plot data and model
plt.scatter(x,y, label = 'Base data')
plt.plot(xfit, yfit, label = 'Linear Fit')
plt.title('Linear Regression \nscikit-learn')

print("Model slope:           ", model.coef_[0].round(3))
print("Model intercept:       ", model.intercept_.round(3))
print('Model r2 score:        ', (100 * r2_score(y, y_pred)).round(3),'%')

plt.savefig('SimpleLinReg.png', dpi = 600)
