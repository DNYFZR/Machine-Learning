#Individual & Voting Regression Predictions
print(__doc__)
import sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes as data
from sklearn.ensemble import GradientBoostingRegressor as GBR, RandomForestRegressor as RFR, VotingRegressor as VR 
from sklearn.linear_model import LinearRegression as LReg 

X, y = data(return_X_y= True)

#Train classifiers
reg1 = GBR(random_state = 2)
reg2 = RFR(random_state = 2)
reg3 = LReg()

reg1.fit(X,y)
reg2.fit(X,y)
reg3.fit(X,y)

ereg = VR([('GBR', reg1), ('RFR', reg2), ('LReg', reg3)])
print(ereg.fit(X,y))

#Predictions
xt = X[:20]

pred1 = reg1.predict(xt)
pred2 = reg2.predict(xt)
pred3 = reg3.predict(xt)
pred4 = ereg.predict(xt)

#Plot
plt.plot()
plt.plot(pred1, label = 'Gradient Boost')
plt.plot(pred2, label = 'Random Forest')
plt.plot(pred3, label = 'Linear')
plt.plot(pred4, ms = 10, label = 'Voting')

plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
plt.ylabel('Predicted')
plt.xlabel('Training Samples')
plt.legend(loc = 'best')
plt.title('Regression Model Predictions & Averages')

plt.show()