#Price forecast model (bitcoin basis)
#Link: https://thecleverprogrammer.com/2020/05/23/bitcoin-price-prediction-with-machine-learning/

import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns;sns.set()

#Raw data
data = pd.read_csv(r'data\bitcoin.csv')
df = pd.DataFrame(data)
print('Raw Data Stats & Top Rows:\n',round(df.describe(),2),'\n', df.head())

#Isolate price data
df.drop(['Date'],1, inplace = True)

#30 day prediction
predDays = 30
df['Prediction'] = df[['Price']].shift(-predDays)

#Create model data
#1. convert price to array 
x = np.array(df.drop(['Prediction'],1))

#2. remove n predDays from test set
x = x[:len(df) - predDays]

#3. convert prediction to array
y = np.array(df['Prediction'])

#4. remove n predDays 
y = y[:-predDays]

#Split into 80% train and 20% tesst
from sklearn.model_selection import train_test_split
testfraction = 0.2
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = testfraction)
predDays_array = np.array(df.drop(['Prediction'], 1))[-predDays:]
#predDays_array.flatten()

#ML Model
from sklearn.svm import SVR

#1. build & train SVM regression using radial basis function
svr_rbf = SVR(C=1000.0, cache_size=200, coef0=0.0, degree=3, epsilon=0.1, gamma=1e-05,
                kernel='rbf', max_iter=-1, shrinking=True, tol=0.001, verbose=False)
svr_rbf.fit(xtrain, ytrain)
print('\n', svr_rbf.fit(xtrain, ytrain))

#2. test model
svr_rbf_conf = svr_rbf.score(xtest, ytest)
print('\nSVR radial accuracy:', svr_rbf_conf)

# print the predicted values
svm_prediction = svr_rbf.predict(xtest)
z = [ytest,svm_prediction]
print(z)
