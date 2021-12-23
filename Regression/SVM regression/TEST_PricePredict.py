import numpy as np
import pandas as pd
df = pd.read_csv(r"data\bitcoin.csv")
df.head()

df.drop(['Date'],1,inplace=True)

predictionDays = 30
# Create another column shifted 'n'  units up
df['Prediction'] = df[['Price']].shift(-predictionDays)
# show the first 5 rows
df.head()

# Create the independent dada set
# Here we will convert the data frame into a numpy array and drp the prediction column
x = np.array(df.drop(['Prediction'],1))
# Remove the last 'n' rows where 'n' is the predictionDays
x = x[:len(df)-predictionDays]
#print(x)

# Create the dependent data set
# convert the data frame into a numpy array
y = np.array(df['Prediction'])
# Get all the values except last 'n' rows
y = y[:-predictionDays]
#print(y)

# Split the data into 80% training and 20% testing
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2)
# set the predictionDays array equal to last 30 rows from the original data set
predictionDays_array = np.array(df.drop(['Prediction'],1))[-predictionDays:]
#print(predictionDays_array)

from sklearn.svm import SVR
# Create and Train the Support Vector Machine (Regression) using radial basis function
svr_rbf = SVR(kernel='rbf', C=1e9, gamma=0.00001)
svr_rbf.fit(xtrain, ytrain)
print('\n', svr_rbf.fit(xtrain, ytrain))
svr_rbf_confidence = svr_rbf.score(xtest,ytest)
print('\nSVR_RBF accuracy :',svr_rbf_confidence)

svm_prediction = svr_rbf.predict(xtest)

# Print the model predictions for the next 30 days
svm_prediction = svr_rbf.predict(predictionDays_array)
#print(svm_prediction)

#print(df['Price'].tail(predictionDays))

import matplotlib.pyplot as plt
plt.scatter(xtest[-30:], ytest[-30:])
plt.scatter(xtest[-30:],svm_prediction);
