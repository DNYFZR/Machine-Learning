#Time Series LTSM
#Link: https://thecleverprogrammer.com/2020/08/29/time-series-with-lstm-in-machine-learning/

import numpy as np, pandas as pd, math, matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#Seed for reproducability
np.random.seed(7)

#Data
data = pd.read_csv(r'data\airline-passengers.csv')
print('\nData Stats:\n', round(data.describe(),2),'\n')
dataset = pd.DataFrame(data['Passengers'])
dataset = dataset.values
dataset = dataset.astype('float32')

#Normalise
scaler = MinMaxScaler(feature_range = (0,1))
dataset = scaler.fit_transform(dataset)

#Train and Test
train_size = int(len(dataset) * (2/3))
test_size = len(dataset) - train_size

train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print(len(train), len(test))

#Time Series
#1. prepare datasets
def create_dataset(dataset, look_back = 1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        dataX.append(dataset[i:(i + look_back), 0])
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

#2. reshape the data for LSTM (X = t, Y = t + 1)
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#3. create model
model = Sequential()
model.add(LSTM(4, input_shape = (1, look_back)))
model.add(Dense(1))
model.compile(loss = 'mean_squared_error', optimizer = 'adam')
model.fit(trainX, trainY, epochs = 100, batch_size = 1, verbose = 2)

#4. predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

#5. error
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:,0]))

#5. Adjust for plot
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:,:] = np.nan
trainPredictPlot[look_back : len(trainPredict) + look_back, :] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:,:] = np.nan
testPredictPlot[(len(trainPredict) + 2 * look_back + 1) : len(dataset) - 1, :] = testPredict

#6. Plot
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show();