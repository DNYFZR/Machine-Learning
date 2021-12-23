""" Multiple Polynomial Regression """
# Naval Propulsion Data Analysis
# Data source: http://archive.ics.uci.edu/ml/datasets/condition+based+maintenance+of+naval+propulsion+plants

import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
sns.set()

# Import / inspect
data = pd.read_csv(r'data.txt', delimiter = '   ', engine = 'python', header = None)
headers = pd.read_csv(r'Features.txt', header = None)
data.columns = [i.split(' - ')[1] for i in headers.iloc[:,0]]
print('Raw data:', '\n') 
print(data.info(), '\n')

# Prepare the data
data['T_delta'] = data['GT Compressor outlet air temperature (T2) [C]'] - data['GT Compressor inlet air temperature (T1) [C]']
data['P_delta'] = data['GT Compressor outlet air pressure (P2) [bar]'] - data['GT Compressor inlet air pressure (P1) [bar]']

data_mlr = data[['Fuel flow (mf) [kg/s]','T_delta', 'P_delta', 'Gas Turbine shaft torque (GTT) [kN m]']]
data_mlr = data_mlr.rename(columns = {'Fuel flow (mf) [kg/s]': 'mass_flow', 'Gas Turbine shaft torque (GTT) [kN m]': 'W_shaft'})
data_mlr['P_delta'] = data_mlr['P_delta'] * 100
print('Model data:', '\n', data_mlr.describe(), '\n')

# Build model
scaler = MinMaxScaler()
scaler.fit(data_mlr)

features = pd.DataFrame([i for i in scaler.transform(data_mlr)], columns = data_mlr.columns)
x = features.iloc[:, :-1]
y = features.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

model = LinearRegression()
model.fit(x_train, y_train)

# Evaluate the model
y_pred = model.predict(x_test)

mean_err = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

print('Mean square error:', mean_err.round(4), '\n', 
        'R squared:', r_squared.round(4), '\n', 
        'Model coefficients:', model.coef_.round(4), '\n', 
        'Model intercept:', model.intercept_.round(4), '\n')

# Plot results
fig, ax = plt.subplots(1, 3, figsize = (18,6))

ax[0].scatter(x_test['P_delta'], y_test, c = 'orange')
ax[0].plot(x_test['P_delta'], y_pred, c = 'b')
ax[0].set_title('Shaft work vs. Pressure change')
ax[0].set_xlabel('Pressure Change (kPa)')
ax[0].set_ylabel('Shaft Work (kJ)')

ax[1].scatter(x_test['T_delta'], y_test, c = 'orange')
ax[1].plot(x_test['T_delta'], y_pred, c = 'b')
ax[1].set_title('Shaft work vs. Temperature change')
ax[1].set_xlabel('Temperature Change (K)')
ax[1].set_ylabel('Shaft Work (kJ)')

ax[2].scatter(x_test['mass_flow'], y_test, c = 'orange')
ax[2].plot(x_test['mass_flow'], y_pred, c = 'b')
ax[2].set_title('Shaft work vs. Mass flow')
ax[2].set_xlabel('Mass Flow (kg/s)')
ax[2].set_ylabel('Shaft Work (kJ)')