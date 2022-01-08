'''Setup'''
import pandas as pd, matplotlib.pyplot as plt, seaborn as sns
sns.set()

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import SGDRegressor

'''Data'''
df = pd.read_csv(r'data\CO2_vs_population.csv', encoding='utf-8')
df.columns = ['Year', 'CO2 (tons)', 'Pop', 'MT CO2', 'Million people']

forecast_pop = pd.read_csv(r'data\pop_forecast.csv', encoding='utf-8')
forecast_pop['Total population'] = forecast_pop['Total population'] / 1e6
forecast_pop = forecast_pop['Total population'].to_numpy()

min_year = 2000# df['Year'].min()

x = df[df['Year'] >= min_year]['Million people'].copy().to_numpy().reshape(-1,1)
y = df[df['Year'] >= min_year]['MT CO2'].copy().to_numpy()

'''ML Model'''
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

applied_model = SGDRegressor(max_iter=1e6)
model_performance = {}
pipeline = Pipeline(
    [
        ('scalar', MinMaxScaler()),
        ('poly feature', PolynomialFeatures(2)),
        ('regression', applied_model),
    ])

pipeline.fit(x_train, y_train)
model_performance = round(100 * pipeline.score(x_test, y_test), 1)
print(f'Model training accuracy: {model_performance}%')

# Forecast on last model
pred_co2 = pipeline.predict(forecast_pop.reshape(-1,1))

# Chart
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (12, 8))
ax.plot(df[df['Year'] >= min_year]['Million people'], df[df['Year'] >= min_year]['MT CO2'])
ax.plot(forecast_pop, pred_co2)

ax.set_xlabel('Million People')
ax.set_ylabel('Million Tons CO2')
ax.set_title(f'Forecast Global CO2 vs. Population')

plt.show()