#Gradient Boosting Regression
#Link: https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html#sphx-glr-auto-examples-ensemble-plot-gradient-boosting-regression-py

import sklearn, numpy as np, matplotlib.pyplot as plt
from sklearn import datasets, ensemble
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

#Data processing
ds = datasets.load_diabetes()
x,y = ds.data, ds.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 13)
params = {'n_estimators':500, 
          'max_depth':4, 
          'min_samples_split':5, 
          'learning_rate':0.01}

#Fit regression model
reg = ensemble.GradientBoostingRegressor(**params)
reg.fit(x_train, y_train)
mse = mean_squared_error(y_test, reg.predict(x_test))
print('Mean square error: {:.1f}'.format(mse))

#Plot training deviation
test_score = np.zeros((params['n_estimators'],), dtype = np.float64)

for i, y_pred in enumerate(reg.staged_predict(x_test)):
    test_score[i] = reg.loss_(y_test, y_pred)

fig = plt.figure(figsize = (6,6))
plt.subplots(1,1)
plt.title('Deviation')
plt.plot(np.arange(params['n_estimators']) +1, reg.train_score_, 'b-', label = 'Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) +1, test_score, 'r-', label = 'Test Set Deviance')
plt.legend(loc = 'upper right')
plt.xlabel('Boosting Itterations')
plt.ylabel('Deviance')
fig.tight_layout()
plt.show()

#Plot feature importance
feat_imp = reg.feature_importances_
sorted_idx = np.argsort(feat_imp)
pos = np.arange(sorted_idx.shape[0]) + 0.5

fig = plt.figure(figsize = (12,6))
plt.subplot(1,2,2)
plt.barh(pos, feat_imp[sorted_idx], align = 'center')
plt.yticks(pos, np.array(ds.feature_names)[sorted_idx])
plt.title('Feature Importance (MDI)')

result = permutation_importance(reg, x_test, y_test, n_repeats = 10, random_state = 42, n_jobs = 2)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1,2,1) 
plt.boxplot(result.importances[sorted_idx].T, vert = False, labels = np.array(ds.feature_names)[sorted_idx])
plt.title('Permutation Importance (test set)')
fig.tight_layout()
plt.show()