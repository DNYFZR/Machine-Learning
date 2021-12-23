""" Logistic Regression (scikit-learn) """

""" Step 1: Build Environment """
import numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
sns.set()

""" Step 2: Import / Generate Data """
x = np.arange(10).reshape(-1, 1)
y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1])

""" Step 3: Build & Train Model """
model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
model.fit(x, y)

""" Step 4: Evaluate The Model """
p_pred = model.predict_proba(x)
y_pred = model.predict(x)
score_ = model.score(x, y)
conf_m = confusion_matrix(y, y_pred)
report = classification_report(y, y_pred)

""" Step 5: Outputs """
print('intercept:', model.intercept_)
print('coef:', model.coef_, end='\n\n')
print('p_pred:', p_pred, sep='\n', end='\n\n')
print('y_pred:', y_pred, end='\n\n')
print('score_:', score_, end='\n\n')
print('conf_m:', conf_m, sep='\n', end='\n\n')
print('report:', report, sep='\n')

""" Step 6: Plot Output """
fig, ax = plt.subplots(1,2, figsize = (10,6))
ax[0].plot(x, y_pred)
ax[0].scatter(x, y)
ax[0].set_title('Logistic Regression Plot')

sns.heatmap(conf_m, annot = True, xticklabels = ['False', 'True'], 
            yticklabels = ['False', 'True'], cbar = False).invert_yaxis()
ax[1].set_title('Logistic Regression Confusion Matrix')
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('Actual')
