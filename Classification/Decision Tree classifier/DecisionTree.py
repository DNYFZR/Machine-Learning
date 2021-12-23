""" Decision Tree Example """
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris

""" Prepare data"""
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
iris_df['target'] = iris.target
x_train, x_test, y_train, y_test = train_test_split(iris_df[iris.feature_names], 
                                                    iris_df['target'], 
                                                    random_state = 0, 
                                                    test_size = 0.3)
""" Model build & testing """
decision_tree = DecisionTreeClassifier(criterion = 'entropy', 
                                       max_depth = 4, random_state = 88)
decision_tree.fit(x_train, y_train)
decision_tree.predict(x_test)

""" Model performance check """
print('Model accuracy: ', 
      round(100 * decision_tree.score(x_test, y_test),1), '%\n'
     )

print('Feature Importance: \n',
      pd.Series(decision_tree.feature_importances_.round(3), 
                index = x_test.columns)
     )

""" Visualise the tree and save""" 
names = iris.feature_names
targets = iris.target_names
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (8,5), dpi=800)

tree.plot_tree(decision_tree,
               feature_names = names,
               class_names = targets,
               filled = True)

fig.savefig('DecisionTree.png')