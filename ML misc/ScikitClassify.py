#ML Native Bayes Classifier
#Gather Data
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as tts 
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

#Organise Data
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

#Review Data
print(label_names, max(labels), min(labels), feature_names[0], features[0], sep = '\n')

#Organise into datasets, 33% test data
train, test, train_labels, test_labels = tts(features, labels, test_size = 0.33, random_state = 42)

#Initialise the classifier
gnb = gnb()
model = gnb.fit(train, train_labels)

#Predict outcome
pred = gnb.predict(test)
print(pred[0:10])

#Evaluate accuracy
print('Model Accuracy:', round(100* accuracy_score(test_labels, pred)),'%')