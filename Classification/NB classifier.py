# Naive Bayes Classifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split as tts 
from sklearn.naive_bayes import GaussianNB as gnb
from sklearn.metrics import accuracy_score, classification_report

# Prepare data
data = load_breast_cancer()
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']

# Split data
train, test, train_labels, test_labels = tts(features, labels, test_size = 0.2, random_state = 8)

# Build classifier
gnb = gnb()
model = gnb.fit(train, train_labels)

# Predict class
pred = gnb.predict(test)

# Measure accuracy
print(classification_report(test_labels, pred))

'''OUTPUT
              precision    recall  f1-score   support

           0       0.95      0.91      0.93        46
           1       0.94      0.97      0.96        68

    accuracy                           0.95       114
   macro avg       0.95      0.94      0.94       114
weighted avg       0.95      0.95      0.95       114
'''