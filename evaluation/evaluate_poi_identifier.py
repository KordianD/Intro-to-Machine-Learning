#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data_dict = pickle.load(open("../final_project/final_project_dataset_unix.pkl", "rb") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson14_keys_unix.pkl')
labels, features = targetFeatureSplit(data)


### your code goes here 


features_train, features_test, labels_train, labels_test = train_test_split(features, labels, random_state=42,
                                                                            test_size=0.3)
																			

classifier = DecisionTreeClassifier()
classifier.fit(features_train, labels_train)
print(classifier.score(features_test, labels_test))

prediction = classifier.predict(features_test)

print('Prediction')
print(prediction)

print('How many elements')
print(len(labels_test))

print('True labels for test_data')
print(labels_test)


counter_tp = 0

for index, elem in enumerate(labels_test):
	if elem == 1 and prediction[index] == 1:
		counter_tp += 1
		
print('True positive')
print(counter_tp)
	
print("Precision: {}".format(metrics.precision_score(labels_test, prediction)))
print("Recall: {}".format(metrics.recall_score(labels_test, prediction)))


