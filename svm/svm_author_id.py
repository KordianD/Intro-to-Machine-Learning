#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn import svm


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()



classifier = svm.SVC(kernel='rbf', C=10000)


t0 = time()
classifier.fit(features_train, labels_train)

print("training time:", round(time()-t0, 3), "s")

t1 = time()
prediction = classifier.predict(features_test)

cris = 0
for i in prediction:
    if i == 1:
        cris += 1

print(cris)

print("predicting time:", round(time()-t1, 3), "s")

print(classifier.score(features_test, labels_test))

