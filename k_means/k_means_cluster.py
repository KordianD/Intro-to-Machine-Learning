#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""


import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cluster import KMeans
import numpy as np


def transform(org, destination):
    outsize = 0
    with open(org, 'rb') as infile:
        content = infile.read()

    with open(destination, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))

original = "../final_project/final_project_dataset.pkl"
destination = "../final_project/final_project_dataset_unix.pkl"

transform(org=original, destination=destination)

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii, pp in enumerate(pred):
        plt.scatter(features[ii][0], features[ii][1], color = colors[pred[ii]])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii, pp in enumerate(pred):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
data_dict = pickle.load(open(destination, "rb"))
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = 'salary'
feature_2 = 'exercised_stock_options'
feature_3 = 'total_payments'
poi = "poi"
features_list = [poi, feature_1, feature_2, feature_3]
data = featureFormat(data_dict, features_list)
poi, finance_features = targetFeatureSplit(data)

def min_max(data, position):
    finance = np.array(finance_features)
    finance = finance[:, position]
    max_value = np.max(finance)

    # Change 0 which was set because of Nan to maximal value
    # It will not affect our minimal value (we must not include 0)
    finance[finance == 0] = max_value
    min_value = np.min(finance)
    return min_value, max_value

min_stock_value, max_stock_value = min_max(finance_features, 1)
print('Min : ' + str(min_stock_value))
print('Max : ' + str(max_stock_value))

min_salary_value, max_salary_value = min_max(finance_features, 0)
print('Min : ' + str(min_salary_value))
print('Max : ' + str(max_salary_value))

### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2, _ in finance_features:
    plt.scatter(f1, f2)
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred

classifier = KMeans(n_clusters=2)
classifier.fit(finance_features)
prediction = classifier.predict(finance_features)

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(prediction, finance_features, poi, mark_poi=False, name="2_clusters", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print("no predictions object named prediction found, no clusters to plot")
