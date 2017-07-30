#!/usr/bin/python
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.


word_data = open("../feature_selection/your_word_data.pkl", "rb")
word_data = pickle.load(word_data, fix_imports=True)

authors = open("../feature_selection/your_email_authors.pkl", "rb")
authors = pickle.load(authors, fix_imports=True)

### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier

features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test = vectorizer.transform(features_test).toarray()


### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train = labels_train[:150]

classifier = DecisionTreeClassifier()
classifier.fit(features_train, labels_train)

prediction = classifier.predict(features_test)

result = classifier.score(features_test, labels_test)
print(result)

feature_number = 0

for elem in classifier.feature_importances_:
    if elem >= 0.2:
        print(feature_number, elem, vectorizer.get_feature_names()[feature_number])

    feature_number += 1

