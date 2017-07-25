#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

original = "../final_project/final_project_dataset.pkl"
destination = "../final_project/final_project_dataset_unix.pkl"

content = ''
outsize = 0
with open(original, 'rb') as infile:
    content = infile.read()

with open(destination, 'wb') as output:
    for line in content.splitlines():
        outsize += len(line) + 1
        output.write(line + str.encode('\n'))

enron_data = pickle.load(open(destination, "rb"))

# How many data points (i.e people) are in the Enron dataset?
print(len(enron_data))

# For each person, how many features are available?
print(len(enron_data['FASTOW ANDREW S']))


# The “poi” feature records whether the person
# is a person of interest, according to our definition.
# How many POIs are there in the E+F dataset?
number_of_poi = 0
for elem in enron_data:
    if enron_data[elem]['poi'] == 1:
        number_of_poi += 1
print(number_of_poi)


# How Many POIs Exist?
names = open('../final_project/poi_names.txt', 'r')
names.readline()
names.readline()

number_of_names = 0
for elem in names:
    number_of_names += 1

print(number_of_names)

# What is the total value of the stock belonging to James Prentice?
print(enron_data['PRENTICE JAMES']['total_stock_value'])

# How many email messages do we have from Wesley Colwell to persons of interest?
print(enron_data['COLWELL WESLEY']['from_this_person_to_poi'])

# What’s the value of stock options exercised by Jeffrey K Skilling?
print(enron_data['SKILLING JEFFREY K']['exercised_stock_options'])

# Of these three individuals (Lay, Skilling and Fastow),
# who took home the most money (largest value of “total_payments” feature)?
people = {'LAY KENNETH L', 'SKILLING JEFFREY K', 'FASTOW ANDREW S'}
money = 0
name = ''

for elem in people:
    if enron_data[elem]['total_payments'] > money:
        name = elem
        money = enron_data[elem]['total_payments']

print(name + ' ' + str(money))

# How is it denoted when a feature doesn’t have a well-defined value?
print('NaN')

# How many folks in this dataset have a quantified salary?
quantified_salary = 0
for elem in enron_data:
    if enron_data[elem]['salary'] != 'NaN':
        quantified_salary += 1

print(quantified_salary)

# Known email address?
email_address = 0
for elem in enron_data:
    if enron_data[elem]['email_address'] != 'NaN':
        email_address += 1

print(email_address)

# How many people in the E+F dataset (as it currently exists) have “NaN” for their total payments?
# What percentage of people in the dataset as a whole is this?
number_of_Nan = 0
for elem in enron_data:
    if enron_data[elem]['total_payments'] == 'NaN':
        number_of_Nan += 1

print(number_of_Nan)
print(number_of_Nan / len(enron_data) * 100)

# How many POIs in the E+F dataset have “NaN” for their total payments?
# What percentage of POI’s as a whole is this?

poi_total_payments_nan = 0

for elem in enron_data:
    if enron_data[elem]['poi'] == 1 and enron_data[elem]['total_payments'] == 'NaN':
        poi_total_payments_nan += 1

print(poi_total_payments_nan)
print(poi_total_payments_nan / number_of_poi * 100)

# What is the new number of people of the dataset?
# What is the new number of folks with “NaN” for total payments?

print(len(enron_data) + 10)
print(number_of_Nan + 10)

# What is the new number of POI’s in the dataset?
# What is the new number of POI’s with NaN for total_payments?

print(number_of_poi + 10)
print(poi_total_payments_nan + 10)


