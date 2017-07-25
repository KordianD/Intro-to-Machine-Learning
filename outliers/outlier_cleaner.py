#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    for i in range(len(predictions)):
        error = abs(net_worths[i] - predictions[i])
        cleaned_data.append((ages[i], net_worths[i], error))

    cleaned_data.sort(key=lambda tup: tup[2])
    cleaned_data = cleaned_data[:int(len(predictions) * 0.9)]

    return cleaned_data

