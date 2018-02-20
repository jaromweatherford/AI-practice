from scipy import stats
import pandas as pd
import nnet
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
"""
list = pd.DataFrame(np.array([1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]).reshape(6, 2))
list.columns = ["lower", "higher"]
print list
print stats.zscore(list)
print stats.zscore(list["lower"])
list["lower"] = stats.zscore(list["lower"])
print list
"""

"""IRIS"""

"""Getting the data set"""
iris = datasets.load_iris()

"""Split the data set"""
train_data, test_data, pre_train_target, pre_test_target = train_test_split(iris.data, iris.target, test_size=0.3)

"""Prep the targets (convert from 0, 1, or 2 to 100, 010, or 001"""
train_target = np.zeros((len(pre_train_target), 3), dtype=int)
for x in xrange(len(pre_train_target)):
    train_target[x][pre_train_target[x]] = 1
test_target = np.zeros((len(pre_test_target), 3), dtype=int)
for x in xrange(len(pre_test_target)):
    test_target[x][pre_test_target[x]] = 1

"""Normalize"""
train_target = stats.zscore(train_target, axis=0)
test_target = stats.zscore(test_target, axis=0)

"""Make the classifier and model"""
classifier = nnet.NNetClassifier(3)
model = classifier.fit(train_data, train_target)

"""Predict"""
prediction = model.predict(test_data)
print "NODES: "
print prediction

"""DIABETES"""

"""Get the data and normalize"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
d_set = pd.read_csv(url, names=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
data = stats.zscore(d_set[["0", "1", "2", "3", "4", "5", "6", "7"]], axis=0)
targets = d_set["8"].as_matrix()

"""Split the data set"""
train_data, test_data, pre_train_target, pre_test_target = train_test_split(data, targets, test_size=0.3)

"""Make the classifier and model"""
classifier = nnet.NNetClassifier(1)
model = classifier.fit(train_data, train_target)

"""Predict"""
prediction = model.predict(test_data)
print "NODES: "
print prediction

#def run(url, header, categories=None, Nan="?", k=3):
#    data = read(url, header)
#    if categories is not None:
#        for key in categories:
#            conversion = categorize(data[key], categories[key], Nan=Nan)
#            data[key] = data[key].map(conversion)
#    else:
#        for head in header:
#            if numpy.issubdtype(data[head].dtype, numpy.number):
#                conversion = categorize(data[head], build_options_list(data[head], Nan))
#                data[head] = data[head].map(conversion)
#    data.apply(pandas.to_numeric)
#    data = data.as_matrix()


#def read(url, header):
#    data = pd.read_csv(url, names=header, skipinitialspace=True)
#    return data

#    data = pd.read_csv(url, names=header, skipinitialspace=True)