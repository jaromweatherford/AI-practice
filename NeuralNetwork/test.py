from scipy import stats
import pandas as pd
import nnet
import numpy as np
from sklearn.model_selection import KFold
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

print
print
print
print "LENGTH OF THE DATA SET: ", len(iris.data)
print
print
print

"""For calculating the total accuracy over all folds"""
total_accuracy = 0.0

"""Split the data set"""
kf = KFold(n_splits=len(iris.data))
for train, test in kf.split(iris.data):
    train_data, test_data, pre_train_target, pre_test_target = iris.data[train], iris.data[test], iris.target[train], iris.target[test]

    """Prep the targets (convert from 0, 1, or 2 to 100, 010, or 001"""
    train_target = np.zeros((len(pre_train_target), 3), dtype=int)
    for x in xrange(len(pre_train_target)):
        train_target[x][pre_train_target[x]] = 1
    test_target = np.zeros((len(pre_test_target), 3), dtype=int)
    for x in xrange(len(pre_test_target)):
        test_target[x][pre_test_target[x]] = 1

    """Normalize"""
    train_target = stats.zscore(train_target, axis=0)
    #test_target = stats.zscore(test_target, axis=0)

    """Make the classifier and model"""
    classifier = nnet.NNetClassifier([3])
    model = classifier.fit(train_data, train_target)

    """Predict"""
    prediction = model.predict(test_data)

    wrong = 0
    for datum in xrange(len(prediction)):
        bad_guess = False
        #print test_target[datum], " - ", prediction[datum]
        for value in xrange(len(prediction[datum])):
            #print prediction[datum][value] < 0.5, " - ", test_target[datum][value] < 0.5, " - ", (prediction[datum][value] < 0.5) != (test_target[datum][value] < 0.5)
            if (prediction[datum][value] < 0.5) != (test_target[datum][value] < 0.5):
                bad_guess = True
        if bad_guess:
            wrong += 1

    #print wrong, ", ", len(prediction)

    accuracy = (len(prediction) - wrong) / float(len(prediction))
    total_accuracy += accuracy
    print "IRIS ACCURACY: ", accuracy

print
print
print
print "IRIS TOTAL ACCURACY: ", total_accuracy / float(len(iris.data))
print
print
print

"""DIABETES"""

"""Get the data and normalize"""
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
d_set = pd.read_csv(url, names=["0", "1", "2", "3", "4", "5", "6", "7", "8"])
data = stats.zscore(d_set[["0", "1", "2", "3", "4", "5", "6", "7"]], axis=0)
targets = d_set["8"].as_matrix()

"""Split the data set"""
train_data, test_data, train_target, test_target = train_test_split(data, targets, test_size=0.3)

train_target = np.reshape(train_target, (len(train_target), 1))
test_target = np.reshape(test_target, (len(test_target), 1))

"""Make the classifier and model"""
classifier = nnet.NNetClassifier([3])
model = classifier.fit(train_data, train_target)

"""Predict"""
prediction = model.predict(test_data)

# print prediction

wrong = 0
for datum in xrange(len(prediction)):
    bad_guess = False
    for value in xrange(len(prediction[datum])):
        if (prediction[datum][value] < 0.5) != (test_target[datum][value] < 0.5):
            bad_guess = True
    if bad_guess == True:
        wrong += 1

print "DIABETES ACCURACY: ", (len(prediction) - wrong) / float(len(prediction))

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

"""
ERROR/UPDATE FOR OUTPUT NODES
error = outputactivation * (1 - outputactivation) * (outputactivation - outputtarget)
newweight = oldweight - (learningrate * error * inputActivation

ej = aj(1 - aj)(aj - tj)
wij = wij - N * ej * ai

ERROR FOR HIDDEN NODES

ej = aj (1 - aj) sum wjk * ek
wij = wij - N * ej * ai


errors = np.array(self.nodes, copy=True)
newWeights = np.array(self.weights, copy=True)
"""