from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from HardCodedClassifier import HardCodedClassifier
from KNNClassifier import KNNClassifier
from KNNClassifier import test_classifier
import numpy

iris = datasets.load_iris()

#print(iris)

file = open("iris.txt", "r")
iris_from_file = file.readlines()
a = numpy.array(iris_from_file[0].split())
#for x in xrange(len(a) - 1):
#    a[x] = float(a[x])
#a[len(a) - 1] = int(a[len(a) - 1])
for x in xrange(len(iris_from_file)):
    nums = iris_from_file[x].split()
    #for y in xrange(len(nums) - 2):
    #    nums[y] = float(nums[y])
    #nums[len(nums) - 1] = int(nums[len(nums) - 1])
    a = numpy.vstack((a, nums))

a = a.astype(float)

file_data = a[:, range(0, (len(a[0]) - 1))]
file_target = a[:, (len(a[0]) - 1)]

#print(file_data)
#print(file_target)

# Show the data (the attributes of each instance)
#print(iris.data)

# Show the target values (in numeric format) of each instance
#print(iris.target)

# Show the actual target names that correspond to each number
#print(iris.target_names)

train_data, test_data, train_target, test_target = train_test_split(iris.data, iris.target, test_size=0.3)

#print(train_data)
#print(test_data)
#print(train_target)
#print(test_target)

gaussclassifier = GaussianNB()
gaussmodel = gaussclassifier.fit(train_data, train_target)
gausstargets_predicted = gaussmodel.predict(test_data)

gausswrong = 0
for x in xrange(len(gausstargets_predicted)):
    if gausstargets_predicted[x] != test_target[x]:
        gausswrong = gausswrong + 1

#print(gausstargets_predicted)
#print(test_target)

#print("gausswrong")
#print(gausswrong)
print("gauss accuracy")
print((len(test_target) - gausswrong) / float(len(test_target)))


classifier = HardCodedClassifier()
model = classifier.fit(train_data, train_target)
targets_predicted = model.predict(test_data)

wrong = 0
for x in xrange(len(targets_predicted)):
    if targets_predicted[x] != test_target[x]:
        wrong = wrong + 1

#print(targets_predicted)
#print(test_target)

#print("wrong")
#print(wrong)
print("accuracy")
print((len(test_target) - wrong) / float(len(test_target)))

train_data, test_data, train_target, test_target = train_test_split(file_data, file_target, test_size=0.3)

#print(train_data)
#print(test_data)
#print(train_target)
#print(test_target)

classifier = HardCodedClassifier()
model = classifier.fit(train_data, train_target)
targets_predicted = model.predict(test_data)

wrong = 0
for x in xrange(len(targets_predicted)):
    if targets_predicted[x] != int(test_target[x]):
        wrong = wrong + 1

#print(targets_predicted)
#print(test_target)

#print("wrong")
#print(wrong)
print("accuracy")
print((len(test_target) - wrong) / float(len(test_target)))

kClassifier = KNNClassifier(3)
kModel = kClassifier.fit(train_data, train_target)
kPredicted = kModel.predict(test_data)

wrong = 0

for x in xrange(len(kPredicted)):
    print (str(kPredicted[x]) + " - " + str(test_target[x]))

for x in xrange(len(kPredicted)):
    if int(kPredicted[x]) != int(test_target[x]):
        wrong += 1

print ("Accuracy of my KNN program:" + str((len(test_target) - wrong) / float(len(test_target))))

from sklearn.neighbors import KNeighborsClassifier

skClassifier = KNeighborsClassifier(n_neighbors=3)
skClassifier.fit(train_data, train_target)

skPredicted = skClassifier.predict(test_data)

wrong = 0

print ("Checking to make sure wrong is reset...:" + str((len(test_target) - wrong) / float(len(test_target))))

for x in xrange(len(skPredicted)):
    if int(skPredicted[x]) != int(test_target[x]):
        wrong += 1

print ("Accuracy of sklearn's KNN program:" + str((len(test_target) - wrong) / float(len(test_target))))


print("Accuracy of my KNN program on the iris data set: " + str(test_classifier(iris.data, iris.target, 3)))

# This one usually produces about 10% accuracy, it probably needs to be scaled.
boston = datasets.load_boston()
print("Accuracy of my KNN program on the boston data set: " + str(test_classifier(boston.data, boston.target, 3)))

cancer = datasets.load_breast_cancer()
print("Accuracy of my KNN program on the breast cancer data set: " + str(test_classifier(cancer.data, cancer.target, 3)))

# This one always gets nearly 0 accuracy.  Maybe it's regressive?  Or it needs scaling.
diabetes = datasets.load_diabetes()
#print diabetes
print("Accuracy of my KNN program on the diabetes data set: " + str(test_classifier(diabetes.data, diabetes.target, 3)))

# this data set seems to be more complex than the others, it causes an error that suggests
# unexpected dimensionality in the arrays
#print("Accuracy of my KNN program on the linnerud data set: " + str(test_classifier(datasets.load_linnerud(), 3)))

wine = datasets.load_wine()
print("Accuracy of my KNN program on the wine data set: " + str(test_classifier(wine.data, wine.target, 3)))


