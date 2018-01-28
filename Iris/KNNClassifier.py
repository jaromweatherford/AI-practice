from scipy.stats import mode
from scipy.stats import tmean
from sklearn.model_selection import train_test_split


class KNNModel:

    def __init__(self, k, data, targets):
        self.k = k
        self.data = data
        self.targets = targets

    def predict(self, data, regressive=False):
        print "Predicting..."
        result = range(len(data))
        for testRow in xrange(len(data)):
            print "For testRow...", testRow
            distances = []
            #print distances
            distances = range(len(self.data))
            for trainRow in xrange(len(self.data)):
                #if trainRow < 10:
                #    print distances
                for col in xrange(len(self.data[testRow])):
                    #print data[testRow][col]
                    distances[trainRow] += float((data[testRow][col] - self.data[trainRow][col]) ** 2)
            topK = []
            for i in xrange(self.k):
                topK.append(-1)
            for i in xrange(len(distances)):
                comparison = i
                for j in xrange(self.k):
                    if (topK[j] == -1) or (distances[topK[j]] > distances[i]):
                        comparison, topK[j] = topK[j], comparison
            #print("len of data, targets: " + str(len(self.data)) + ", " + str(len(self.targets)))
            #print("topK: " + str(topK))
            #print("self.targets: " + str(self.targets))
            #print("self.targets[topK, :]: " + str(self.targets[topK]))
            if not regressive:
                result[testRow] = int(mode(self.targets[topK])[0][0])
            else:
                result[testRow] = int(tmean(self.targets[topK])[0][0])
        return result


class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, data, targets):
        return KNNModel(self.k, data, targets)


def test_classifier(data, targets, k):
    training_data, testing_data, training_target, testing_target = train_test_split(data, targets, test_size=0.3)
    cla = KNNClassifier(k)
    mod = cla.fit(training_data, training_target)
    prediction = mod.predict(testing_data)

    num_wrong = 0

    for x in xrange(len(prediction)):
        if int(prediction[x]) != int(testing_target[x]):
            num_wrong += 1

    return (len(prediction) - num_wrong) / float(len(prediction))
