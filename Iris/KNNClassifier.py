from scipy.stats import mode


class KNNModel:

    def __init__(self, k, data, targets):
        self.k = k
        self.data = data
        self.targets = targets

    def predict(self, data):
        result = []
        for testRow in xrange(len(data)):
            result.append(0)
            distances = []
            for trainRow in xrange(len(self.data)):
                distances.append(0)
                for col in xrange(len(self.data[testRow])):
                    distances[trainRow] += (data[testRow][col] - self.data[trainRow][col]) ** 2
            topK = []
            for i in xrange(self.k):
                topK.append(-1)
            for i in xrange(len(distances)):
                comparison = i
                for j in xrange(self.k):
                    if topK[j] == -1 or distances[topK[j]] > distances[i]:
                        comparison, topK[j] = topK[j], comparison
            #print("len of data, targets: " + str(len(self.data)) + ", " + str(len(self.targets)))
            #print("topK: " + str(topK))
            #print("self.targets: " + str(self.targets))
            #print("self.targets[topK, :]: " + str(self.targets[topK]))
            result[testRow] = int(mode(self.targets[topK])[0][0])
        return result


class KNNClassifier:
    def __init__(self, k):
        self.k = k

    def fit(self, data, targets):
        return KNNModel(self.k, data, targets)
