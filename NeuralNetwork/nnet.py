import numpy as np

class NNetModel:

    def __init__(self, data, targets, num_nodes):
        self.nodes = np.zeros(num_nodes, dtype=float)
        self.weights = np.full((len(data[0]) + 1, num_nodes), 0.1, dtype=float)
        self.inputs = np.full(len(data[0]) + 1, -1.0, dtype=float)

    def predict(self, data):
        result = np.empty((len(data), len(self.nodes)))
        for i in xrange(len(data)):
            self.inputs = np.append([-1], data[i])
            self.run()
            result[i] = self.nodes
        return result

    def run(self):
        for i in xrange(len(self.inputs)):
            for n in xrange(len(self.weights[i])):
                self.nodes[n] += self.inputs[i] * self.weights[i][n]
        self.squash()

    def squash(self):
        for n in xrange(len(self.nodes)):
            if self.nodes[n] < 0:
                self.nodes[n] = 0.
            else:
                self.nodes[n] = 1.


class NNetClassifier:
    def __init__(self, num_nodes=1):
        self.num_nodes = num_nodes

    def fit(self, data, targets):
        return NNetModel(data, targets, self.num_nodes)