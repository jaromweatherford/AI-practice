import numpy as np
import math
from sklearn.model_selection import train_test_split


class NNetModel:

    def __init__(self, data, targets, num_nodes, learning_rate):
        self.n = learning_rate
        self.num_layers = len(num_nodes) + 2                      # Add two for the input and output
        self.num_nodes = np.append([len(data[0])], num_nodes)     # Keep track of the number of nodes in each layer
        self.num_nodes = np.append(self.num_nodes, [len(targets[0])])  # Add the output layer
        for n in xrange(len(self.num_nodes)):
            self.num_nodes[n] += 1
        max_layer = max(self.num_nodes)
        self.nodes = [np.full((self.num_nodes[a]), -1, dtype=float) for a in xrange(self.num_layers)]
        self.errors = [np.full((self.num_nodes[a]), -1, dtype=float) for a in xrange(self.num_layers)]
        #self.weights = np.random.random_sample((self.num_layers - 1, len(self.nodes[0]), len(self.nodes[0]))) / 5
        self.weights = [np.random.random((self.num_nodes[a], self.num_nodes[a + 1])) - 0.5 for a in xrange(self.num_layers - 1)]
        self.train(data, targets)

    def train(self, data, targets):
        train_d, test_d, train_t, test_t = train_test_split(data, targets, test_size=0.3)
        epochs = 0
        accuracy, last_accuracy = 0, -1
        run = True
        saved_weights = np.copy(self.weights)
        saved_accuracy = 0
        while run:
            for datum in xrange(len(train_d)):
                self.run(train_d[datum])
                self.calc_errors(train_t[datum])
                self.update_weights()

            """
            # calculate current accuracy
            wrong = 0.0
            predictions = self.predict(test_d)
            for prediction in xrange(len(predictions)):
                for value in xrange(len(predictions[prediction])):
                    wrong += abs((predictions[prediction][value] - targets[prediction][value]))
            """
            wrong = 0
            predictions = self.predict(test_d)
            for datum in xrange(len(predictions)):
                bad_guess = False
                for value in xrange(len(predictions[datum])):
                    if (predictions[datum][value] < 0.5) != (test_t[datum][value] < 0.5):
                        bad_guess = True
                if bad_guess == True:
                    wrong += 1

            # update accuracy values
            last_accuracy = accuracy
            accuracy = (len(test_d) - wrong) / float(len(test_d))
            # Add to the Epochs
            epochs += 1

            if accuracy > saved_accuracy:
                saved_weights = self.weights
                saved_accuracy = accuracy
                #print accuracy

            if epochs >= 300:
                run = False
        print "Final accuracy: ", saved_accuracy
        self.weights = saved_weights

    def update_weights(self):
        """Updates the weights according to the current weights, errors, activation values, and learning rate"""
        # This calculation doesn't depend on any other weights, so order doesn't matter much
        for layer in xrange(self.num_layers - 1):
            for node in xrange(self.num_nodes[layer]):
                for next_node in xrange(self.num_nodes[layer + 1]):
                    self.weights[layer][node][next_node] = \
                        self.weights[layer][node][next_node] - \
                        self.n * self.errors[layer + 1][next_node] * self.nodes[layer][node]

    def calc_errors(self, target):
        """Updates the errors variable according to the current nodes and a given target"""
        # update the output error
        for output in xrange(1, self.num_nodes[self.num_layers - 1]):
            a = self.nodes[self.num_layers - 1][output]
            self.errors[self.num_layers - 1][output] = a * (1 - a) * (a - target[output - 1])

        # update the hidden errors
        layer = self.num_layers - 2
        while layer > 0:                                         # Each layer other than the input has error
            for node in xrange(1, self.num_nodes[layer]):        # Each node other than the bias has error
                weighted_error_sum = 0
                for next_node in xrange(1, self.num_nodes[layer + 1]):
                    weighted_error_sum += self.weights[layer][node][next_node] * self.errors[layer + 1][next_node]

                # activation of the node in question
                a = self.nodes[layer][node]

                # assign the error value
                self.errors[layer][node] = a * (1 - a) * weighted_error_sum
            layer -= 1
        #print self.errors

    def predict(self, data):
        """Returns a numpy array of the predicted targets for a given set of data"""
        result = np.empty((len(data), self.num_nodes[self.num_layers - 1] - 1))

        for datum in xrange(len(data)):
            self.run(data[datum])

            for classification in xrange(self.num_nodes[self.num_layers - 1] - 1):
                result[datum][classification] = self.nodes[self.num_layers - 1][classification + 1]
        return result

    def run(self, datum):
        """Calculates the nodes given the current weights and a given instance of data"""
        # zero it all out
        for layer in xrange(len(self.nodes)):
            self.nodes[layer].fill(0)

        # set the bias nodes to -1
        for layer in xrange(0, self.num_layers):
            self.nodes[layer][0] = -1

        # Set the input nodes equal to the data
        for value in xrange(len(datum)):
            self.nodes[0][value + 1] = datum[value]

        # Update the nodes
        for layer in xrange(self.num_layers - 1):
            for output_node in xrange(1, self.num_nodes[layer + 1]):
                for input_node in xrange(self.num_nodes[layer]):
                    self.nodes[layer + 1][output_node] += self.weights[layer][input_node][output_node] * \
                                                          self.nodes[layer][input_node]
                self.nodes[layer + 1][output_node] = sigmoid(self.nodes[layer + 1][output_node])


class NNetClassifier:
    def __init__(self, num_nodes=[3]):
        self.num_nodes = num_nodes

    def fit(self, data, targets, learning_rate=0.4):
        return NNetModel(data, targets, self.num_nodes, learning_rate)


def sigmoid(n):
    return 1 / (1 + math.e ** (0 - n))
