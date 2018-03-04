import numpy as np


def sigmoid(val, deriv=False):
    if deriv:
        return val * (1 - val)

    return 1 / (1 + np.exp(-val))


# input
x = np.array([[0, 0, 1],
             [0, 1, 1],
             [1, 0, 1],
             [1, 1, 1]])

# output
y = np.array([[0],
              [1],
              [1],
              [0]])

# random
np.random.seed(1)

# synapses
syn0 = 2 * np.random.random((3, 4)) - 1
syn1 = 2 * np.random.random((4, 1)) - 1
print syn0

# training
for j in xrange(60000):
    l0 = x
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))

    l2_error = y - l2

    #if(j % 10000) == 0:
        #print "Error: " + str(np.mean(np.abs(l2_error)))
        #print "Output: ", l2

    l2_delta = l2_error * sigmoid(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * sigmoid(l1, deriv=True)

    # update weights
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print "output after training"
print l2
