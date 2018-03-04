import numpy as np
import nnet

test_data = np.arange(8).reshape((4, 2))
test_data[0] = [0, 0]
test_data[1] = [1, 0]
test_data[2] = [0, 1]
test_data[3] = [1, 1]

test_target = np.arange(12).reshape((4, 3))
test_target[0] = [1, 0, 0]
test_target[1] = [0, 1, 0]
test_target[2] = [0, 1, 0]
test_target[3] = [0, 0, 1]

classifier = nnet.NNetClassifier([2])
model = classifier.fit(test_data, test_target)

prediction = model.predict(test_data)

print prediction
print test_target
