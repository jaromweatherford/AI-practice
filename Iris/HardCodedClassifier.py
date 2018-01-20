class HardCodedModel:
    def __init__(self):
        pass

    def predict(self, data):
        result = []
        for x in data:
            result.append(0)
        return result


class HardCodedClassifier:
    def __init__(self):
        pass

    @staticmethod
    def fit(data, targets):
        return HardCodedModel()