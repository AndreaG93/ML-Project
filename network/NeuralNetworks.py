from network import NeuralSubnetworks
import numpy


class FridgeNeuralNetwork(object):

    def __init__(self):
        self._regression_subnetwork = NeuralSubnetworks.get_regression_subnetwork()

    def fit(self, data, labels):
        self._regression_subnetwork.fit(data, labels, epochs=1)

    def predict(self, data):
        output = []

        prediction_from_regression = self._regression_subnetwork.predict(data)

        for index in range(0, prediction_from_regression.shape[0]):
            output.append(prediction_from_regression[index][0][0])

        return output


class DishwasherNeuralNetwork(object):

    def __init__(self):
        self._regression_subnetwork = NeuralSubnetworks.get_regression_subnetwork()
        self._classification_subnetwork = NeuralSubnetworks.get_classification_subnetwork()

    def fit(self, data, labels):

        labels_cls = []

        for x in range(0, len(labels)):
            labels_cls.append(1 if labels[x] > 0 else 0)

        labels_cls = numpy.asarray(labels_cls)

        self._regression_subnetwork.fit(data, labels, epochs=1)
        self._classification_subnetwork.fit(data, labels_cls, epochs=1)

    def predict(self, data):
        output = []

        prediction_from_regression = self._regression_subnetwork.predict(data)
        prediction_from_classification = self._classification_subnetwork.predict(data)

        for index in range(0, prediction_from_regression.shape[0]):
            output.append(prediction_from_regression[index][0][0] * prediction_from_classification[index][0][0])

        return output
