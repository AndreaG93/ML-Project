from network import NeuralSubnetworks


class ApplianceNeuralNetwork(object):

    def __init__(self, appliance_name):
        self.name = "Neural Network for '{}' appliance".format(appliance_name)

        self.classification_subnetwork = NeuralSubnetworks.get_classification_subnetwork(appliance_name)
        self.regression_subnetwork = NeuralSubnetworks.get_regression_subnetwork(appliance_name)


class NeuralNetwork(object):

    def __init__(self, appliances_names):
        self._appliance_neural_networks = {}

        for appliance_name in appliances_names:
            neural_network = ApplianceNeuralNetwork(appliance_name)
            self._appliance_neural_networks[appliance_name] = neural_network

    def fit(self, appliance_name, data_rgs, labels_rgs, data_cls, labels_cls):
        appliance_neural_network = self._appliance_neural_networks[appliance_name]

        appliance_neural_network.classification_subnetwork.fit(data_cls, labels_cls, epochs=1)
        appliance_neural_network.regression_subnetwork.fit(data_rgs, labels_rgs, epochs=1)

    def prediction(self, appliance_name, data):

        output = []

        appliance_neural_network = self._appliance_neural_networks[appliance_name]

        pred_from_regression = appliance_neural_network.regression_subnetwork.predict(data)
        pred_from_classification = appliance_neural_network.classification_subnetwork.predict(data)

        for index in range(0, pred_from_regression.shape[0]):
            output.append(pred_from_regression[index][0][0] * pred_from_classification[index][0][0])

        return output