from DatasetRegistry import DatasetRegistry
from SlidingWindowWalkForward import SlidingWindowWalkForward
from network.NeuralNetworks import FridgeNeuralNetwork
from network.NeuralNetworks import DishwasherNeuralNetwork
from utility.AccuracyMetrics import AccuracyMetrics
from utility.Plot import Plot
import utility.Common as common


def show_results(appliance_name, real_data, predicted_data):
    results_plot = Plot(appliance_name)
    results_plot.insert("Predicted Data", predicted_data)
    results_plot.insert("Real Data", real_data)

    results_plot.show()

    accuracy_metrics = AccuracyMetrics(real_data, predicted_data)
    f1_score = accuracy_metrics.get_f1()

    print(f1_score)


def perform_training(registry, appliance_name, appliance_neural_network):
    dataset_size = registry.get_datasets_size()

    sliding_window_walk_forward = SlidingWindowWalkForward(dataset_size)

    while True:

        # 'x1' represents the first observation which belongs to training set
        # 'x2' represents the last observation which belongs to training set
        # 'y1' represents the first observation which belongs to testing set
        # 'y1' represents the last observation which belongs to testing set

        x1, x2, y1, y2 = sliding_window_walk_forward.get_next_iteration_indexes(debug=True)

        data = registry.get_values_range(appliance_name, x1, x2)
        labels = registry.get_values_range('Aggregate Power', x1, x2)

        appliance_neural_network.fit(data, labels)

        if y2 == dataset_size:
            break

        data_to_predict = registry.get_values_range('Aggregate Power', y1, y2)
        real_data = registry.get_values_range(appliance_name, y1, y2)

        predicted_data = appliance_neural_network.predict(data_to_predict)

        common.show_shape(real_data, real_data)

        x = []
        for value in real_data:
            x.append(value[0])
        real_data = x

        show_results(appliance_name, real_data, predicted_data)


if __name__ == '__main__':
    dataset_registry = DatasetRegistry("Train Datasets")

    dataset_registry.insert('Aggregate Power', './datasets/main_train.csv')
    dataset_registry.insert('Dishwasher Power', './datasets/dishwasher_train.csv')
    dataset_registry.insert('Fridge Power', './datasets/fridge_train.csv')

    dataset_registry.plot()

    dishwasher_neural_network = DishwasherNeuralNetwork()
    fridge_neural_network = FridgeNeuralNetwork()

    perform_training(dataset_registry, 'Dishwasher Power', dishwasher_neural_network)
