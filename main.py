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


def train_dishwasher_neural_network(train_dataset_registry):
    neural_network = DishwasherNeuralNetwork()

    fold = 40000

    data = train_dataset_registry.get_values_range('Aggregate Power',
                                                   fold,
                                                   0)

    labels = train_dataset_registry.get_values_range('Dishwasher Power',
                                                     fold,
                                                     0)

    neural_network.fit(data, labels)

    data_to_predict = train_dataset_registry.get_values_range('Aggregate Power',
                                                              fold,
                                                              fold)

    real_data = train_dataset_registry.get_values_range('Dishwasher Power',
                                                        fold,
                                                        fold)

    predicted_data = neural_network.predict(data_to_predict)

    common.show_shape(real_data, real_data)

    x = []
    for value in real_data:
        x.append(value[0])
    real_data = x

    show_results('Dishwasher', real_data, predicted_data)


if __name__ == '__main__':

    train_dataset_registry = DatasetRegistry("Train Datasets")

    train_dataset_registry.insert('Aggregate Power', './datasets/main_train.csv')
    train_dataset_registry.insert('Dishwasher Power', './datasets/dishwasher_train.csv')
    train_dataset_registry.insert('Fridge Power', './datasets/fridge_train.csv')

    train_dataset_registry.plot()

    dataset_size = train_dataset_registry.get_datasets_size()

    sliding_window_walk_forward = SlidingWindowWalkForward(dataset_size)

    while True:

        # 'x1' represents the first observation which belongs to training set
        # 'x2' represents the last observation which belongs to training set
        # 'y1' represents the first observation which belongs to testing set
        # 'y1' represents the last observation which belongs to testing set

        x1, x2, y1, y2 = sliding_window_walk_forward.get_next_iteration_indexes(debug=True)
        if y2 == dataset_size:
            break













    exit(43423)





    train_dishwasher_neural_network(train_dataset_registry)

"""
    training_dataset_registry = EnergyConsumptionRegistry(appliances_name)
    application_neural_network = NeuralNetwork(appliances_name)

    for appliance_name in appliances_name:
        fold = 40000

        data_rgs, labels_rgs = training_dataset_registry.get_rgs_nn_inputs(appliance_name,
                                                                           fold,
                                                                           0)

        data_cls, labels_cls = training_dataset_registry.get_cls_nn_inputs(appliance_name,
                                                                           fold,
                                                                           0)

        application_neural_network.fit(appliance_name, data_rgs, labels_rgs, data_cls, labels_cls)

        data, labels = training_dataset_registry.get_rgs_nn_inputs(appliance_name,
                                                               fold,
                                                               fold)

        predicted_data = application_neural_network.prediction(appliance_name, data)

        results_plot = Plot(appliance_name)
        results_plot.insert("Predicted Data", predicted_data)
        results_plot.insert("Real Data", labels)

        results_plot.show()

        accuracy_metrics = AccuracyMetrics(labels, predicted_data)
        f1_score = accuracy_metrics.get_f1()

        print(f1_score)


"""
""" 
    

    classificationNetwork = ClassificationNetwork()
    regressionNetwork = RegressionNetwork()

    fold = 400000

    data, labels = datasets.get_network_input_for_regression(fold, 0)

    show_shape(data, labels)

    # labels = labels.reshape((labels.shape[0], labels.shape[1], 1))

    show_shape(data, labels)
    print(data)
    print(labels)

    h = regressionNetwork._model.fit(data, labels, epochs=10)

    # 1D Conv

    # data = data.reshape((data.shape[0], data.shape[1], 1))
    # labels = labels.reshape((labels.shape[0], labels.shape[1], 1))

    # print(data.shape)
    # print(labels.shape)

    data, labels = datasets.get_network_input_for_regression(fold, fold)

    pred_y = regressionNetwork._model.predict(data)
    print(pred_y)
    print(pred_y.shape)

    dd = []
    for elem in pred_y:
        dd.append([elem[0][1], elem[0][0]])

    dd = numpy.asarray(dd)
    plot_results(dd, labels)
"""
"""
    standrard = datasets.get_standardize_dataset(400000, debug=True)

    univariate_past_history = 20
    univariate_future_target = 0

    print(standrard.shape)

    x_train_uni, y_train_uni = univariate_data(standrard, 0, 300000,
                                               univariate_past_history,
                                               univariate_future_target)

    if True:
        print('Input train-data shape:      {}'.format(x_train_uni.shape))
        print('Input validation-data shape: {}'.format(y_train_uni.shape))

"""
