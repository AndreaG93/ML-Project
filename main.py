import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import numpy
import matplotlib.pyplot as plt
import pandas
import datetime
import pytz

from tensorflow.keras.layers import Dense

from network.ClassificationNetwork import ClassificationNetwork
from network.RegressionNetwork import RegressionNetwork
from Dataset import Dataset


def print_installed_libraries():
    print(f'Installed \'TensorFlow\' version: {tensorflow.__version__}')
    print(f'Installed \'Keras\' version:      {keras.__version__}')
    print(f'Installed \'NumPy\' version:      {numpy.__version__}')
    print(f'Installed \'Pandas\' version:      {pandas.__version__}')


def plot_results(predicted_data, real_data):
    plt.figure()
    subplots_number = predicted_data.shape[-1]

    for subplot_index in range(0, subplots_number):
        plt.subplot(subplots_number, 1, subplot_index + 1)
        plt.plot(predicted_data[:, subplot_index])
        plt.plot(real_data[:, 1])

    plt.show()


def show_shape(input_data, output_label): # can make yours to take inputs; this'll use local variable values
    print("Expected ==(num_samples, timestamps, channels)==")
    print("Input Data   -> {}".format(input_data.shape))
    print("Output Label -> {}".format(output_label.shape))

if __name__ == '__main__':

    plt.interactive(True)

    dataset = Dataset(['dishwasher', 'fridge'], debug=True)

    classificationNetwork = ClassificationNetwork()
    regressionNetwork = RegressionNetwork()

    fold = 400000

    data, labels = dataset.get_network_input_for_regression(fold, 0)

    show_shape(data, labels)

    labels = labels.reshape((labels.shape[0], labels.shape[1], 1))

    show_shape(data, labels)
    print(data)
    print(labels)

    h = regressionNetwork._model.fit(data, labels, epochs=10)


    # 1D Conv

    #data = data.reshape((data.shape[0], data.shape[1], 1))
    #labels = labels.reshape((labels.shape[0], labels.shape[1], 1))

    #print(data.shape)
    #print(labels.shape)


    data, labels = dataset.get_network_input_for_regression(fold, fold)

    pred_y = regressionNetwork._model.predict(data)
    print(pred_y)
    print(pred_y.shape)

    dd = []
    for elem in pred_y:
        dd.append([elem[0][1], elem[0][0]])

    dd = numpy.asarray(dd)
    plot_results(dd, labels)

"""
    standrard = dataset.get_standardize_dataset(400000, debug=True)

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
