import calendar

import tensorflow
import tensorflow.keras as keras
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import numpy
import matplotlib.pyplot as plt
import pandas
import datetime
from dateutil.parser import parse
import pytz

import numpy as np
import os
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, LSTM, Attention, RepeatVector, TimeDistributed, Bidirectional, Flatten, Dropout, \
    Reshape

# 2019-01-31 19:35:34+01:00
from ClassificationNetwork import ClassificationNetwork
from DatasetForClassification import DatasetForClassification


def print_installed_libraries():
    print(f'Installed \'TensorFlow\' version: {tensorflow.__version__}')
    print(f'Installed \'Keras\' version:      {keras.__version__}')
    print(f'Installed \'NumPy\' version:      {numpy.__version__}')
    print(f'Installed \'Pandas\' version:      {pandas.__version__}')


def split_data(filename):
    train_set_datetime_lower_bound = datetime.datetime(year=2019, month=1, day=1, hour=0, minute=0, second=0,
                                                       tzinfo=pytz.timezone("Etc/GMT-1"))

    train_set_datetime_upper_bound = datetime.datetime(year=2019, month=3, day=14, hour=23, minute=59, second=59,
                                                       tzinfo=pytz.timezone("Etc/GMT-1"))

    test_set_datetime_lower_bound = datetime.datetime(year=2019, month=3, day=15, hour=0, minute=0, second=0,
                                                      tzinfo=pytz.timezone("Etc/GMT-1"))

    test_set_datetime_upper_bound = datetime.datetime(year=2019, month=3, day=31, hour=23, minute=59, second=59,
                                                      tzinfo=pytz.timezone("Etc/GMT-1"))

    dtypes = {'timestamp': 'str', 'power': 'float'}
    parse_dates = ['timestamp']
    dataset = pandas.read_csv(filename, header=0, dtype=dtypes, date_parser=pandas.to_datetime, parse_dates=parse_dates)

    train_set = dataset[
        (dataset['timestamp'] >= train_set_datetime_lower_bound) &
        (dataset['timestamp'] <= train_set_datetime_lower_bound)]

    test_set = dataset[
        (dataset['timestamp'] >= test_set_datetime_lower_bound) &
        (dataset['timestamp'] <= test_set_datetime_lower_bound)]

    for object in train_set.values:
        print(object)

    # test_set = dataset[
    #    test_set_datetime_lower_bound <= dataset['timestamp'] <= test_set_datetime_upper_bound]

    print(train_set.shape())
    # print(test_set.shape())
    training_set_index = -1


def combine_dataset(aggregate_dataset_file, appliance_dataset_file):
    types = {'timestamp': 'str', 'power': 'float'}

    aggregate_dataset = pandas.read_csv(aggregate_dataset_file, header=0, dtype=types, date_parser=pandas.to_datetime,
                                        parse_dates=['timestamp'])
    appliance_dataset = pandas.read_csv(appliance_dataset_file, header=0, dtype=types, date_parser=pandas.to_datetime,
                                        parse_dates=['timestamp'])

    aggregate_dataset.rename(columns={'power': 'aggregate_power'}, inplace=True)
    appliance_dataset.rename(columns={'power': 'appliance_power'}, inplace=True)

    return aggregate_dataset.join(appliance_dataset.set_index('timestamp'), on='timestamp')


def retrieve_sets(dataset):
    train_size = int(0.7 * dataset.shape[0])
    output_train, output_validation = dataset[0:train_size], dataset[train_size:],

    return output_train, output_validation


def build_disaggregation_mode(train_set, validation_set):
    readings = Input(shape=(12,))
    x = Dense(8, activation="relu", kernel_initializer="glorot_uniform")(readings)
    benzene = Dense(1, kernel_initializer="glorot_uniform")(x)

    model = Model(inputs=[readings], outputs=[benzene])
    model.compile(loss="mse", optimizer="adam")

    print(model.summary())

    NUM_EPOCHS = 5
    BATCH_SIZE = 10

    model.fit(train_set, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

    test_scores = model.evaluate(validation_set, verbose=2)
    print('Test loss:', test_scores[0])
    print('Test accuracy:', test_scores[1])


def get_train_prova(dataset_values, train_data_size, start_from_index=0):
    data = []
    labels = []

    for index in range(start_from_index, start_from_index + train_data_size):
        new_data = [dataset_values[index][0]]

        data.append(new_data)
        labels.append(dataset_values[index][1:])

    return np.array(data), np.array(labels)


def dsdsad():
    x = numpy.asarray([[1, 2, 5], [3, 4, 5], [3, 5, 5], [3, 5, 5], [3, 5, 5], [3, 23, 4]])

    r, f = get_train_prova(x, 3)

    print(r.shape)
    print(f.shape)

    model = Sequential()

    r = r.reshape((r.shape[0], r.shape[1], 1))
    f = f.reshape((f.shape[0], f.shape[1], 1))

    print(r.shape)
    print(f.shape)



    # 1D Conv
    model.add(Conv1D(16, 4, strides=1, activation="relu", padding='same', input_shape=(1, 1)))
    model.add(Conv1D(16, 4, strides=1, activation="relu", padding='same'))
    model.add(Conv1D(16, 4, strides=1, activation="relu", padding='same'))
    model.add(Conv1D(16, 4, strides=1, activation="relu", padding='same'))

    model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False, activation="tanh"), merge_mode='concat'))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))

    model.compile(optimizer='adam', loss='mae')
    model.summary()

    print(r.shape)
    print(f.shape)

    h = model.fit(r, f, epochs=10)


def plot_results(predicted_data, real_data):
    plt.figure()
    subplots_number = predicted_data.shape[-1]

    for subplot_index in range(0, subplots_number):
        plt.subplot(subplots_number, 1, subplot_index + 1)
        plt.plot(predicted_data[:, subplot_index])
        plt.plot(real_data[:, 1])

    plt.show()


if __name__ == '__main__':


    plt.interactive(True)

    dataset = DatasetForClassification(['dishwasher', 'fridge'], debug=True)

    TRAIN_SPLIT = 400000

    fd = dataset._dataset.values

    data, labels = get_train_prova(fd, 40000, 2)

    model = Sequential()

    # 1D Conv

    data = data.reshape((data.shape[0], data.shape[1], 1))
    labels = labels.reshape((labels.shape[0], labels.shape[1], 1))

    print(data.shape)
    print(labels.shape)

    # 1D Conv
    model.add(Conv1D(16, 4, strides=1, activation="relu", padding='same', input_shape=(1, 1)))
    model.add(Conv1D(16, 4, strides=1, activation="relu", padding='same'))
    model.add(Conv1D(16, 4, strides=1, activation="relu", padding='same'))
    model.add(Conv1D(16, 4, strides=1, activation="relu", padding='same'))

    model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False, activation="tanh"), merge_mode='concat'))

    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='linear'))

    model.compile(optimizer='adam', loss='mae')
    model.summary()


    h = model.fit(data, labels, epochs=10)

    data, labels = get_train_prova(fd, 40000, start_from_index=400)

    data = data.reshape((data.shape[0], data.shape[1], 1))

    pred_y = model.predict(data)
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
