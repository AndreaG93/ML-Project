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


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size

    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(dataset[indices])
        labels.append(dataset[i + target_size])

    return np.array(data), np.array(labels)


def dsadas():
    plt.interactive(True)

    zip_path = tensorflow.keras.utils.get_file(
        origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
        fname='jena_climate_2009_2016.csv.zip',
        extract=True)
    csv_path, _ = os.path.splitext(zip_path)

    df = pd.read_csv(csv_path)

    print(df)
    print(df.shape)

    uni_data = df['T (degC)']
    uni_data.index = df['Date Time']
    uni_data.head()

    # uni_data.plot(subplots=True)
    print(type(uni_data))

    TRAIN_SPLIT = 300000

    # index = uni_data.index

    # print(index)

    uni_data = uni_data.values

    uni_train_mean = uni_data[:TRAIN_SPLIT].mean()
    uni_train_std = uni_data[:TRAIN_SPLIT].std()
    uni_data = (uni_data - uni_train_mean) / uni_train_std

    print(uni_data)
    print(uni_data.shape)
    print(type(uni_data))

    univariate_past_history = 20
    univariate_future_target = 0

    x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                               univariate_past_history,
                                               univariate_future_target)

    x_val_uni, y_val_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                           univariate_past_history,
                                           univariate_future_target)

    simple_model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Dense(32, input_shape=x_train_uni.shape[-1:]),
        tensorflow.keras.layers.Dense(1)
    ])

    simple_model.compile(optimizer='adam', loss='mae')

    print(simple_model.summary())

    h = simple_model.fit(x_train_uni, y_train_uni, epochs=10, batch_size=256)
    print(type(h))

    pred_y = simple_model.predict(x_val_uni)

    # %%
    plt.interactive(True)
    plt.plot(y_val_uni[0:100])
    plt.plot(pred_y[0:100])
    plt.show()


def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data = []
    labels = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        data.append(dataset[indices])
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)


def get_train_ds(dataset_values, train_data_size, appliance_target_index, start_from_index=0):
    data = []
    labels = []

    for index in range(start_from_index, start_from_index + train_data_size):
        new_data = [dataset_values[index][0]]

        data.append(new_data)
        labels.append(dataset_values[index][appliance_target_index])

    return np.array(data), np.array(labels)


def dsdsad():
    x = numpy.asarray([[1, 2, 5], [3, 4, 5], [3, 5, 5], [3, 5, 5], [3, 5, 5], [3, 23, 4]])

    y = numpy.asarray([3, 4, 5])

    r, f = get_train_ds(x, 3, 2)

    simple_model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Dense(16, input_shape=(r.shape[1],)),
        tensorflow.keras.layers.Dense(1)
    ])

    simple_model.compile(optimizer='adam', loss='mae')
    simple_model.summary()

    print(r.shape)
    print(f.shape)

    h = simple_model.fit(r, f, epochs=10)


if __name__ == '__main__':
    plt.interactive(True)

    dataset = DatasetForClassification(['dishwasher', 'fridge'], debug=True)

    TRAIN_SPLIT = 400000

    fd = dataset._dataset.values

    data, labels = get_train_ds(fd, 400, 2)

    simple_model = tensorflow.keras.models.Sequential([
        tensorflow.keras.layers.Dense(16, input_shape=(data.shape[1],)),
        tensorflow.keras.layers.Dense(1)
    ])

    simple_model.compile(optimizer='adam', loss='mae')
    simple_model.summary()

    print(data.shape)
    print(labels.shape)

    h = simple_model.fit(data, labels, epochs=10)

    data, labels = get_train_ds(fd, 400, 2, start_from_index=400)

    pred_y = simple_model.predict(data)

    plt.plot(pred_y)
    plt.plot(labels)
    plt.show()

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
