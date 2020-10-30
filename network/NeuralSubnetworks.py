from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, Bidirectional, LSTM


def get_classification_subnetwork(appliance_name):
    output = Sequential(name="ClassificationSubNetwork{}".format(appliance_name))

    output.add(
        Conv1D(filters=30, kernel_size=10, strides=1, activation="relu", padding='same', input_shape=(1, 1)))

    output.add(Conv1D(filters=30, kernel_size=8, strides=1, activation="relu", padding='same'))
    output.add(Conv1D(filters=40, kernel_size=6, strides=1, activation="relu", padding='same'))
    output.add(Conv1D(filters=50, kernel_size=5, strides=1, activation="relu", padding='same'))
    output.add(Conv1D(filters=50, kernel_size=5, strides=1, activation="relu", padding='same'))
    output.add(Conv1D(filters=50, kernel_size=5, strides=1, activation="relu", padding='same'))

    output.add(Dense(1024, activation='relu'))
    output.add(Dense(1, activation='sigmoid'))

    output.compile(optimizer='adam', loss='mae')

    return output


def get_regression_subnetwork(appliance_name):
    output = Sequential(name="RegressionSubNetwork{}".format(appliance_name))

    k = 4
    f = 16
    h = 256

    output.add(
        Conv1D(filters=f, kernel_size=k, strides=1, activation="relu", padding='same', input_shape=(1, 1)))

    output.add(Conv1D(filters=f, kernel_size=k, strides=1, activation="relu", padding='same'))
    output.add(Conv1D(filters=f, kernel_size=k, strides=1, activation="relu", padding='same'))
    output.add(Conv1D(filters=f, kernel_size=k, strides=1, activation="relu", padding='same'))
    output.add(Conv1D(filters=f, kernel_size=k, strides=1, activation="relu", padding='same'))

    output.add(
        Bidirectional(LSTM(h, return_sequences=True, stateful=False, activation="tanh"), merge_mode='concat'))

    output.add(Dense(h, activation='relu'))
    output.add(Dense(1, activation='linear'))

    output.compile(optimizer='adam', loss='mae')

    return output
