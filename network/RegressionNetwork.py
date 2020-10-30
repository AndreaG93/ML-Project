from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense
from tensorflow.python.keras.layers import Bidirectional, LSTM


class RegressionNetwork(object):

    def __init__(self, debug=False):
        self._model = Sequential()

        k = 4
        f = 16
        h = 256

        self._model.add(
            Conv1D(filters=f, kernel_size=k, strides=1, activation="relu", padding='same', input_shape=(1, 2)))

        self._model.add(Conv1D(filters=f, kernel_size=k, strides=1, activation="relu", padding='same'))
        self._model.add(Conv1D(filters=f, kernel_size=k, strides=1, activation="relu", padding='same'))
        self._model.add(Conv1D(filters=f, kernel_size=k, strides=1, activation="relu", padding='same'))
        self._model.add(Conv1D(filters=f, kernel_size=k, strides=1, activation="relu", padding='same'))

        self._model.add(
            Bidirectional(LSTM(h, return_sequences=True, stateful=False, activation="tanh"), merge_mode='concat'))

        self._model.add(Dense(h, activation='relu'))
        self._model.add(Dense(2, activation='linear'))

        self._model.compile(optimizer='adam', loss='mae')
        self._model.summary()

        if debug:
            print(self._model.summary())
