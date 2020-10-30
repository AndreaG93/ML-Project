from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense


class ClassificationNetwork(object):

    def __init__(self, debug=False):
        self._model = Sequential()

        self._model.add(
            Conv1D(filters=30, kernel_size=10, strides=1, activation="relu", padding='same', input_shape=(1, 2)))

        self._model.add(Conv1D(filters=30, kernel_size=8, strides=1, activation="relu", padding='same'))
        self._model.add(Conv1D(filters=40, kernel_size=6, strides=1, activation="relu", padding='same'))
        self._model.add(Conv1D(filters=50, kernel_size=5, strides=1, activation="relu", padding='same'))
        self._model.add(Conv1D(filters=50, kernel_size=5, strides=1, activation="relu", padding='same'))
        self._model.add(Conv1D(filters=50, kernel_size=5, strides=1, activation="relu", padding='same'))

        self._model.add(Dense(1024, activation='relu'))
        self._model.add(Dense(2, activation='sigmoid'))

        self._model.compile(optimizer='adam', loss='mae')
        self._model.summary()

        if debug:
            print(self._model.summary())
