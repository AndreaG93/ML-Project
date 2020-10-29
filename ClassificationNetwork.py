from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, LSTM, Bidirectional


class ClassificationNetwork(object):

    def __init__(self, dataset_size, debug=False):
        self._model = Sequential()

        # 1D Conv
        # self._model.add(Conv1D(32, (1, 1, 1), input_shape=(dataset_size, 3, 1), strides=1))

        # Fully Connected Layers
        self._model.add(Dense(1024, activation='relu'))
        self._model.add(Dense(3, activation='sigmoid'))

        self._model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        inputs = keras.Input(shape=(3,))

        if debug:
            print(self._model.summary())

    def baseline_model(self):
        model = Sequential()
        model.add(Dense(8, input_dim=4, activation='relu'))
        model.add(Dense(3, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model
