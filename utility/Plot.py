import matplotlib.pyplot as plt


class Plot(object):
    """
    A very simple utility class to manage plots.
    """

    def __init__(self, name, x_label="", y_label=""):
        self._name = name
        self._data_registry = {}
        self._x_label = x_label
        self._y_label = y_label

    def insert(self, data_label, data):
        self._data_registry[data_label] = data

    def show(self):
        figure, axes = plt.subplots()

        for data_label, data in self._data_registry.items():
            axes.plot(data, label=data_label)

        axes.legend(loc='upper center', shadow=True)

        plt.title(self._name)
        plt.xlabel(self._x_label)
        plt.ylabel(self._y_label)
        plt.show()
