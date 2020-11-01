import numpy

from Dataset import Dataset
from utility.Plot import Plot


class DatasetRegistry(object):

    def __init__(self, name):

        self.name = name
        self._registry = {}
        self._size = -1

    def insert(self, dataset_name, dataset_source_file):

        dataset = Dataset(dataset_name, dataset_source_file)
        size = dataset.get_values().size

        if self._size == -1:
            self._size = size
        elif self._size != size:
            raise ValueError("Error: All datasets must have same size!")

        self._registry[dataset_name] = dataset

    def get_datasets_size(self):

        if self._size == -1:
            raise RuntimeError("Error: 'DatasetRegistry' NOT initialized!")

        return self._size

    def get_values_range(self, dataset_name, first_value, last_value):

        output = []

        values = self._registry[dataset_name].get_values()

        for index in range(first_value, last_value):
            new_data = values[index]
            output.append([new_data])

        return numpy.asarray(output)

    def plot(self):

        summary_plot = Plot(self.name, x_label='timestamp', y_label='power')

        for dataset_name, dataset in self._registry.items():
            summary_plot.insert(dataset_name, dataset.get_values())

        summary_plot.show()
