import pandas
import numpy
import matplotlib.pyplot as plot


class Dataset(object):

    def __init__(self, appliances_names, debug=False):

        self._appliances_names = appliances_names
        self._dataset = pandas.read_csv('dataset/main_train.csv', header=0)

        for appliance_name in self._appliances_names:
            filename = './dataset/{}_train.csv'.format(appliance_name)

            appliance_dataset = pandas.read_csv(filename, header=0)

            appliance_dataset.rename(columns={'power': '{}_power'.format(appliance_name)}, inplace=True)
            self._dataset = pandas.merge(self._dataset, appliance_dataset)

        if debug:
            self._debug_plot(self._dataset.values)

    def get_standardize_dataset(self, upper_bound, debug=False):
        output = self._dataset.values

        for appliance in range(output.shape[1]):
            mean = output[:upper_bound, appliance].mean()
            std = output[:upper_bound, appliance].std()
            output[:, appliance] = (output[:, appliance] - mean) / std

        if debug:
            print(self._dataset.head(10))
            self._debug_plot(self._dataset.values)

        return output

    def _debug_plot(self, values):

        plot.figure()

        subplots_number = len(self._appliances_names) + 2

        for subplot_index in range(1, subplots_number):
            plot.subplot(subplots_number - 1, 1, subplot_index)
            plot.ylim(0, 5000)
            plot.plot(values[:, subplot_index])
            plot.title(self._dataset.columns[subplot_index], loc='right')

        plot.show()

    def get_network_input_for_regression(self, train_data_size, start_from_index=0):

        dataset_values = self._dataset.values

        output_data = []
        output_labels = []

        for index in range(start_from_index, start_from_index + train_data_size):
            new_data = [dataset_values[index][:2]]

            output_data.append(new_data)
            output_labels.append(dataset_values[index][2:])

        return numpy.asarray(output_data), numpy.asarray(output_labels)

    def get_network_input_for_classification(self, train_data_size, start_from_index=0):

        dataset_values = self._dataset.values

        output_data = []
        output_labels = []

        for index in range(start_from_index, start_from_index + train_data_size):
            new_data = [dataset_values[index][:2]]

            output_data.append(new_data)

            label = dataset_values[index][2:]
            for x in range(0, len(label)):
                label[x] = 1 if label[x] > 0 else 0

            output_labels.append(label)

        return numpy.asarray(output_data), numpy.asarray(output_labels)
