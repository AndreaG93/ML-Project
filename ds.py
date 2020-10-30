import pandas
import numpy


def get_network_input_for_regression(self, train_data_size, start_from_index=0):
    dataset_values = self._dataset.values

    output_data = []
    output_labels = []

    for index in range(start_from_index, start_from_index + train_data_size):
        ff = dataset_values[index][:2]
        ff[0] = 1.89
        new_data = [ff]

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
