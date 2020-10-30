import pandas
import numpy
import matplotlib.pyplot as plot


class ApplianceDataset(object):

    def __init__(self, aggregate_power_dataset, appliance_name):
        appliance_dataset_filename = './dataset/data/{}_train.csv'.format(appliance_name)

        appliance_dataset = pandas.read_csv(appliance_dataset_filename, header=0, index_col='timestamp')
        appliance_dataset.rename(columns={'power': '{}_power'.format(appliance_name)}, inplace=True)

        self._dataset = appliance_dataset.join(aggregate_power_dataset)
        self._appliance_name = appliance_name

    def get_name(self):
        return self._appliance_name

    def get_values(self):
        return self._dataset.values


class EnergyConsumptionRegistry(object):

    def __init__(self, appliances_names):

        self._aggregate_dataset = pandas.read_csv('./dataset/data/main_train.csv', header=0, index_col='timestamp')
        self._aggregate_dataset_values = self._aggregate_dataset.values
        self._appliance_dataset_registry = {}

        for appliance_name in appliances_names:
            dataset = ApplianceDataset(self._aggregate_dataset, appliance_name)
            self._appliance_dataset_registry[appliance_name] = dataset

    def _get_nn_inputs(self, appliance_name, train_data_size, start_from_index, is_for_classification):

        output_data = []
        output_labels = []

        values = self._appliance_dataset_registry[appliance_name].get_values()

        for index in range(start_from_index, start_from_index + train_data_size):
            new_data = self._aggregate_dataset_values[index]
            new_label = values[index][0]

            if is_for_classification:
                new_label = 1 if new_label > 0 else 0

            output_data.append([new_data])
            output_labels.append(new_label)

        return numpy.asarray(output_data), numpy.asarray(output_labels)

    def get_rgs_nn_inputs(self, appliance_name, train_data_size, start_from_index=0):
        return self._get_nn_inputs(appliance_name, train_data_size, start_from_index, False)

    def get_cls_nn_inputs(self, appliance_name, train_data_size, start_from_index=0):
        return self._get_nn_inputs(appliance_name, train_data_size, start_from_index, True)

    def plot(self):

        if not plot.isinteractive():
            plot.interactive(True)

        plot.figure()

        subplot_index = 1

        for name, dataset in self._appliance_dataset_registry.items():
            values = dataset.get_values()
            self._make_subplot(subplot_index, name, values[:, 0])

            subplot_index += 1

        values = self._aggregate_dataset.values
        self._make_subplot(subplot_index, 'Aggregate', values[:, 0])

        plot.show()

    def _make_subplot(self, index, name, values):

        plot.subplot(len(self._appliance_dataset_registry) + 1, 1, index)
        plot.ylim(0, 5000)
        plot.plot(values)
        plot.title(name, loc='right')
