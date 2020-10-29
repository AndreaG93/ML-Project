import pandas
import calendar

import matplotlib.pyplot as plot


class DatasetForClassification(object):

    def __init__(self, appliances_names, debug=False):

        self._appliances_names = appliances_names
        self._dataset = pandas.read_csv('./dataset/main_train.csv',
                                        header=0,
                                        index_col='timestamp')

        for appliance_name in self._appliances_names:
            filename = './dataset/{}_train.csv'.format(appliance_name)

            appliance_dataset = pandas.read_csv(filename,
                                                header=0,
                                                index_col='timestamp')

            appliance_dataset.rename(columns={'power': '{}_power'.format(appliance_name)}, inplace=True)
            self._dataset = self._dataset.join(appliance_dataset)

        if debug:
            self.debug_plot(self._dataset.values)

    def get_standardize_dataset(self, upper_bound, debug=False):
        output = self._dataset.values

        for appliance in range(output.shape[1]):
            mean = output[:upper_bound, appliance].mean()
            std = output[:upper_bound, appliance].std()
            output[:, appliance] = (output[:, appliance] - mean) / std

        if debug:
            self.debug_plot(self._dataset.values)

        return output

    def debug_plot(self, values):

        plot.figure()

        subplots_number = len(self._appliances_names) + 1

        max_value = 0

        for subplot_index in range(0, subplots_number):

            values_to_visualize = values[:, subplot_index]
            current_max_value = values_to_visualize.max()

            if current_max_value > max_value:
                max_value = current_max_value

        for subplot_index in range(0, subplots_number):

            plot.subplot(subplots_number, 1, subplot_index + 1)
            plot.ylim(0, max_value)
            plot.plot(values[:, subplot_index])
            plot.title(self._dataset.columns[subplot_index], loc='right')

        plot.show()
