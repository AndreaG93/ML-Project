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
            self.debug_plot()

    def get_numpy_array(self):
        return self._dataset.to_numpy()

    def debug_plot(self):

        plot.figure()

        values = self._dataset.values

        subplots_number = len(self._appliances_names) + 1

        for subplot_index in range(0, subplots_number):

            plot.subplot(subplots_number, 1, subplot_index + 1)
            plot.plot(values[:, subplot_index])
            plot.title(self._dataset.columns[subplot_index], y=0.5, loc='right')

        plot.show()
