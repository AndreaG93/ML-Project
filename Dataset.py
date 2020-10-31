import pandas


class Dataset(object):

    def __init__(self, name, source_filename):
        self._name = name
        self._dataset = pandas.read_csv(source_filename, header=0, index_col='timestamp')

        self._dataset.rename(columns={'power': '{}_power'.format(self._name)}, inplace=True)

        self._values = self._dataset.values

    def get_values(self):
        return self._values

    def get_name(self):
        return self._name
