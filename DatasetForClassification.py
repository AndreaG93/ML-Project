import pandas
import calendar


class DatasetForClassification(object):

    def __init__(self, aggregate_dataset_file, appliance_dataset_file, debug=False):
      
        aggregate_dataset = pandas.read_csv(aggregate_dataset_file, header=0, parse_dates=['timestamp'])
        aggregate_dataset['timestamp'] = aggregate_dataset['timestamp'].apply(lambda x: calendar.timegm(x.timetuple()))
        print(aggregate_dataset.dtypes)


        #appliance_dataset = pandas.read_csv(appliance_dataset_file, header=0, dtype=types, parse_dates=['timestamp'])

        #aggregate_dataset.rename(columns={'power': 'aggregate_power'}, inplace=True)

        #aggregate_dataset['is_active'] = appliance_dataset.power.map(lambda x: True if x > 0.0 else False)

        self._dataset = aggregate_dataset

        if debug:
            print(self._dataset.head(10))

    def get_numpy_array(self):
        return self._dataset.to_numpy()