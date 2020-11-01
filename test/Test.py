import unittest

from utility.AccuracyMetrics import AccuracyMetrics
import numpy
import FeatureScaling


class Test(unittest.TestCase):

    def test_score(self):
        real_values = [2, 2, 4, 2]
        predicted_values = [2, 2, 2, 2]

        accuracy_metrics = AccuracyMetrics(real_values, predicted_values)
        print(accuracy_metrics.get_f1())

    def test_feature_scaling(self):
        values = numpy.asarray([60, 40, 40, 50, 52])
        values = FeatureScaling.perform_feature_scaling(values, 'normalization')
        print(values)

        values = numpy.asarray([3.0, 3.0, 4.0, 4.5, 4.2])
        values = FeatureScaling.perform_feature_scaling(values, 'standardization')
        print(values)


if __name__ == '__main__':
    unittest.main()
