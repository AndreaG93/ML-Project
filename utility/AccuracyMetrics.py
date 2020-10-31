class AccuracyMetrics:
    """
    An utility class to calc accuracy's metrics.
    """

    def __init__(self, real_values, predicted_values):

        if len(predicted_values) != len(real_values):
            raise ValueError("Error: 'predicted_values' and 'real_values' have different size!")

        if len(predicted_values) == 0:
            raise ValueError("Error: 'predicted_values' and 'real_values' are empty!")

        self._real_values = real_values
        self._predicted_values = predicted_values
        self._data_length = len(real_values)

    def get_f1(self):
        """
        Compute F1 score.

        :return: A 'float'.
        """
        numerator = 0.0

        for x in range(0, self._data_length):
            numerator += min(self._predicted_values[x], self._real_values[x])

        numerator_p = numerator
        denominator_p = sum(self._predicted_values)

        numerator_r = numerator
        denominator_r = sum(self._real_values)

        p = numerator_p / denominator_p
        r = numerator_r / denominator_r

        return 2 * (p * r) / (p + r)
