class Score:

    @staticmethod
    def _sum(values):

        output = 0.0

        for value in values:
            output += value

        return output

    @staticmethod
    def compute_f1(predicted_values, real_values):

        if len(predicted_values) != len(real_values):
            raise ValueError("Error: 'predicted_values' and 'real_values' have different size!")

        if len(predicted_values) == 0:
            raise ValueError("Error: 'predicted_values' and 'real_values' are empty!")

        numerator = 0.0

        for x in range(0, len(predicted_values)):
            numerator += min(predicted_values[x], real_values[x])

        numerator_p = numerator
        denominator_p = Score._sum(predicted_values)

        numerator_r = numerator
        denominator_r = Score._sum(real_values)

        p = numerator_p / denominator_p
        r = numerator_r / denominator_r

        return 2 * (p * r) / (p + r)
