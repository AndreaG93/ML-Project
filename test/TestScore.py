import unittest

from Score import Score


class TestScore(unittest.TestCase):

    # Returns True or False.
    def test(self):
        predicted_values = [2, 2, 2, 2]
        real_values = [2, 2, 4, 2]

        score_f1 = Score.compute_f1(predicted_values, real_values)
        print(score_f1)


if __name__ == '__main__':
    unittest.main()
