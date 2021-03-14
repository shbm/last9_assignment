from forecast.TripleExponentialSmoothing import TripleExponentialSmoothing
import unittest
import numpy as np


class TestTripleExponentialSmoothing(unittest.TestCase):
    def test_TripleExponentialSmoothing(self):
        size = 100
        n_preds = 20
        x = np.arange(size)
        tes = TripleExponentialSmoothing(ts=x,
                                         seasonality=2,
                                         alpha=0.2,
                                         beta=0.2,
                                         gamma=0.2,
                                         n_preds=n_preds)
        forecast = tes.triple_exponential_smoothing()
        self.assertEqual(len(forecast.shape) == 1, True)
        self.assertEqual(forecast[0] == size+n_preds, True)