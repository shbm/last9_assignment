import unittest
from outlier_detection.Outliers import ESDOutlier, ZScoreOutlier


class TestOutliers(unittest.TestCase):
    def test_ESDOutlier(self):
        l = [1, 2, 3, 4, 6, 21 , 6, 7]
        e = ESDOutlier()
        out_index = e.evaluate(l)
        self.assertCountEqual(out_index, [3, 4, 5])

        with self.assertRaises(ValueError):
            out_index = e.evaluate([1, 2, 3, 4, 5, 6, 7, 8, None, 10])

    def test_ZScoreOutlier(self):
        l = [1, 2, 3, 4, 5, 6, 21 , 6, 7]
        outlier_idx_test = [6]
        z = ZScoreOutlier(l, 2)
        out_index = z.detect_outliers()
        self.assertEqual(out_index, outlier_idx_test)

        l = [1, 2, 3, 4, 5, None]
        z = ZScoreOutlier(l, 2)
        with self.assertRaises(ValueError):
            out_index = z.detect_outliers()
