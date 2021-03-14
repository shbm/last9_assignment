import unittest
from data_gen.GenerateData import ElectricityData


class TestGenData(unittest.TestCase):
    def test_gen_data(self):
        n_days = 30
        min_units = 10
        max_units = 20
        e = ElectricityData(2019, 1, 1, n_days, min_units, max_units)
        x = e.compute()

        self.assertEqual('units' in x, True)
        self.assertEqual('dates' in x, True)
        self.assertEqual(len(x['units']), len(x['dates']))
        self.assertEqual(len(x['units']), n_days)
        self.assertEqual(max(x['units']) <= max_units, True)
        self.assertEqual(min(x['units']) >= min_units, True)
