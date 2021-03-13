import numpy as np
import datetime


class ElectricityData:
    """
    To generate fake electricity data using RNG
    """

    def __init__(self, year: int, date: int, month: int, n_days: int, base_units: float, max_units: float):
        """

        :param year: int
        :param date: int
        :param month: int
        :param n_days: int, number of days
        :param base_units: float, minimum units
        :param max_units: int, maximum units
        """
        self.dates = []
        self.units = []
        self.base_units = float(base_units)
        self.max_units = float(max_units)
        self.year = year
        self.date = date
        self.month = month
        self.n_days = n_days
        self.init_date = None

    def compute(self):
        """
        Computes fake electricity data from init_date upto n_days

        :return:  dict[str, list], contains 'dates' and electricity 'units'
        """
        self.init_date = datetime.date(self.year, self.month, self.date)
        base_units = self.base_units
        ceil = self.max_units
        init_date = self.init_date
        for i in range(self.n_days):
            new_date = init_date + datetime.timedelta(days=1)
            init_date = new_date
            self.dates.append(new_date)

            if new_date.month in (3, 4, 5, 6, 7):
                r = base_units * (1.5 + new_date.month / 10)
            else:
                r = base_units

            units = (np.random.randint(r*10, ceil*10))/10
            self.units.append(units)

        return {
            'dates': self.dates,
            'units': self.units
        }


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ed = ElectricityData(2019, 1, 1, 730, 10.0, 30.0)
    d = ed.compute()
    print((d['dates']))
    print((d['units']))
    plt.plot(d['dates'], d['units'])
    plt.show()
