import numpy as np
import matplotlib.pyplot as plt
from outlier_detection.Outliers import ZScoreOutlier, ESDOutlier
from forecast.TripleExponentialSmoothing import TripleExponentialSmoothing
from data_gen.GenerateData import ElectricityData
from tqdm import tqdm

def root_mean_square_error(x, y):
    """
    Simple Root Mean Square implementation.
    Shape of X and Y should be the same.

    :param x: list
    :param y: list
    :return: float
    """
    mse = np.square(np.subtract(x, y)).mean()
    rmse = np.sqrt(mse)
    return rmse


if __name__ == '__main__':

    e = ElectricityData(2019, 1, 3, 730, 10, 30)
    d = e.compute()
    dates = d['dates']
    units = d['units']
    ts = units
    d = {}
    best_tes = None
    min_rmse = float('inf')
    new_dates = dates[-1]
    n_preds = 15
    ne = ElectricityData(new_dates.year, new_dates.day, new_dates.month, n_preds, 1, 10)
    ne = ne.compute()
    ned = np.append(np.array(dates), ne['dates'])
    print(ned.shape)
    for season in tqdm(range(50, int(len(ts)/2), 10)):
        for alpha in range(0,10):
            alpha = alpha/10
            for beta in range(0,10):
                beta = beta/10
                for gamma in range(0,10):
                    gamma = gamma/10
                    tes = TripleExponentialSmoothing(ts, season, alpha, beta, gamma, n_preds=15)
                    forecast = tes.triple_exponential_smoothing()
                    rmse_ = root_mean_square_error(forecast[:len(ts)], ts)
                    if rmse_ < min_rmse:
                        min_rmse = rmse_
                        best_tes = tes
                    d[rmse_] = (season, alpha, beta, gamma)

    forecast = best_tes.triple_exponential_smoothing()
    print(forecast.shape)
    print(ned.shape)
    plt.plot(ned, forecast)
    plt.plot(dates, units)
    plt.vlines(x=dates[-1], ymin=min(units)-1, ymax=max(units)+1)
    plt.show()
