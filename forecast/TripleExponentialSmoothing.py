import numpy as np
from tqdm import tqdm


class TripleExponentialSmoothing:
    """
    Triple Exponential Smoothing. It works for data which has trend and seasonality.

    Examples
    --------

    >>> size = 10
    >>> x = np.arange(size)
    >>> tes = TripleExponentialSmoothing(ts=x,seasonality=2,alpha=0.2,beta=0.2,gamma=0.2,n_preds=5)
    >>> forecast = tes.triple_exponential_smoothing()
    >>> print(forecast.shape)
    (15,)

    References
    ----------

    https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc435.htm
    https://otexts.com/fpp2/holt-winters.html
    https://mlcourse.ai/
    """

    def __init__(self, ts: list, seasonality=20, alpha=0.2, beta=0.2, gamma=0.2, n_preds=14):
        """
        :param ts: timeseries data, list or np.array
        :param seasonlity: season length, int
        :param alpha: alpha coefficient of Triple Exponential Smoothing, float
        :param beta: beta coefficient of Triple Exponential Smoothing, float
        :param gamma: gamma coefficient of Triple Exponential Smoothing, float
        :param n_preds: number of future predictions, int
        """
        self.ts = ts
        self.seasonality = seasonality
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_preds = n_preds
        self.alpha_opt = None
        self.beta_opt = None
        self.gamma_opt = None
        self.seasonality_opt = None
        self.forecast = None

    def initial_trend(self):
        """
        To calculate the trend when we are processing the first value in the timeseries based on the seasonlity passed

        :return: float
        """
        seasonality = self.seasonality
        s = 0.0
        try:
            for i in range(seasonality):
                s += float(self.ts[i + seasonality] - self.ts[i]) / seasonality

        except IndexError as e:
            raise ValueError(f"Seasonality Value {seasonality} is too high. "
                             "Decrease the value of seasonality")
        return s / seasonality

    def initial_seasonal_components(self):
        """
        Returns the seasonality component of each index less than passed seasonlity
        :return: dict[int, float]
        """
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.ts) / self.seasonality)
        for j in range(n_seasons):
            season_averages.append(
                sum(self.ts[self.seasonality * j:self.seasonality * j + self.seasonality]) / float(self.seasonality))
        for i in range(self.seasonality):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += self.ts[self.seasonality * j + i] - season_averages[j]
            seasonals[i] = sum_of_vals_over_avg / n_seasons
        return seasonals

    def triple_exponential_smoothing(self):
        """
        Computes the forecast which is the next n_pred values
        :return: np.ndarray
        """
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma

        result = []
        seasonals = self.initial_seasonal_components()

        for i in range(len(self.ts) + self.n_preds):
            if i == 0:  # initial values
                smooth = self.ts[0]
                trend = self.initial_trend()
                result.append(self.ts[0])
                continue
            if i >= len(self.ts):  # we are forecasting
                m = i - len(self.ts) + 1
                result.append((smooth + m * trend) + seasonals[i % self.seasonality])
            else:
                val = self.ts[i]
                last_smooth = smooth
                smooth = alpha * (val - seasonals[i % self.seasonality]) + (1 - alpha) * (smooth + trend)
                trend = beta * (smooth - last_smooth) + (1 - beta) * trend
                seasonals[i % self.seasonality] = gamma * (val - smooth) + (1 - gamma) * seasonals[i % self.seasonality]
                result.append(smooth + trend + seasonals[i % self.seasonality])
        self.forecast = np.array(result)
        return self.forecast

    def root_mse(self, x, y):
        mse = np.square(np.subtract(x, y)).mean()
        rmse = np.sqrt(mse)
        return rmse

if __name__ == '__main__':
    size = 100
    x = np.arange(size)
    tes = TripleExponentialSmoothing(ts=x,
                                     seasonality=2,
                                     alpha=0.2,
                                     beta=0.2,
                                     gamma=0.2,
                                     n_preds=20)
    forecast = tes.triple_exponential_smoothing()
    print(forecast)
    print(forecast.shape)
