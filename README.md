# LAST9 ASSIGNMENT

## Problem Statement
```
Generate time series data for the Electricity meter reading of a household.
The recording is done every day at 6 PM, Daily

Example:
Time                    Units
1/3/2020  6 PM  18.2
2/3/2020 6 PM   38.1
..
23/3/2020 6 PM  540.5
....
1/4/2020 6 PM   717.5


Summertime will reflect more units being consumed, maybe humid days of monsoon as well.
So have to consider that. Generate two years dataset.
That will yield a total of 730 data points(365*2).
Forecast reading from 15th March to 28th March of 2021, to detect if it is an anomaly.

```

## Solution Approach

It is mainly a three part solution.
* Generate time series data ([GenerateData.py](<./last9py/data_gen/GenerateData.py>))
* Perform Forecast for next 15 days ([TripleExponentialSmoothing.py](<./last9py/data_gen/TripleExponentialSmoothing.py>))
* Perform Anomaly Detection ([Outliers.py](<./last9py/outlier_detection/Outliers.py>))

#### Generate time series data

This is done using a naive approach of generating random number from minimum units of electricity consumed
up to a maximum number of electricity consumed with some extra units being consumed in between the month
of March to July. An approach to improve this step would be perform a weighted linear combination of
temperature and humidity to get a real world simulated data. It would look like
`units[i] = w_temp*temp[i] + w_humidity*huimdty[i] + base_units`. Here `temp` and `humidity` is real data collected.

#### Perform Forecast for next 15 days

Here I used an implementation of Triple Exponential Smoothing also called Holt-Winters' Method. It is 
an additive combination of level, trend and seasonality. https://otexts.com/fpp2/holt-winters.html is a great
resource about Holt-Winters'. I tried using Double Exponential Smoothing but it does not capture the 
seasonality. An improvement of this approach can be calculation of lag features, adding new boolen features 
such as "whether the day is a weekday or not", "national holiday or not". These features will effect the 
amount if electricity units being consumed. On top of this data we do regression modelling using 
Linear Regression, XGBoost Regressor, Random Forest Regressor etc.

#### Perform Anomaly Detection

Here I have used only classical statistical approaches like Z-Score and ESD Outlier test which is a combination
of various statistical methods like Moving Average, t distribution p-value. I plot both outliers in the 
final output. Z Score gives a clear output but ESD sometimes gives ambiguous output as it's dependent on
the moving average, and student's t distribution and alpha of the p-value. There is a lot of score of tuning
parameters here if we have ground truth of outliers. Other advanced approach would be using ML models like
Isolation Forest, Local Outlier Factors etc, which can works quite well but required some hyper-parameter
tuning.


#### How to Run

```
$ pip install -r requirements.txt
$ python run.py
```

`run.py` has two functions. One uses TimeSeriesCV and scipy's optimize to find the optimized parameters.
The other functions uses multiple for loop and iterates over a range of possible values to find the
apt. alpha, beta, gamma and seasonality params. I used *rmse*  as the optimizing metric. [Click here to view the notebook
at nbviewer](https://nbviewer.jupyter.org/github/shbm/last9_assignment/blob/master/Run.ipynb)