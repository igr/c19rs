import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# Saves figures to output folder
def savefig(name):
    plt.xlabel('Vreme')
    plt.savefig('./out/' + name + '.png')


# Region to watch
REGION = 'Serbia'
LOGY = False

# Style
FONT_DICT = {'fontsize': 24, 'fontweight': 'bold', 'family': 'serif'}
FIG_SIZE = (10, 4)

# Open-data URL
BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/' \
          'csse_covid_19_data/' \
          'csse_covid_19_time_series'


confirmed = pd.read_csv(BASE_URL + '/time_series_19-covid-Confirmed.csv', error_bad_lines=False)
death = pd.read_csv(BASE_URL + '/time_series_19-covid-Deaths.csv', error_bad_lines=False)
recover = pd.read_csv(BASE_URL + '/time_series_19-covid-Recovered.csv', error_bad_lines=False)


def format_region(s):
    s = str(s)
    if s == 'nan':
        return ''
    return '_' + s


# merge regions names
confirmed['region'] = confirmed['Country/Region'].map(str) + confirmed['Province/State'].map(format_region)
death['region'] = death['Country/Region'].map(str) + death['Province/State'].map(format_region)
recover['region'] = recover['Country/Region'].map(str) + '_' + recover['Province/State'].map(format_region)


def create_ts(data_frame):
    ts = data_frame
    ts = ts.drop(['Province/State', 'Country/Region', 'Lat', 'Long'], axis=1)
    ts.set_index('region')
    ts = ts.T
    ts.columns = ts.loc['region']
    ts = ts.drop('region')
    ts = ts.fillna(0)
    ts = ts.reindex(sorted(ts.columns), axis=1)
    return ts


# create time series
ts_c = create_ts(confirmed)
ts_d = create_ts(death)
ts_r = create_ts(recover)

# plot time series (no China_Hubei)
p_c = ts_c.reindex(ts_c.max().sort_values(ascending=False).index, axis=1)
p_c.iloc[:, 1:11].plot(marker='x', figsize=FIG_SIZE, logy=LOGY)
p_c.loc[:, REGION].plot(marker='o', figsize=FIG_SIZE, logy=LOGY).set_title('Potrvđeno', fontdict=FONT_DICT)
savefig('confirmed-10')

p_d = ts_d.reindex(ts_c.mean().sort_values(ascending=False).index, axis=1)
p_d.iloc[:, 1:11].plot(marker='x', figsize=FIG_SIZE)
p_d.loc[:, REGION].plot(marker='o', figsize=FIG_SIZE).set_title('Smrtnost', fontdict=FONT_DICT)
savefig('death-10')


# kalman filter
class Filter:
    def __init__(self, x0):
        self.kf = KalmanFilter(dim_x=2, dim_z=1)

        # Initial state estimate
        self.kf.x = np.array([x0, 0])

        # Initial Covariance matrix
        # An uncertainty must be given for the initial state.
        self.kf.P = np.eye(2) * 10 ** 2

        # State transition function
        self.kf.F = np.array([
            [1., 1.],
            [0., 1.]
        ])

        # Process noise
        self.kf.Q = Q_discrete_white_noise(dim=2, dt=1, var=1)

        # Measurement noise
        # This measurement uncertainty indicates
        # how much one trusts the measured values ​​of the sensors.
        # If the sensor is very accurate, small values ​​should be used here.
        # If the sensor is relatively inaccurate, large values ​​should be used here.
        self.kf.R = np.array([[0.1]])

        # Measurement function
        # The filter must also be told what is measured
        # and how it relates to the state vector
        self.kf.H = np.array([[1., 0.]])

    def run(self, z):
        self.kf.predict()
        self.kf.update(z)

    def next(self):
        self.kf.predict()
        return self.kf.x[0]

    def predicted(self):
        return int(self.kf.x[0])


def create_prediction(ts, region):
    m = ts.reset_index()
    m = m.rename(columns={'index': 'date'})
    m['date'] = pd.to_datetime(m['date'], errors='coerce')

    # result
    r = pd.DataFrame(columns=["date", "values", "p"])
    r['date'] = m['date']
    r['values'] = m[region]

    f = Filter(0)
    index = 0
    last_date = r.at[index, 'date']
    for val in m[region].tolist():
        if val > 0:
            f.run(val)
        next_val = f.predicted()
        r.at[index, 'p'] = next_val

        last_date = r.at[index, 'date']
        index += 1

    # prediction for the next day
    next_val = int(f.next())
    r.at[index, 'p'] = next_val
    r.at[index, 'date'] = last_date + pd.DateOffset(1)

    print('Predictions')
    print(r.at[index, 'date'])
    print(next_val)

    # prepare data frame for plotting and reindex
    r = r.rename(columns={'p': 'prediction'})
    r = r.set_index('date')
    return r


p = create_prediction(ts_c, REGION)
p.to_csv('./out/prediction.csv')
p.plot(marker='x', figsize=FIG_SIZE).set_title('Predikcija: ' + REGION, fontdict=FONT_DICT)
savefig('prediction')
