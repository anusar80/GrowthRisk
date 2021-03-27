# %% Import modules
from math import sqrt
import os
import pandas as pd
import numpy as np
from numpy.linalg import inv
from scipy.stats import f
from scipy import stats
from scipy.optimize import least_squares
from dwtest import dwtest
from fredapi import Fred
import quandl
from matplotlib import pyplot as plt
from PartialLeastSquares import PartialLeastSquares as PLS
import seaborn as sns
from statsmodels.tsa.api import VAR
from sklearn.preprocessing import RobustScaler
import statsmodels.api as sm
import statsmodels.formula.api as smf
import sstudentt as student
import pymc3
from statsmodels.regression.quantile_regression import QuantReg
from pingouin import multivariate_ttest as hotelling
from statsmodels.tsa import seasonal
from statsmodels.tsa import filters
from scipy import signal
from statsmodels.tsa.seasonal import STL
# from stldecompose import decompose, forecast
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeCV
from statsmodels.api import Logit
from pingouin import corr, partial_corr
import xlsxwriter
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import dendrogram, linkage, distance
from scipy.cluster import hierarchy
from scipy.stats import ttest_ind as ttest
from sklearn.decomposition import PCA
from pandas.plotting import register_matplotlib_converters
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler, RobustScaler, KBinsDiscretizer
from scipy.stats import norm, skewnorm, skew
from matplotlib import dates as mdates
from sklearn.decomposition import PCA
from numpy import matlib as mb
from pyppca import ppca
import plotly.graph_objects as go
import warnings
import time
from beepy import beep
from datetime import date, datetime
from matplotlib.dates import DateFormatter
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)
sns.set()
# %% To fix the date fuck-ups
old_epoch = '0000-12-31T00:00:00'
new_epoch = '1970-01-01T00:00:00'
mdates.set_epoch(old_epoch)
plt.rcParams['date.epoch'] = '000-12-31'
plt.rcParams['axes.facecolor'] = 'w'
plt.style.use('seaborn')
register_matplotlib_converters()


# %% Clock utility function
def TicTocGenerator():
    ti = 0
    tf = time.time()
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti


TicToc = TicTocGenerator()


def toc(tempBool=True):
    tempTimeInterval = next(TicToc)
    if tempBool:
        print("Elapsed time: %f seconds.\n" % tempTimeInterval)


def tic():
    toc(False)


def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, ':')


def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a - a.T) < tol)


def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], Str(point['val']))


def GRS(alpha, resids, mu):
    # GRS test statistic
    # N assets, T factors, and T time points
    # alpha is Nx1 vector of intercepts of the time-series regressions,
    # resids is TxN matrix of residuals,
    # mu is TxL matrix of factor returns
    T, N = resids.shape
    L = mu.shape[1]
    mu_mean = np.array(np.nanmean(mu, axis=0))
    cov_resids = resids.transpose().dot(resids) / (T - L - 1)
    cov_fac = np.array(mu - np.nanmean(mu, axis=0)).transpose().dot(np.array(mu - np.nanmean(mu, axis=0))) / T - 1
    GRS = (T / N) * ((T - N - L) / (T - L - 1)) * ((alpha.transpose().dot(inv(cov_resids))).dot(alpha) / (
            1 + (mu_mean.transpose().dot(inv(cov_fac))).dot(mu_mean)))
    pVal = 1 - f.cdf(GRS, N, T - N - L)
    return GRS, pVal


# %% Set directory
os.chdir('/Users/anusarfarooqui/Docs/Matlab/GrowthRisk/Python')
# %% Get response
api_key = ''
fred = Fred(api_key=api_key)
start = '1968-12-31'
end = '2021-3-31'

url= 'https://www.philadelphiafed.org/-/media/frbp/assets/surveys-and-data/survey-of-professional-forecasters/data-files/files/mean_rgdp_growth.xlsx?la=en&hash=15495A46612CF5D3767AB74D530C6E50'

Mean_RGDP_growth = pd.read_excel(url, usecols=['YEAR', 'QUARTER', 'DRGDP2', 'DRGDP3', 'DRGDP4', 'DRGDP5', 'DRGDP6'])
Mean_RGDP_growth.index = pd.date_range(start=start, end=end, freq='Q')
Mean_RGDP_growth = Mean_RGDP_growth.melt(id_vars=['YEAR', 'QUARTER'], ignore_index=False)
Mean_RGDP_growth.drop(columns=['YEAR', 'QUARTER'], inplace=True)
Mean_RGDP_growth = Mean_RGDP_growth.pivot(columns='variable', values='value')
Mean_RGDP_growth = Mean_RGDP_growth.asfreq('Q', method='pad')
Mean_RGDP_growth = (Mean_RGDP_growth / 100 + 1)**0.25 - 1

RGDP_growth = fred.get_series(series_id='GDPC1', observation_start=start, observation_end=end, frequency='q')
RGDP_growth = pd.DataFrame({'RGDP': np.log(RGDP_growth).diff()})
RGDP_growth = RGDP_growth.asfreq('Q', method='pad')

Growth = RGDP_growth.merge(Mean_RGDP_growth, how='inner', left_index=True, right_index=True)
Error = pd.DataFrame({'Q'+np.str_(num+1): Growth['RGDP'] - Growth['DRGDP'+np.str_(num+2)].shift(periods=num+1) for num in range(5)})
Error['Qtr'] = Error.index
Error = Error.melt(id_vars='Qtr', var_name='Horizon', value_name='Error')
# %% Get features
# EA = fred.get_series(series_id='BOGZ1FL153064476Q', observation_start=start, observation_end=end, frequency='q') / 100
# EA = EA.asfreq('Q', method='pad')
# EA = pd.DataFrame({'Q'+np.str_(num+1): EA.diff().shift(num+1) for num in range(5)}, index=EA.index)
# EA['Qtr'] = EA.index
# EA = EA.melt(id_vars='Qtr', var_name='Horizon', value_name='EA', ignore_index=True)
#
#
# T10Y2Y = fred.get_series(series_id='T10Y2Y', observation_start=start, observation_end=end, frequency='q') / 100
# T10Y2Y = T10Y2Y.asfreq('Q', method='pad')
# T10Y2Y = pd.DataFrame({'Q'+np.str_(num+1): T10Y2Y.diff().shift(num+1) for num in range(5)}, index=T10Y2Y.index)
# T10Y2Y['Qtr'] = T10Y2Y.index
# T10Y2Y = T10Y2Y.melt(id_vars='Qtr', var_name='Horizon', value_name='T10Y2Y', ignore_index=True)
#
#
# BAA10Y = fred.get_series(series_id='BAA10Y', observation_start=start, observation_end=end, frequency='q') / 100
# BAA10Y = BAA10Y.asfreq('Q', method='pad')
# BAA10Y = pd.DataFrame({'Q'+np.str_(num+1): BAA10Y.diff().shift(num+1) for num in range(5)}, index=BAA10Y.index)
# BAA10Y['Qtr'] = BAA10Y.index
# BAA10Y = BAA10Y.melt(id_vars='Qtr', var_name='Horizon', value_name='BAA10Y', ignore_index=True)
#
#
# VIX = fred.get_series(series_id='VXOCLS', observation_start=start, observation_end=end, frequency='q') / 100
# VIX = VIX.asfreq('Q', method='pad')
# VIX = pd.DataFrame({'Q'+np.str_(num+1): VIX.diff().shift(num+1) for num in range(5)}, index=VIX.index)
# VIX['Qtr'] = VIX.index
# VIX = VIX.melt(id_vars='Qtr', var_name='Horizon', value_name='VIX', ignore_index=True)

NFCI = fred.get_series(series_id='NFCI', observation_start=start, observation_end=end, frequency='q') / 100
NFCI = NFCI.asfreq('Q', method='pad')
NFCI = pd.DataFrame({'Q'+np.str_(num+1): NFCI.diff().shift(num+1) for num in range(5)}, index=NFCI.index)
NFCI['Qtr'] = NFCI.index
NFCI = NFCI.melt(id_vars='Qtr', var_name='Horizon', value_name='NFCI', ignore_index=True)
# %% Create combined dataset
data = Error.merge(NFCI, how='inner', left_on=['Qtr', 'Horizon'], right_on=['Qtr', 'Horizon'])
# data = data.merge(T10Y2Y, how='inner', left_on=['Qtr', 'Horizon'], right_on=['Qtr', 'Horizon'])
# data = data.merge(BAA10Y, how='inner', left_on=['Qtr', 'Horizon'], right_on=['Qtr', 'Horizon'])
# data = data.merge(VIX, how='inner', left_on=['Qtr', 'Horizon'], right_on=['Qtr', 'Horizon'])
# data = data.merge(EA, how='inner', left_on=['Qtr', 'Horizon'], right_on=['Qtr', 'Horizon'])
# %% Quantile regressions
warnings.simplefilter('ignore')

features = ['NFCI']
formula = 'Error~1'
for i in range(len(features)):
    formula = formula + '+' + features[i] + '*Horizon - Horizon'
model = smf.quantreg(formula=formula, data=data[np.logical_and(data.Qtr.dt.year<2020, data.Qtr.dt.year>1984)], missing='drop')
quantiles = [0.05, 0.25, 0.75, 0.95]


def get_tvals(q):
    res = model.fit(q=q)
    return res.tvalues


tStat = [get_tvals(x) for x in quantiles]
tStat = pd.DataFrame(np.array(tStat), columns=['Const', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5'], index=['q05', 'q25', 'q75', 'q95'])

plt.figure(dpi=300, figsize=(10, 8))
sns.heatmap(tStat.iloc[:, 1:])
plt.xlabel('forecast horizon')
plt.title(features[0]+': 1985Q1-2019Q4')
plt.savefig('tStat_1985-2019.png')
plt.show()
# %% Get Quantiles
warnings.simplefilter('ignore')

features = ['NFCI']
formula = 'Error~1'
for i in range(len(features)):
    formula = formula + '+' + features[i] + '*Horizon - Horizon'
model = smf.quantreg(formula=formula, data=data[np.logical_and(data.Qtr.dt.year<2020, data.Qtr.dt.year>1984)], missing='drop')
quantiles = [0.05, 0.25, 0.75, 0.95]


def fit_model(q):
    res = model.fit(q=q)
    return res.params

models = [fit_model(x) for x in quantiles]
models = np.array(models)


X = NFCI[np.logical_and(NFCI.Qtr.dt.year<2020, NFCI.Qtr.dt.year>1984)].pivot(index='Qtr', columns='Horizon', values='NFCI')

Q = np.empty(shape=(len(X), 4, 5), dtype='float64')
for i in range(4):
    for j in range(5):
        Q[:, i, j] = models[i, 0] + models[i, 1+j] * X.iloc[:, j]
# %% fit skew-t distribution
precision = 1e-6
horizon = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

def skew_qq(x, mu, sigma, nu, tau):
    return student.SST(mu=mu, sigma=sigma, nu=nu, tau=tau).q(x)


def loss_fun(x, qq, qq_):
    return qq - skew_qq(qq_, mu=x[0], sigma=x[1], nu=x[2], tau=x[3])


qq_ = [0.05, 0.25, 0.75, 0.95]

for num in range(5):
    loc0 = data[data.Horizon==horizon[num]].Error.mean()
    scale0 = data[data.Horizon==horizon[num]].Error.std()
    nu0 = 1
    shape0 = 2.15

    x0 = np.array([loc0, scale0, nu0, shape0])

    bounds = np.array([[-0.01, 0.25 * scale0, 0, 2], [0.05, 3 * scale0, 1.25, 3]])

    SkewParams = np.empty(shape=(len(Q), 4), dtype='float64')

    for i in range(len(Q)):
        print('day', X.index[i])
        tic()
        qq = np.reshape(Q[i, :, num], 4)
        result = least_squares(fun=loss_fun, args=(qq, qq_), x0=x0, bounds=bounds, method='trf', ftol=precision, xtol=precision,
                               gtol=precision, loss='linear', tr_solver='lsmr')
        SkewParams[i, :] = result.x
        toc()

    SkewParams = pd.DataFrame(SkewParams, index=X.index)
    SkewParams.to_pickle('SkewParams_'+horizon[num]+'.pkl')

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, dpi=400, figsize=(12, 10))
    sns.lineplot(x=SkewParams.index, y=SkewParams.iloc[:, 0], ax=ax[0, 0])
    ax[0, 0].set_title('location')
    ax[0, 0].set_ylabel('mu')
    ax[0, 0].set_xlabel('')
    sns.lineplot(x=SkewParams.index, y=SkewParams.iloc[:, 1], ax=ax[0, 1])
    ax[0, 1].set_title('scale')
    ax[0, 1].set_ylabel('sigma')
    ax[0, 1].set_xlabel('')
    sns.lineplot(x=SkewParams.index, y=SkewParams.iloc[:, 2], ax=ax[1, 0])
    ax[1, 0].set_title('skew parameter')
    ax[1, 0].set_ylabel('nu')
    ax[1, 0].set_xlabel('')
    sns.lineplot(x=SkewParams.index, y=SkewParams.iloc[:, 3], ax=ax[1, 1])
    ax[1, 1].set_title('kurtosis parameter')
    ax[1, 1].set_ylabel('tau')
    ax[1, 1].set_xlabel('')
    plt.savefig('parameters_'+horizon[num]+'.png')
    plt.show()
# %% Generate densities
n_pts = 1000

for num in range(5):
    Params = pd.read_pickle('SkewParams_'+horizon[num]+'.pkl')
    y = np.empty(shape=(len(Params), n_pts))
    for i in range(len(Params)):
        x = np.linspace(start=-0.02, stop=0.03, num=n_pts)
        dist = student.SST(mu=Params.iloc[i, 0], sigma=Params.iloc[i, 1], nu=Params.iloc[i, 2], tau=Params.iloc[i, 3])
        y[i, :] = dist.d(x)
    pd.DataFrame(y, index=Params.index).to_excel('density_'+horizon[num]+'.xlsx')