# %% markdown
# Portfolio Optimization - Risk
# %% add path
if __name__ == '__main__' and __package__ is None:
    import sys, os.path
    sys.path
    # append parent of the directory the current file is in
    inputfilename1 = r"C:\Users\Liam Morrow\Documents\Onedrive\Python scripts\_01 Liam Stock Analysis Project\stock_analysis\Python_Core"
    inputfilename2 = r"C:\Users\l.morrow\OneDrive\Python scripts\_01 Liam Stock Analysis Project\stock_analysis\Python_Core"
    sys.path.append(inputfilename1)
    sys.path.append(inputfilename2)

# %% imports
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas import ExcelWriter
from pandas import ExcelFile
from scipy.optimize import minimize
from stock_fetch import stock_utilities as sf
import scipy.optimize as sco
# import cvxpy as cp
# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt.expected_returns import mean_historical_return
# from pypfopt.risk_models import CovarianceShrinkage
# from pypfopt import Plotting
# import pypfopt
sns.set(style="darkgrid")
%matplotlib inline


# %% fetch stock data
tickers="AFDIX,FXAIX,JLGRX,MEIKX,PGOYX,HFMVX,FCVIX,FSSNX,WSCGX,CVMIX,DOMOX,FSPSX,ODVYX,MINJX,FGDIX,CMJIX,FFIVX,FCIFX,FFVIX,FDIFX,FIAFX,BPRIX,CBDIX,OIBYX,PDBZX"
# tickers="AFDIX,FXAIX,JLGRX,MEIKX"
start_date = datetime(2015,1,1)
end_date = datetime(2020,6,1)
stock_df = sf.yahoo_stock_fetch(tickers, start_date, end_date)


# %% make df

analysis_df = {}
for t in stock_df.keys():
    analysis_df[t] = pd.DataFrame()
    analysis_df[t]['Adj Close'] = (stock_df[t]['Adj Close'])
    analysis_df[t]['Simple Returns']  = (stock_df[t]['Adj Close'].pct_change(1))
    analysis_df[t]['Total ROI %'] = ((stock_df[t]['Adj Close']-stock_df[t]['Adj Close'].iloc[0])/stock_df[t]['Adj Close'].iloc[0])*100
    analysis_df[t]['Log Returns'] = np.log(stock_df[t]['Adj Close']/stock_df[t]['Adj Close'].shift(1))

adj_close_df = pd.DataFrame()
for t in stock_df.keys():
    adj_close_df[t] = analysis_df[t]['Adj Close']
adj_close_df


#%% covariance  matrix class
# TODO add to a utilities file
def cov_to_corr(cov_matrix):
    """
    Convert a covariance matrix to a correlation matrix.
    :param cov_matrix: covariance matrix
    :type cov_matrix: pd.DataFrame
    :return: correlation matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(cov_matrix, pd.DataFrame):
        warnings.warn("cov_matrix is not a dataframe", RuntimeWarning)
        cov_matrix = pd.DataFrame(cov_matrix)

    Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
    corr = np.dot(Dinv, np.dot(cov_matrix, Dinv))
    return pd.DataFrame(corr, index=cov_matrix.index, columns=cov_matrix.index)


class covariance_models:

    def __init__(self, prices, returns_data=False, frequency=252):
        """
        :param prices: adjusted closing prices of the asset, each row is a date and each column is a ticker/id.
        :type prices: pd.DataFrame
        :param returns_data: if true, the first argument is returns instead of prices.
        :type returns_data: bool, defaults to False.
        :param frequency: number of time periods in a year, defaults to 252 (the number of trading days in a year)
        :type frequency: int, optional
        """
        # Optional import
        try:
            from sklearn import covariance

            self.covariance = covariance
        except (ModuleNotFoundError, ImportError):
            raise ImportError("Please install scikit-learn via pip")

        if not isinstance(prices, pd.DataFrame):
            warnings.warn("Data is not in a dataframe", RuntimeWarning)
            prices = pd.DataFrame(prices)

        self.frequency = frequency


        if returns_data:
            self.X = prices.dropna(how="all")
        else:
            self.X = prices.pct_change().dropna(how="all")

        self.S = self.X.cov().values

        self.delta = None  # shrinkage constant

    def sample_covariance(self):
        cov_mat = self.X.cov() * self.frequency
        return cov_mat

    def shrunk_covariance(self, delta=0.2):
        """
        Shrink a sample covariance matrix to the identity matrix (scaled by the average
        sample variance). This method does not estimate an optimal shrinkage parameter,
        it requires manual input.
        :param delta: shrinkage parameter, defaults to 0.2.
        :type delta: float, optional
        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        """
        self.delta = delta
        N = self.S.shape[1]
        # Shrinkage target
        mu = np.trace(self.S) / N
        F = np.identity(N) * mu
        # Shrinkage
        shrunk_cov = delta * F + (1 - delta) * self.S
        return self._format_and_annualize(shrunk_cov)

    def _format_and_annualize(self, raw_cov_array):
        """
        Helper method which annualises the output of shrinkage calculations,
        and formats the result into a dataframe
        :param raw_cov_array: raw covariance matrix of daily returns
        :type raw_cov_array: np.ndarray
        :return: annualised covariance matrix
        :rtype: pd.DataFrame
        """
        assets = self.X.columns
        cov = pd.DataFrame(raw_cov_array, index=assets, columns=assets) * self.frequency
        #pyportfolioopt added a fix_nonpositive_semidefinite function here
        return cov


#%% plot covariance
matrix =cov_to_corr(covariance_models(adj_close_df).sample_covariance())
fig, ax = plt.subplots(figsize=(8,8))

cax = ax.imshow(matrix)
fig.colorbar(cax)

ax.set(title='Covariance Matrix')
ax.set_xticks(np.arange(0, matrix.shape[0], 1))
ax.set_xticklabels(matrix.index)
ax.set_yticks(np.arange(0, matrix.shape[0], 1))
ax.set_yticklabels(matrix.index)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

#%% plot covariance
matrix = cov_to_corr(covariance_models(adj_close_df).shrunk_covariance())
fig, ax = plt.subplots(figsize=(8,8))

cax = ax.imshow(matrix2)
fig.colorbar(cax)

ax.set(title='Covariance Matrix')
ax.set_xticks(np.arange(0, matrix.shape[0], 1))
ax.set_xticklabels(matrix.index)
ax.set_yticks(np.arange(0, matrix.shape[0], 1))
ax.set_yticklabels(matrix.index)
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
