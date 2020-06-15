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
sns.set(style="darkgrid")

# %% fetch stock data
# tickers="AFDIX,FXAIX,JLGRX,MEIKX,PGOYX,HFMVX,FCVIX,FSSNX,WSCGX,CVMIX,DOMOX,FSPSX,ODVYX,MINJX,FGDIX,CMJIX,FFIVX,FCIFX,FFVIX,FDIFX,FIAFX,BPRIX,CBDIX,OIBYX,PDBZX"
tickers="AFDIX,FXAIX,JLGRX,MEIKX"
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

    def ledoit_wolf(self, shrinkage_target="constant_variance"):
        """
        Calculate the Ledoit-Wolf shrinkage estimate for a particular
        shrinkage target.
        :param shrinkage_target: choice of shrinkage target, either ``constant_variance``,
                                 ``single_factor`` or ``constant_correlation``. Defaults to
                                 ``constant_variance``.
        :type shrinkage_target: str, optional
        :raises NotImplementedError: if the shrinkage_target is unrecognised
        :return: shrunk sample covariance matrix
        :rtype: np.ndarray
        """
        if shrinkage_target == "constant_variance":
            X = np.nan_to_num(self.X.values)
            shrunk_cov, self.delta = self.covariance.ledoit_wolf(X)
        elif shrinkage_target == "single_factor":
            shrunk_cov, self.delta = self._ledoit_wolf_single_factor()
        elif shrinkage_target == "constant_correlation":
            shrunk_cov, self.delta = self._ledoit_wolf_constant_correlation()
        else:
            raise NotImplementedError(
                "Shrinkage target {} not recognised".format(shrinkage_target)
            )

        return self._format_and_annualize(shrunk_cov)

    def _ledoit_wolf_single_factor(self):
        """
        Helper method to calculate the Ledoit-Wolf shrinkage estimate
        with the Sharpe single-factor matrix as the shrinkage target.
        See Ledoit and Wolf (2001).
        :return: shrunk sample covariance matrix, shrinkage constant
        :rtype: np.ndarray, float
        """
        X = np.nan_to_num(self.X.values)

        # De-mean returns
        t, n = np.shape(X)
        Xm = X - X.mean(axis=0)
        xmkt = X.mean(axis=1).reshape(t, 1)

        # compute sample covariance matrix
        sample = np.cov(np.append(Xm, xmkt, axis=1), rowvar=False) * (t - 1) / t
        betas = sample[0:n, n].reshape(n, 1)
        varmkt = sample[n, n]
        sample = sample[:n, :n]
        F = np.dot(betas, betas.T) / varmkt
        F[np.eye(n) == 1] = np.diag(sample)

        # compute shrinkage parameters
        c = np.linalg.norm(sample - F, "fro") ** 2
        y = Xm ** 2
        p = 1 / t * np.sum(np.dot(y.T, y)) - np.sum(sample ** 2)

        # r is divided into diagonal
        # and off-diagonal terms, and the off-diagonal term
        # is itself divided into smaller terms
        rdiag = 1 / t * np.sum(y ** 2) - sum(np.diag(sample) ** 2)
        z = Xm * np.tile(xmkt, (n,))
        v1 = 1 / t * np.dot(y.T, z) - np.tile(betas, (n,)) * sample
        roff1 = (
            np.sum(v1 * np.tile(betas, (n,)).T) / varmkt
            - np.sum(np.diag(v1) * betas.T) / varmkt
        )
        v3 = 1 / t * np.dot(z.T, z) - varmkt * sample
        roff3 = (
            np.sum(v3 * np.dot(betas, betas.T)) / varmkt ** 2
            - np.sum(np.diag(v3).reshape(-1, 1) * betas ** 2) / varmkt ** 2
        )
        roff = 2 * roff1 - roff3
        r = rdiag + roff

        # compute shrinkage constant
        k = (p - r) / c
        delta = max(0, min(1, k / t))

        # compute the estimator
        shrunk_cov = delta * F + (1 - delta) * sample
        return shrunk_cov, delta

    def _ledoit_wolf_constant_correlation(self):
        """
        Helper method to calculate the Ledoit-Wolf shrinkage estimate
        with the constant correlation matrix as the shrinkage target.
        See Ledoit and Wolf (2003)
        :return: shrunk sample covariance matrix, shrinkage constant
        :rtype: np.ndarray, float
        """
        X = np.nan_to_num(self.X.values)
        t, n = np.shape(X)

        S = self.S  # sample cov matrix

        # Constant correlation target
        var = np.diag(S).reshape(-1, 1)
        std = np.sqrt(var)
        _var = np.tile(var, (n,))
        _std = np.tile(std, (n,))
        r_bar = (np.sum(S / (_std * _std.T)) - n) / (n * (n - 1))
        F = r_bar * (_std * _std.T)
        F[np.eye(n) == 1] = var.reshape(-1)

        # Estimate pi
        Xm = X - X.mean(axis=0)
        y = Xm ** 2
        pi_mat = np.dot(y.T, y) / t - 2 * np.dot(Xm.T, Xm) * S / t + S ** 2
        pi_hat = np.sum(pi_mat)

        # Theta matrix, expanded term by term
        term1 = np.dot((X ** 3).T, X) / t
        help_ = np.dot(X.T, X) / t
        help_diag = np.diag(help_)
        term2 = np.tile(help_diag, (n, 1)).T * S
        term3 = help_ * _var
        term4 = _var * S
        theta_mat = term1 - term2 - term3 + term4
        theta_mat[np.eye(n) == 1] = np.zeros(n)
        rho_hat = sum(np.diag(pi_mat)) + r_bar * np.sum(
            np.dot((1 / std), std.T) * theta_mat
        )

        # Estimate gamma
        gamma_hat = np.linalg.norm(S - F, "fro") ** 2

        # Compute shrinkage constant
        kappa_hat = (pi_hat - rho_hat) / gamma_hat
        delta = max(0.0, min(1.0, kappa_hat / t))

        # Compute shrunk covariance matrix
        shrunk_cov = delta * F + (1 - delta) * S
        return shrunk_cov, delta

    def variation_over_time (self,covariance_matrix, periods, correlation=False):
        stock_df = sf.yahoo_stock_fetch(tickers, start_date, end_date)


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
matrix = cov_to_corr(covariance_models(adj_close_df).ledoit_wolf())
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
