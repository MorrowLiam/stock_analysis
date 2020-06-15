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


adj_close_df = pd.DataFrame()
for t in stock_df.keys():
    adj_close_df[t] = analysis_df[t]['Adj Close']
adj_close_df

# %% function
frequency=252
periods=8
covariance_matrix = adj_close_df.pct_change().dropna(how="all").cov()*frequency
covariance_matrix

date_delta = end_date - start_date
divided_days = date_delta/periods
times = pd.date_range(start_date, periods=periods, freq=divided_days,normalize=True)


for i in times:
    print(i)
    
#need to replace the datetime time with 0,0,0,0 so I can match with the datetimes in the stockdf. I can then create a dictionary from each of those points and make a covariance matrix from that.

# cov_dict = {}


# %% test
stock_df['MEIKX']

# %%
analysis_df = {}
for t in stock_df.keys():
    analysis_df[t] = pd.DataFrame()
    analysis_df[t]['Adj Close'] = (stock_df[t]['Adj Close'])
    analysis_df[t]['Simple Returns']  = (stock_df[t]['Adj Close'].pct_change(1))
    analysis_df[t]['Total ROI %'] = ((stock_df[t]['Adj Close']-stock_df[t]['Adj Close'].iloc[0])/stock_df[t]['Adj Close'].iloc[0])*100
    analysis_df[t]['Log Returns'] = np.log(stock_df[t]['Adj Close']/stock_df[t]['Adj Close'].shift(1))