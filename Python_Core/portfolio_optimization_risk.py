# %% markdown
# # Portfolio Optimization - Risk
# %% imports
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
# %% fetch stock data
from stock_fetch import stock_fetch as sf
tickers = 'PG,MSFT,F,GE'
start_date = '2012-01-01'
end_date = '2017-01-01'
stock_df = sf.yahoo_stock_fetch(tickers, start_date, end_date)
# %% Pull stock portfolio weights
# need to code into the stock fetch module
#excel pull and try to get an etrade pull

# %% compile returns
analysis_df = {}
for t in stock_df.keys():
    analysis_df[t] = pd.DataFrame()
    analysis_df[t]['Adj Close'] = (stock_df[t]['Adj Close'])
    analysis_df[t]['Simple Returns']  = (stock_df[t]['Adj Close'].pct_change(1))
    analysis_df[t]['Total ROI %'] = ((stock_df[t]['Adj Close']-stock_df[t]['Adj Close'].iloc[0])/stock_df[t]['Adj Close'].iloc[0])*100
    analysis_df[t]['Log Returns'] = np.log(stock_df[t]['Adj Close']/stock_df[t]['Adj Close'].shift(1))



# %% Find Covariance and Correllation


# %% calculate variance and volatility


# %% calculate diversifiable risk


# %% calculate un-diversifiable risk


# %% Recomend allocation adjustments
