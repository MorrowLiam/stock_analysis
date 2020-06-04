# %% markdown
# # Portfolio Optimization - Risk
# %% imports
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
from datetime import datetime
from pandas import ExcelWriter
from pandas import ExcelFile

if __name__ == '__main__' and __package__ is None:
    import sys, os.path
    sys.path
    # append parent of the directory the current file is in
    inputfilename = r"C:\Users\Liam Morrow\Documents\Onedrive\Python scripts\_01 Liam Stock Analysis Project\stock_analysis\Python_Core"
    inputfilename = r"C:\Users\Liam Morrow\.conda\pkgs"
    inputfilename = r"C:\Users\Liam Morrow\anaconda3"

    sys.path.append(os.path.dirname(os.path.abspath(inputfilename)))
    sys.path.append(inputfilename)


# %% fetch stock data
from stock_fetch import stock_fetch as sf
#tickers = 'PG,MSFT,F,GE'
#tickers = 'FSPSX,FSSNX,FXAIX,JLGRX'
tickers="AFDIX,FXAIX,JLGRX,MEIKX,PGOYX,FSMDX,HFMVX,FCVIX,FSSNX,WSCGX,CVMIX,DOMOX,FSPSX,ODVYX,MINJX,FGDIX,CMJIX,FFIVX,FCIFX,FFVIX,FDIFX,FIAFX,BPRIX,CBDIX,OIBYX,PDBZX"

start_date = datetime(2010,1,1)
end_date = datetime(2020,6,1)
stock_df = sf.yahoo_stock_fetch(tickers, start_date, end_date)
# %% Pull stock portfolio weights
# need to code into the stock fetch module
#excel pull and try to get an etrade pull

# %% print to excel
excel_name = pd.ExcelWriter('funds.xlsx', engine = 'xlsxwriter')
print('Saving:')
for i in stock_df.keys():
    print (i)
    stock_df[i].to_excel(excel_name,str(i))
excel_name.save()

# %% Read from excel
all_sheets_df = pd.read_excel('funds.xlsx', sheet_name=None)
for i in all_sheets_df.keys():
    all_sheets_df[i].set_index('Date', inplace=True)

# %% compile returns
analysis_df = {}
for t in stock_df.keys():
    analysis_df[t] = pd.DataFrame()
    analysis_df[t]['Adj Close'] = (stock_df[t]['Adj Close'])
    analysis_df[t]['Simple Returns']  = (stock_df[t]['Adj Close'].pct_change(1))
    analysis_df[t]['Total ROI %'] = ((stock_df[t]['Adj Close']-stock_df[t]['Adj Close'].iloc[0])/stock_df[t]['Adj Close'].iloc[0])*100
    analysis_df[t]['Log Returns'] = np.log(stock_df[t]['Adj Close']/stock_df[t]['Adj Close'].shift(1))


# %% table plot
plt.figure(figsize=(16,8))
for t in stock_df.keys():
    plt.plot(analysis_df[t]['Total ROI %'])
plt.ylabel('Log Return')
plt.xlabel('Year')
plt.title('Log Return')
plt.legend(stock_df.keys())
plt.tight_layout()
plt.show()

# %% Portfolio Test Weights
num_assets = len(stock_df.keys())
num_ports = 1000
# create random numberes that equal 1. these will be used to test the different percentages of the portfolios

pfolio_returns = []
pfolio_volatilities = []
log_ret_df = pd.DataFrame()

all_weights = np.zeros((num_ports,num_assets))
ret_arr = np.zeros(num_ports)
vol_arr = np.zeros(num_ports)
sharpe_arr = np.zeros(num_ports)

for t in stock_df.keys():
    log_ret_df[t] = analysis_df[t]['Log Returns']

for x in range(num_ports):
    weights = np.random.random(num_assets)
    weights /= np.sum(weights)
    all_weights[x,:] = weights
    # Expected Return
    ret_arr[x] = np.sum((log_ret_df.mean() * weights) *252)

    # Expected Variance
    vol_arr[x] = np.sqrt(np.dot(weights.T, np.dot(log_ret_df.cov() * 252, weights)))

    # Sharpe Ratio
    sharpe_arr[x] = ret_arr[x]/vol_arr[x]

max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]

plt.figure(figsize=(12,8))
plt.scatter(vol_arr,ret_arr,c=sharpe_arr,cmap='plasma')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')

# Add red dot for max SR
eff_frontier = plt.scatter(max_sr_vol,max_sr_ret,c='red',s=50,edgecolors='black')


# %% Find Covariance and Correllation
# Stock Columns
print('Stocks: ')
print(stock_df.keys())
print('\n')

# Rebalance Weights
print('Weights:')
print((all_weights[sharpe_arr.argmax(),:])*100)
print('\n')

# Expected Return
print('Expected Portfolio Return: ')
exp_ret = (np.sum(log_ret_df.mean() * weights) *252)*100
print("%.2f%%" % exp_ret)
print('\n')

# Expected Variance
print('Expected Volatility: ')
exp_vol = (np.sqrt(np.dot(weights.T, np.dot(log_ret_df.cov() * 252, weights))))*100
print("%.2f%%" % exp_vol)
print('\n')

# Sharpe Ratio
print('Sharpe Ratio: ')
print(sharpe_arr.max())
# %% codecell
log_ret_df
def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio
    """
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])
# %% codecell
 %% markdown
# ### Functionalize Return and SR operations
# %% codecell
def get_ret_vol_sr(weights):
    """
    Takes in weights, returns array or return,volatility, sharpe ratio
    """
    weights = np.array(weights)
    ret = np.sum(log_ret.mean() * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(log_ret.cov() * 252, weights)))
    sr = ret/vol
    return np.array([ret,vol,sr])
# %% codecell
from scipy.optimize import minimize
# %% markdown
# To fully understand all the parameters, check out:
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
# %% codecell
help(minimize)
# %% markdown
# Optimization works as a minimization function, since we actually want to maximize the Sharpe Ratio, we will need to turn it negative so we can minimize the negative sharpe (same as maximizing the postive sharpe)
# %% codecell
def neg_sharpe(weights):
    return  get_ret_vol_sr(weights)[2] * -1
# %% codecell
# Contraints
def check_sum(weights):
    '''
    Returns 0 if sum of weights is 1.0
    '''
    return np.sum(weights) - 1
# %% codecell
# By convention of minimize function it should be a function that returns zero for conditions
cons = ({'type':'eq','fun': check_sum})
# %% codecell
# 0-1 bounds for each weight
bounds = ((0, 1), (0, 1), (0, 1), (0, 1))
# %% codecell
# Initial Guess (equal distribution)
init_guess = [0.25,0.25,0.25,0.25]
# %% codecell
# Sequential Least SQuares Programming (SLSQP).
opt_results = minimize(neg_sharpe,init_guess,method='SLSQP',bounds=bounds,constraints=cons)
# %% codecell
opt_results
# %% codecell
opt_results.x
# %% codecell
get_ret_vol_sr(opt_results.x)

# %% Find Covariance and Correllation


# %% calculate variance and volatility


# %% calculate diversifiable risk


# %% calculate un-diversifiable risk


# %% Recomend allocation adjustments
