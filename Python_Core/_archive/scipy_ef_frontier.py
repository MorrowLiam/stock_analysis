# %% markdown
# Portfolio Optimization - Risk
# %% imports
import numpy as np
import pandas as pd
from pandas_datareader import data as wb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pandas import ExcelWriter
from pandas import ExcelFile
import scipy.optimize as sco
from stock_fetch import stock_utilities as sf
# from pypfopt.efficient_frontier import EfficientFrontier
# from pypfopt.expected_returns import mean_historical_return
# from pypfopt.risk_models import CovarianceShrinkage
# from pypfopt import Plotting
# import pypfopt
sns.set(style="darkgrid")
%matplotlib inline

# if __name__ == '__main__' and __package__ is None:
#     import sys, os.path
#     sys.path
#     # append parent of the directory the current file is in
#     inputfilename1 = r"C:\Users\Liam Morrow\Documents\Onedrive\Python scripts\_01 Liam Stock Analysis Project\stock_analysis\Python_Core"
#     inputfilename2 = r"C:\Users\l.morrow\OneDrive\Python scripts\_01 Liam Stock Analysis Project\stock_analysis\Python_Core"
#     #inputfilename = r"C:\Users\Liam Morrow\.conda\pkgs"
#     #inputfilename = r"C:\Users\Liam Morrow\anaconda3"
#     sys.path.append(inputfilename1)
#     sys.path.append(inputfilename2)

# %% fetch stock data
# tickers="AFDIX,FXAIX,JLGRX,MEIKX,PGOYX,HFMVX,FCVIX,FSSNX,WSCGX,CVMIX,DOMOX,FSPSX,ODVYX,MINJX,FGDIX,CMJIX,FFIVX,FCIFX,FFVIX,FDIFX,FIAFX,BPRIX,CBDIX,OIBYX,PDBZX"
tickers="AFDIX,FXAIX,JLGRX,MEIKX"
start_date = datetime(2015,1,1)
end_date = datetime(2020,6,1)
stock_df = sf.yahoo_stock_fetch(tickers, start_date, end_date)

# %% Excel Functions
# sf.write_to_excel(stock_df)
# new_df = sf.read_from_excel(r"C:\Users\Liam Morrow\Documents\Onedrive\Python scripts\_01 Liam Stock Analysis Project\stock_analysis\Python_Core\funds.xlsx")



# %% fetch stock data

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


# %%----------------------------------------------------------------------------------------------
trading_days = 252
risk_free_rate=0.02
returns_df = adj_close_df.pct_change().dropna()
avg_returns = returns_df.mean() * trading_days
cov_mat = returns_df.cov() * trading_days
tickers = returns_df.keys()


#added these to determine range
max_ann_avg_return = returns_df.mean().max()*trading_days
min_ann_avg_return = returns_df.mean().min()*trading_days


# %%
def reduce_stock_selections(tickers,weights):
    reduce_df = pd.DataFrame({'tickers':tickers,'weights':weights},columns=['tickers','weights'])
    #filter stocks/funds to remove any below .1% allocation
    masked_df = reduce_df.mask(weights<=.005)
    masked_df = masked_df.dropna()
    return masked_df

def get_portf_rtn(weights, avg_returns):
    return np.sum(avg_returns * weights)

def get_portf_vol(weights, avg_returns, cov_mat):
    return np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))

def get_metrics(efficient_portfolios,avg_returns,cov_mat):
    """
    Takes in weights, average returns, a covariance matrix and returns array of return,volatility, sharpe ratio
    """
    weights = np.array([x['x'] for x in efficient_portfolios])
    vol= [x['fun'] for x in efficient_portfolios]
    ret = []
    for i in np.arange(0,(len(weights))):
        ret.append(np.sum((avg_returns) * weights[i]))
    sr = [ret / vol for ret, vol in zip(ret, vol)]
    results = pd.DataFrame({'returns':ret,'volatility':vol,'sharpe':sr},columns=['returns','volatility','sharpe'])
    return (results,weights)

def get_efficient_frontier(avg_rtns, cov_mat, rtns_range):
    efficient_portfolios = []
    n_assets = len(avg_returns)
    args = (avg_returns, cov_mat)
    bounds = tuple((0,1) for asset in range(n_assets))
    initial_guess = n_assets * [1. / n_assets, ]

    for ret in rtns_range:
        constraints = ({'type': 'eq',
                        'fun': lambda x: get_portf_rtn(x, avg_rtns) - ret},
                       {'type': 'eq',
                        'fun': lambda x: np.sum(x) - 1})
        efficient_portfolio = sco.minimize(get_portf_vol, initial_guess,
                                           args=args, method='SLSQP',
                                           constraints=constraints,
                                           bounds=bounds)
        efficient_portfolios.append(efficient_portfolio)

    return efficient_portfolios

def neg_sharpe_ratio(weights, avg_rtns, cov_mat, risk_free_rate):
    portf_returns = np.sum(avg_rtns * weights)
    portf_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_mat, weights)))
    portf_sharpe_ratio = (portf_returns - risk_free_rate) / portf_volatility
    return -portf_sharpe_ratio

# %%
rtns_range = np.linspace(min_ann_avg_return, max_ann_avg_return, 200)
efficient_portfolios = get_efficient_frontier(avg_returns,cov_mat,rtns_range)


# %% codecell
n_assets = len(avg_returns)
args = (avg_returns, cov_mat, risk_free_rate)
constraints = ({'type': 'eq',
                'fun': lambda x: np.sum(x) - 1})
bounds = tuple((0,1) for asset in range(n_assets))
initial_guess = n_assets * [1. / n_assets]
max_sharpe_portf = sco.minimize(neg_sharpe_ratio,
                                x0=initial_guess,
                                args=args,
                                method='SLSQP',
                                bounds=bounds,
                                constraints=constraints)


# %% min vol port
min_vol_metrics = get_metrics(efficient_portfolios,avg_returns,cov_mat)
min_vol_portf = min_vol_metrics[0].sort_values('volatility',ascending=True)
min_vol_portf_index = min_vol_portf.index[0]
min_vol_port_weight = pd.DataFrame({'tickers':tickers,'weights':min_vol_metrics[1][min_vol_portf_index]},columns=['tickers','weights'])

min_vol_metrics[0]

# %% max sharpe port
max_sharpe_weights = np.array(max_sharpe_portf.x)
max_sharpe_weights_df = pd.DataFrame({'tickers':tickers,'weights':max_sharpe_weights},columns=['tickers','weights'])
m_s_returns = get_portf_rtn(max_sharpe_weights, avg_returns)
m_s_vol = get_portf_vol(max_sharpe_weights, avg_returns, cov_mat)
m_s_sr = -float(max_sharpe_portf.fun)
max_sharpe_metrics = pd.DataFrame([{'returns':m_s_returns,'volatility':m_s_vol,'sharpe':m_s_sr}],columns=['returns','volatility','sharpe'])




# %% tear sheet
print('Minimum Volatility portfolio ----')
print('Performance:')
for index, value in min_vol_portf.iloc[0].items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print('\nWeights:')
for x, y in zip(reduce_stock_selections(tickers,min_vol_port_weight.weights).tickers, reduce_stock_selections(tickers,min_vol_port_weight.weights).weights):
    print(f'{x}: {100*y:.2f}% ', end="\n", flush=True)
print('\n')

print('Maximum Sharpe Ratio portfolio ----')
print('Performance')
for index, value in max_sharpe_metrics.iloc[0].items():
    print(f'{index}: {100 * value:.2f}% ', end="", flush=True)
print('\nWeights')
for x, y in zip(reduce_stock_selections(tickers,max_sharpe_weights_df.weights).tickers, reduce_stock_selections(tickers,max_sharpe_weights_df.weights).weights):
    print(f'{x}: {100*y:.2f}% ', end="\n", flush=True)




# %% Plotting
fig, ax = plt.subplots(figsize=(12,6))
ax.plot(metrics[0]['volatility'], metrics[0]['returns'], 'b--', linewidth=3)
plt.scatter(metrics[0]['volatility'],metrics[0]['returns'],c=metrics[0]['sharpe'],cmap='plasma', alpha=1, s=.5)
plt.colorbar(label='Sharpe Ratio')
ax.set(xlabel='Volatility',
       ylabel='Expected Returns',
       title='Efficient Frontier')
ax.scatter(x=max_sharpe_metrics['volatility'], y=max_sharpe_metrics['returns'], c='black', marker='*', s=200, label='Max Sharpe Ratio')
ax.scatter(x=min_vol_portf.iloc[0]['volatility'],y=min_vol_portf.iloc[0]['returns'],c='black',marker='P',s=200, label='Minimum Volatility')
ax.legend()
plt.tight_layout()
plt.show()
