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
from scipy.optimize import minimize
from stock_fetch import stock_utilities as sf
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage
from pypfopt import Plotting
import pypfopt
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
# tickers="AFDIX,FXAIX,JLGRX,MEIKX,PGOYX,FSMDX,HFMVX,FCVIX,FSSNX,WSCGX,CVMIX,DOMOX,FSPSX,ODVYX,MINJX,FGDIX,CMJIX,FFIVX,FCIFX,FFVIX,FDIFX,FIAFX,BPRIX,CBDIX,OIBYX,PDBZX"
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


# %% Efficient Frontier

class efficient_frontier_models: 

    def pyfolio_eff_frontier(adj_close_df,cov_type = "ledoit_wolf",returns = False,risk_free_rate=0.02):
        """Use pyfolio to generate a efficient frontier. Comes with less control over the plot. Returns values to use for analysis.

        Args:
            adj_close_df ([DataFrame]): A dataframe of either adj close prices or returns
            cov_type (str, optional): Type of covariance to consider. Inputs are "simple" for a simple covariance calulation and "ledoit_wolf" for taking shrinkage into account. Defaults to "ledoit_wolf".
            returns (bool, optional): True for returns False for adjusted close prices. Defaults to False.
            risk_free_rate (float, optional): Risk Free rate modifier. Defaults to 0.02.

        Returns:
            [tuple]: [0] weights of the max sharpe value [1] mean historical return [2] Covariance matrix
        """
        mu = mean_historical_return(adj_close_df, returns_data=returns)

        if cov_type == "simple":
            S = pypfopt.risk_models.sample_cov(adj_close_df, returns_data=returns, frequency=252)
            
        elif cov_type == "ledoit_wolf":
            S = CovarianceShrinkage(adj_close_df, returns_data=returns).ledoit_wolf()
        else:
            S = CovarianceShrinkage(adj_close_df, returns_data=returns).ledoit_wolf()

        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe(risk_free_rate=risk_free_rate)
        cleaned_weights = ef.clean_weights()
        
        #Turn off deprecated warnings
        import warnings

        def fxn():
            warnings.warn("deprecated", DeprecationWarning)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
        fxn()
        return cleaned_weights,mu,S


    def monte_carlo_eff_frontier(adj_close_df, n_portfolios = 1000, trading_days = 252, seed = 1, n_points_on_curve = 100):
        """Use Monte Carlo approach to generate a efficient frontier. Returns values to use for analysis.

        Args:
            adj_close_df ([DataFrame]): A dataframe of adj close prices 
            n_portfolios (int, optional): Number of random portfolios to generate. Defaults to 1000.
            trading_days (int, optional): Number of assumed trading days. Defaults to 252.
            seed (int, optional): Random seed. Defaults to 1.
            n_points_on_curve (int, optional): Number of points on the eff frontier curve. Defaults to 100.

        Returns:
            [tuple]: [0] weights of all portfolios, [1] results for the optimal portfolios along the efficient frontier, [2] results of the random portfolios including returns, volatility, and sharpe ratio, [3]stock tickers, [4] covariance matrix of all the stocks analyzed
        """
        #annualized average returns and the corresponding standard deviation
        returns_df = adj_close_df.pct_change().dropna()
        avg_returns = returns_df.mean() * trading_days
        cov_mat = returns_df.cov() * trading_days
        tickers = returns_df.keys()

        #random weights for portfolios
        np.random.seed(seed)
        weights = np.random.random(size=(n_portfolios, len(returns_df.keys())))
        weights /=  np.sum(weights, axis=1)[:, np.newaxis]

        #random portfolio metrics
        portf_rtns = np.dot(weights, avg_returns)
        portf_vol = []
        for i in range(0, len(weights)):
            portf_vol.append(np.sqrt(np.dot(weights[i].T, 
                                            np.dot(cov_mat, weights[i]))))
        portf_vol = np.array(portf_vol)  
        portf_sharpe_ratio = portf_rtns / portf_vol

        #df for random portfolios
        portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                        'volatility': portf_vol,
                                        'sharpe_ratio': portf_sharpe_ratio})

        #locate points to plot efficient frontier
        portf_vol_ef = []
        indices_to_skip = []

        portf_rtns_ef = np.linspace(portf_results_df.returns.min(), 
                                    portf_results_df.returns.max(), 
                                    n_points_on_curve)
        portf_rtns_ef = np.round(portf_rtns_ef, 2)    
        portf_rtns = np.round(portf_rtns, 2)

        for point_index in range(n_points_on_curve):
            if portf_rtns_ef[point_index] not in portf_rtns:
                indices_to_skip.append(point_index)
                continue
            matched_ind = np.where(portf_rtns == portf_rtns_ef[point_index])
            portf_vol_ef.append(np.min(portf_vol[matched_ind]))
            
        portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip)
        plot_results_df = pd.DataFrame({'returns': portf_rtns_ef,
                                        'volatility': portf_vol_ef})
        

        #print the max sharpe portfolio and min volatility portfolio
        max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
        max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]
        min_vol_ind = np.argmin(portf_results_df.volatility)
        min_vol_portf = portf_results_df.loc[min_vol_ind]

        print('Maximum Sharpe Ratio Portfolio ----')
        print('Performance')
        #!!! Sharpe is a ratio not a percent. needs fixed below.
        for index, value in max_sharpe_portf.items():
            print(f'{index}: {100 * value:.2f}%   ', end="", flush=True)
        print('\nWeights')
        for x, y in zip(returns_df.keys(),
        weights[max_sharpe_ind]):
            print(f'{x}: {100*y:.2f}%   ', end="", flush=True)

        print('\n\nMinimum Volatility Portfolio ----')
        print('Performance')
        for index, value in min_vol_portf.items():
            print(f'{index}: {100 * value:.2f}%   ', end=" ", flush=True)
        print('\nWeights')
        for x, y in zip(returns_df.keys(),
        weights[min_vol_ind]):
            print(f'{x}: {100*y:.2f}%   ', end="", flush=True)


        return weights, plot_results_df, portf_results_df, tickers, cov_mat, max_sharpe_portf, min_vol_portf,avg_returns
        


# %% Monte Carlo Run
monte_carlo_ef = efficient_frontier_models.monte_carlo_eff_frontier(adj_close_df,n_portfolios = 3000,n_points_on_curve=100)


# %% Monte Carlo Plot
fig, ax = plt.subplots(figsize=(12,10))
#plot the random portfolios
monte_carlo_ef[2].plot(kind='scatter', x='volatility', 
                      y='returns', c='sharpe_ratio',
                      cmap='RdYlGn', edgecolors='black', 
                      ax=ax)
ax.set(xlabel='Volatility', 
       ylabel='Expected Returns', 
       title='Efficient Frontier')
#plot the efficient frontier line
ax.plot(monte_carlo_ef[1]['volatility'], monte_carlo_ef[1]['returns'], 'b--')
for asset_index in range(len(monte_carlo_ef[3])):
    ax.scatter(x=np.sqrt(monte_carlo_ef[4].iloc[asset_index, asset_index]),y=monte_carlo_ef[7][asset_index],marker='o',s=150,color='black',edgecolors='red',label=monte_carlo_ef[3][asset_index])
    ax.annotate((monte_carlo_ef[3][asset_index]), (np.sqrt(monte_carlo_ef[4].iloc[asset_index, asset_index]), monte_carlo_ef[7][asset_index]), xytext=(10,10), textcoords='offset points')


ax.scatter(x=monte_carlo_ef[5].volatility, y=monte_carlo_ef[5].returns, c='black', marker='*', s=200, label='Max Sharpe Ratio')
ax.scatter(x=monte_carlo_ef[6].volatility,y=monte_carlo_ef[6].returns,c='black',marker='P',s=200, label='Minimum Volatility')
ax.legend()
plt.tight_layout()
plt.show()





# %% Pyfolio Plot
# TODO work out a better way to graph this.
pyfo_info = efficient_frontier_models.pyfolio_eff_frontier(adj_close_df,cov_type="ledoit_wolf",returns = False,risk_free_rate=0.02)
plt.figure(figsize=(10,10))
Plotting.plot_efficient_frontier(pypfopt.cla.CLA(pyfo_info[1],pyfo_info[2],weight_bounds=(0, 1)), points=100, show_assets=True)

plt.tight_layout()
plt.show()


# %%
