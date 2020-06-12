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


# %% Efficient Frontier Class

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
        #find the average historical returns from list of returns or prices
        mu = mean_historical_return(adj_close_df, returns_data=returns)
        # choose the type of covariance matrix to run analysis on. simple is most basic. ledoit_wolf can be used to select portfolios with significantly lower out-of-sample variance than a set of existing estimators
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

    def monte_carlo_eff_frontier(adj_close_df, n_portfolios = 1000, trading_days = 252, seed = 1, n_points_on_curve = 100,risk_free_rate=0.02):
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
        #update to call for different cov
        cov_mat = returns_df.cov() * trading_days
        tickers = returns_df.keys()

        #random weights for portfolios
        np.random.seed(seed)
        weights = np.random.random(size=(n_portfolios, len(returns_df.keys())))
        #divide portfolios by themselves to make sure the portfolios = 1
        weights /=  np.sum(weights, axis=1)[:, np.newaxis]

        #random portfolio metrics
        portf_rtns = np.dot(weights, avg_returns)
        portf_vol = []
        for i in range(0, len(weights)):
            #create a range of volatility by taking the squareroot(to cancel out the exponent and make sure the result is positive)
            #daily volatility is sum(average closing-closing price/number of data points)^2
            #std of the portfolio = (w1^2*std1^2)+2(w1*std1*w2*std2*(cor or cov))+(w2^2*std2^2), which can be achieved by multiplying matrices
            portf_vol.append(np.sqrt(np.dot(weights[i].T,np.dot(cov_mat, weights[i]))))
        #converts vol to a np array
        portf_vol = np.array(portf_vol)
        #calculate the sharpe ratio. (return-risk_free_rate)/volatility
        portf_sharpe_ratio = (portf_rtns-risk_free_rate) / portf_vol

        #df for random portfolios
        portf_results_df = pd.DataFrame({'returns': portf_rtns,
                                        'volatility': portf_vol,
                                        'sharpe_ratio': portf_sharpe_ratio})

        #locate points to plot efficient frontier
        portf_vol_ef = []
        indices_to_skip = []
        # take the min and max of the returns and generate a range of numbers between
        portf_rtns_ef = np.linspace(portf_results_df.returns.min(),
                                    portf_results_df.returns.max(),
                                    n_points_on_curve)
        # round our returns to smooth the line
        portf_rtns_ef = np.round(portf_rtns_ef, 3)
        portf_rtns = np.round(portf_rtns, 3)

        #create a np array of the min volatility that matches the range between the min and max returns from our random portfolios
        for point_index in range(n_points_on_curve):
            if portf_rtns_ef[point_index] not in portf_rtns:
                indices_to_skip.append(point_index)
                continue
            matched_ind = np.where(portf_rtns == portf_rtns_ef[point_index])
            portf_vol_ef.append(np.min(portf_vol[matched_ind]))
        # remove any excess numbers from the list that did not match the range created
        portf_rtns_ef = np.delete(portf_rtns_ef, indices_to_skip)
        plot_results_df = pd.DataFrame({'returns': portf_rtns_ef,
                                        'volatility': portf_vol_ef})

        #print the max sharpe portfolio and min volatility portfolio
        #find the index of the max sharpe ratio portfolio and select the return,volatility,and sharpe_ratio
        max_sharpe_ind = np.argmax(portf_results_df.sharpe_ratio)
        max_sharpe_portf = portf_results_df.loc[max_sharpe_ind]
        #find the index of the min volatility portfolio and select the return,volatility,and sharpe_ratio
        min_vol_ind = np.argmin(portf_results_df.volatility)
        min_vol_portf = portf_results_df.loc[min_vol_ind]

        # plot the information of the max sharp and min volatility
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

    def scipy_eff_frontier(adj_close_df, n_portfolios = 1000, trading_days = 252, seed = 1, n_points_on_curve = 100,risk_free_rate=0.02):
        """Use scipy approach to generate a efficient frontier. Returns values to use for analysis.
        """
        returns_df = adj_close_df.pct_change().dropna()
        avg_returns = returns_df.mean() * trading_days
        cov_mat = returns_df.cov() * trading_days
        tickers = returns_df.keys()
        #added these to determine range to test for
        max_ann_avg_return = returns_df.mean().max()*trading_days
        min_ann_avg_return = returns_df.mean().min()*trading_days
        rtns_range = np.linspace(min_ann_avg_return, max_ann_avg_return, n_points_on_curve)

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

        def get_highest_sharpe(avg_returns, cov_mat, rtns_range, risk_free_rate):
            args = (avg_returns, cov_mat, risk_free_rate)
            n_assets = len(avg_returns)
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
            return max_sharpe_portf


        #run find efficient frontier
        efficient_portfolios = get_efficient_frontier(avg_returns,cov_mat,rtns_range)

        # min vol port
        min_vol_metrics = get_metrics(efficient_portfolios,avg_returns,cov_mat)
        min_vol_portf = min_vol_metrics[0].sort_values('volatility',ascending=True)
        min_vol_portf_index = min_vol_portf.index[0]
        min_vol_port_weight = pd.DataFrame({'tickers':tickers,'weights':min_vol_metrics[1][min_vol_portf_index]},columns=['tickers','weights'])

        # max sharpe port
        max_sharpe_portf = get_highest_sharpe(avg_returns, cov_mat, rtns_range, risk_free_rate)
        max_sharpe_weights = np.array(max_sharpe_portf.x)
        max_sharpe_weights_df = pd.DataFrame({'tickers':tickers,'weights':max_sharpe_weights},columns=['tickers','weights'])
        m_s_returns = get_portf_rtn(max_sharpe_weights, avg_returns)
        m_s_vol = get_portf_vol(max_sharpe_weights, avg_returns, cov_mat)
        m_s_sr = -float(max_sharpe_portf.fun)
        max_sharpe_metrics = pd.DataFrame([{'returns':m_s_returns,'volatility':m_s_vol,'sharpe':m_s_sr}],columns=['returns','volatility','sharpe'])

        # tear sheet
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

        return min_vol_metrics,tickers,cov_mat,max_sharpe_metrics,min_vol_portf

    def cvxpy_eff_frontier(adj_close_df, n_portfolios = 1000, trading_days = 252, seed = 1, n_points_on_curve = 100,risk_free_rate=0.02):

        return

# %% Monte Carlo Run
mc_ef = efficient_frontier_models.monte_carlo_eff_frontier(adj_close_df,n_portfolios = 3000,n_points_on_curve=100)
#weights of random portfolios
mc_weights = mc_ef[0]
#plot results for efficient frontier line
mc_plot_results_df = mc_ef[1]
#plot results for random portfolios;annualized returns, volatility, sharpe_ratio
mc_portf_results_df = mc_ef[2]
#list of stock tickers used
mc_tickers = mc_ef[3]
#simple covariance matrix; no shrinkage acounted for
mc_cov_mat = mc_ef[4]
#max sharpe portfolio; annualized returns, volatility, sharpe_ratio
mc_max_sharpe_portf = mc_ef[5]
#min volatility portfolio; annualized returns, volatility, sharpe_ratio
mc_min_vol_portf = mc_ef[6]
#fund/stock average annualized return
mc_avg_returns = mc_ef[7]


# %% Scipy Run
sc_ef = scipy_eff_frontier(adj_close_df)
#weights on the eff frontier
sc_ef_weights = sc_ef[0][1]
#returns, volatility, sharpe_ratio
sc_ef_plot_results_df = sc_ef[0][0]
#list of stock tickers used
sc_ef_tickers = sc_ef[1]
#simple covariance matrix; no shrinkage acounted for
sc_ef_cov_mat = sc_ef[2]
#max sharpe portfolio; annualized returns, volatility, sharpe_ratio
sc_ef_max_sharpe_portf = sc_ef[3]
#min volatility portfolio; annualized returns, volatility, sharpe_ratio
sc_ef_min_vol_portf = sc_ef[4].iloc[0]

# %% Monte Carlo Plot
fig, ax = plt.subplots(figsize=(12,6))
#plot the random portfolios
mc_portf_results_df.plot(kind='scatter', x='volatility',
                      y='returns', c='sharpe_ratio',
                      cmap='RdYlGn', edgecolors='black',
                      ax=ax)
ax.set(xlabel='Volatility',
       ylabel='Expected Returns',
       title='Efficient Frontier')
#plot the efficient frontier line
ax.plot(mc_plot_results_df['volatility'], mc_plot_results_df['returns'], 'b--')
#stock/funds plotted on the graph
for asset_index in range(len(mc_tickers)):
    ax.scatter(x=np.sqrt(mc_cov_mat.iloc[asset_index, asset_index]),y=mc_avg_returns[asset_index],marker='o',s=150,color='black',edgecolors='red',label=mc_tickers[asset_index])
    ax.annotate((mc_tickers[asset_index]), (np.sqrt(mc_cov_mat.iloc[asset_index, asset_index]), (mc_avg_returns[asset_index])), xytext=(10,10), textcoords='offset points')

ax.scatter(x=mc_max_sharpe_portf.volatility, y=mc_max_sharpe_portf.returns, c='black', marker='*', s=200, label='Max Sharpe Ratio')
ax.scatter(x=mc_min_vol_portf.volatility,y=mc_min_vol_portf.returns,c='black',marker='P',s=200, label='Minimum Volatility')
ax.legend()
plt.tight_layout()
plt.show()



# %% scipy Plot

fig, ax = plt.subplots(figsize=(12,6))
ax.plot(sc_ef_plot_results_df.volatility, sc_ef_plot_results_df.returns, 'b--', linewidth=3)
plt.scatter(sc_ef_plot_results_df.volatility,sc_ef_plot_results_df.returns,c=sc_ef_plot_results_df.sharpe,cmap='RdYlGn', alpha=1, s=.5)
plt.colorbar(label='Sharpe Ratio')
ax.set(xlabel='Volatility',
       ylabel='Expected Returns',
       title='Efficient Frontier')
#plot stock/funds on chart
for asset_index in range(len(mc_tickers)):
    ax.scatter(x=np.sqrt(mc_cov_mat.iloc[asset_index, asset_index]),y=mc_avg_returns[asset_index],marker='o',s=150,color='black',edgecolors='red',label=mc_tickers[asset_index])
    ax.annotate((mc_tickers[asset_index]), (np.sqrt(mc_cov_mat.iloc[asset_index, asset_index]), (mc_avg_returns[asset_index])), xytext=(10,10), textcoords='offset points')
#plot monte_carlo_eff_frontier overlay
ax.plot(mc_plot_results_df['volatility'], mc_plot_results_df['returns'], 'b--')

#plot scipy max sharpe min vol portfolios
ax.scatter(x=sc_ef_max_sharpe_portf['volatility'], y=sc_ef_max_sharpe_portf['returns'], c='white',edgecolors='black', marker='*', s=200, label='Max Sharpe Ratio')
ax.scatter(x=sc_ef_min_vol_portf['volatility'],y=sc_ef_min_vol_portf['returns'],c='white',edgecolors='black',marker='P',s=200, label='Minimum Volatility')
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
