# %% markdown
# # Simple Fetch (Yahoo)
# %% markdown
# Multpiple Df in a Dictionary
# %% codecell
import pandas as pd
import pandas_datareader as wb
class stock_fetch:    #function to pull multiple stocks into several dataframes from yahoo.
    def yahoo_stock_fetch(tickers,start_date,end_date):
        #remove commas
        #should add in a check for unique, remove spaces, and check to see if these are needed
        tickers = tickers.split(',')
        #create a dictionary to sort the dataframes
        d = {}
        for tickers in tickers:
            d[tickers] = pd.DataFrame()
            d[tickers] = wb.DataReader(tickers, data_source='yahoo', start=(start_date), end=(end_date))
        return d
# #Sample Required Input
# tickers = 'PG,MSFT,F,GE'
# start_date = '2012-01-01'
# end_date = '2017-01-01'
#
# #call for stock information.
# stock_df = stock_fetch.yahoo_stock_fetch(tickers, start_date, end_date)
# #stock list
# stock_df.keys()
# #dataframes
# for t in stock_df.keys():
#     print(stock_df[t])
