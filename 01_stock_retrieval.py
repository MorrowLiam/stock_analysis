#required imports
import pandas as pd
import pandas_datareader as wb
#function to pull multiple stocks into several dataframes from yahoo.
def stock_retrieval(tickers,start_date,end_date):
    #remove commas
    #should add in a check for unique, remove spaces, and check to see if these are needed
    tickers = tickers.split(',')
    #create a dictionary to sort the dataframes
    d = {}
    for tickers in tickers:
        d[tickers] = pd.DataFrame()
        d[tickers] = wb.DataReader(tickers, data_source='yahoo', start=(start_date), end=(end_date))
    return d
#required input
tickers = 'PG,MSFT,F,GE'
start_date = '2012-01-01'
end_date = '2017-01-01'


#call for stock information.
stock_df = stock_retrieval(tickers, start_date, end_date)
stock_df
