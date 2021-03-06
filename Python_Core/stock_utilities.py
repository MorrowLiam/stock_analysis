# %% [markdown]
# Utility Functions for analysis

# %% Import
import pandas as pd
import pandas_datareader as wb
from datetime import datetime
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import bs4 as bs
import pickle
import requests
import os
import time


# %% Stock Utilities Helper Functions

#fetch and cull functions
def yahoo_stock_fetch(tickers,start_date,end_date,input='list'):
    """ Helper function to pull multiple stocks into several dataframes from yahoo.
    Args:
        tickers ([str]): [stock or fund tickers]
        start_date ([datetime]): [date time]
        end_date ([datetime]): [date time]
        input ([str]):describe type of data for tickers options are 'list' or 'pickle'
    Returns:
        [DataFrame]: [Dataframe with date, open, close, high, low, adj close]
    """
    # TODO: should add in a check for unique, remove spaces, and check to see if these are needed
    if input=="list":
        tickers = tickers.split(',')
        #create a dictionary to sort the dataframes
        df = {}
        for tickers in tickers:
            df[tickers] = pd.DataFrame()
            df[tickers] = wb.DataReader(tickers, data_source='yahoo', start=(start_date), end=(end_date))
        return df
    elif input=="pickle":
        with open(tickers, "rb") as f:
            tickers = pickle.load(f)

        if not os.path.exists('stock_dfs'):
            os.makedirs('stock_dfs')

        for ticker in tickers:
            # just in case your connection breaks, we'd like to save our progress!
            if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
                ticker = ticker.replace('\n', '')
                try:
                    df = wb.DataReader(ticker, data_source='yahoo', start=(start_date), end=(end_date))
                except:
                    print (ticker + "was not able to be fetched")
                    pass
                df.reset_index(inplace=True)
                df.set_index("Date", inplace=True)
                df.to_csv('stock_dfs/{}.csv'.format(ticker))
                time.sleep(0.5)#to throttle speed and avoid yahoo stoppping the data request
            else:
                print('Already have {}'.format(ticker))

def write_to_excel(df,file_name="funds.xlsx"):
    """Save dataframe(s) to excel

    Args:
        df ([DataFarme(s)]): DataFrame or Dictionary with multiple dataframes to send to excel
        file_name ([str], optional): File name to save excel as. Defaults to funds.xlsx.
    """
    excel_name = pd.ExcelWriter(file_name, engine = 'xlsxwriter')
    print('Saving:')
    for i in df.keys():
        print (i)
        df[i].to_excel(excel_name,str(i))
    excel_name.save()

def read_from_excel(file_path):
    """Read excel sheet to a DataFrame.
    !!Issues with large files!!

    Args:
        file_path ([str]): Path with a filename to read from.
        parse (bool, optional): Parse the DataFrame to use the dates as the index. Defaults to False.

    Returns:
        [DataFrame]: Excel DataFrame
    """
    all_sheets_df = pd.read_excel(file_path, sheet_name=None)
    return all_sheets_df

def read_sp500_csv(tickers, dfs_file_paths='stock_dfs/', ticker_input='pickle'):
    """ Helper function to put the entire sp500 adj close into a single table

    Args:
        tickers ([str]): [Stock or fund tickers. Can be a pickle of list.]
        dfs_file_paths ([filepath]): [folder path to all the csv files of the stocks]
        input ([str]):describe type of data for tickers options are 'list' or 'pickle'
    Returns:
        [DataFrame]: [Single Dataframe of all stocks inputted with adj close]
    """
    try:
        #determine input type and set tickers variable
        if ticker_input=='pickle':
            #open the pickle and read the tickers
            with open(tickers, "rb") as f:
                tickers = pickle.load(f)
                #remove odd char and prep data
                ticker = []
                for count,i in enumerate(tickers):
                    i = i.replace('\n', '')
                    ticker.append(i)
                tickers = ticker
        else:
            #split a list of tickers at commas
            tickers = tickers.split(',')

        #create a df to host the data
        main_df = pd.DataFrame()

        #single table for comparision
        for count, ticker in enumerate(tickers):
            df = pd.read_csv(dfs_file_paths+'{}.csv'.format(ticker))
            # Prep the df. Set the date as the index, Rename Adj Close, Drop Other Columns.
            df.set_index('Date', inplace=True)
            df.rename(columns={'Adj Close': ticker}, inplace=True)
            df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

            #check if the df is empty, if not join to the df.
            if main_df.empty:
                main_df = df
            else:
                main_df = main_df.join(df, how='outer')

        print(main_df.head())
        main_df.to_csv('sp500_joined_closes.csv')

    except Exception as e:
        print(e)

def scrape_sp500_tickers():
    """Scrape the wikipedia page for the latest list of sp500 companies

    Returns:
        [pickle]: [list of sp500 companies]
    """
    #set get file to look at wikipedia's list of sp500 companies
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    #cycle through wiki table to find get all of the stock tickers
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)
    #save to pickle to speed up process
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
    print ('Scraping Complete')
    return tickers

def iex_stock_fetch(tickers,start_date,end_date):
    """All of this data can be taken from the etrade wrapper.Save in case it is useful later."""
    import configparser
    import json
    config = configparser.ConfigParser()
    config_path=r"config.ini"
    config.read(config_path)
    secret_key = config.get('DEFAULT', 'IEX_SECRET_KEY')
    base_url = config.get('DEFAULT', 'IEX_BASE_URL')
    
    tickers = tickers.split(',')

    results = {}
    for count,ticker in enumerate(tickers):
        stats_url = base_url+f'stock/{ticker}/stats/stats?token={secret_key}'
        stats_response = requests.get(stats_url)
        results[ticker] = (stats_response.json())

    df= pd.DataFrame(results)
    return df

# Operations on analysis
def cov_to_corr(cov_matrix):
    """
    Convert a covariance matrix to a correlation matrix.
    :param cov_matrix: covariance matrix
    :type cov_matrix: pd.DataFrame
    :return: correlation matrix
    :rtype: pd.DataFrame
    """
    if not isinstance(cov_matrix, pd.DataFrame):
        cov_matrix = pd.DataFrame(cov_matrix)

    Dinv = np.diag(1 / np.sqrt(np.diag(cov_matrix)))
    corr = np.dot(Dinv, np.dot(cov_matrix, Dinv))
    return pd.DataFrame(corr, index=cov_matrix.index, columns=cov_matrix.index)

# dataframe functions
def typ_tech_analysis_df(stock_df, RS_n_days=14):
    """Creates a full dataframe for specific stock statistics. Uses OHLC stock dataframe.

    Args:
        stock_df ([DataFrame]): Stock OHLC DataFrame
        RS_n_days (int, optional): Defined period for RS index. Defaults to 14.

    Returns:
        DataFrame: DataFrame including adj close, volume, close delta, 
        simple return %, total ROI%, log returns, RSI, bolinger bands, 
        52 week high and low.
    """
    analysis_df = {}
    for t in stock_df.keys():
        analysis_df[t] = pd.DataFrame()
        analysis_df[t]['Adj Close'] = (stock_df[t]['Adj Close'])
        analysis_df[t]['Volume'] = (stock_df[t]['Volume'])
        analysis_df[t]['Close Delta']  = (stock_df[t]['Adj Close'].diff(1))
        analysis_df[t]['Simple Return %']  = (stock_df[t]['Adj Close'].pct_change(1))
        analysis_df[t]['Total ROI %'] = ((stock_df[t]['Adj Close']-stock_df[t]['Adj Close'].iloc[0])/stock_df[t]['Adj Close'].iloc[0])*100
        analysis_df[t]['Log Returns'] = np.log(stock_df[t]['Adj Close']/stock_df[t]['Adj Close'].shift(1))

        #calculate RSI
        gains = analysis_df[t]['Close Delta'].mask(analysis_df[t]['Close Delta']<0,0)
        losses = analysis_df[t]['Close Delta'].mask(analysis_df[t]['Close Delta']>0,0)
        analysis_df[t][str(RS_n_days) + ' RS Average Gains'] = gains.ewm(com = RS_n_days -1, min_periods=RS_n_days).mean()
        analysis_df[t][str(RS_n_days) + ' RS Average Losses'] = losses.ewm(com = RS_n_days -1, min_periods=RS_n_days).mean()
        RS=abs(analysis_df[t][str(RS_n_days) + ' RS Average Gains']/analysis_df[t][str(RS_n_days) + ' RS Average Losses'])
        analysis_df[t]['RSI'] = 100 - 100/ (1+RS)

        #Calculate bolinger bands and rolling mean
        analysis_df[t]['Close: 30 Day Mean'] = analysis_df[t]['Adj Close'].rolling(window=30).mean()
        analysis_df[t]['Close: 50 Day Mean'] = analysis_df[t]['Adj Close'].rolling(window=50).mean()
        analysis_df[t]['Close: 150 Day Mean'] = analysis_df[t]['Adj Close'].rolling(window=150).mean()
        analysis_df[t]['Close: 200 Day Mean'] = analysis_df[t]['Adj Close'].rolling(window=200).mean()
        analysis_df[t]['30 Day Upper Band'] = analysis_df[t]['Close: 30 Day Mean'] + 2*analysis_df[t]['Adj Close'].rolling(window=20).std()
        analysis_df[t]['30 Day Lower Band'] = analysis_df[t]['Close: 30 Day Mean'] - 2*analysis_df[t]['Adj Close'].rolling(window=20).std()

        #calc 52 week low/high
        analysis_df[t]['52 Wk High'] = analysis_df[t]['Adj Close'].rolling(min_periods=1, window=252, center=False).max()
        analysis_df[t]['52 Wk Low'] = analysis_df[t]['Adj Close'].rolling(min_periods=1, window=252, center=False).min()
    return analysis_df