# %% Import
import pandas as pd
import pandas_datareader as wb
from datetime import datetime
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np

# %% Stock_utilities Class
class stock_utilities:
    def yahoo_stock_fetch(tickers,start_date,end_date):
        """ function to pull multiple stocks into several dataframes from yahoo

        Args:
            tickers ([str]): [stock or fund tickers]
            start_date ([datetime]): [date time]
            end_date ([datetime]): [date time]

        Returns:
            [DataFrame]: [Dataframe with date, open, close, high, low, adj close]
        """
        # TODO: should add in a check for unique, remove spaces, and check to see if these are needed
        tickers = tickers.split(',')
        #create a dictionary to sort the dataframes
        df = {}
        for tickers in tickers:
            df[tickers] = pd.DataFrame()
            df[tickers] = wb.DataReader(tickers, data_source='yahoo', start=(start_date), end=(end_date))
        return df

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

    def typ_tech_analysis_df(stock_df, RS_n_days=14):
        """TODO Add doc string
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
