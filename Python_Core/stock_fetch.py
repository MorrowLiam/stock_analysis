# %% markdown
# # Simple Fetch (Yahoo)
# %% markdown
# Multpiple Df in a Dictionary
# %% codecell
import pandas as pd
import pandas_datareader as wb
from datetime import datetime
from pandas import ExcelWriter
from pandas import ExcelFile

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
        d = {}
        for tickers in tickers:
            d[tickers] = pd.DataFrame()
            d[tickers] = wb.DataReader(tickers, data_source='yahoo', start=(start_date), end=(end_date))
        return d

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
        

