# %% markdown
# Stock Screener
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
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from stock_utilities import stock_utilities as su
sns.set(style="darkgrid")
from statsmodels.tsa.seasonal import seasonal_decompose
from fbprophet import Prophet

# %% fetch data
tickers="T,AAPL,FB"
start_date = pd.to_datetime('1/1/2019', utc=True)
end_date = pd.to_datetime('6/16/2020', utc=True)
#fetch yahoo stock information and save to a variable.
stock_df = su.yahoo_stock_fetch(tickers, start_date, end_date)
#Calculate a number of typical techincal indicators
analysis_df = su.typ_tech_analysis_df(stock_df)


# %% Plot data
#test plot of RSI,Close,and BOlinger Bands
fig, ax = plt.subplots(figsize=(16,10))
analysis_df['FB']['RSI'].plot()
analysis_df['FB']['Adj Close'].plot(figsize=(16,6))
analysis_df['FB']['Close: 30 Day Mean'].plot(figsize=(16,6))
analysis_df['FB']['30 Day Upper Band'].plot(figsize=(16,6))
analysis_df['FB']['30 Day Lower Band'].plot(figsize=(16,6))
ax.legend()
plt.tight_layout()
plt.show()

# %% Plot data

df = pd.DataFrame()

df['adj_close'] = analysis_df['FB']['Adj Close']
n_period = 30
df['rolling_mean'] = analysis_df['FB']['Adj Close'].rolling(window=n_period).mean()
df['rolling_std'] = analysis_df['FB']['Adj Close'].rolling(window=n_period).std()


res = decomposition_results = seasonal_decompose(df['adj_close'],model='multiplicative', period=n_period)

fig, (ax1,ax2,ax3,ax4) = plt.subplots(4,1, figsize=(15,12))
ax1.set(title='Observed')
ax2.set(title='Trend')
ax3.set(title='Seasonal')
ax4.set(title='Residual')

res.observed.plot(ax=ax1)
res.trend.plot(ax=ax2)
res.seasonal.plot(ax=ax3)
res.resid.plot(ax=ax4)

plt.tight_layout()
plt.show()

# %% pull data

# exportList = pd.DataFrame(columns=['Stock', "RS_Rating", "50 Day MA", "150 Day Ma", "200 Day MA", "52 Week Low", "52 week High"])

# for stock in stocklist:
#     n += 1
#     time.sleep(1)

#     print ("\npulling {} with index {}".format(stock, n))
#     # rsi value
#     start_date = datetime.datetime.now() - datetime.timedelta(days=365)
#     end_date = datetime.date.today()

#     df = pdr.get_data_yahoo(stock, start=start_date, end=end_date)
#     df = df.reset_index()
#     df['Date'] = pd.to_datetime(df.Date)
#     data = df.sort_values(by="Date", ascending=True).set_index("Date").last("59D")
#     df = df.set_index('Date')
#     rsi_period = 14
#     chg = data['Close'].diff(1)
#     gain = chg.mask(chg < 0, 0)
#     data['gain'] = gain
#     loss = chg.mask(chg > 0, 0)
#     data['loss'] = loss
#     avg_gain = gain.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
#     avg_loss = loss.ewm(com=rsi_period - 1, min_periods=rsi_period).mean()
#     data['avg_gain'] = avg_gain
#     data['avg_loss'] = avg_loss
#     rs = abs(avg_gain/avg_loss)
#     rsi = 100-(100/(1+rs))
#     rsi = rsi.reset_index()
#     rsi = rsi.drop(columns=['Date'])
#     rsi.columns = ['Value']
#     rsi_list = rsi.Value.to_list()
#     RS_Rating = rsi['Value'].mean()

# try:
#         sma = [50, 150, 200]
#         for x in sma:
#             df["SMA_"+str(x)] = round(df.iloc[:,4].rolling(window=x).mean(), 2)

#         currentClose = df["Adj Close"][-1]
#         moving_average_50 = df["SMA_50"][-1]
#         moving_average_150 = df["SMA_150"][-1]
#         moving_average_200 = df["SMA_200"][-1]
#         low_of_52week = min(df["Adj Close"][-260:])
#         high_of_52week = max(df["Adj Close"][-260:])

#         try:
#             moving_average_200_20 = df["SMA_200"][-20]

#         except Exception:
#             moving_average_200_20 = 0

#         # Condition 1: Current Price > 150 SMA and > 200 SMA
#         if(currentClose > moving_average_150 > moving_average_200):
#             condition_1 = True
#         else:
#             condition_1 = False
#         # Condition 2: 150 SMA and > 200 SMA
#         if(moving_average_150 > moving_average_200):
#             condition_2 = True
#         else:
#             condition_2 = False
#         # Condition 3: 200 SMA trending up for at least 1 month (ideally 4-5 months)
#         if(moving_average_200 > moving_average_200_20):
#             condition_3 = True
#         else:
#             condition_3 = False
#         # Condition 4: 50 SMA> 150 SMA and 50 SMA> 200 SMA
#         if(moving_average_50 > moving_average_150 > moving_average_200):
#             #print("Condition 4 met")
#             condition_4 = True
#         else:
#             #print("Condition 4 not met")
#             condition_4 = False
#         # Condition 5: Current Price > 50 SMA
#         if(currentClose > moving_average_50):
#             condition_5 = True
#         else:
#             condition_5 = False
#         # Condition 6: Current Price is at least 30% above 52 week low (Many of the best are up 100-300% before coming out of consolidation)
#         if(currentClose >= (1.3*low_of_52week)):
#             condition_6 = True
#         else:
#             condition_6 = False
#         # Condition 7: Current Price is within 25% of 52 week high
#         if(currentClose >= (.75*high_of_52week)):
#             condition_7 = True
#         else:
#             condition_7 = False
#         # Condition 8: IBD RS rating >70 and the higher the better
#         if(RS_Rating > 70):
#             condition_8 = True
#         else:
#             condition_8 = False

#         if(condition_1 and condition_2 and condition_3 and condition_4 and condition_5 and condition_6 and condition_7 and condition_8):
#             final.append(stock)
#             index.append(n)

#             dataframe = pd.DataFrame(list(zip(final, index)), columns =['Company', 'Index'])

#             dataframe.to_csv('stocks.csv')

#             exportList = exportList.append({'Stock': stock, "RS_Rating": RS_Rating, "50 Day MA": moving_average_50, "150 Day Ma": moving_average_150, "200 Day MA": moving_average_200, "52 Week Low": low_of_52week, "52 week High": high_of_52week}, ignore_index=True)
#             print (stock + " made the requirements")
#     except Exception as e:
#         print (e)
#         print("No data on "+stock)

# print(exportList)

# writer = ExcelWriter("ScreenOutput.xlsx")
# exportList.to_excel(writer, "Sheet1")
# writer.save()

# # %%
