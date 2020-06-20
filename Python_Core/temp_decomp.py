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
start_date = pd.to_datetime('1/1/2010', utc=True)
end_date = pd.to_datetime('6/16/2020', utc=True)
#fetch yahoo stock information and save to a variable.
stock_df = su.yahoo_stock_fetch(tickers, start_date, end_date)
#Calculate a number of typical techincal indicators
analysis_df = su.typ_tech_analysis_df(stock_df)

# %%

df = pd.DataFrame()
df['Value'] = analysis_df['FB']['Adj Close']
df.reset_index(inplace=True)
df['Date'] = pd.to_datetime(df['Date'],utc=True)
train_date =  pd.to_datetime('1/1/2015', utc=True)
train_df = df.mask(df['Date'] < train_date)
train_df.dropna(inplace=True)
train_df.reset_index(drop=True, inplace=True)
# train_df['Date'].replace(tzinfo=None)
time_series = train_df['Date'].apply(lambda d: d.replace(tzinfo=None))
train_df['Date'] = time_series
train_df.rename(columns={'Date': 'ds', 'Value': 'y'}, inplace=True)
train_df

# %%
model_prophet = Prophet(seasonality_mode='additive')
model_prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model_prophet.fit(train_df)

# %%
df_future = model_prophet.make_future_dataframe(periods=365)
df_pred = model_prophet.predict(df_future)
model_prophet.plot(df_pred)
plt.tight_layout()
plt.show()

# %%
model_prophet.plot_components(df_pred)
