import pandas as pd
import os
import yfinance as yf
import datetime
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import matplotlib.colors as mcolors
import mplcursors
import math
import mpld3




#VARIABLES
#crypto = input("Crypto ? : [Y/N]")
#time_period = int(input("Time period : "))
#ticker = input("Ticker : ")



def tech_analysis(time_period, ticker, crypto):
   long_name = yf.Ticker(f"{ticker}").info['longName']
   #DIR OF SAVE
   if crypto == "Y":
      currency = "CRYPTO"
      frequency = "D"
   else : 
      currency = yf.Ticker(f"{ticker}").info['financialCurrency']
      frequency = "B"
   path = f"C:\Simsim\!!!!TRADE\CODE\RESULTS PREDICTIONS\{ticker}"
   # Check whether the specified path exists or not
   isExist = os.path.exists(path)
   if not isExist:

      # Create a new directory because it does not exist
      os.makedirs(path)
      print("The new directory is created!")

   today = date.today()

   d1 = today.strftime("%Y-%m-%d")
   end_date = d1
   d2 = date.today() - timedelta(days=time_period)
   d2 = d2.strftime("%Y-%m-%d")
   start_date = d2

   stock_df = yf.download(ticker, 
                        start=start_date, 
                        end=end_date, 
                        progress=False)
   stock_df["Date"] = stock_df.index
   stock_df = stock_df[["Date", "Open", "High", "Low", "Close", "Volume"]]
   stock_df.reset_index(drop=True, inplace=True)

   def get_tech_ind(data):
      data['MA10'] = data.iloc[:,4].rolling(window=10).mean() #Close column
      data['MA20'] = data.iloc[:,4].rolling(window=20).mean() #Close column
      data['MA50'] = data.iloc[:,4].rolling(window=50).mean() #Close Column
      data['MA100'] = data.iloc[:,4].rolling(window=100).mean() #Close Column

      data['MACD'] = data.iloc[:,4].ewm(span=26).mean() - data.iloc[:,1].ewm(span=12,adjust=False).mean()
      #This is the difference of Closing price and Opening Price

      # Create Bollinger Bands
      data['20SD'] = data.iloc[:, 4].rolling(20).std()
      data['upper_band'] = data['MA20'] + (data['20SD'] * 2)
      data['lower_band'] = data['MA20'] - (data['20SD'] * 2)

      # Create Exponential moving average
      data['EMA'] = data.iloc[:,4].ewm(com=0.5).mean()

      # Create LogMomentum
      data['logmomentum'] = np.log(data.iloc[:,4] - 1)

      return data
   
   def get_days(data):
      #DAYS
      now = date.today() - timedelta(days=time_period)
      now = now.strftime("%Y-%m-%d").split("-")
      noww= datetime.datetime(int(now[0]), int(now[1]), int(now[2]), 0, 0)
      then = datetime.datetime.now()
      then = then.strftime("%Y-%m-%d").split("-")
      thenn= datetime.datetime(int(then[0]), int(then[1]), int(then[2]), 0, 0)
      pd.set_option('display.max_rows', None)
      print(data)
      print(len(data), data.iloc[-1]["Date"])
      days = pd.date_range(start=noww, end=thenn, freq=frequency)
      leap = []
      for each in days:
         if (each not in data["Date"].to_list()) & (each < data.iloc[-1]["Date"].iloc[0]):
            leap.append(each)
      days = days.drop(leap)
      days = days[:-1]
      return days


   days_timeline = get_days(stock_df)

   tech_df = get_tech_ind(stock_df)
   dataset = tech_df.iloc[20:,:].reset_index(drop=True)
   dataset.head()

   def tech_ind(dataset):
      plt.figure(figsize=(15, 8))

      plt.plot(days_timeline, dataset['Close'], label='Closing Price', color='#000', linewidth=0.7)
      plt.plot(days_timeline, dataset['upper_band'], color=[0.8,0.7,0.2], linestyle='-', linewidth=0.2)
      plt.plot(days_timeline, dataset['lower_band'], color=[0.8,0.7,0.2], linestyle='-', linewidth=0.2)
      plt.fill_between(days_timeline, dataset['lower_band'], dataset['upper_band'], label="Bandes de Bollinger", alpha=0.35, color=[0.8,0.7,0.2])
      plt.plot(days_timeline, dataset['MA10'], label='Moving Average (10 days)', color='b', linestyle='-', linewidth=1.5)
      plt.plot(days_timeline, dataset['MA50'], label='Moving Average (50 days)', color='g', linestyle='-', linewidth=1.5)
      plt.plot(days_timeline, dataset['MA100'], label='Moving Average (100 days)', color='r', linestyle='-', linewidth=1.5)

      plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%d-%m-%Y"))
      plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=round(time_period/20))) 
      plt.gca().xaxis.set_tick_params(rotation = 30)

      plt.title(f"{long_name} Stock Price Predictions")
      plt.ylabel(f"Closing Price ({currency})")
      plt.legend()
      plt.tight_layout()

      def annotation_data(sel):
         date = (pd.to_timedelta(sel.target[0], unit='D') + pd.Timestamp('1970-1-1')).strftime('%Y-%m-%d')
         price = round(dataset.loc[dataset["Date"] == date]["Close"].values[0],2)
         ma10 = round(dataset.loc[dataset["Date"] == date]["MA10"].values[0],2)
         ma50 = round(dataset.loc[dataset["Date"] == date]["MA50"].values[0],2)
         ma100 = round(dataset.loc[dataset["Date"] == date]["MA100"].values[0],2)
         lower_band = round(dataset.loc[dataset["Date"] == date]["lower_band"].values[0],2)
         upper_band = round(dataset.loc[dataset["Date"] == date]["upper_band"].values[0],2)

         bollinger = f'{lower_band}-{upper_band}'

         text = f"Date : {datetime.datetime.strptime(date, '%Y-%m-%d').strftime('%d-%m-%Y')}\nPrice : {price}"

         if(math.isnan(ma10) == False):
            text += f"\nMA 10 Days : {ma10}"
         if(math.isnan(ma50) == False):
            text += f"\nMA 50 Days : {ma50}"
         if(math.isnan(ma100) == False):
            text += f"\nMA 100 Days : {ma100}"
         if(math.isnan(lower_band) == False and math.isnan(upper_band) == False):
            text += f"\nBollinger Bands : {bollinger}"

                  

         return sel.annotation.set_text(text)


      def create_mplcursor_for_points_on_line(lines, ax=None, annotation_func=None, **kwargs):
         ax = ax or plt.gca()
         scats = ax.scatter(x=lines[0].get_xdata(), y=lines[0].get_ydata(), color='none')
         cursor = mplcursors.cursor(scats, **kwargs)
         annotation_func = lambda sel: annotation_data(sel)
         if annotation_func is not None:
            cursor.connect('add', annotation_func)
         return cursor
         
      create_mplcursor_for_points_on_line(plt.gca().get_lines(),hover=True)

   tech_ind(tech_df)
   return plt
