import pandas as pd
import os
import yfinance as yf
import datetime
from datetime import date, timedelta
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplcursors


#VARIABLES
#time_period = 300
#predictions_time = 20
#ticker = "PEPE24478-USD"
#pru = 0.000023
#crypto = "Y"

def pred_sarimax(time_period, predictions_time, ticker, pru, crypto):
   tick_space = round(time_period/20)
   today = date.today()

   d1 = today.strftime("%Y-%m-%d")
   end_date = d1
   d2 = date.today() - timedelta(days=time_period)
   d2 = d2.strftime("%Y-%m-%d")
   start_date = d2

   data = yf.download(ticker, 
                        start=start_date, 
                        end=end_date, 
                        progress=False)
   data["Date"] = data.index
   data = data[["Date", "Open", "High", "Low", "Close", "Volume"]]
   data.reset_index(drop=True, inplace=True)
   print(data.tail())

   if crypto == "Y":
      currency = "CRYPTO"
      frequency = "D"
   else : 
      currency = yf.Ticker(f"{ticker}").info['financialCurrency']
      frequency = "B"
   long_name = yf.Ticker(f"{ticker}").info['longName']
   #DAYS
   now = date.today() - timedelta(days=time_period)
   now = now.strftime("%Y-%m-%d").split("-")
   noww= datetime.datetime(int(now[0]), int(now[1]), int(now[2]), 0, 0)
   then = datetime.datetime.now() + timedelta(days=predictions_time)
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
   print(days)
   #VARIABLESDEUX
   print(len(days), days.searchsorted(data.iloc[-1]["Date"]))
   cut_train = days.searchsorted(data.iloc[-1]["Date"]) +1
   cut_train = int(cut_train[0])
   prediction_model = len(days)-cut_train
   print(prediction_model, cut_train, len(days), len(days[cut_train:]), len(days[:cut_train]), len(data["Close"]))


   #MODEL

   p, d, q = 3, 1, 5


   import statsmodels.api as sm
   from statsmodels.tsa.arima_model import ARIMA
   import warnings
   model=sm.tsa.statespace.SARIMAX(data['Close'],
                                 order=(p, d, q),
                                 seasonal_order=(p, d, q, 12))
   model=model.fit()
   print(model.summary())

   predictions = model.predict(len(data), len(data)+prediction_model-1)
   ress = ""
   #PREDICTIONS VALUES
   if pru == None:
      ress = f"Min : {min(predictions)}"
      print(f"Buying price minimum for {ress}")
   else:
      ress = f"Max : {max(predictions)}"
      print(f"Selling price maximum for {ress}")

   #PLOT
   plt.style.use('fivethirtyeight')
   plt.figure(figsize=(16,8))

   plt.plot(days[:cut_train], data["Close"], color="blue", label="Training Data", marker="o", linewidth=2)
   plt.plot(days[cut_train:], predictions, color="red", label="Predictions", marker="o", linewidth=2)
   # Format the date into months & days
   plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%Y')) 

   # Change the tick interval
   plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=tick_space)) 

   plt.gca().xaxis.set_tick_params(rotation = 30)
   plt.title(f"{long_name} Stock Price Predictions")
   plt.ylabel(f"Closing Price ({currency})")
   plt.legend()
   plt.tight_layout()

   def create_mplcursor_for_points_on_line(lines, ax=None, annotation_func=None, **kwargs):
    ax = ax or plt.gca()
    scats = [ax.scatter(x=line.get_xdata(), y=line.get_ydata(), color='none') for line in lines]
    cursor = mplcursors.cursor(scats, **kwargs)
    annotation_func = lambda sel: sel.annotation.set_text(f"Date : {(pd.to_timedelta(sel.target[0], unit='D') + pd.Timestamp('1970-1-1')).strftime('%d-%m-%Y')}\nPrice : {round(sel.target[1],2)}")
    if annotation_func is not None:
        cursor.connect('add', annotation_func)
    return cursor
   
   create_mplcursor_for_points_on_line(plt.gca().get_lines(),hover=True)
   plt.show()


   #plt.savefig(f"{path}\{ticker}_predictions_{time_period}_{predictions_time}.pdata["Close"]", bbox_inches='tight')
   
   return plt