from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
from PRED import pred_sarimax
from Technical import tech_analysis
from tweet import retrieving_tweets_polarity

ticker = input("Enter the ticker of the stock you want to analyze: ")
time_period = int(input("Enter the time period you want to analyze: "))
predictions_time = int(input("Enter the time period you want to predict: "))
pru = float(input("Enter the PRU: "))
num_of_tweets = int(input("Enter the number of tweets you want to analyze: "))
crypto = input("Crypto ? : [Y/N] ")


if ticker == "":
    print(ticker)
else:
    plt.ion()
    #sentiment = retrieving_tweets_polarity(ticker,num_of_tweets)
    predi = pred_sarimax(time_period, predictions_time, ticker, pru, crypto)
    tech = tech_analysis(time_period, ticker, crypto)
    print(tech, plt.get_fignums())
    path = f"C:\Simsim\TRADING\CODE\RESULTS PREDICTIONS\{ticker}"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    print(isExist)
    if not isExist:

        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

    # Instantiating PDF document
    pdf = PdfPages(os.path.join(path, f"{ticker}_predictions_{time_period}_{predictions_time}_{num_of_tweets}.pdf"))
    for i in plt.get_fignums():
        t= plt.figure(i)
        pdf.savefig(t, bbox_inches='tight')
        print(f"Figure {i} saved as pdf")
    pdf.close()
    print("Predictions saved as pdf")