#**************** SENTIMENT ANALYSIS **************************
import re
import pandas as pd
import preprocessor as p
from textblob import TextBlob
import matplotlib.pyplot as plt
from Tweet_class import Tweet
import pandas as pd
import yfinance as yf
from ntscraper import Nitter



def retrieving_tweets_polarity(symbol, num_of_tweets):
    symbol = yf.Ticker(f"{symbol}").info['longName']
    print(symbol)
    # Creating list to append tweet data to
    scraper = Nitter(log_level=0)
    tweets = scraper.get_tweets(f"{symbol}", mode="term",number=num_of_tweets, max_retries=3)

    
    print(len(tweets["tweets"]))
    tweet_list = [] #List of tweets alongside polarity
    global_polarity = 0 #Polarity of all tweets === Sum of polarities of individual tweets
    tw_list=[] #List of tweets only => to be displayed on web page
    #Count Positive, Negative to plot pie chart
    pos=0 #Num of pos tweets
    neg=0 #Num of negative tweets

    for tweet in tweets["tweets"]:
        count=20 #Num of tweets to be displayed on web page
        #Convert to Textblob format for assigning polarity
        tw2 = tweet["text"]
        tw = tweet["text"]
        #Clean
        tw=p.clean(tw)
        #print("-------------------------------CLEANED TWEET-----------------------------")
        #print(tw)
        #Replace &amp; by &
        tw=re.sub('&amp;','&',tw)
        #Remove :
        tw=re.sub(':','',tw)
        #print("-------------------------------TWEET AFTER REGEX MATCHING-----------------------------")
        #print(tw)
        #Remove Emojis and Hindi Characters
        tw=tw.encode('ascii', 'ignore').decode('ascii')

        #print("-------------------------------TWEET AFTER REMOVING NON ASCII CHARS-----------------------------")
        #print(tw)
        blob = TextBlob(tw)
        polarity = 0 #Polarity of single individual tweet
        for sentence in blob.sentences:
                
            polarity += sentence.sentiment.polarity
            if polarity>0:
                pos=pos+1
            if polarity<0:
                neg=neg+1
            
            global_polarity += sentence.sentiment.polarity
        if count > 0:
            tw_list.append(tw2)
            
        tweet_list.append(Tweet(tw, polarity))
        count=count-1
    if len(tweet_list) != 0:
        global_polarity = global_polarity / len(tweet_list)
    else:
        global_polarity = global_polarity
    print(pos, neg, num_of_tweets-pos-neg)
    neutral=num_of_tweets-pos-neg

    print()
    print("##############################################################################")
    print("Positive Tweets :",pos,"Negative Tweets :",neg,"Neutral Tweets :",neutral)
    print("##############################################################################")
    labels=['Positive','Negative','Neutral']
    sizes = [pos,neg,neutral]
    explode = (0, 0, 0)
    fig1, ax1 = plt.subplots(figsize=(7.2,4.8),dpi=65)

    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{:.1f}%\n({v:d})'.format(pct, v=val)
        return my_format

    ax1.pie(sizes, explode=explode, labels=labels, autopct=autopct_format(sizes), startangle=90)
    # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.axis('equal')  
    plt.tight_layout()
    plt.title(f"{symbol} Sentiment")
    if global_polarity>0:
        print()
        print("##############################################################################")
        print("Tweets Polarity: Overall Positive")
        print("##############################################################################")
        tw_pol="Overall Positive"
    else:
        print()
        print("##############################################################################")
        print("Tweets Polarity: Overall Negative")
        print("##############################################################################")
        tw_pol="Overall Negative"
    return plt

retrieving_tweets_polarity("TSLA", 10)