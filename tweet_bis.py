#libraries needed
import pandas as pd
import snscrape.modules.twitter as sntwitter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
nltk.download('stopwords') #run once and comment it out to avoid it downloading multiple times
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import string
import re
import textblob
from textblob import TextBlob

from wordcloud import WordCloud, STOPWORDS
from emot.emo_unicode import UNICODE_EMOJI

porter = PorterStemmer()

lemmatizer = WordNetLemmatizer() 

from wordcloud import ImageColorGenerator
from PIL import Image

import warnings

# comment it out once you have gathered your data 
# to avoid running out of the wait time gather the data per month
query = "(Peter Obi OR Tinubu OR Atiku OR #PeterObiForPresident2023 OR #TinubuForPresident2023 OR #AtikuForPresident2023 OR #BAT2023 OR #Obi2023 OR #Atiku2023 OR #OBIdient OR #Atikulated OR Batified OR PDP OR APC OR LP OR LABOUR PARTY) since:2022-07-01 until:2022-03-30"
tweets = []
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
    if i>20000:
        break
    else:
        tweets.append([tweet.date, tweet.id, tweet.url, tweet.user.username, tweet.sourceLabel, tweet.user.location, tweet.content, tweet.likeCount, tweet.retweetCount])
df = pd.DataFrame(tweets, columns = ['Date', 'ID', 'url', 'username', 'source', 'location', 'tweet', 'num_of_likes', 'num_of_retweet'])

df =  pd.read_csv('sentiment.csv', encoding = 'unicode_escape')
df.head(2)

df.location.isna().sum()

df['location'] = df['location'].fillna('Unknown')

eng_stop_words = list(stopwords.words('english'))
emoji = list(UNICODE_EMOJI.keys())

# function for preprocessing tweet in preparation for sentiment analysis
def ProcessedTweets(text):
    #changing tweet text to small letters
    text = text.lower()
    # Removing @ and links 
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) |(\w+:\/\/\S+)", " ", text).split())
    # removing repeating characters
    text = re.sub(r'\@\w+|\#\w+|\d+', '', text)
    # removing punctuation and numbers
    punct = str.maketrans('', '', string.punctuation+string.digits)
    text = text.translate(punct)
    # tokenizing words and removing stop words from the tweet text
    tokens = word_tokenize(text)  
    filtered_words = [w for w in tokens if w not in eng_stop_words]
    filtered_words = [w for w in filtered_words if w not in emoji]
    # lemmetizing words
    lemmatizer = WordNetLemmatizer() 
    lemma_words = [lemmatizer.lemmatize(w) for w in filtered_words]
    text = " ".join(lemma_words)
    return text

# Generate a new column called 'Processed Tweets' by applying preprocessed tweets function to the 'Tweet' column.
df['Processed_Tweets'] = df['tweet'].apply(ProcessedTweets)

df.head(5)

#another data wrangling process
#replacing some words in the processed tweet
df['Processed_Tweets'] = df['tweet'].replace(to_replace ='parties', value = 'party')
df['Processed_Tweets'] = df['tweet'].replace(to_replace ='Labor party', value = 'Labour party')

#another data wrangling process
# extracting the parties: apc, pdp, labour party and the 3 top candidate name into new columns
def extract_apc(n):
  n = str(n)
  resp = n.lower().find("apc")
  if resp != -1:
    return "APC"
  else:
    return None

def extract_pdp(n):
  n = str(n)
  resp = n.lower().find("pdp")
  if resp != -1:
    return "PDP"
  else:
    return None

def extract_labour(n):
  n = str(n)
  resp = n.lower().find("labour party")
  if resp != -1:
    return "Labour Party"
  else:
    return None

def extract_obi(n):
  n = str(n)
  resp = n.lower().find("obi")
  if resp != -1:
    return "Peter Obi"
  else:
    return None

def extract_atiku(n):
  n = str(n)
  resp = n.lower().find("atiku")
  if resp != -1:
    return "Atiku"
  else:
    return None

def extract_tinubu(n):
  n = str(n)
  resp = n.lower().find("tinubu")
  if resp != -1:
    return "Tinubu"
  else:
    return None
  
#applying the function
df['Obi'] = df['tweet'].apply(extract_obi) 
df['labour'] = df['tweet'].apply(extract_labour) 
df['pdp'] = df['tweet'].apply(extract_pdp) 
df['apc'] = df['tweet'].apply(extract_apc) 
df['atiku'] = df['tweet'].apply(extract_atiku) 
df['tinubu'] = df['tweet'].apply(extract_tinubu)


#another data wrangling process
# replacing empty values with Empty
df['Obi'] = df['Obi'].fillna('Empty')
df['labour'] = df['labour'].fillna('Empty')
df['pdp'] = df['pdp'].fillna('Empty')
df['apc'] = df['apc'].fillna('Empty')
df['atiku'] = df['atiku'].fillna('Empty')
df['tinubu'] = df['tinubu'].fillna('Empty')

#another data wrangling process
# convert the tweet text into a string separate with " "
tweets_string = df['Processed_Tweets'].tolist()
tweets_string = " ".join(tweets_string)


# Function for polarity score
def polarity(tweet):
    return TextBlob(tweet).sentiment.polarity

# Function to get sentiment type
#setting the conditions
def sentimenttextblob(polarity):
    if polarity < 0:
        return "Negative"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Positive"  
    
# using the functions to get the polarity and sentiment
df['Polarity'] = df['Processed_Tweets'].apply(polarity)
df['Sentiment'] = df['Polarity'].apply(sentimenttextblob)
sent = df['Sentiment'].value_counts()
sent


# plotting bar chart for the Sentiment
plt.subplot(1,2,1)
sent.plot(kind='bar', color=['green'], figsize=(15,5))
plt.title('Sentiment percieved for the forthcoming election', fontsize=16)
plt.xlabel('Types of sentiment')
plt.ylabel('Number of sentiment');

df.head(5)

# Displaying the most talked about word in a word cloud 
# some stop words were still evident but was removed during visualization on Power BI
# Instantiate the Twitter word cloud object
w_cloud = WordCloud(collocations = False,max_words=200, background_color = 'white', width = 7000, height = 5000).generate(tweets_string)

# Display the generated Word Cloud
plt.imshow(w_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()

# masking the word cloud into Nigeria's flag colour

#Create the mask
colosseum_mask = np.array(Image.open('/Users/fresh/Documents/Overall python/Twitter project/Flag-Nigeria.jpg'))

#Grab the mask colors
colors = ImageColorGenerator(colosseum_mask)

#Instantiate the wordcloud using color_func argument
cloud = WordCloud(mask=colosseum_mask,
                  background_color='white',
                  color_func=colors).generate_from_text(tweets_string)

#Plot the wordcloud
plt.figure(figsize=(10,8))
plt.imshow(cloud)
plt.axis('off')
plt.title('title')
plt.show()