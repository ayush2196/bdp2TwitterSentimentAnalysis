#importing required libraries

import tweepy
import json
import time
from dotenv import load_dotenv
import boto3
import os
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import pickle

# Load the environment variables from .env file
load_dotenv()

# Get the Bearer token value from the environment variable
bearer_token = os.getenv('BEARER_TOKEN')

# Authenticate with Twitter API using API keys, API secret key, and Bearer token
auth = tweepy.OAuth2BearerHandler(bearer_token)
api = tweepy.API(auth)

# Define the search query
query = 'covid'

# Define the fields to include in the response
fields = ['created_at', 'id', 'full_text', 'user']

# Extract data using search API
tweets = []
start_time = time.time()
seconds = 1800
while True:
    elapsed_time = time.time() - start_time
    if elapsed_time >= seconds:
        break
    try:
        for status in tweepy.Cursor(api.search_tweets,
                                    q=query,
                                    tweet_mode='extended',
                                    lang='en',
                                    result_type='recent',
                                    count=100).items():
             # Create a new dictionary with only the desired fields
            tweet_data = {field: status._json[field] for field in fields}
            tweets.append(tweet_data)
            #print(elapsed_time)
            #time.sleep(60) # Wait for 60 seconds and try again
    except Exception as e:
        print(f"Error occurred: {e}")
        time.sleep(30) # Wait for 60 seconds and try again

# Save the extracted data as JSON
with open('tweets.json', 'w') as f:
    json.dump(tweets, f)

# Print confirmation message
print(f"Saved {len(tweets)} tweets to tweets.json")

# assuming `tweets` is the list of dictionaries
df = pd.DataFrame(tweets, columns=['id', 'created_at', 'full_text'])

# Define a function to clean tweets
def clean_tweet(tweet):
    # Remove Twitter-specific entities
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)

    # Remove punctuation and special characters
    tweet = re.sub(r'[^\w\s]', '', tweet)

    # Tokenize the tweet
    tokens = word_tokenize(tweet)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in tokens if not token.lower() in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    # Convert tokens to lowercase and join them into a string
    cleaned_tweet = " ".join([token.lower() for token in lemmatized_tokens])

    return cleaned_tweet


# Clean the tweets
df['cleaned_tweet'] = df['full_text'].apply(clean_tweet)

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Compute the sentiment scores for each tweet
df['sentiment_scores'] = df['cleaned_tweet'].apply(lambda x: sia.polarity_scores(x))

# Map the sentiment scores to categories
def map_sentiment_score_to_category(sentiment_score):
    if sentiment_score >= 0.6:
        return 'very good'
    elif sentiment_score >= 0.2:
        return 'good'
    elif sentiment_score > -0.2:
        return 'neutral'
    elif sentiment_score > -0.6:
        return 'bad'
    else:
        return 'very bad'

df['sentiment_category'] = df['sentiment_scores'].apply(lambda x: map_sentiment_score_to_category(x['compound']))