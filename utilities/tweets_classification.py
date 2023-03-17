import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

tweets = pd.read_csv('cleaned_tweets.csv')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Compute the sentiment scores for each tweet
tweets['sentiment_scores'] = tweets['cleaned_tweet'].astype(str).apply(lambda x: sia.polarity_scores(x))
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

tweets['sentiment_category'] = tweets['sentiment_scores'].apply(lambda x: map_sentiment_score_to_category(x['compound']))

# Print the results
#print(tweets[['full_text', 'sentiment_category']])

tweets.to_csv('classified_tweets.csv')