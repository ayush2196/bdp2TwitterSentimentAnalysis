import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import json

# Open the file and read each line as a separate JSON object
tweets = []
with open('raw_tweets.json', 'r') as f:
    for line in f:
        try:
            tweet_data = json.loads(line)
            tweets.append(tweet_data)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON object: {e}")

# Print the number of tweets
print(f"Number of tweets: {len(tweets)}")

# Extract the relevant columns from the DataFrame
df = pd.DataFrame(tweets, columns=['id', 'created_at', 'full_text'])

# dropping duplicates
df = df.drop_duplicates(subset='full_text')

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

df.to_csv('cleaned_tweets.csv',index=False)