import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pickle
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class BestModelToUse:
    def init(self):
        self.sentiment = ""
    
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

    def pickleModel(self, userInput):
        # Load the tokenizer and model from file
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)

        # Load the trained model
        loaded_model = load_model('sentiment_analysis_model.h5', compile=False)

        # Preprocess the input text
        input_sequence = tokenizer.texts_to_sequences([userInput])
        max_length = loaded_model.layers[0].input_length
        input_data = pad_sequences(input_sequence, maxlen=max_length, padding='post')

        # Make a prediction using the loaded model
        score = loaded_model.predict(input_data)[0][0]
        if score >= 0.6:
            self.sentiment = "Very Good"
        elif score >= 0.2:
            self.sentiment = "Good"
        elif score > -0.2:
            self.sentiment = "Neutral"
        elif score > -0.6:
            self.sentiment = "Bad"
        else:
            self.sentiment = "Very Bad"
        
        return self.sentiment
