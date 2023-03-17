import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from keras.utils import to_categorical
from imblearn.under_sampling import RandomUnderSampler
import pickle

# Load the CSV file
df = pd.read_csv('classified_tweets.csv')

# Tokenize the tweets
tokenizer = Tokenizer()
df['cleaned_tweet'] = df['cleaned_tweet'].astype(str)
tokenizer.fit_on_texts(df['cleaned_tweet'])
word_index = tokenizer.word_index

# Convert the tweets to sequences
sequences = tokenizer.texts_to_sequences(df['cleaned_tweet'])

# Pad the sequences
max_length = max(len(sequence) for sequence in sequences)
data = pad_sequences(sequences, maxlen=max_length, padding='post')

# Convert labels to categorical
test_categories = ['very good', 'good', 'neutral', 'bad', 'very bad']
labels = np.array(df['sentiment_category'].apply(lambda x: test_categories.index(x)))
labels = to_categorical(labels)

# Define the undersampler
undersampler = RandomUnderSampler()

# Undersample the data
data_resampled, labels_resampled = undersampler.fit_resample(data, labels)

# Define the model
model = Sequential()
model.add(Embedding(len(word_index) + 1, 128, input_length=max_length))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])



# Define a checkpoint callback
checkpoint_callback = ModelCheckpoint('best_model.h5', 
                                       monitor='val_accuracy', 
                                       mode='max', 
                                       save_best_only=True)

# Train the model with different hyperparameters
history = model.fit(data_resampled, labels_resampled, validation_split=0.3, epochs=5, batch_size=128, callbacks=[checkpoint_callback])

# Test the model
test_text = ['I really love this product', 'I hate this product', 'This product is just okay']
test_sequences = tokenizer.texts_to_sequences(test_text)
test_data = pad_sequences(test_sequences, maxlen=max_length, padding='post')
test_scores = model.predict(test_data)
print(test_scores)
test_labels = ['very good', 'good', 'neutral', 'bad', 'very bad']
test_categories = []
for score in test_scores:
    test_categories.append(test_labels[np.argmax(score)])
print(test_categories)

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the model
model.save('sentiment_analysis_model.h5')