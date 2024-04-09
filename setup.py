import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

# Load data from CSV file
df = pd.read_csv('data.csv')

# Map sentiment labels to numerical values
sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
df['Sentiment'] = df['Sentiment'].map(sentiment_mapping)

# Tokenize sentences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['Sentence'])
vocab_size = len(tokenizer.word_index) + 1

# Convert sentences to sequences
sequences = tokenizer.texts_to_sequences(df['Sentence'])

# Pad sequences to make them of uniform length
max_len = max(len(seq) for seq in sequences)
sequences_padded = pad_sequences(sequences, maxlen=max_len, padding='post')

# Define model
model = Sequential([
    Embedding(vocab_size, 100),
    Conv1D(128, 5, activation='relu'),
    MaxPooling1D(5),
    Conv1D(128, 5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # Use softmax activation for multiclass classification
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Define callbacks to save the best model weights
checkpoint = ModelCheckpoint('sentiment_model.keras', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# Split dataset into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(sequences_padded, df['Sentiment'], test_size=0.2, random_state=42)

# Train model
history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val), callbacks=[checkpoint])

# Optionally, you can save the whole model instead of just the weights
model.save('sentiment_analysis_model.h5')


