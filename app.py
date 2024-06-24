import os
import streamlit as st
import numpy as np
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Check if the model file exists
model_path = 'sentiment_analysis_model.h5'
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please ensure the file is uploaded and the path is correct.")
else:
    # Load the model
    model = load_model(model_path)

    # Define a tokenizer and fit it on your training data
    tokenizer = Tokenizer(num_words=10000)
    # Tokenizer fitting should be done with the same data it was originally trained on

    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'<br />', ' ', text)
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text

    st.title('Sentiment Analysis')
    st.write('Enter a movie review below to predict its sentiment.')

    review = st.text_area('Review:')
    if st.button('Predict'):
        review = preprocess_text(review)
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequence, maxlen=200)
        prediction = model.predict(padded_sequence)
        sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
        st.write(f'Sentiment: {sentiment}')
