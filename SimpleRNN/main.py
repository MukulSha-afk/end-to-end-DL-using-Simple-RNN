import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model


word_index = imdb.get_word_index()
reverse_word_index = {value:key for key ,value in word_index.items()}

model = load_model('simple_rnn_imdb.h5')

word_index = imdb.get_word_index()

def preprocess_text(text):
    words = text.lower().split()

    encoded_review = [
        word_index.get(word, 2) + 3 for word in words
    ]

    padded_review = pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review


# step 3 predction function

def predict_sentiment(review):
    preprocess_input = preprocess_text(review)

    prediction = model.predict(preprocess_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]

# streamlit app 

import streamlit as st
st.title('IMDB Moive review Sentiment Analysis')
st.write('Enter a moive review to classify it as a positive or negative .')

user_input = st.text_area('Moive_review')

if st.button('Classify'):
    preprocess_input = preprocess_text(user_input)
    
    ##Make prediction 
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0] >0.5 else 'Negative'
    
    # display the result 
    st.write(f'Sentiment:{sentiment}')
    st.write(f'Prediction Score:{prediction[0][0]}')
    
else:
    st.write('Please enter the moive name.')
    
    