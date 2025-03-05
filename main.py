import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}
MAX_VOCAB_SIZE = 10000  # Ensure indices stay within valid range

# Load the pre-trained model
model = load_model('rnn_imdb.h5')

# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words if word_index.get(word, 2) + 3 < MAX_VOCAB_SIZE]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app
st.set_page_config(page_title='IMDB Sentiment Analyzer', page_icon='🎬', layout='centered')
st.markdown(
    """
    <style>
    .main {background-color: #f0f2f6; padding: 20px; border-radius: 10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('🔍 Enter a movie review below to classify it as **positive** or **negative**!')

# User input
user_input = st.text_area('✍️ Movie Review:', height=150, placeholder='Type your review here...')

if st.button('🚀 Analyze Sentiment'):
    with st.spinner('Analyzing review... Please wait!'):
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        score = prediction[0][0]
        
        if score > 0.75:
            sentiment = 'Highly Positive 😊'
        elif score > 0.5:
            sentiment = 'Slightly Positive 🙂'
        elif score > 0.25:
            sentiment = 'Slightly Negative 😕'
        else:
            sentiment = 'Highly Negative 😞'
        
        st.success(f'Sentiment: **{sentiment}**')
        st.write(f'🧠 **Confidence Score:** {score:.2f}')
        
        # Show decoded review
        decoded = decode_review(preprocessed_input[0])
        with st.expander("🔍 View Processed Review"):
            st.write(decoded)
else:
    st.info('💡 Enter a movie review and click "Analyze Sentiment" to get the result!')
