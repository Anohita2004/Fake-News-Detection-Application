import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model = tf.keras.models.load_model("lstm_fake_news_model.h5")

# Load the tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Set max_len same as in training
MAX_LEN = 500  # change if your notebook used a different value

# App title
st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout="centered")
st.title("📰 Fake News Classifier")
st.markdown("Enter a news article or statement below, and I'll predict whether it's **Real** or **Fake**.")

# Text input
news_text = st.text_area("Enter news text here", height=200)

if st.button("Classify"):
    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        seq = tokenizer.texts_to_sequences([news_text])
        padded = pad_sequences(seq, maxlen=MAX_LEN)

        # Predict
        pred = model.predict(padded)[0][0]

        if pred >= 0.5:
            st.error(f"❌ Fake News (Confidence: {pred:.2f})")
        else:
            st.success(f"✅ Real News (Confidence: {1 - pred:.2f})")
