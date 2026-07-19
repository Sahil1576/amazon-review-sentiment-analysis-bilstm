import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import gdown

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Amazon Review Prediction",
    page_icon="🛒",
    layout="centered"
)

# -----------------------------
# Load Model & Tokenizer
# -----------------------------
MODEL_PATH = "amazon_reviews_prediction_model.keras"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        "https://drive.google.com/file/d/1jL6MjSEfMSwq7wmbRy7qGZ-sF060qlKq/view?usp=sharing",
        MODEL_PATH,
        quiet=False
    )
@st.cache_resource
def load_files():
    model = load_model(MODEL_PATH)

    with open("tokenizer.pkl", "rb") as file:
        tokenizer = pickle.load(file)

    return model, tokenizer

model, tokenizer = load_files()

MAX_LEN = 224

# -----------------------------
# Custom CSS
# -----------------------------
st.markdown("""
<style>

body{
    background-color:#f5f5f5;
}

.title{
    text-align:center;
    color:#ff9900;
    font-size:40px;
    font-weight:bold;
}

.subtitle{
    text-align:center;
    font-size:18px;
    color:gray;
}

.stButton>button{
    width:100%;
    background-color:#ff9900;
    color:black;
    font-size:18px;
    font-weight:bold;
    border-radius:8px;
    height:50px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# Title
# -----------------------------
st.markdown('<p class="title">🛒 Amazon Review Prediction</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Predict whether a review is Positive, Negative or Neutral.</p>',
    unsafe_allow_html=True
)

st.write("")

# -----------------------------
# Review Input
# -----------------------------
review = st.text_area(
    "✍️ Enter Amazon Review",
    height=180,
    placeholder="Example: This product is amazing and worth every penny..."
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("⚠️ Please enter a review.")
    else:

        sequence = tokenizer.texts_to_sequences([review])

        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(padded, verbose=0)

        predicted_class = np.argmax(prediction)

        confidence = prediction[0][predicted_class] * 100

        labels = {
            0: "😊 Positive",
            1: "😔 Negative",
            2: "😐 Neutral"
        }

        st.success(f"### Prediction : {labels[predicted_class]}")
        st.info(f"Confidence : **{confidence:.2f}%**")

        st.write("---")

        st.subheader("Prediction Probabilities")

        st.write(f"😊 Positive : **{prediction[0][0]*100:.2f}%**")
        st.progress(float(prediction[0][0]))

        st.write(f"😔 Negative : **{prediction[0][1]*100:.2f}%**")
        st.progress(float(prediction[0][1]))

        st.write(f"😐 Neutral : **{prediction[0][2]*100:.2f}%**")
        st.progress(float(prediction[0][2]))
