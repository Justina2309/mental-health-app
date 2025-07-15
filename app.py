import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np

# Load model and encoder
model = joblib.load("mental_health_model.pkl")
bert = SentenceTransformer("bert_encoder")

# UI
st.set_page_config(page_title="Mental Health Detector", layout="centered")
st.title("🧠 Mental Health Detection from Social Media")
st.markdown("Enter a social media post to detect if it indicates signs of depression.")

# Text input
user_input = st.text_area("Enter a post or comment:", height=150)

# Predict
if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        embedding = bert.encode([user_input])
        prediction = model.predict(embedding)[0]

        if prediction == 1:
            st.error("⚠️ This post may indicate signs of depression.")
        else:
            st.success("✅ This post does not indicate signs of depression.")
