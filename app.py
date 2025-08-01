# app.py

import streamlit as st
from transformers import pipeline

# Load zero-shot classification pipeline using BART
@st.cache_resource
def load_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

classifier = load_model()

st.title("📰 Fake News Detector using BART")
st.write("Enter a news headline or sentence below to check if it might be fake or true.")

# User input
news_input = st.text_area("✍️ Enter a news statement", height=150)

# Hypothesis labels for classification
candidate_labels = ["real", "fake"]

# Check button
if st.button("Analyze"):
    if news_input.strip() == "":
        st.warning("Please enter a news statement.")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(news_input, candidate_labels)
            label = result["labels"][0]
            score = result["scores"][0]

        st.markdown("### 🧠 Prediction")
        st.success(f"**Prediction**: {label.upper()}  \n**Confidence**: {score:.2%}")
        st.write("### 🔍 Full Result")
        st.json(result)
