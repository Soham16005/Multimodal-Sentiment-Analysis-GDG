import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
# Streamlit title
st.title("Fine-Tuning BERT for IMDB Sentiment Classification")

# Specify the model path (replace with your actual path if necessary)
model_path = "D:/bert-base-uncased-sentiment-model"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Create the pipeline with both the model and tokenizer
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)

# User input for text
text = st.text_area("Enter Your Tweet Here")

# Predict button
if st.button("Predict"):
    if text.strip():  # Check if the input is not empty
        result = classifier(text)
        st.write("Prediction Result:", result)
    else:
        st.write("Please enter a valid tweet!")
