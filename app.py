import streamlit as st
import numpy as np
import pandas as pd
from sklearn_crfsuite import CRF  # Replace this with your trained CRF model
from tensorflow.keras.models import load_model  # For BiLSTM model (if you use this)
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy

# Assuming you have a pre-trained CRF model or BiLSTM model
# Load the pre-trained model (use your trained models here)
# Example for CRF (replace with your trained model)
crf_model = CRF()  # Load your actual CRF model here (this is just a placeholder)

# Load your spaCy model for NER if you want to use it as an alternative
nlp = spacy.load("en_core_web_sm")

def process_text_for_pos_tagging(text):
    # Tokenize and preprocess the text for POS tagging
    tokens = text.split()
    features = [word2features(tokens, i) for i in range(len(tokens))]
    return features

def word2features(tokens, i):
    word = tokens[i]
    features = {
        'word': word,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],  # Last 3 characters of the word
        'word[:3]': word[:3],    # First 3 characters of the word
        'is_capitalized': word[0].upper() == word[0],
    }
    if i > 0:
        features['prev_word'] = tokens[i - 1]
    if i < len(tokens) - 1:
        features['next_word'] = tokens[i + 1]
    return features

def predict_pos_tagging(text):
    features = process_text_for_pos_tagging(text)
    # Replace with your trained CRF or BiLSTM model prediction
    predicted_tags = crf_model.predict([features])[0]  # This is a placeholder
    return predicted_tags

def predict_named_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

def display_result(text, task):
    if task == 'POS Tagging':
        pos_tags = predict_pos_tagging(text)
        st.write("POS Tagging Results:")
        st.write(pd.DataFrame(list(zip(text.split(), pos_tags)), columns=["Word", "POS Tag"]))
    elif task == 'NER':
        entities = predict_named_entities(text)
        st.write("Named Entity Recognition Results:")
        if entities:
            st.write(pd.DataFrame(entities, columns=["Entity", "Label"]))
        else:
            st.write("No named entities found.")

# Streamlit Interface
st.title('POS Tagging and Named Entity Recognition (NER)')

# Text Input
input_text = st.text_area("Enter your text here:")

# Task Selection
task = st.radio("Select a task:", ("POS Tagging", "NER"))

# Process Input and Display Results
if st.button("Analyze"):
    if input_text:
        display_result(input_text, task)
    else:
        st.warning("Please enter some text to analyze.")
