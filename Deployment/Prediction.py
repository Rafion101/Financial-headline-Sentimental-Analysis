import streamlit as st
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
import re

from nltk.tokenize import word_tokenize
import tqdm
import unicodedata
import contractions
from bs4 import BeautifulSoup


load_model = tf.keras.models.load_model('model_func.h5')




def user_input():

        # Make a function using beutifulsoup to parse and strip white space and enters
    def strip_html_tags(text):
      soup = BeautifulSoup(text, "html.parser")
      [s.extract() for s in soup(['iframe', 'script'])]
      stripped_text = soup.get_text()
      stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
      return stripped_text
    
    #  remove accented characters from text strings
    def remove_accented_chars(text):
      text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
      return text
    
    # Using stopword liblary to remove stop words like the and or ect.
    def stopwords_removal(words):
        list_stopwords = nltk.corpus.stopwords.words('english')
        return [word for word in words if word not in list_stopwords]
    
    # Making a fucntion to run previouse functions and other symbols
    def pre_process_corpus(docs):
      norm_docs = []
      for doc in tqdm.tqdm(docs):
        #case folding
        doc = doc.lower()
        #remove special characters\whitespaces
        doc = strip_html_tags(doc)
        doc = doc.translate(doc.maketrans("\n\t\r", "   "))
        doc = remove_accented_chars(doc)
        doc = contractions.fix(doc)
        doc = re.sub(r'[^a-zA-Z0-9\s]', '', doc, re.I|re.A)
        doc = re.sub(' +', ' ', doc)
        doc = doc.strip()
        #tokenize
        doc = word_tokenize(doc)
        #filtering
        doc = stopwords_removal(doc)
        norm_docs.append(doc)
      
      norm_docs = [" ".join(word) for word in norm_docs]
      return norm_docs
    
    st.title("MODEL PREDICTION")

    # Divider
    st.markdown('<div style="text-align: laeft;"><h2>Dataframe</h2></div>', unsafe_allow_html=True)

    # Divider
    st.markdown('<div style="text-align: left;"><h2>Input Data</h2></div>', unsafe_allow_html=True)

    # Customer data
    headline = st.text_input("News Headline", value='Tech-Company has a huge layoff after covid-19 ended')


    # Create a dictionary with the entered data
    data = {
        'headline' : headline
    }

    features = pd.DataFrame(data, index=[0])


    features.headline = pre_process_corpus(features.headline)

    # Text Vectorization

    text_vectorization = TextVectorization(max_tokens=5157,
                                           standardize="lower_and_strip_punctuation",
                                           split="whitespace",
                                           ngrams=None,
                                           output_mode="int",
                                           output_sequence_length=35,
                                           input_shape=(1,)) 
    
    # Adapt the vectorization based on X_train
    text_vectorization.adapt(features)

    # Vectorize the text data in features
    features = text_vectorization(np.array(features['headline']))  # Assuming 'text_column' is the column containing text data
    
    # # Perform inference using the model
    # results_inference = load_model.predict(features)
    

    # Button to input data into DataFrame
    col1, col2 = st.columns((8, 4))
    with col1:
        input_button = st.button("Input Headline", key="input_button", help="Click this button to input data to see the sentimental analysis.")

    with col2:
        predict_button = st.button("Run Model to Make Prediction", key="predict_button", help="Click this button to make a prediction.")

    if input_button:
        st.subheader('User Input')
        st.write(features)

    if predict_button:
        prediction = load_model.predict(features) # Get the first element of the prediction array
        st.write(prediction)
        pred_inf_th07= tf.where(prediction >0.70, 1,0)
       
        if pred_inf_th07 == 1:
            prediction_text = 'Positive News'
        else:
            prediction_text = 'Negative News'

        st.subheader('Results')
        st.write(prediction_text)

user_input()