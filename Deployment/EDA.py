# Import Liblaries
import pandas as pd
import numpy as np
from scipy import stats
import streamlit as st
from nltk.corpus import stopwords
from wordcloud import WordCloud,STOPWORDS
import re

# Liblaries for Visualization
import matplotlib.pyplot as plt
import seaborn as sns



def graph():

  st.title("Credit Card Attrition Model Prediction")
  # Load Data
  copy_df = pd.read_csv('all_data.csv', delimiter=',',encoding='latin-1')
  copy_df = copy_df.rename(columns={'neutral':'sentiment','According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .':'headline'})

  # Create a boolean mask where sentiment column is not equal to 'neutral'
  mask = copy_df['sentiment'] != 'neutral'

  # Apply the mask to filter out rows where sentiment is not 'neutral'
  copy_df = copy_df[mask]

  copy_df['sentiment'].value_counts()

  # Set Web Title
  st.title("EXPLORATORY DATA ANALYSIS (EDA)")


  # Divider
  st.markdown('--------------')


  # Divider 1
  st.markdown('<div style="text-align: center;"><h2>EDA 1</h2></div>', unsafe_allow_html=True)

  # Divider
  st.markdown('--------------')


  # Seaborn count plot
  fig, ax = plt.subplots()
  ax = sns.countplot(x='sentiment', data=copy_df, palette='Set2')

  # Calculate percentages
  total_count = len(copy_df)
  for p in ax.patches:
      percentage = '{:.1f}%'.format(100 * p.get_height() / total_count)
      x = p.get_x() + p.get_width() / 2
      y = p.get_height() + 0.02 * total_count
      ax.annotate(percentage, (x, y), ha='center')

  # Set plot titles and labels
  plt.title('Sentiment Count')
  plt.xlabel('Sentiment')
  plt.ylabel('Count')

  # Display the plot using Streamlit
  st.pyplot(fig)

  # Graph Info
  st.markdown('<div style="text-align: center;"><p>Sentiment Count</p></div>', unsafe_allow_html=True)
  
  description1 = """
    Check Sentiment Data (Target)

    From the bar plot above we can see that there are more postive news than there are negative news, where positive is 69.29% and 30.71% negative values. `The data is moderatly imbalanced but since we cannot duplicate negative values or remove positive value we will leave them be.`

    Note: The reason we cannot conduct balancing is because :

    - if we oversample we will get duplicate values which we cannot use.
    - if we undersample we will lower the already low values even further.
  """

  st.markdown(description1)


  # Divider 2
  st.markdown('<div style="text-align: center;"><h2>EDA 2</h2></div>', unsafe_allow_html=True)

  # Divider
  st.markdown('--------------')


  # Combines all tweet text into one string
  all_headline_text = ' '.join(copy_df['headline'])  

  # Create WordCloud
  wordcloud = WordCloud(width=800, height=400, background_color='white', min_font_size=10, max_words=50).generate(all_headline_text) 

  # Displays WordCloud
  fig, ax = plt.subplots(figsize=(10, 5))
  ax.imshow(wordcloud, interpolation='bilinear')
  ax.axis("off")
  st.pyplot(fig)

    # Graph Info
  st.markdown('<div style="text-align: center;"><p>Word Cloud</p></div>', unsafe_allow_html=True)
  
  description2 = """
    Using the wordcloud liblary we can see the top 50 words are the ones that show up the most `We can see that the words that are correlated to companies and EUR currency, and many other words like sales, year, ect`. 

    There are still words that needs to be cleared such as S which is a single letter and U as well, this will be cleaned in the feature engineering section.
  """
  st.markdown(description2)
