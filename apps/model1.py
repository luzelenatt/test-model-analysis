import streamlit as st
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud, ImageColorGenerator
from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
def app():
    st.header('Análisis de sentimientos de tweets sobre Pedro Castillo')
    import snscrape.modules.twitter as sntwitter
    cantidad_tweets = st.number_input('Cantidad de tweets a evaluar: ',500,1000)
    
    # query = "(from:BarackObama) until:2022-01-01 since:2002-01-01"
    query = "(from:PedroCastilloTe) until:2022-12-22 since:2010-01-01"
    tweets = []
    limit = cantidad_tweets
    
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        if len(tweets)==limit:
            break
        else:
            tweets.append([tweet.date.date(), tweet.content]) 
        
    df = pd.DataFrame(tweets, columns=['Date', 'Tweet'])
    
    def clean_text(text):  
        pat1 = r'@[^ ]+'                   #signs
        pat2 = r'https?://[A-Za-z0-9./]+'  #links
        pat3 = r'\'s'                      #floating s's
        pat4 = r'\#\w+'                     # hashtags
        pat5 = r'&amp '
        #pat6 = r'[^A-Za-z\s]'         #remove non-alphabet
        combined_pat = r'|'.join((pat1, pat2,pat3,pat4,pat5))
        text = re.sub(combined_pat,"",text).lower()
        return text.strip()
    #limpiar tweets
    df['Tweet']=df['Tweet'].apply(clean_text)

    #mostrar los tweets limpiados
    st.subheader('Datos limpiados (sin menciones, links, hashtags o retweets)') 
    st.write(df) 
    
    #crear una funcion para obtener la subjetividad
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    #crear una funcion para obtener la polaridad
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity

    #crear dos columnas 'Subjectivity' y 'Polarity'
    df['Subjectivity']=df['Tweet'].apply(getSubjectivity)
    df['Polarity']=df['Tweet'].apply(getPolarity)

    #mostrar el nuevo dataframe con las dos nuevas columnas 'Subjectivity' y 'Polarity'
    st.subheader('Se añadieron columnas de subjetividad y polaridad') 
    st.write(df)
    
