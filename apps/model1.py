import streamlit as st
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import WordPunctTokenizer
from textblob import TextBlob
import plotly.express as px
import asyncio
import sys
#from config import load_tweet
import gensim
from gensim import matutils, models
import scipy.sparse
from gensim import interfaces, utils, matutils
sys.path.append('../')

def app():
    st.header('Análisis de sentimientos de tweets sobre Pedro Castillo')
    import snscrape.modules.twitter as sntwitter
    st.subheader('EXTRACCIÓN DE LA DATA')    
    query = st.text_input('Ingresar la busqueda para scrapear','pedro castillo (to:PedroCastilloTe) (@PedroCastilloTe) lang:es until:2022-12-16 since:2022-12-01')
    st.caption('Se recomienda acceder a la búsqueda avanzada de twitter (https://twitter.com/search-advanced?lang=es) y pegar la consulta generada')
    
    cantidad_tweets = st.number_input('Cantidad de tweets a evaluar:', 500, step=20)
    tweet_data = None 
    df = None   
    # query = "(from:BarackObama) until:2022-01-01 since:2002-01-01"
    #query = "(from:PedroCastilloTe) until:2022-12-22 since:2010-01-01"
    tweets = []
    limit = cantidad_tweets
    
    for tweet in sntwitter.TwitterSearchScraper(query).get_items():
        
        if len(tweets) == limit:
            break
        else:
            tweets.append([tweet.date.date(), tweet.user.username, tweet.content])
    st.subheader('INFORMACIÓN DE LA DATA')        
    #mostrar los tweets extraídos
    st.subheader('Datos extraídos de Twitter: Fecha, Nombre de usuario y Tweet')
    df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet'])  
    st.write(df) 
     
    # Check Columns
    st.subheader('Columnas de la data')
    st.write('Referencia de cada columna: Índice 0: Fecha, Índice 1: Nombre de usuario, Índice 2: Tweet')
    st.write(df.columns)   
    
    st.subheader('Gráfica de barras - Distribución de los tweets por usuario')     
    # Source/ Value Count/Distribution of the Users
    #st.write('Mostramos los usuarios únicos que hacen más tweets relacionados con el tema')   
    #st.write(df['User'].unique())
    # Plot the top value_counts
    st.write('Mostramos los 25 primeros usuarios que hacen más tweets relacionados con el tema')   
    df1 = df['User'].value_counts().nlargest(25)
    st.write(df1)
    # graficar en streamlit
    st.write('Obtenemos la gráfica de barras en donde el eje x son los nombres usuarios de Twitter y el eje Y es la cantidad o frecuencia de tweets')
    fig = px.bar(df1,height=800)
    st.plotly_chart(fig)
    
    st.subheader('PREPROCESAMIENTO DE TEXTO')

    @st.cache
    def tweet_cleaner(text):
        tok = WordPunctTokenizer()
        pat1 = r'@[^ ]+'
        pat2 = r'https?://[A-Za-z0-9./]+'
        pat3 = r'\'s'
        pat4 = r'\#\w+' 
        pat5 = r'&amp'
        combined_pat = r'|'.join((pat1, pat2, pat3, pat4, pat5))
        text = re.sub(combined_pat,"",text).lower()
        return text.strip()
    #limpiar tweets
    df['Tweet']=df['Tweet'].apply(tweet_cleaner)
    st.write(df)
        # Cleaning Text : remove_userhandles
    import neattext.functions as nfx
    df['Tweet'] = df['Tweet'].apply(lambda x: nfx.remove_userhandles(x))
    # Cleaning Text: Multiple WhiteSpaces
    df['Tweet'] = df['Tweet'].apply(nfx.remove_multiple_spaces)
    # Cleaning Text : remove_urls
    df['Tweet'] = df['Tweet'].apply(nfx.remove_urls)
    # Cleaning Text: remove_punctuations
    df['Tweet'] = df['Tweet'].apply(nfx.remove_punctuations)
    # Cleaning Text: remove_special_characters
    #df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_special_characters)
    # Cleaning Text: remove_shortwords
    df['Tweet'] = df['Tweet'].apply(nfx.remove_shortwords)
    # Cleaning Text: remove_emojis
    df['Tweet'] = df['Tweet'].apply(nfx.remove_emojis)
    # Cleaning Text: remove_punctuations
    df['Tweet'] = df['Tweet'].apply(nfx.remove_punctuations)
    # Cleaning Text: remove_punctuations
    df['Tweet'] = df['Tweet'].apply(nfx.remove_terms_in_bracket)
    # Cleaning Text: remove_shortwords
    df['Tweet'] = df['Tweet'].apply(nfx.remove_shortwords)
    # Cleaning Text: remove_stopwords
    df['Tweet'] = df['Tweet'].apply(nfx.remove_stopwords)
    
    
    st.subheader('ANÁLISIS DE SENTIMIENTO')
    #crear una funcion para calcular el sentimiento
    def detect_polarity(text):
        return TextBlob(text).sentiment.polarity

    #crear una funcion para calcular la subjetividad
    def detect_subjectivity(text):
        return TextBlob(text).sentiment.subjectivity

    # aplicamos la funcion a la columna de tweets
    df['polarity'] = df['Tweet'].apply(detect_polarity)
    df['subjectivity'] = df['Tweet'].apply(detect_subjectivity)
        
    #mostrar el dataframe con los tweets y sus sentimientos
    st.subheader('Resultado de twetts con sentimiento (polaridad y subjetividad')
    #selecciona las columnas que nos interesan
    st.write(df[['Tweet', 'polarity', 'subjectivity']])
    
    #hacer una nube de palabras con los tweets positivos
    #seleccionar los tweets positivos
    df_pos = df[df['polarity'] > 0]
    # Join the different processed titles together.
    long_string = ','.join(list(df_pos['Tweet'].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')       
    # Generate a word cloud
    wordcloud.generate(long_string)       
    # Visualize the word cloud in streamlit
    st.subheader('Nube de palabras de tweets positivos')
    st.image(wordcloud.to_array())
    
    #hacer una nube de palabras con los tweets negativos
    #seleccionar los tweets negativos
    df_neg = df[df['polarity'] < 0]
    # Join the different processed titles together.
    long_string = ','.join(list(df_neg['Tweet'].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud in streamlit
    st.subheader('Nube de palabras de tweets negativos')
    st.image(wordcloud.to_array())

    #hacer una nube de palabras con los tweets neutros
    #seleccionar los tweets negativos
    df_neg = df[df['polarity'] == 0]
    # Join the different processed titles together.
    long_string = ','.join(list(df_neg['Tweet'].values))
    # Create a WordCloud object
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    # Generate a word cloud
    wordcloud.generate(long_string)
    # Visualize the word cloud in streamlit
    st.subheader('Nube de palabras de tweets neutros')
    st.image(wordcloud.to_array())

    st.subheader('Grafico de Dispersión: Polaridad y Subjetividad de sentimientos')
    with st.spinner('Cargando grafica de sentimiento'):
        #grafico de sentimiento y subjetividad con plotly
        fig = px.scatter(df, x="polarity", y="subjectivity", hover_data=['Tweet'])
        st.write("Eje horizontal Polaridad: -1 <-- Negativa --------- 0 Neutral ------------ Positiva --> 1")
        st.write("Eje verticalSubjetividad: 1 <-- Hechos --------------------- Opiniones --> 0")
        st.plotly_chart(fig)
    