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
    
    cantidad_tweets = st.number_input('Cantidad de tweets a evaluar:', 50, step=20)
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
        pat1 = r'@[A-Za-z0-9]+'
        pat2 = r'https?://[A-Za-z0-9./]+'
        pat3 = r'pic.twitter.com/[A-Za-z0-9./]+'
        #pat4 = r'\'s' 
        #pat5 = r'\#\w+'
        combined_pat = r'|'.join((pat1, pat2, pat3))
        soup = BeautifulSoup(text, 'lxml')
        souped = soup.get_text()
        stripped = re.sub(combined_pat, '', souped)
        try:
            clean = stripped.decode("utf-8-sig").replace(u"\ufffd", "?")
        except:
            clean = stripped
        letters_only = re.sub("[^a-zA-Z]", " ", clean)
        lower_case = letters_only.lower()
        words = tok.tokenize(lower_case)
        return (" ".join(words)).strip()
   
    st.subheader("Eliminar enlaces, menciones, hastags y espacios de los tweets")
    removing_data = st.text('Eliminando caracteres innecesarios...')
    
    df['Tweet'] = df['Tweet'].apply(lambda text: tweet_cleaner(text))
    df['Tweet'].head(25)
    st.write(df)
    
    st.subheader('ANÁLISIS DE SENTIMIENTO')
    @st.cache
    def get_sentiment(text):
        blob = TextBlob(text)
        sentiment_polarity = blob.sentiment.polarity
        sentiment_subjectivity = blob.sentiment.subjectivity
        if sentiment_polarity > 0:
            sentiment_label = 'Positive'
        elif sentiment_polarity < 0:
            sentiment_label = 'Negative'
        else:
            sentiment_label = 'Neutral'
        result = {'polarity':sentiment_polarity,
                'subjectivity':sentiment_subjectivity,
                'sentiment':sentiment_label}
        return result

    st.text('Resultados de la clasificación de sentimientos')
    df["sentiment"] = df["Tweet"].apply(get_sentiment)
    df['Tweet'].head(25)
    st.write(df)
    
    df4 = df['sentiment'].value_counts()
    st.write(df4)
    
    positive_tweet = df[df['sentiment'] == 'Positive']['Tweet']
    neutral_tweet = df[df['sentiment'] == 'Neutral']['Tweet']
    negative_tweet = df[df['sentiment'] == 'Negative']['Tweet']
    
    st.write('Cantidad de tweets positivos: ', positive_tweet.shape[0]/ df.shape[0] * 100, 5)
    st.write('Cantidad de tweets neutrales: ', neutral_tweet.shape[0]/ df.shape[0] * 100, 5)
    st.write('Cantidad de tweets negativos: ', negative_tweet.shape[0]/ df.shape[0] * 100, 5)
    
    st.subheader('Gráfica circular (%) que indica la distribución de los sentimientos de tipo positivo, negativo y neutro')
    labels = 'Positivo', 'Neutral', 'Negativo'
    positive = df[df.sentiment == 1].shape[0]
    neutral = df[df.sentiment == 0].shape[0]
    negative = df[df.sentiment == -1].shape[0]
    
    sizes = [positive, neutral, negative]
    colors = ['lightcoral', 'gold', 'lightskyblue']
    explode = (0.1, 0, 0)
    # Plot
    plt.pie(sizes, explode=explode  , labels=labels, colors=colors,
    autopct='%1.1f%%', startangle=140)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()
    
    st.subheader('Gráfica de Análisis de Sentimientos Subjetividad - Polaridad')
    with st.spinner('Cargando grafica de sentimiento'):
        #grafico de sentimiento y subjetividad con plotly
        st.subheader('Grafico de sentimiento y subjetividad')
        fig = px.scatter(df_tweets, x="polarity", y="subjectivity", color="sentiment",
                            hover_data=['Tweet'])
        st.write("Eje horizontal - Polaridad: Mientras más cercano a 1, más positivo es el comentario, mientras más cercano a -1, más negativo es el sentimiento.")
        st.write("Eje vertical - Subjetividad: Mientras más cercano a 1, más relacionado a la opinión es el comentario, mientras más cercano a 0, más relacionado a hechos es el comentario.")
        st.plotly_chart(fig)    
