import streamlit as st
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import re
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import asyncio
import sys
#usar textblob para analizar el sentimiento de cada tweet
from textblob import TextBlob
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
    fig = px.bar(df1,height=800)
    st.plotly_chart(fig)
    
    st.subheader('Preprocesamiento de texto')   
    #hacemos una funcion para limpiar los tweets
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
    # creamos una nueva columna para los tweets limpios
    df['Tweet']=df['Tweet'].apply(clean_text)
    st.write('Removemos las menciones, enlaces, numeros, hashtags y caracteres especiales')  
    st.write(df)
    
    st.subheader('ANÁLISIS DE SENTIMIENTOS')      
    #crear una funcion para obtener la polaridad y subjetividad
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
    df['sentiment_results'] = df['clean_tweet'].apply(get_sentiment)
    df = df.join(pd.json_normalize(df['sentiment_results']))
    #mostrar el nuevo dataframe con las dos nuevas columnas 'Subjectivity' y 'Polarity'
    st.write('Se añadieron columnas de subjetividad y polaridad') 
    st.write(df)
    
    st.write('Contabilizando sentimientos positivos, negativos y neutros') 
    df2=df['sentiment'].value_counts()
    st.write(df2)
    
    # graficar en streamlit
    fig2 = px.bar(df2,height=800)
    st.plotly_chart(fig2)
    
    positive_tweet = df[df['sentiment'] == 'Positive']['clean_tweet']
    neutral_tweet = df[df['sentiment'] == 'Neutral']['clean_tweet']
    negative_tweet = df[df['sentiment'] == 'Negative']['clean_tweet']
    
    st.write('Visualizamos con mayor detalle los tweets con sentimiento positivo')
    st.write(positive_tweet)
    
    st.write('Visualizamos con mayor detalle los tweets con sentimiento neutro') 
    st.write(neutral_tweet)
    
    st.write('Visualizamos con mayor detalle los tweets con sentimiento negativo')  
    st.write(negative_tweet)
    
    #gráfia circular de los sentimientos
    ptweet = df[df.sentiment == 'Neutral']
    pteet = ptweet['clean_tweet']
    round(ptweet.shape[0] / df.shape[0] * 100, 1)
    
    ptweet = df[df.sentiment == 'Negative']
    pteet = ptweet['clean_tweet']
    round(ptweet.shape[0] / df.shape[0] * 100, 1)   
    
    ptweet = df[df.sentiment == 'Positive']
    pteet = ptweet['clean_tweet']
    round(ptweet.shape[0] / df.shape[0] * 100, 1)   
    
    st.subheader('Gráfica circular que contabiliza los comentarios positivos, negantivos y negativos')
    fig = px.pie(df['sentiment'], values='total', names='sentimiento', title='Sentimientos')
    st.plotly_chart(fig)