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
    # Ahora aplicaremos la siguiente técnica: Eliminar algunas puntuaciones, textos o palabras que no tengan sentido
    def clean_text_round2(text):
        '''Suprimir algunos signos de puntuación adicionales y texto sin sentido.'''
        text = re.sub('[‘’“”…]', '', text)
        text = re.sub('\n', '', text)
        return text

    round2 = lambda x: clean_text_round2(x)
    df['clean_tweet'] = df['clean_tweet'].apply(round2)
    #funcion para eliminar todos los caracteres que no sean letras en ingles
    def remove_non_ascii_1(text):
        '''Remove non-ASCII characters from list of tokenized words'''
        return re.sub(r'[^\x00-\x7f]',r'', text)
    #aplicamos la funcion a la columna de tweets
    df['clean_tweet'] = pd.DataFrame(df.clean_tweet.apply(remove_non_ascii_1))
    #las filas que no tienen tweets se eliminan
    df = df[df['clean_tweet'] != '']
    st.subheader('Tweets extraidos')
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
    
