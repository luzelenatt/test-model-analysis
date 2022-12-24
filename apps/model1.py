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
    
    st.subheader('PREPROCESAMIENTO DE TEXTO')

    @st.cache
    def tweet_cleaner(text):
        tok = WordPunctTokenizer()
        pat1 = r'@[A-Za-z0-9]+'
        pat2 = r'https?://[A-Za-z0-9./]+'
        pat3 = r'pic.twitter.com/[A-Za-z0-9./]+'
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
   
    st.subheader("Remover caracteres")
    df['Tweet'] = df['Tweet'].apply(lambda text: tweet_cleaner(text))
    st.table(df['Tweet'].head(5))
    st.write(df)
    
    
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

    
    