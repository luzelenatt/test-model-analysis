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
    st.write(df_tweets[['tweet', 'polarity', 'subjectivity']])
    
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

    with st.spinner('Cargando grafica de sentimiento'):
        #grafico de sentimiento y subjetividad con plotly
        st.subheader('Grafico de sentimiento y subjetividad')
        fig = px.scatter(df_tweets, x="polarity", y="subjectivity", color="sentiment",
                            hover_data=['tweet'])
        st.write("Eje horizontal: Mientras más cercano a 1, más positivo es el comentario Mientras más cercano a -1, más negativo es el sentimiento.")
        st.write("Eje vertical: Mientras más cercano a 1, más subjetivo es el comentario Mientras más cercano a 0, más objetivo es el comentario.")
        st.plotly_chart(fig)
    with st.spinner('Contando comentarios positivos y negativos'):
        #hacer un grafico circular de los sentimientos positivos y negativos con plotly
        # si el sentimiento es mayor a 0, es positivo, si es menor a 0 es negativo 
        #contar los tweets positivos y negativos
        df_tweets['label'] = df_tweets['polarity'].apply(lambda x: 'Positivo' if x > 0 else 'Negativo')
        #crear un dataframe con los sentimientos
        df_sent = df_tweets['label'].value_counts().reset_index()
        df_sent.columns = ['sentimiento', 'total']
        #grafico circular
        st.subheader('Contador de comentarios positivos y negativos')
        fig = px.pie(df_sent, values='total', names='sentimiento', title='Sentimientos')
        st.plotly_chart(fig)

    num_temas = st.slider('Numero de temas', 1, 10, 5)
    with st.spinner('Analizando temas de los tweets'):
        # crear un diccionario de palabras para el modelo
        cv = CountVectorizer(stop_words='spanish')
        data_cv = cv.fit_transform(df_tweets.clean_tweet)
        data_stop = pd.DataFrame(data_cv.toarray(), columns=cv.get_feature_names())
        data_stop.index = df_tweets.index
        #crear el modelo de LDA
        # Convertir una matriz dispersa de conteos en un corpus gensim
        corpus = matutils.Sparse2Corpus(scipy.sparse.csr_matrix(data_stop.transpose()))

        # Gensim también requiere un diccionario de todos los términos y su ubicación respectiva en la matriz de documentos de términos
        id2word = dict((v, k) for k, v in cv.vocabulary_.items())

        # Crear modelo lda (equivalente a "fit" en sklearn)
        lda = models.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_temas, passes=40)
        #guardar cada topico como la combinacion de las palabras
        # de cada topico
        topics = lda.show_topics(formatted=False)
        # estraee solo la palabra de cada topico
        topics_words = [(tp[0], [wd[0] for wd in tp[1]]) for tp in topics]
        topics_words
        #armar una string con las palabras de cada topico unidad porcomas
        topics_string = []
        for topic in topics_words:
            topics_string.append(' '.join(topic[1]))
        topics_string
        #renombrar la columna de los topico
        topics_string = pd.DataFrame(topics_string, columns=['topic'])
        topics_string
        #armar un dataframe con los topico y las palabras
        df_topics_names= pd.DataFrame(topics_string)

        # Ver los temas en el modelo LDA
        st.subheader('Temática en los tweets')
        for i in range(0, df_topics_names.shape[0]):
            st.write('Tema', i, ':', df_topics_names.iloc[i, 0])
        # Echemos un vistazo a los temas que contiene cada tweet
        # y guardarlo en un dataframe
        corpus_transformed = lda[corpus]
        topics = [sorted(topics, key=lambda record: -record[1])[0] for topics in corpus_transformed]
        df_topics = pd.DataFrame(topics, columns=['Topico', 'Importancia'])
        #grafucar los topico con plotly
        st.subheader('Gráfico de los topicos de los tweets')
        fig = px.histogram(df_topics, x="Topico", y="Importancia", color="Topico", height=400)
        st.plotly_chart(fig)
        
    #mostrar que usuario tiene el comentario mas positivo
    with st.spinner('Calculando el usuario con el comentario mas positivo'):
        #seleccionar el tweet mas positivo
        tweet_positivo = df.loc[df['polarity'].idxmax()]
        #mostrar el tweet mas positivo
        st.subheader('El tweet más positivo y su usuario')
        st.write(tweet_positivo['Tweet'])
        st.write("Usuario 😉: "+tweet_positivo['User'])
        
    #mostrar que usuario tiene el comentario mas negativo
    with st.spinner('Calculando el usuario con el comentario mas negativo'):
        #seleccionar el tweet mas negativo
        tweet_negativo = df.loc[df['polarity'].idxmin()]
        #mostrar el tweet mas negativo
        st.subheader('El tweet más negativo y su usuario')
        st.write(tweet_negativo['Tweet'])
        st.write("Usuario 😔: "+tweet_negativo['User'])
