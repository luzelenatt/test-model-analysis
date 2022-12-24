import streamlit as st
#import tweepy
#from textblob import TextBlob
#from wordcloud import WordCloud, ImageColorGenerator
#from nltk.tokenize import WordPunctTokenizer
import pandas as pd
import numpy as np
import plotly.express as px
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import nfx
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
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
    
    dir(nfx)
    # Cleaning Text: Multiple WhiteSpaces
    df['clean_tweet'] = df['Tweet'].apply(nfx.remove_multiple_spaces)
    # Cleaning Text : Remove urls
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_urls)
    # Cleaning Text : remove_userhandles
    df['clean_tweet'] = df['clean_tweet'].apply(lambda x: nfx.remove_userhandles(x))
    # Cleaning Text: Multiple WhiteSpaces
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_multiple_spaces)
    # Cleaning Text : remove_urls
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_urls)
    # Cleaning Text: remove_punctuations
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_punctuations)
    # Cleaning Text: remove_special_characters
    #df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_special_characters)
    # Cleaning Text: remove_shortwords
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_shortwords)
    # Cleaning Text: remove_emojis
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_emojis)
    # Cleaning Text: remove_punctuations
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_punctuations)
    # Cleaning Text: remove_punctuations
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_terms_in_bracket)
    # Cleaning Text: remove_shortwords
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_shortwords)
    # Cleaning Text: remove_stopwords
    df['clean_tweet'] = df['clean_tweet'].apply(nfx.remove_stopwords)
    
    stopwords = set(stopwords.words('spanish', 'english')) 
    stopwords.update(['la','lo','los','las','le','les','un','unos','una','unas','el','tu','mi','si','te','se','de','mas','al','del','él','tú','mí','sí','té','sé','dé','más',
                    'este','esta','estos','estas','ese','esa','esos','esos','esas','aquel','aquella','aquellos','aquellas','mío','mía','nuestro','nuestra','tuyo','tuya','su','suyo','suya'
                    'mis','míos','mías','nuestros','nuestras','vuestros','vuestras','tuyos','tuyas','sus','suyos','suyas','algún','alguna','algunos','algunas','varios','mucho','muchas','poco','poca'
                    'pocos','pocas','demasiado','demasiados','demasiadas','a','e','i','o','u','ante','bajo','cabe','con','contra','desde','durante','en','entre','hacia','hasta','mediante','para','por',
                    'según','sin','so','sobre','tras','versus','vía'
                    'pero','como','esta','porque','y','nos'])
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
    
