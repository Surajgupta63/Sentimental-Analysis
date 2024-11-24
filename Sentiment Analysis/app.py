import re
import pickle
import joblib
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()

model = pickle.load(open('model.mdl', 'rb'))
tfidf = joblib.load('tfidf_model.joblib')

st.title("Sentiment Analysis")

def preprocess_single_tweet(tweet):
    tweet = re.sub(r'[^a-zA-Z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = [word for word in tweet.split(' ') if word and word not in stopwords.words('english')]
    tweet = [stemmer.stem(word) for word in tweet]
    tweet = ' '.join(tweet)
    return tweet

tweet = st.text_input("write...")

if st.button("Sentiment"):

    processed_tweet = preprocess_single_tweet(tweet)


    encoded_tweet = tfidf.transform([processed_tweet])

    result = encoded_tweet.toarray()

    prediction = model.predict(result)[0]

    if prediction == 1:
        st.success("Positive")
    else:
        st.error('Negative')

    st.write('Text:', tweet)