import streamlit as st
import pandas as pd
import tweepy
import plotly as pt
import plotly.graph_objects as go
from tweepy import OAuthHandler
import json
import csv
import re
from textblob import TextBlob
import string
import preprocessor as p
import os
import time
from datetime import datetime, timedelta, date
import sys
import numpy as np
import pycountry
from bs4 import BeautifulSoup
from PIL import Image
import nltk
nltk.downloader.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from langdetect import detect
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

#authorize tweepy
consumer_key = 'e62eKZbH5PiOqscveQE2iqEZV'
consumer_secret = 'Mi7E6XlxmQXoRy6wpMkhzPPc2FRGMdklepcUDGBycpUAoGrvMZ'
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

redirect_url = auth.get_authorization_url()
# Get access token
key = '1165311351868100609-cSKLvQQW1Qc5dQjnHLBy7gbPvXk5Og'
secret = 'kEfF616sXSZ2GUnQHqnrfuMOs2B7RT6gqyJUMVUyf9fJ1'
auth.set_access_token(key, secret)
# Construct the API instance
api = tweepy.API(auth)

#twitter scraping function

def scraptweets(search_words, date_since, numTweets, numRuns):
    # Define a for-loop to generate tweets at regular intervals
    # We cannot make large API call in one go. Hence, let's try T times

    # Define a pandas dataframe to store the date:
    db_tweets = pd.DataFrame(columns=['username', 'acctdesc', 'location', 'following',
                                      'followers', 'totaltweets', 'usercreatedts', 'tweetcreatedts',
                                      'retweetcount', 'text', 'hashtags']
                             )
    program_start = time.time()
    for i in range(0, numRuns):
        # We will time how long it takes to scrape tweets for each run:
        start_run = time.time()

        # Collect tweets using the Cursor object
        # .Cursor() returns an object that you can iterate or loop over to access the data collected.
        # Each item in the iterator has various attributes that you can access to get information about each tweet
        if option == 'Username':
            utweets = api.user_timeline(screen_name=search_words, count=numTweets, lang='en', tweet_mode='extended')
            db_tweets = pd.DataFrame(columns=['text'])
            for t in utweets:
                text = t.full_text
                db_tweets = db_tweets.append({'text' : text}, ignore_index = True)


        else:
            tweets = tweepy.Cursor(api.search, q=search_words, lang="en", until=date_since, tweet_mode='extended', count=numTweets).items(
                numTweets)

            # Store these tweets into a python list

            tweet_list = [tweet for tweet in tweets]

            # Obtain the following info (methods to call them out):
            # user.screen_name - twitter handle
            # user.description - description of account
            # user.location - where is he tweeting from
            # user.friends_count - no. of other users that user is following (following)
            # user.followers_count - no. of other users who are following this user (followers)
            # user.statuses_count - total tweets by user
            # user.created_at - when the user account was created
            # created_at - when the tweet was created
            # retweet_count - no. of retweets
            # (deprecated) user.favourites_count - probably total no. of tweets that is favourited by user
            # retweeted_status.full_text - full text of the tweet
            # tweet.entities['hashtags'] - hashtags in the tweet
            # Begin scraping the tweets individually:
            noTweets = 0

            for tweet in tweet_list:
                # Pull the values
                username = tweet.user.screen_name
                acctdesc = tweet.user.description
                location = tweet.user.location
                following = tweet.user.friends_count
                followers = tweet.user.followers_count
                totaltweets = tweet.user.statuses_count
                usercreatedts = tweet.user.created_at
                tweetcreatedts = tweet.created_at
                retweetcount = tweet.retweet_count
                hashtags = tweet.entities['hashtags']
                try:
                    text = tweet.retweeted_status.full_text
                except AttributeError:  # Not a Retweet
                    text = tweet.full_text
                    # Add the 11 variables to the empty list - ith_tweet:
                ith_tweet = [username, acctdesc, location, following, followers, totaltweets,
                             usercreatedts, tweetcreatedts, retweetcount, text, hashtags]
                # Append to dataframe - db_tweets
                db_tweets.loc[len(db_tweets)] = ith_tweet
                # increase counter - noTweets
                noTweets += 1



        # Run ended:
        end_run = time.time()
        duration_run = round((end_run - start_run) / 60, 2)

        print('no. of tweets scraped for run {} is {}'.format(i + 1, numTweets))
        print('time take for {} run to complete is {} mins'.format(i + 1, duration_run))

        # Once all runs have completed, save them to a single csv file:

        # Define working path and filename
        path = os.getcwd()
        filename = path + '/data/' + 'test_data_tweets.csv'
        # Store dataframe in csv with creation date timestamp

        db_tweets.to_csv(filename, index=False)

        program_end = time.time()
        print('Scraping has completed!')
        print('Total time taken to scrap is {} minutes.'.format(round(program_end - program_start) / 60, 2))



# streamlit front-end
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


local_css('style.css')

head_img = st.beta_container()
header = st.beta_container()
dataset = st.beta_container()
footer = st.beta_container()

option = st.sidebar.selectbox('Analyse by:', ('Hashtag', 'Username'))
st.sidebar.header('How many tweets should be extracted?')
numTweets = st.sidebar.slider(' ', min_value=10, max_value=500, step=50, value=50)
st.sidebar.header('Extract tweets from how many days ago?')
days_to_subtract = st.sidebar.slider(' ', min_value=0, max_value=7, step=1, value=0)
date_since = date.today() - timedelta(days=days_to_subtract)
numRuns = 1


def percentage(part, whole):
    return 100 * float(part)/float(whole)

positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []

with header:
    st.title('Welcome to our Data Science Project')
    st.subheader('We will analyse data obtained by scrapping tweets using the twitter API')
    st.header('')
    if option == 'Hashtag':
        st.header('Enter the hashtag:')
    else:
        st.header('Enter the Username:')

with dataset:
    if option == 'Hashtag':
        search_words = st.text_input('', '#')
    else:
        search_words = st.text_input('', '@')
    col1, col2 = st.beta_columns(2)
    pressed = col1.button('Analyse')
    press = col2.button('Show raw tweet data')
    if press:
        st.subheader('Tweets:')
        try:
            data = pd.read_csv('data/test_data_tweets.csv')
            data.index+=1
            st.write(data.head(numTweets))
            st.success("Success")
        except FileNotFoundError:
            st.error('Please perform scraping first')

#Sentiment Analysis:
    if search_words != '#' and search_words != '' and search_words != '@':
        if pressed:
            scraptweets(search_words, date_since, numTweets, numRuns)
            st.success('Scraping done successfully ')
            tweets = pd.read_csv('data/test_data_tweets.csv')
            column1, column2 = st.beta_columns(2)
            tweetList= tweets.text
            for tweet in tweetList:
                tweet_list.append(tweet)
                analysis = TextBlob(tweet)
                score = SentimentIntensityAnalyzer().polarity_scores(tweet)
                neg = score['neg']
                neu = score['neu']
                pos = score['pos']
                comp = score['compound']
                polarity += analysis.sentiment.polarity

                if neg > pos:
                    negative_list.append(tweet)
                    negative += 1
                elif pos > neg:
                    positive_list.append(tweet)
                    positive += 1
                elif pos == neg:
                    neutral_list.append(tweet)
                    neutral += 1

            positive = percentage(positive, numTweets)
            negative = percentage(negative, numTweets)
            neutral = percentage(neutral, numTweets)
            polarity = percentage(polarity, numTweets)
            positive = format(positive, '.1f')
            negative = format(negative, '.1f')
            neutral = format(neutral, '.1f')

            # Number of Tweets (Total, Positive, Negative, Neutral)
            tweet_list = pd.DataFrame(tweet_list)
            tweet_list.index+=1
            neutral_list = pd.DataFrame(neutral_list)
            negative_list = pd.DataFrame(negative_list)
            positive_list = pd.DataFrame(positive_list)
            pos_num = len(positive_list)
            neg_num = len(negative_list)
            neu_num = len(neutral_list)
            with column1:
                st.header('')
                st.header('')
                st.write("total number of tweets: ", len(tweet_list))
                st.write("number of positive tweets: ", pos_num)
                st.write("number of negative tweets: ", neg_num)
                st.write("number of neutral tweets: ", neu_num)

            column2.write('Tweet List:')
            column2.write(tweet_list)


            # Cleaning Text (RT, Punctuation etc)
            tweet_list.drop_duplicates(inplace=True)
            # Creating new dataframe and new features
            tw_list = pd.DataFrame(tweet_list)
            tw_list["text"] = tw_list[0]

            # Removing Punctuation
            def remove_punctuation(text):
                no_punct = [words for words in text if words not in string.punctuation]
                words_wo_punct = ''.join(no_punct)
                return words_wo_punct

            tw_list['text'] = tw_list['text'].apply(lambda x: remove_punctuation(x))

            #tokenization
            def tokenize(text):
                split = re.split("\W+", text) #Here, "\W+" splits on one or more non-word character
                return split

            tw_list['text'] = tw_list['text'].apply(lambda x: tokenize(x.lower()))


            #removing stopwords
            stopword = nltk.corpus.stopwords.words('english')

            def remove_stopwords(text):
                text = [word for word in text if word not in stopword]
                return text

            tw_list['text'] = tw_list['text'].apply(lambda x: remove_stopwords(x))


            #lemmetize text
            lemmatizer = WordNetLemmatizer()

            def word_lemmatizer(text):
                lem_text = [lemmatizer.lemmatize(i) for i in text]
                return lem_text

            tw_list['text'] = tw_list['text'].apply(lambda x: word_lemmatizer(x))

            #converting class list to string
            def list_to_string(texts):
                sentence = '-'.join(texts)
                sentence = ' '.join(texts)
                return sentence

            tw_list['text'] = tw_list['text'].apply(lambda x: list_to_string(x))

            # Calculating Negative, Positive, Neutral and Compound values again
            tw_list[['polarity', 'subjectivity']] = tw_list['text'].apply(
                lambda Text: pd.Series(TextBlob(Text).sentiment))
            for index, row in tw_list['text'].iteritems():
                score = SentimentIntensityAnalyzer().polarity_scores(row)
                neg = score['neg']
                neu = score['neu']
                pos = score['pos']
                comp = score['compound']
                if neg > pos:
                    tw_list.loc[index, 'sentiment'] = "negative"
                elif pos > neg:
                    tw_list.loc[index, 'sentiment'] = "positive"
                else:
                    tw_list.loc[index, 'sentiment'] = "neutral"
                tw_list.loc[index, 'neg'] = neg
                tw_list.loc[index, 'neu'] = neu
                tw_list.loc[index, 'pos'] = pos
                tw_list.loc[index, 'compound'] = comp


            # Creating new data frames for all sentiments (positive, negative and neutral)
            tw_list_negative = tw_list[tw_list["sentiment"] == "negative"]
            tw_list_positive = tw_list[tw_list["sentiment"] == "positive"]
            tw_list_neutral = tw_list[tw_list["sentiment"] == "neutral"]

            tw_neutral_list = pd.DataFrame(tw_list_neutral)
            tw_negative_list = pd.DataFrame(tw_list_negative)
            tw_positive_list = pd.DataFrame(tw_list_positive)
            pos_num = len(tw_positive_list)
            neg_num = len(tw_negative_list)
            neu_num = len(tw_neutral_list)

            st.header('')
            st.header('After cleaning data:')
            st.write("total number of tweets: ", len(tw_list))
            st.write("number of positive number: ", pos_num)
            st.write("number of negative number: ", neg_num)
            st.write("number of neutral number: ", neu_num)

            st.write('Cleaned Tweet List:')
            tw_list.reset_index(inplace=True)
            tw_list.index += 1
            st.write(tw_list)

            # Visualizing Data
            st.header('')
            bar_chart = ['positive', 'neutral', 'negative']

            fig = go.Figure([go.Bar(x=bar_chart, y=[pos_num, neu_num, neg_num])])
            fig.update_traces(marker_color='rgb(158,202,225)', marker_line_color='rgb(8,48,107)',
                              marker_line_width=1.5, opacity=0.6)
            fig.update_layout(title_text='SENTIMENTS OF TWEETS FETCHED:')
            st.write(fig)

            fig2 = go.Figure(data=go.Scatter(
                x = tw_list['polarity'],
                y = tw_list['subjectivity'],

                mode='markers',
                marker=dict(
                    size=16,
                    color=tw_list['polarity'],  # set color equal to a variable
                    colorscale='Viridis',  # one of plotly colorscales
                    showscale=True
                ),
                ))
            fig2.update_layout(title='Polarity And Subjectivity:')
            fig2.update_xaxes(title_text='Polarity')
            fig2.update_yaxes(title_text='Subjectivity')

            st.write(fig2)

    else:
        if option == 'Hashtag':
            st.error('Please enter a hashtag')
        else:
            st.error('Please enter a username')

