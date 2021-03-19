import streamlit as st
import pandas as pd
import tweepy
from PIL import Image
import plotly as pt
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import csv
import re
from textblob import TextBlob
import string
import preprocessor as p
import os
import time
from datetime import datetime, timedelta, date

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

        print('no. of tweets scraped for run {} is {}'.format(i + 1, noTweets))
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

st.sidebar.header('How many tweets should be extracted?')
numTweets = st.sidebar.slider(' ', min_value=10, max_value=500, step=50, value=50)
st.sidebar.header('Extract tweets from how many days ago?')
days_to_subtract = st.sidebar.slider(' ', min_value=0, max_value=7, step=1, value=0)
date_since = date.today() - timedelta(days=days_to_subtract)
numRuns = 1

with header:
    st.title('Welcome to our Data Science Project')
    st.subheader('We will analyse data obtained by scrapping tweets using the twitter API')
    st.header('')
    st.header('')
    st.header('Enter the hashtag:')
with dataset:
    search_words = st.text_input('', '#')
    col1, col2 = st.beta_columns(2)
    pressed = col1.button('Analyse')
    press = col2.button('Show raw tweet data')
    if press:
        st.subheader('Tweets:')
        try:
            data = pd.read_csv('data/test_data_tweets.csv')
            st.write(data.head(numTweets))
            st.success("Success")
        except FileNotFoundError:
            st.error('Please perform scraping first')


    if pressed:

        scraptweets(search_words, date_since, numTweets, numRuns)
        st.success('Scraping done successfully ')



