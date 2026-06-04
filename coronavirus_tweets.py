# Part 3: Text mining.

import pandas as pd
import numpy as np
import requests
from collections import Counter
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

# Return a pandas dataframe containing the data set.
# Specify a 'latin-1' encoding when reading the data.
# data_file will be populated with a string 
# corresponding to a path containing the coronavirus_tweets.csv file.
def read_csv_3(data_file):
	df = pd.read_csv(data_file, encoding='latin-1')
	return df

# Return a list with the possible sentiments that a tweet might have.
def get_sentiments(df):
	return list(df['Sentiment'].unique())

# Return a string containing the second most popular sentiment among the tweets.
def second_most_popular_sentiment(df):
	counts = df['Sentiment'].value_counts()
	return counts.index[1]

# Return the date (string as it appears in the data) with the greatest number of extremely positive tweets.
def date_most_popular_tweets(df):
	extremely_positive = df[df['Sentiment'] == 'Extremely Positive']
	return extremely_positive['TweetAt'].value_counts().index[0]

# Modify the dataframe df by converting all tweets to lower case. 
def lower_case(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.lower()

# Modify the dataframe df by replacing each characters which is not alphabetic or whitespace with a whitespace.
def remove_non_alphabetic_chars(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'[^a-zA-Z\s]', ' ', regex=True)

# Modify the dataframe df with tweets after removing characters which are not alphabetic or whitespaces.
def remove_multiple_consecutive_whitespaces(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.replace(r'\s+', ' ', regex=True).str.strip()

# Given a dataframe where each tweet is one string with words separated by single whitespaces,
# tokenize every tweet by converting it into a list of words (strings).
def tokenize(df):
	df['OriginalTweet'] = df['OriginalTweet'].str.split()

# Given dataframe tdf with the tweets tokenized, return the number of words in all tweets including repetitions.
def count_words_with_repetitions(tdf):
	return int(tdf['OriginalTweet'].apply(len).sum())

# Given dataframe tdf with the tweets tokenized, return the number of distinct words in all tweets.
def count_words_without_repetitions(tdf):
	all_words = [word for tweet in tdf['OriginalTweet'] for word in tweet]
	return len(set(all_words))

# Given dataframe tdf with the tweets tokenized, return a list with the k distinct words that are most frequent in the tweets.
def frequent_words(tdf, k):
	all_words = [word for tweet in tdf['OriginalTweet'] for word in tweet]
	counter = Counter(all_words)
	return [word for word, count in counter.most_common(k)]

# Given dataframe tdf with the tweets tokenized, remove stop words and words with <=2 characters from each tweet.
# The function should download the list of stop words via:
# https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt
def remove_stop_words(tdf):
	url = 'https://raw.githubusercontent.com/fozziethebeat/S-Space/master/data/english-stop-words-large.txt'
	response = requests.get(url)
	stop_words = set(response.text.splitlines())
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(
		lambda tweet: [word for word in tweet if word not in stop_words and len(word) > 2]
	)

# Given dataframe tdf with the tweets tokenized, reduce each word in every tweet to its stem.
def stemming(tdf):
	ps = PorterStemmer()
	tdf['OriginalTweet'] = tdf['OriginalTweet'].apply(
		lambda tweet: [ps.stem(word) for word in tweet]
	)

# Given a pandas dataframe df with the original coronavirus_tweets.csv data set,
# build a Multinomial Naive Bayes classifier. 
# Return predicted sentiments (e.g. 'Neutral', 'Positive') for the training set
# as a 1d array (numpy.ndarray). 
def mnb_predict(df):
	X = df['OriginalTweet']
	y = df['Sentiment']
	vectorizer = CountVectorizer(min_df=1, max_df=1.0, max_features=500000, ngram_range=(1,5))
	X_transformed = vectorizer.fit_transform(X)
	clf = MultinomialNB(alpha=0.0001)
	clf.fit(X_transformed, y)
	return clf.predict(X_transformed)

# Given a 1d array (numpy.ndarray) y_pred with predicted labels (e.g. 'Neutral', 'Positive') 
# by a classifier and another 1d array y_true with the true labels, 
# return the classification accuracy rounded in the 3rd decimal digit.
def mnb_accuracy(y_pred, y_true):
	accuracy = (y_pred == y_true).mean()
	return round(accuracy, 3)
