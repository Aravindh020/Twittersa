# Twittersa
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from textblob import TextBlob

import re

from nltk import punkt

from nltk.corpus import stopwords

from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn.metrics import confusion_matrix, classification_report,accuracy_score

train_tweets = pd.read_csv('finalizedtrain.csv')

test_tweets = pd.read_csv('finalizedtest.csv')

train_tweets = train_tweets[['senti','tweet']]

test = test_tweets['tweet']

train_tweets['length'] = train_tweets['tweet'].apply(len)

fig1 = sns.barplot('senti','length',data = train_tweets,palette='PRGn')

plt.title('Average Word Length vs senti')

plot = fig1.get_figure()

plot.savefig('Barplot.png')

fig2 = sns.countplot(x= 'senti',data = train_tweets)

plt.title('Sentiment Counts')

plot = fig2.get_figure()

plot.savefig('Count Plot.png')

def text_processing(tweet):

def form_sentence(tweet):

tweet_blob = TextBlob(tweet)

return ' '.join(tweet_blob.words)

new_tweet = form_sentence(tweet)

def no_user_alpha(tweet):

tweet_list = [ele for ele in tweet.split() if ele != 'user']

clean_tokens = [t for t in tweet_list if re.match(r'[^\W\d]*$', t)]

clean_s = ' '.join(clean_tokens)
       
	   clean_mess = [word for word in clean_s.split() if word.lower() not in stopwords.words('english')]
       
	   return clean_mess
    
	no_punc_tweet = no_user_alpha(new_tweet) 
    
	def normalization(tweet_list):
    
	lem = WordNetLemmatizer()
    
	normalized_tweet = []
    
	for word in tweet_list:
    
	normalized_text = lem.lemmatize(word,'v')
    
	normalized_tweet.append(normalized_text)
    
	return normalized_tweet
  
  train_tweets['tweet_list'] = train_tweets['tweet'].apply(text_processing)

test_tweets['tweet_list'] = test_tweets['tweet'].apply(text_processing)

train_tweets[train_tweets['senti']==1].drop('tweet',axis=1).head()

X = train_tweets['tweet']

y = train_tweets['senti']

test = test_tweets['tweet']

from sklearn.model_selection import train_test_split

msg_train, msg_test, label_train, label_test = train_test_split(train_tweets['tweet'], train_tweets["senti"], test_size=0.2)

#MISSSING


def normalization(tweet_list):

lem = WordNetLemmatizer()

normalized_tweet = []

for word in tweet_list:

normalized_text = lem.lemmatize(word,'v')

normalized_tweet.append(normalized_text)

return normalized_tweet
    
tweet_list = 'I was playing with my friends with whom I used to play, when you called me yesterday'.split()

print(normalization(tweet_list))

predictions = pipeline.predict(msg_test)

print(classification_report(predictions,label_test))

print ('\n')

print(confusion_matrix(predictions,label_test))

print(accuracy_score(predictions,label_test))

