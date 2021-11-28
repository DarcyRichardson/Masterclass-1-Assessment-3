# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 13:09:55 2021

@author: Darcy Richardson
"""
#data structures
import numpy  as np
import pandas as pd

# from nltk.tokenize import sent_tokenize
# text = comments['comment'][7]
# tokenised=(sent_tokenize(text))

#1. Data wrangling
#Get forum comments
comments=pd.read_csv('comments.csv')

#Get keywords
keywords=pd.read_csv('keyword.csv')

#Make all comments lowercase

comments['comment'].str.lower()


#Keyword sentence extraction
#txt = comments['comment'][3]

keyword_sentences={'sentence':[],'keyword':[]}

#loop through comments
for i in range(len(comments)):
    comment=comments['comment'].iloc[i]

    #loop through keywords to extract sentences
    for x in range(len(keywords)):
        keyword=keywords['keywords'].iloc[x]
        
        #perform tokenisation and extract sentences with keywords
        sentence_strings=[sentence + '.' for sentence in comment.split('.') if keyword in sentence]
        
        #loop through append sentences to dictionary
        for y in range(len(sentence_strings)):
            keyword_sentences['sentence'].append(sentence_strings[y])
            keyword_sentences['keyword'].append(keyword)
            
#Remove duplicate sentences
keyword_sentences=pd.DataFrame(keyword_sentences)        
keyword_sentences=keyword_sentences.drop_duplicates()

keyword_sentences=keyword_sentences.reset_index()
keyword_sentences=keyword_sentences.drop('index',axis=1)

#2. Sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()

#Complete sentiment analysis of comments
comments_sent = keyword_sentences.apply(lambda r: analyser.polarity_scores(r.sentence), axis=1)

df = pd.DataFrame(list(comments_sent))
compiled_sentiment = keyword_sentences.join(df)
compiled_sentiment['comp_score'] = compiled_sentiment['compound'].apply(lambda c: '1' if c >=0 else '0')

#3. Plot results for each keyword

#compiled_sentiment.groupby('keyword').hist(column=["compound"])

summary = compiled_sentiment.loc[:,['keyword', 'compound']]

summary=summary.groupby('keyword').describe()











