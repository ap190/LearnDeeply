#!/usr/bin/env python3.6.5

import json
import numpy as np

from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EmotionOptions, EntitiesOptions,SentimentOptions,KeywordsOptions

'''
Conducts sentiment analysis over a given set of text
Args:
    target_text - a string of text to do sentiment analysis on
Returns:
    sentiscore - a score from -1 to 1 to rank negative to positive
'''
# natural language understanding package and version requried.
def SentimentClassify(target_text):
    try:
        natural_language_understanding = NaturalLanguageUnderstandingV1(
            version='2018-03-16',
            iam_apikey='KNU1uoHR7W2C44UOJoKYGGyCajGRAosybOEVEPHlzkzn',
            url='https://gateway.watsonplatform.net/natural-language-understanding/api'
        )

        # as of right now, this doesn't work for Chinese.
        language = TextBlob(target_text).detect_language()

        response_senti = natural_language_understanding.analyze(
            text= target_text,
            features=Features(sentiment=SentimentOptions()),language=language).get_result()
        
        sentiscore = response_senti["sentiment"]["document"]["score"]
        return sentiscore
    except:
        return 0

'''
Iterative deal with sentiment analysis to hand a list of inputs rather than a single string
Args:
    target_text - list of string inputs or a single string of text
Returns:
    sent_score - a list of their scores or score for single string
'''
def sentiment_analysis(target_text):
    if isinstance(target_text, list):
        sent_score = []
        for text in target_text:
            score = SentimentClassify(text)
            sent_score.append(score)
    elif isinstance(target_text, str):
        sent_score = SentimentClassify(target_text)

    return sent_score


with open('compiled_data.json') as jsonfile:
    jsondata = json.load(jsonfile)

count = 0
for user in jsondata:
    for image in user['images']:
        desc = image['description']
        desc = " ".join(filter(lambda x:x[0]!='#' and x[0]!='@', desc.split()))
        sentiment = sentiment_analysis(desc)

        print('(%.3f) %s' % (sentiment, desc))
        image['sentiment'] = sentiment

with open('compiled_data_wsent.json', 'w') as outfile:
        json.dump(jsondata, outfile, ensure_ascii=False, indent=2)
