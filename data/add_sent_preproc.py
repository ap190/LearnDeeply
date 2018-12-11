#!/usr/bin/env python3.6.5

import json
import numpy as np


from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger
from langdetect import detect
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
natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-03-16',
    iam_apikey='KNU1uoHR7W2C44UOJoKYGGyCajGRAosybOEVEPHlzkzn',
    url='https://gateway.watsonplatform.net/natural-language-understanding/api'
)
true0 = 0
err0 = 0

def SentimentClassify(target_text):
    supportinglist = ['ar','en','fr','de','it','ja','ko','pt','ru','es']
    try:
        response_senti = natural_language_understanding.analyze(
            text= target_text,
            features=Features(sentiment=SentimentOptions())
            ).get_result()
        
        sentiscore = response_senti["sentiment"]["document"]["score"]
        if sentiscore==0:
            global true0
            true0 = true0+1
        return sentiscore
    except:
        # print("discription: ",target_text)
        global err0
        err0 = err0+1
        print("a language error")
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


with open('compiled_data.json',encoding='UTF-8') as jsonfile:
    jsondata = json.load(jsonfile)
    # print(jsondata)

count = 0
for user in jsondata:
    for image in user['images']:
        count = count+1
        if count%10==0:
            print(count)
        desc = image['description']
        desc = " ".join(filter(lambda x:x[0]!='#' and x[0]!='@', desc.split()))
        sentiment = sentiment_analysis(desc)
        image['sentiment'] = sentiment
        
        if count%1000==0 and count>14000:
            new_json = 'compiled_data_wsent_%.1f.json ' %count
            print("printing count: ",new_json)
            with open(new_json, 'w',encoding='UTF-8') as outfile:
                json.dump(jsondata, outfile, ensure_ascii=False, indent=2)
                # outfile.truncate() 
print("0 due to mutual: ",true0)
print("0 due to language exception: ",err0)
        


