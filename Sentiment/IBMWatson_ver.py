import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EmotionOptions, EntitiesOptions,SentimentOptions,KeywordsOptions

from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-03-16',
    iam_apikey='KNU1uoHR7W2C44UOJoKYGGyCajGRAosybOEVEPHlzkzn',
    url='https://gateway.watsonplatform.net/natural-language-understanding/api'
)
text1= 'Bruce Banner is the Hulk and Bruce Wayne is BATMAN! '
text2= 'Today is wonderful'
text3 = 'Today is bad'
text4 = "i hate you so much"
text_japan = "指原莉乃です。初心者です。猫2匹と生活してます"
text_japan2 = "愛しているが愛してないわよ。□ポロンポロン□□愛あい出会いよポロンポロン□□オマジナイヤデ〜。以後、キオツケヤ〜。☆☆☆"
text_chinese = "自由人在ShangHai"
text_french = "tu as besoin de gants de boxe"
text_korean = "이지금"

def createTargetlist(text):
    targets = text.split()
    print("targets: ",targets)
    return targets

#Emotion - need targets
"""
@ only support English
@ return respective scores of 5 different emotions 
@range [0,1]
"""
def EmotionClassify(target_text):
    response_emo = natural_language_understanding.analyze(
    text=target_text,
    features=Features(emotion=EmotionOptions(document=True))
    ).get_result()
    emotionscore = response_emo["emotion"]["document"]["emotion"]
    return emotionscore


#Sentiment - [(bool)document, (list)target]
"""
@ support all except Chinese
@ return scores 
@ range [-1,1]
"""
def SentimentClassify(target_text):
    response_senti = natural_language_understanding.analyze(
        text= target_text,
        features=Features(sentiment=SentimentOptions()),language=TextBlob(target_text).detect_language()
        ).get_result()
    sentiscore = response_senti["sentiment"]["document"]["score"]
    return sentiscore

print(EmotionClassify(text1))
print(SentimentClassify(text4))
