import json
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EmotionOptions, EntitiesOptions,SentimentOptions,KeywordsOptions

natural_language_understanding = NaturalLanguageUnderstandingV1(
    version='2018-03-16',
    iam_apikey='KNU1uoHR7W2C44UOJoKYGGyCajGRAosybOEVEPHlzkzn',
    url='https://gateway.watsonplatform.net/natural-language-understanding/api'
)
text1= 'Bruce Banner is the Hulk and Bruce Wayne is BATMAN! '
text2= 'Today is wonderful'
text3 = 'Today is bad'
text4 = "i do love you"
text_japan = "指原莉乃です。初心者です。猫2匹と生活してます"
text_chinese = "自由人在ShangHai"
text_french = "tu as besoin de gants de boxe"
text_korean = "이지금"

# #Entities detect- [limit,mentions,model,sentiment,emotion]
# response1 = natural_language_understanding.analyze(
#     # html="<html><head><title>Fruits</title></head><body><h1>Apples and Oranges</h1><p>I love apples! I don't like oranges.</p></body></html>",
#     text=text2,
#     features=Features(entities=EntitiesOptions(sentiment=True,limit=1,emotion=True))
#     ).get_result()
# #Key words detect
# response = natural_language_understanding.analyze(
#     text=text3,
#     features=Features(keywords=KeywordsOptions(sentiment=True,emotion=True,limit=2))).get_result()

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
    features=Features(emotion=EmotionOptions(document=True,targets=createTargetlist(target_text)))
    ).get_result()
    return response_emo


#Sentiment - [(bool)document, (list)target]
"""
@ support all except Chinese
@ return scores 
@ range [-1,1]
"""
def SentimentClassify(target_text):
    response_senti = natural_language_understanding.analyze(
        text= target_text,
        features=Features(sentiment=SentimentOptions(createTargetlist(target_text)))).get_result()
    sentiscore = response_senti["sentiment"]["document"]["score"]
    return sentiscore



print(json.dumps(EmotionClassify(text4), indent=2))
print("=========================================================")
print(SentimentClassify(text4))
