from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger

text1= 'Bruce Banner is the Hulk and Bruce Wayne is BATMAN! '
text2= 'Today is wonderful'
text3 = 'Today is bad'
text4 = "i love you"
text_japan = "指原莉乃です。初心者です。猫2匹と生活してます"
text_chinese = "自由人在ShangHai"
text_french = "tu as besoin de gants de boxe"
text_korean = "이지금"

testimonial = TextBlob(text4)
testimonial2 = TextBlob(text4)


print(text4)
print(testimonial.sentiment)
print("language: ",testimonial2.detect_language())