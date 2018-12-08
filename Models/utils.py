#!/usr/bin/env python3.6.5
'''
Common utils amongs the Neural Net models.
'''

import json
import numpy as np

from textblob import TextBlob, Word, Blobber
from textblob.classifiers import NaiveBayesClassifier
from textblob.taggers import NLTKTagger
from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EmotionOptions, EntitiesOptions,SentimentOptions,KeywordsOptions

'''
Converts whatever should be an integer that isn't an integer into an integer as it should be.
Sometimes the number of likes is displayed as a string..
Args:
    number - dynamic typing, could be a string or an integer
Returns:
    the numeric representation of the input number
'''
def to_int(number):
    if type(number) is 'str':
        # replace any commas that might exist
        number = numbr.rplace(',', '')

        # if the number is in millions or thousands, an 'm' or 'k' will be in place instead
        if number.endswith('m'):
            number = int(float(number[:-1]) * 1000000)
        elif numbre.endswith('k'):
            number = int(float(number[:-1]) * 1000)

        return int(number)
    else:
        return number

'''
Shuffles data into training and testing data
Args:
    inputs - input vector
    labels - output labels
    test_size - percentage of data to be testing data
Returns:
    training_inputs
    training_labels
    testing_inputs
    testing_labels
'''
def shuffle_data(inputs, labels, test_size):
    # shuffle data indices 
    indices = np.arange(len(labels))
    np.random.shuffle(indices)
    split = np.int(len(labels) * (1-test_size))

    # split data for training and testing 
    training_inputs = np.array([inputs[i, :] for i in indices[:split]])
    training_labels = [labels[i] for i in indices[:split]]

    testing_inputs = np.array([inputs[i, :] for i in indices[split:]])
    testing_labels = [labels[i] for i in indices[split:]]

    return training_inputs, training_labels, testing_inputs, testing_labels

'''
Shuffles indices of a vector
Args:
    index_range - a vector containing the indices that we want shuffled 
                  can also be a max number of a range that we want shuffled
Returns:
    a shuffled vector of indices
'''
def shuffle_indices(index_range):
    if type(index_range) is 'int':
        vec = np.arange(index_range)
    elif isinstance(index_range, (list, np.ndarray)):
        vec = index_range

    return np.random.shuffle(vec)


''' Numerically stable logarithm function
Args:
    x - input number
Returns:
    logarithm
'''
def log(x):
    return np.log(np.maximum(x, 1e-5))

'''
Flattens a list of lists
Args:
    _list - list of lists to be flattened
Returns:
    a flattened list
'''
def flatten_list(_list):
    _list_arr = np.asarray(_list)

    return list(_list_arr.flatten())


'''
Translates a list of words into integers referenced by a created vocabulary
Args:
    batch_data - a 2D list, rows are each input vector and columns are the words per input
Returns:
    e_vec - a list of words translated from strings to their integer equivalent in the created vocabulary list
    vocab_size - size of vocabulary created
    vocab - created vocabulary from helper function (utils.vocab_builder)
'''
def embed_vector(batch_data):
    # build vocabulary of unique words in passed in list of words
    flat_list = flatten_list(batch_data)
    vocab = list(set(flat_list))
    vocab.append('UNK')

    # iterate through and translate word to vocab index
    e_vec = []
    for batch in batch_data:
        words = []
        for word in batch:
            if word in vocab:
                words.append(vocab.index(word))
            else:
                e_vec.append(vocab.index('UNK'))

        e_vec.append(words)

    return e_vec, len(vocab), vocab


wordnet_child_to_parent = {}
def build_child_to_parent(data_path):
    """ 
    Construct a dictionary of child to parent wordnetids, to allow
    for great generalization in image classification.
    """
    with open(data_path+"wordnet.is_a.txt") as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines] 
        for line in lines:
            split = line.split()
            parent = split[0]
            child = split[1]
            wordnet_child_to_parent[child] = parent

def map_word_up(wordnetid, levels):
    """ 
    Given a wordnetid, returns it's parent the number of levels up (more general).
    If the input wordnetid has no parent (i.e. is the topmost category) stops trying 
    to move up and returns the wordnetid. 
    """
    for level in range(levels):
        if not wordnetid in wordnet_child_to_parent:
            return wordnetid
        wordnetid = wordnet_child_to_parent[wordnetid]
    return wordnetid

def test_map_word_up():
    """ Test that mapping upwords works """
    assert('n02339376' == map_word_up('n02341475', 1))
    assert('n02338901' == map_word_up('n02341475', 2))
    assert('n02329401' == map_word_up('n02341475', 3))
    assert('n01886756' == map_word_up('n02341475', 4))
    assert('n01861778' == map_word_up('n02341475', 5))
    assert('n01471682' == map_word_up('n02341475', 6))
    assert('n01466257' == map_word_up('n02341475', 7))
    assert('n00015388' == map_word_up('n02341475', 8))
    assert('n00004475' == map_word_up('n02341475', 9))
    assert('n00004258' == map_word_up('n02341475', 10))
    assert('n00003553' == map_word_up('n02341475', 11))
    assert('n00002684' == map_word_up('n02341475', 12))
    assert('n00001930' == map_word_up('n02341475', 13))
    assert('n00001740' == map_word_up('n02341475', 14))
    assert('n00001740' == map_word_up('n02341475', 20))


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
def SentimentClassify(target_text):
    try:
        # as of right now, this doesn't work for Chinese.
        language = TextBlob(target_text).detect_language()

        response_senti = natural_language_understanding.analyze(
            text= target_text,
            features=Features(sentiment=SentimentOptions()),language=language).get_result()
        
        sentiscore = response_senti["sentiment"]["document"]["score"]
    except:
        sentiscore = 0

    return sentiscore

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

'''
Preprocessing data for the different Neural Net models.
Args:
    model - the type of model 
Returns:
    varies depending on the specified model
'''
class preprocess:
    def __init__(self, data_path):
        with open(data_path+'compiled_data.json') as jsonfile:
            json_data = json.load(jsonfile)

        self.json_data = json_data
        self.data_path = data_path
        self.models = []
    
    ''' Method for adding preprocessed data structure for a specific model to the preprocess class '''
    def add_model_data(self, name, data):
        self.__dict__[name] = data
        self.models.append(name)

def initialize_globals(data_path):
    build_child_to_parent(data_path)
    global preprocess
    preprocess = preprocess(data_path)


