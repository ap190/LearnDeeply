#!/usr/bin/env python3.6.5

'''
=========================

Common utility functions

=========================
'''

import json
import numpy as np

from keras import backend as kb

'''
Converts whatever should be an integer that isn't an integer into an integer as it should be.
(Sometimes the number of likes is displayed as a string)
'''
def to_int(number):
    try:
        # if number is a string, need to clean it up a bit
        if isinstance(number, (str)):
            number = number.replace(',', '')

            if number.endswith('m'):
                ret = int(float(number[:-1]) * 1000000)
            elif number.endswith('k'):
                ret = int(float(number[:-1]) * 1000)
        # if number is some other type of number, why might have to round
        elif isinstance(number, (float)):
            ret = int(number)
        # if number is already an integer then we're okay
        elif isinstance(number, (int)):
            ret = number

        return ret
    except:
        # ruh roh, passed in number to convert to integer isn't what we expect 
        # (number isn't a string that is a number or a some sort of numeric thing..)
        stop


'''
Numerically stabel logarithm function
'''
def log(x):
    return np.log(np.maximum(x, 1e-05))



'''
Shuffle indices vector
'''
def shuffle_indices(index_range):
    if isinstance(index_range, (list, np.ndarray)):
        vec = index_range
    else:
        vec = np.arange(index_range)
    
    np.random.shuffle(vec)
    return vec


'''
Hour grouping 
'''
def hour_categorize(hour):
    hour_ind = 0
    if hour >= 0 and hour < 8:
        hour_ind = 0
    elif hour >= 8 and hour < 12:
        hour_ind = 1
    elif hour >= 12 and hour < 14:
        hour_ind = 2
    elif hour >= 14 and hour < 17:
        hour_ind = 3
    elif hour >= 17 and hour < 20:
        hour_ind = 4
    else:
        hour_ind = 5

    return hour_ind


'''
Translates a list of words into integers referenced by a created vocabulary
'''
def embed_vector(batch_data):
    print('starting the thing with embedding')
    count = 0
    # build vocabulary of unique words in passed in list of words
    flat_list = list(np.array(batch_data).flatten())
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

        count += 1
        print(count)

        e_vec.append(words)

    return e_vec, len(vocab), vocab


'''
Construct a dictionary of child to parent wordnetIDs to allow for great
generalization in image classification.
'''
__wordnet_child_to_parent = {}
def build_child_to_parent(datapath):
    with open(datapath + 'wordnet.is_a.txt') as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines] 
        for line in lines:
            split = line.split()
            parent = split[0]
            child = split[1]
            __wordnet_child_to_parent[child] = parent


'''
Given a wordnetid, returns it's parent the number of levels up (more general).
If the input wordnetid has no parent (i.e. is the topmost category) stops trying 
to move up and returns the wordnetid.
'''
def map_word_up(wordnetID, levels):
    for level in range(levels):
        if not wordnetID in __wordnet_child_to_parent:
            return wordnetID
        wordnetID = __wordnet_child_to_parent[wordnetID]

    return wordnetID


'''
Root mean squared error
'''
def rmse(y_true, y_pred):
    return kb.sqrt(kb.mean(kb.square(y_pred-y_true), axis=-1))


'''
Use a passed in keras model for predictions
'''
def keras_predict(model, inputs, targets, tolerance=0.1, verbose=1, correct_only=0):
    predictions = model.predict(inputs)

    correct = 0
    for prediction, target in zip(predictions, targets):
        prediction, target = np.exp(prediction), np.exp(target)
        mark = ''

        if np.abs(prediction-target) <= target*tolerance:
            correct += 1
            mark = 'O'

        if verbose:
                if correct_only and mark is 'O':
                    print('prediction: %20.3f  |  target: %20.3f  |  %s' % (prediction, target, mark))
                else: 
                    print('prediction: %20.3f  |  target: %20.3f  |  %s' % (prediction, target, mark))

    print('Total number correct = %d / %d' % (correct, len(targets)), end='\n')

    return np.exp(predictions)


'''
Preprocessing data for the different Neural Net models
This will be a global class that will be passed between the modules and their specific required
structured will be added to this class.
'''
class preprocess:
    def __init__(self, data_path):
        with open(data_path + 'compiled_data.json', 'rb') as json_file:
            json_data = json.load(json_file)

        with open(data_path + 'top_hashtags.json', 'rb') as tagweight_file:
            tagweight_data = json.load(tagweight_file)

        self.data_path    = data_path
        self.json_data    = json_data
        self.hash_weights = tagweight_data

        self.preproc_functs = {}
        self.models = []

    ''' Append preprocessed data structure for an invididual module '''
    def add_model_data(self, name, data):
        self.__dict__[name] = data
        self.models.append(name)

    ''' Add a preprocessing function '''
    def add_preprocess_function(self, name, funct):
        self.preproc_functs[name] = funct

def initialize_globals(data_path):
    build_child_to_parent(data_path)
    global preprocess
    preprocess = preprocess(data_path)



