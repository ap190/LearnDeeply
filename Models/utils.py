#!/usr/bin/env python3.6.5
'''
Common utils amongs the Neural Net models.
'''

import json
import numpy as np

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
        self.models = []

    def wut(self, name):
        self.__dict__[name] = 'hello'
    
    ''' Method for adding preprocessed data structure for a specific model to the preprocess class '''
    def add_model_data(self, name, data):
        self.__dict__[name] = data
        self.models.append(name)
def initialize_globals(data_path):
    global preprocess
    preprocess = preprocess(data_path)


