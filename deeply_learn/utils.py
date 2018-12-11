#!/usr/bin/env python3.6.5

'''
=========================

Common utility functions

=========================
'''

import json
import keras
import numpy as np

from keras import backend as kb
from keras.applications import mobilenet
from keras.preprocessing import image
from tqdm import tqdm

'''
Converts whatever should be an integer that isn't an integer into an integer as it should be.
(Sometimes the number of likes is displayed as a string)
'''
def to_int(number):
    try:
        # if number is a string, need to clean it up a bit
        if isinstance(number, (str)):
            ret = number.replace(',', '')

            if ret.endswith('m'):
                ret = int(float(ret[:-1]) * 1000000)
            elif ret.endswith('k'):
                ret = int(float(ret[:-1]) * 1000)
            else:
                ret = int(ret)
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
Creates keras data generators for training and testing
'''
def create_data_generators(inputs, labels, train_batch_size=30, test_size=0.33):
    indices = shuffle_indices(len(labels))
    split = np.int(len(indices) * (1-test_size))

    train_index, test_index = indices[:split], indices[split:]

    train_labels = [labels[i] for i in train_index]
    test_labels = [labels[i] for i in test_index]

    train_set, test_set = [], []
    splitting_points = []
    for count, item in enumerate(inputs):
        train_data = [item[i] for i in train_index]
        train_data = np.array(train_data)

        test_data = [item[i] for i in test_index]
        test_data = np.array(test_data)

        if len(train_data.shape) == 1 and len(test_data.shape) == 1:
            train_data = train_data.reshape((len(train_data), 1))
            test_data = test_data.reshape((len(test_data), 1))

        if count > 0:
            splitting_points.append(splitting_points[count-1] + train_data.shape[1])
        else:
            splitting_points.append(train_data.shape[1])

        train_set.append(train_data)
        test_set.append(test_data)

    train_set = np.concatenate(tuple(train_set), axis=1)
    test_set = np.concatenate(tuple(test_set), axis=1)

    train_gen = DataGenerator(train_index, train_set, splitting_points, train_labels, train_batch_size)
    test_gen = DataGenerator(test_index, test_set, splitting_points, test_labels, 1)
    
    return train_gen, test_gen


'''
Use a passed in keras model for predictions
'''
def keras_predict(generator, model, tolerance=0.1, verbose=1, correct_only=0):
    correct = []
    history = {'prediction': [], 'target': []}
    for i in tqdm(range(len(generator.IDs)), desc='generating predictions', ncols=100):
        x, y = generator.__getitem__(i)
        prediction = model.predict(x)

        prediction, target = np.exp(prediction), np.exp(y)

        if np.abs(prediction-target) <= target*tolerance:
            correct.append(1)
        else:
            correct.append(0)

        history['prediction'].append(prediction)
        history['target'].append(target)


    if verbose:
        for i in range(len(correct)):
            if correct_only:
                if correct[i]:
                    print('prediction: %20.3f  |  target: %20.3f  |  %s' % (history['prediction'][i], history['target'][i], 'O'))
            else:
                if correct[i]:
                    print('prediction: %20.3f  |  target: %20.3f  |  %s' % (history['prediction'][i], history['target'][i], 'O'))
                else:
                    print('prediction: %20.3f  |  target: %20.3f  | ' % (history['prediction'][i], history['target'][i]))

    print('Total number correct = %d / %d' % (np.sum(correct), len(generator.IDs)), end='\n')
    return history


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


'''
Data generator for keras models
This will allow us to load in images and generate batch data on the fly, solving the problem
of running out of memory when trying to load in too many images
(order of input data MUST be [meta, image classification, image path]!!!)
'''
class DataGenerator(keras.utils.Sequence):
    def __init__(self, IDs, inputs, splitting_points, labels, batch_size=30, shuffle=True):
        self.IDs = IDs
        self.inputs = inputs
        self.split = splitting_points
        self.labels = labels

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.IDs) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size : (index+1)*self.batch_size]

        temp_IDs = [self.indices[i] for i in indices]

        inputs, labels = self.__data_generation(temp_IDs)

        return inputs, labels

    def on_epoch_end(self):
        self.indices = np.arange(len(self.IDs))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data_generation(self, temp_IDs):
        meta = []
        image_class = []
        image_array = []
        outputs = []

        for ID in temp_IDs:
            meta.append(self.inputs[ID, :self.split[0]])

            image_class.append(self.inputs[ID, self.split[0]:self.split[1]])
            
            img = image.load_img(self.inputs[ID, -1], target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = mobilenet.preprocess_input(np.expand_dims(img_array, axis=0))
            img_array = np.squeeze(img_array, axis=0)

            image_array.append(img_array)

            outputs.append(self.labels[ID])

        return [np.array(meta),  np.array(image_array)], outputs


