#!/usr/bin/env python3.6.5

'''
=========================

Neural Network for processing and learning weights of the the metadata of 
user's instagram post.

=========================
'''

import json 
import numpy as np
import keras
import utils

# ==================== DATA PREPROCESSING (WILL AUTO RUN ON IMPORT)
def meta_data(_data, _dict):
    user_info = _data['user']
    image_info = _data['image']

    tw = 0
    for tag in image_info['tags']:
        if tag[1:] in utils.preprocess.hash_weights:
            tw += utils.preprocess.hash_weights[tag[1:]]

    one_hot_weekday = [0] * 7
    one_hot_weekday[image_info['weekday']] = 1

    one_hot_hour = [0] * 6
    hour = image_info['hour']
    hour_ind = utils.hour_categorize(hour)
    one_hot_hour[hour_ind] = 1

    _dict['followers'].append(utils.to_int(user_info['following']))
    _dict['following'].append(utils.to_int(user_info['following']))
    _dict['num_posts'].append(utils.to_int(user_info['posts']))
    _dict['avg_likes'].append(utils.to_int(user_info['avg_likes']))
    _dict['num_tags'].append(len(image_info['tags']))
    _dict['len_desc'].append(len(image_info['description']))
    _dict['num_ments'].append(len(image_info['mentions']))
    _dict['tag_weight'].append(tw)
    _dict['weekday'].append(one_hot_weekday)
    _dict['hour'].append(one_hot_hour)

utils.preprocess.add_preprocess_function('metadata', meta_data)
# ====================


# ==================== KERAS CREATED NEURAL NETWORK
''' Neural Network for learning how the metadata of a post influences the number of likes on instagram it receives. '''
class Graph:
    def __init__(self, input_length, output_length=0, intermediate_layer=128, dropout=0.0, hidden_layers=0, hidden_sizes=[]):
        self.input_length = input_length

        self.intermediate_layer = intermediate_layer
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.hidden_sizes = hidden_sizes
        if output_length == 0:
            self.output_length = self.input_length
        else:
            self.output_length = output_length

        self.inputs, self.outputs = self.construct_graph()

    def construct_graph(self):
        # instantiate input tensors
        metadata = keras.layers.Input(shape=(self.input_length, ))

        # first dense layer
        inputs = keras.layers.Dense(units=self.intermediate_layer, kernel_initializer='random_normal', activation='relu')(metadata)
        inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        # hidden layers
        if self.hidden_layers:
            for layer in range(self.hidden_layers):
                inputs = keras.layers.Dense(units=self.hidden_sizes[layer], kernel_initializer='random_normal', activation='relu')(inputs)
                inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        # last dense layer
        outputs = keras.layers.Dense(units=self.output_length, kernel_initializer='random_normal', activation='linear')(inputs)

        return metadata, outputs

# ====================


