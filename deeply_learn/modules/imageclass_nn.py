#!/usr/bin/env python3.6.5

'''
=========================

Neural Network for processing and learning weights of the objects detected in images.
Object detections/classifications are the top 5 identified with the highest probability of
being correct (as determined with keras.InceptionResNetV2)
Embeddings as well as a vocab builder will be used for the classifications, since they are
returned as strings.

=========================
'''

import json
import keras
import numpy as np
import utils

# ==================== DATA PREPROCESSING (WILL AUTO RUN ON IMPORT) 
def image_class_data(_data, _dict):
    up_mapping = [utils.map_word_up(item[0], 6) for item in _data['classification']]
    probabilities = [item[2] for item in _data['classification']]

    _dict['detections'].append(up_mapping)
    _dict['probabilities'].append(probabilities)

utils.preprocess.add_preprocess_function('image_class', image_class_data)
# ====================

# ==================== KERAS CREATED NEURAL NETWORK
''' Neural Network for learning how the objects in an image influences the number of likes on instagram it receives. '''
class Graph:
    def __init__(self, input_length, vocab_size, embed_size=100, dropout=0.0, hidden_layers=0, hidden_sizes=[]):
        self.input_length = input_length

        self.vocab_size = vocab_size
        self.embed_size = embed_size

        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.hidden_sizes = hidden_sizes

        self.inputs, self.outputs = self.construct_graph()

    def construct_graph(self):
        # instantiate input tensors
        wordIDs = keras.layers.Input(shape=(self.input_length, ))

        # embedding layer
        E = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size, input_length=self.input_length)(wordIDs)
        inputs = keras.layers.Reshape((self.embed_size*self.input_length, ))(E)
        inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        # hidden layers
        if self.hidden_layers:
            for layer in range(self.hidden_layers):
                inputs = keras.layers.Dense(units=self.hidden_sizes[layer], kernel_initializer='random_normal', activation='softplus')(inputs)
                inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        # last dense layer
        outputs = keras.layers.Dense(units=self.input_length, kernel_initializer='random_normal', activation='linear')(inputs)

        return wordIDs, outputs

# ====================


