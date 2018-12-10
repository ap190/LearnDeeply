#!/usr/bin/env python3.6.5

'''
=========================

Neural Network for combining multiple models

=========================
'''

import json
import keras
import numpy as np
import utils

# ====================
''' Neural Network to process the combined system as well as adding in sentiment analysis of the captions '''
class Graph:
    def __init__(self, input_tensor, intermediate_layer=128, dropout=0.0, hidden_layers=0, hidden_sizes=[]):
        self.input_tensor = input_tensor

        self.intermediate_layer = intermediate_layer
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.hidden_sizes = hidden_sizes

        self.inputs, self.outputs = self.construct_graph()

    def construct_graph(self):
        # first dense layer
        inputs = keras.layers.Dense(units=self.intermediate_layer, kernel_initializer='random_normal', activation='softplus')(self.input_tensor)
        inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        # hidden layers
        if self.hidden_layers:
            for layer in range(self.hidden_layers):
                inputs = keras.layers.Dense(units=self.hidden_sizes[layer], kernel_initializer='random_normal', activation='softplus')(inputs)
                inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        outputs = keras.layers.Dense(units=1, kernel_initializer='random_normal', activation='linear')(inputs)

        return self.input_tensor, outputs

# ====================


