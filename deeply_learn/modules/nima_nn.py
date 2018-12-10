#!/usr/bin/env python3.6.5

'''
=========================

Neural Network for processing and learning weights of the the metadata of 
user's instagram post.

=========================
'''

import numpy as np
import keras
import utils
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet


# ==================== DATA PREPROCESSING (WILL AUTO RUN ON IMPORT)
def nima_data(_data, _dict):
    _dict['images'].append(_data)

utils.preprocess.add_preprocess_function('nima', nima_data)
# ====================


# ==================== NIMA KNOCKOFF
class Graph:
    def __init__(self, dropout=0.0):
        self.dropout = dropout

        self.inputs, self.outputs = self.construct_graph()

    def construct_graph(self):
        # load images to target size
        image_size = 224
        base_model = MobileNet(input_shape=(image_size, image_size, 3), alpha=1, include_top=False, pooling='avg')
        # incorrect layers, without the top layer we have 54 layers but this weights file has 55 layers of weights
        # base_model.load_weights('modules/mobilenet_weights.h5')

        for layer in base_model.layers:
            layer.trainable = False

        outputs = Dropout(rate=self.dropout)(base_model.output)
        outputs = Dense(10, activation='softmax')(outputs)
        outputs.trainable = False

        NIMA = keras.Model(base_model.input, outputs)
        NIMA.load_weights('modules/mobilenet_weights.h5')

        return base_model.input, NIMA.output

# ====================


