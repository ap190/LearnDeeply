#!/usr/bin/env python3.6.5

'''
=========================

Neural Network for processing and learning weights of the the metadata of 
user's instagram post.

=========================
'''

import numpy as np
import utils
from keras import backend as K 
from keras.applications import mobilenet as M
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Dense, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.applications.mobilenet import MobileNet


# ==================== DATA PREPROCESSING (WILL AUTO RUN ON IMPORT)
def nima_data(_data, _dict):
    img = image.load_img(_data, target_size=(224, 224))

    im_array = image.img_to_array(img)
    im_array = np.expand_dims(im_array, axis=0)

    pim = M.preprocess_input(im_array)
    pim = pim.flatten()

    _dict['images'].append(pim.tolist())

utils.preprocess.add_preprocess_function('nima', nima_data)
# ====================


# ==================== NIMA KNOCKOFF
class Graph:
    def __init__(self, input_shape, dropout=0.0):
        self.dropout = dropout

        self.inputs, self.outputs, self.layers = self.construct_graph()

    def construct_graph(self):
        images = keras.layers.Input(shape=(input_shape[0]*input_shape[1]*input_shape[2], ))

        image_size = 224
        base_model = MobileNet(input_shape=(image_size, image_size, 3), input_tensor=images, alpha=1, include_top=False, pooling='avg')

        for layer in base_model.layers:
            layer.trainable = False

        outputs = Dropout(rate=self.dropout)(base_model.output)
        outputs = Dense(10, activation='softmax')(outputs)

        return images, outputs, base_model.layers

# ====================


