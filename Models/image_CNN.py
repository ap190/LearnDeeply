#!/usr/bin/env python3.6.5
'''
Convolutional Neural Network for downsampling the image as well as provide a 
learning mechanism for the image features themselves.
'''

import keras
import numpy as np
import utils
from PIL import Image

# ==================== DATA PREPROCESSING FOR IMAGE CLASSIFICATION NN (WILL AUTO RUN ON IMPORT)
json_data = utils.preprocess.json_data

image, num_likes = [], []
for user in json_data:
    for image in user['images']:
        if not utils.to_int(images['likes']) > 0:
            continue

        im = Image.open(utils.preprocess.data_path + image['picture']).convert('L')
        im = np.array(im)
        im = np.float32(np.reshape(im, (im.shape[0]*im.shape[1])))

        image.append(list(im))
        num_likes.append(image['likes'])

model_data = {
    'inputs': image,
    'labels': num_likes
}

# add processed data structure for this NN model into global pre-processed data class
utils.preprocess.add_model_data('image', model_data)
# ====================


# ====================
''' Convolutional Neural Network for learning and downsampling an image to see the influence of the actual image
    on the number of likes an instagram post recieves '''
class Model:
    def __init__(self, inputs, labels, output_size=100, 
        learning_rate=0.001, test_size=0.33,
        dropout=0.0, hidden_layers=0, hidden_sizes=[], epochs=10, batch_size=100):

        # split data into training and testing datasets
        self.train_inputs, self.train_labels, self.test_inputs, self.test_labels = utils.shuffle_data(inputs, labels, test_size)

        # derived parameters
        self.input_length = len(self.train_inputs[0])

        # optional parameters that have defaulted values
        self.learning_rate = learning_rate
        self.output_size = output_size
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.batch_size = batch_size

        # computation graph construction
        self.construct_graph()

    ''' Construct computation graph with keras '''
    def construct_graph(self):
        # instantiate input tensors
        image = keras.layers.Input(shape=(self.input_length, ))

        # if number of hidden layers and their respective sizes are specified
        if self.hidden_layers:
            # iterate through hidden layer construction
            for layer in range(self.hidden_layers):
                inputs = keras.layers.Dense(units=self.hidden_sizes[layer], kernel_initializer='random_normal', activation='relu')(inputs)
                inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        ouput = keras.layers.Dense(units=self.output_size, kernel_initializer='random_normal')(inputs)

# ====================