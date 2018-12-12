#!/usr/bin/env python3.6.5
'''
Convolutional Neural Network for downsampling the image as well as provide a 
learning mechanism for the image features themselves.
'''

import json
import keras
import numpy as np
import utils
from keras.applications.inception_v3 import InceptionV3
from PIL import Image

# ==================== DATA PREPROCESSING FOR IMAGE CNN (WILL AUTO RUN ON IMPORT)
json_data = utils.preprocess.json_data

image_data, num_likes = [], []
print('opening images')
for user in json_data:
    for image in user['images']:
        if not utils.to_int(image['likes']) > 0:
            continue

        im = Image.open(utils.preprocess.data_path + image['picture']).convert('L')
        im = np.array(im)
        im = np.float32(im.flatten())

        image_data.append(im)
        num_likes.append(image['likes'])

print('finished images thing')
model_data = {
    'inputs': image_data,
    'labels': num_likes
}

print(image_data[0])
stop

# add processed data structure for this NN model into global pre-processed data class
utils.preprocess.add_model_data('image', model_data)
# ====================


# ====================
''' Convolutional Neural Network for image something'''
class Model:
    def __init__(self, inputs, labels, output_size=100, 
        learning_rate=0.001, test_size=0.33,
        dropout=0.0, hidden_layers=0, hidden_sizes=[], epochs=10, batch_size=100):

        # intialize InceptionNetV3
        self.IncepV3 = InceptionV3(weights='imagenet', include_top=False)

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

        # send image through InceptionV3
        inputs = self.IncepV3.predict(image)

        # if number of hidden layers and their respective sizes are specified
        if self.hidden_layers:
            # iterate through hidden layer construction
            for layer in range(self.hidden_layers):
                inputs = keras.layers.Dense(units=self.hidden_sizes[layer], kernel_initializer='random_normal', activation='relu')(inputs)
                inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        ouput = keras.layers.Dense(units=self.output_size, kernel_initializer='random_normal', activation='softmax')(inputs)

        # specify optimizer and intialize model for training
        optimizer = keras.optimizer.Adam(lr=self.learning_rate)
        self.model = keras.model.Model(inputs=image, outputs=output)

        # compile keras computation graph
        self.model.compile(loss='mae', optimizer=optimizer)

    ''' Train constructed model '''
    def train_model(self, verbose=0):
        if not verbose:
            print('Training image classification embeddings NN . . . ')

        self.model.fit(x=self.train_inputs, y=self.train_labels, 
            batch_size=self.batch_size, epochs=self.epochs, verbose=verbose)

        if not verbose:
            print(' . . . Finished training.')

    ''' Make predictions with constructed model '''
    def predict(self, inputs, verbose=0):
        return self.model.predict(x=inputs, verbose=verbose)

    ''' Test constructed model '''
    def test_model(self, tolerance=0.10, verbose=0):
        predictions = self.predict(self.test_inputs, self.test_probabilities)

        print('Testing with tolerance of +/- %.3f %% of correct value:' % (tolerance*100))
        correct = 0
        for prediction, target in zip(predictions, self.test_labels):
            prediction, target = np.exp(prediction), np.exp(target)
            mark = ''

            if np.abs(prediction-target) <= target*tolerance:
                correct += 1
                mark = 'O'

            if verbose:
                print('prediction: %20.3f  |  target: %20.3f  |  %s' % (prediction, target, mark))

        print('Total number correct = %d / %d' % (correct, len(self.test_labels)), end='\n')

# ====================