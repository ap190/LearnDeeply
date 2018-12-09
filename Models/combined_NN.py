#!/usr/bin/env python3.6.5
'''
Neural Network for combining multiple models
'''

import json
import keras
import numpy as np
import utils

# ==================== DATA PREPROCESSING
json_data = utils.preprocess.json_data

captions, likes = [], []
for user in json_data:
    for image in user['images']:
        if not utils.to_int(image['likes']) > 0:
            continue

        desc = image['description']
        desc = " ".join(filter(lambda x:x[0]!='#' and x[0]!='@', desc.split()))

model_data = {
    'inputs': np.array(desc),
    'labels': utils.log(np.array(likes))
}
utils.preprocess.add_model_data('captions', model_data)
# ====================


# ====================
''' Neural Network to process the combined system as well as adding in sentiment analysis of the captions '''
class Model:
    def __init__(self, inputs, labels,
        learning_rate=0.001, test_size=0.33,
        dropout=0.0, hidden_layers=0, hidden_sizes=[], epochs=10, batch_size=100):

        # split data into training and testing datasets
        self.train_inputs, self.train_labels, self.test_inputs, self.test_labels = utils.shuffle_data(inputs, labels, test_size)
        print(self.train_inputs)
        # derived parameters
        self.input_length = self.train_inputs.shape[0]

        # optional parameters that have defaulted values
        self.learning_rate = learning_rate
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
        merged_inputs = self.train_inputs

        # feed through at least 1 dense layer 
        inputs = keras.layers.Dense(units=100, kernel_initializer='random_normal', activation='relu')(merged_inputs)
        inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        # if number of hidden layers and their respective sizes are specified
        if self.hidden_layers:
            # iterate through hidden layer construction
            for layer in range(self.hidden_layers):
                inputs = keras.layers.Dense(units=self.hidden_sizes[layer], kernel_initializer='random_normal', activation='relu')(inputs)
                inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        output = keras.layers.Dense(units=1, kernel_initializer='random_normal')(inputs)

        # specify optimizer and initialize model for training
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        self.model = keras.models.Model(inputs=merged_inputs, outputs=output)

        # comiple keras computation graph
        self.model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])

    ''' Train constructed model '''
    def train_model(self, verbose=0):
        if not verbose:
            print('Training combined NN . . . ')

        self.model.fit(x=self.train_inputs, y=self.train_labels, 
            batch_size=self.batch_size, epochs=self.epochs, verbose=verbose)

        if not verbose:
            print(' . . . Finished training.')

    ''' Make predictions with constructed model '''
    def predict(self, inputs, verbose=0):
        return self.model.predict(x=inputs, verbose=verbose)

    ''' Test constructed model '''
    def test_model(self, tolerance=0.10, verbose=0, correct_only=0):
        predictions = self.predict(self.test_inputs)

        print('Testing with tolerance of +/- %.3f %% of correct value:' % (tolerance*100))
        correct = 0
        for prediction, target in zip(predictions, self.test_labels):
            prediction, target = np.exp(prediction), np.exp(target)
            mark = ''

            if np.abs(prediction-target) <= target*tolerance:
                correct += 1
                mark = 'O'

            if verbose:
                if correct_only:
                    if mark is 'O':
                        print('prediction: %20.3f  |  target: %20.3f  |  %s' % (prediction, target, mark))
  
                else:
                    print('prediction: %20.3f  |  target: %20.3f  |  %s' % (prediction, target, mark))

        print('Total number correct = %d / %d' % (correct, len(self.test_labels)), end='\n')

# ====================