#!/usr/bin/env python3.6.5
'''
Neural Network for processing and learning weights of the objects detected in images.
Object detections/classifications are the top 5 identified with the highest probability of
being correct (as determined with keras.InceptionResNetV2)
Embeddings as well as a vocab builder will be used for the classifications, since they are
returned as strings.
'''

import json
import keras
import numpy as np
import utils

# ==================== DATA PREPROCESSING FOR IMAGE CLASSIFICATION NN (WILL AUTO RUN ON IMPORT)
json_data = utils.preprocess.json_data

detections, probabilities, num_likes = [], [], []
for user in json_data:
    for image in user['images']:
        if not utils.to_int(image['likes']) > 0:
            continue

        detection_upmap = [utils.map_word_up(item[0], 7) for item in image['classification']]
        
        detections.append(detection_upmap)
        probabilities.append([item[2] for item in image['classification']])
        num_likes.append(utils.to_int(image['likes']))

e_vec, vocab_size, _ = utils.embed_vector(detections)
print(vocab_size)
model_data = {
    'inputs': np.array(e_vec),
    'labels': utils.log(np.array(num_likes)),
    'vocab_size': vocab_size,
    'probabilities': np.array(probabilities)
}

# add processed data structure for this NN model into global pre-processed data class
utils.preprocess.add_model_data('image_class', model_data)
# ====================


# ====================
''' Neural Network for learning how the objects in an image influences the number of likes on instagram it receives. '''
class Model:
    def __init__(self, inputs, labels, probabilities, vocab_size, 
        embed_size=100, learning_rate=0.001, test_size=0.33,
        dropout=0.0, hidden_layers=0, hidden_sizes=[], epochs=10, batch_size=100):

        # shuffle data indices 
        indices = np.arange(len(labels))
        np.random.shuffle(indices)
        split = np.int(len(labels) * (1-test_size))

        # required inputs to the model
        self.train_inputs = [inputs[i, :] for i in indices[:split]]
        self.train_labels = [labels[i] for i in indices[:split]]
        self.train_probabilities = [probabilities[i, :] for i in indices[:split]]

        self.test_inputs = [inputs[i, :] for i in indices[split:]]
        self.test_labels = [labels[i] for i in indices[split:]]

        self.test_probabilities = [probabilities[i, :] for i in indices[split:]]

        self.vocab_size = vocab_size

        # dervied parameters
        self.input_length = len(self.train_inputs[0])
        self.probabilities_length = len(self.train_probabilities[0])

        # optional parameters that have defaulted values
        self.embed_size = embed_size
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
        wordIDs = keras.layers.Input(shape=(self.input_length, ))
        probabilities = keras.layers.Input(shape=(self.probabilities_length, ))

        # encode inputs with embedding tensor
        E = keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embed_size, input_length=self.input_length)(wordIDs)
        inputs = keras.layers.Reshape((self.embed_size*self.input_length, ))(E)
        
        # adding in probabilities of classification from InceptionResNetV2 to model
        # inputs = keras.layers.concatenate([inputs, probabilities], axis=1)
        inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        # if number of hidden layers and their respective sizes are specified
        if self.hidden_layers:
            # iterate through hidden layer construction
            for layer in range(self.hidden_layers):
                inputs = keras.layers.Dense(units=self.hidden_sizes[layer], kernel_initializer='random_normal', activation='relu')(inputs)
                inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        # add in last dense layer to computation graph
        output = keras.layers.Dense(units=self.input_length, kernel_initializer='random_normal')(inputs)
        self.output = output

        # specify optimizer and initialize model for training
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)
        self.model = keras.models.Model(inputs=[wordIDs, probabilities], outputs=output)

        # compile keras computation graph 
        self.model.compile(loss='mae', optimizer=optimizer, metrics=["mae", "mse"])

    def get_output_layer(self):
        return self.output

    ''' Train constructed model '''
    def train_model(self, verbose=0):
        if not verbose:
            print('Training image classification embeddings NN . . . ')

        self.model.fit(x=[self.train_inputs, self.train_probabilities], y=self.train_labels, 
            batch_size=self.batch_size, epochs=self.epochs, verbose=verbose)

        if not verbose:
            print(' . . . Finished training.')

    ''' Make predictions with constructed model '''
    def predict(self, inputs, probabilities, verbose=0):
        return self.model.predict(x=[inputs, probabilities], verbose=verbose)

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
