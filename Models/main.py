#!/usr/bin/env python3.6.5

'''
Main models file that calls on the other individual models.
Serves as a rendezvous point for all the individual models of the multi-modal model for
training predicting using all the modes altogether.
'''

# import utility functions and initialize global data preprocessing object
import utils
utils.initialize_globals('../data/')
data = utils.preprocess

# import packages, libraries, and python files
import keras
import numpy as np

import image_CNN
#import imageclass_NN
#import metadata_NN
#import combined_NN

# ==================== METADATA
def metadata():
    inputs, labels = data.metadata['inputs'], data.metadata['labels']
    metaNN = metadata_NN.Model(inputs=inputs, labels=labels)
    metaNN.train_model(verbose=1)
    metaNN.test_model(verbose=1)

# ====================

# ==================== IMAGE CLASSES
def imageclass():
    inputs, labels = data.image_class['inputs'], data.image_class['labels']

    probabilities, vocab_size = data.image_class['probabilities'], data.image_class['vocab_size']
    imageNN = imageclass_NN.Model(inputs=inputs, labels=labels, probabilities=probabilities, vocab_size=vocab_size,
        learning_rate=0.001, embed_size=100, dropout=0.15, hidden_layers=3, hidden_sizes=[100, 150, 100])
    imageNN.train_model(verbose=1)
    imageNN.test_model(verbose=1)

# ====================

# ==================== IMAGE CNN
def imagecnn():
    print('starting to extract the inputs and stuff now')
    inputs, labels = data.image['inputs'], data.image['labels']

    print('initializing model')
    imageNN = image_CNN.Model(inputs=inputs, labels=labels,
        learning_rate=0.001, dropout=0.15, hidden_layers=3, hidden_sizes=[100, 150, 100])
    
    print('training model right meow')
    imageNN.train_model(verbose=1)
    imageNN.test_model(verbose=1)

# ====================

# ==================== MAIN.PY
def main():
    # meta data NN
    inputs, labels = data.metadata['inputs'], data.metadata['labels']
    metaNN = metadata_NN.Model(inputs=inputs, labels=labels, 
        learning_rate=0.001, dropout=0, hidden_layers=3, hidden_sizes=[100, 150, 100])

    meta_output = metaNN.predict(inputs)

    # image classification NN
    inputs, labels = data.image_class['inputs'], data.image_class['labels']
    probabilities, vocab_size = data.image_class['probabilities'], data.image_class['vocab_size']
    imageNN = imageclass_NN.Model(inputs=inputs, labels=labels, probabilities=probabilities, vocab_size=vocab_size,
        learning_rate=0.001, embed_size=50, dropout=0, hidden_layers=3, hidden_sizes=[100, 150, 100])

    image_output = imageNN.predict(inputs, probabilities)

    # combined NN
    inputs = data.captions['inputs']
    combined_inputs = np.concatenate((meta_output, image_output, np.reshape(inputs, (inputs.shape[0], 1))), axis=1)
    combined_labels = labels
    combinedNN = combined_NN.Model(inputs=combined_inputs, labels=combined_labels, 
        learning_rate=0.001, dropout=0.15, hidden_layers=5, hidden_sizes=[150, 250, 400, 250, 150], epochs=5)
    combinedNN.train_model(verbose=1)
    combinedNN.test_model(verbose=1, correct_only=1)

# ==================== 

if __name__ == '__main__':
    imagecnn()


