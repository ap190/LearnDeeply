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

import imageclass_NN
import metadata_NN
import combined_NN

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

# ==================== MAIN.PY
def main():
    inputs, labels = data.metadata['inputs'], data.metadata['labels']
    metaNN = metadata_NN.Model(inputs=inputs, labels=labels, 
        learning_rate=0.001, dropout=0, hidden_layers=3, hidden_sizes=[100, 150, 100])

    print(metaNN.model.get_weights())

    meta_output = metaNN.get_output_layer()

    inputs, labels = data.image_class['inputs'], data.image_class['labels']
    probabilities, vocab_size = data.image_class['probabilities'], data.image_class['vocab_size']
    imageNN = imageclass_NN.Model(inputs=inputs, labels=labels, probabilities=probabilities, vocab_size=vocab_size,
        learning_rate=0.001, embed_size=50, dropout=0, hidden_layers=3, hidden_sizes=[100, 150, 100])

    image_output = imageNN.get_output_layer()

    combined_inputs = keras.layers.concatenate([meta_output, image_output])
    combined_labels = labels
    combinedNN = combined_NN.Model(inputs=combined_inputs, labels=combined_labels, 
        learning_rate=0.001, dropout=0.15, hidden_layers=5, hidden_sizes=[150, 250, 400, 250, 150], epochs=5)
    combinedNN.train_model(verbose=1)
    print(metaNN.model.get_weights())

    # combinedNN.test_model(verbose=1, correct_only=0)

# ==================== 

if __name__ == '__main__':
    main()
