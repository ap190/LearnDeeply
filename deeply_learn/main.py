#!/usr/bin/env python3.6.5

'''
=========================

Main models file that calls on the other individual models.
Serves as a rendezvous point for all the individual models of the multi-modal model for
training predicting using all the modes altogether.

=========================
'''

# import utility functions and initialize global data preprocessing object
import utils
utils.initialize_globals('data/')
data = utils.preprocess

# import packages, libraries, and python files
import keras
import numpy as np
import tensorflow as tf
import sys

sys.path.append(sys.path[0]+'/modules/')
import imageclass_nn
import metadata_nn
import nima_nn
import combined_nn
import preprocess_data

def main():
    # establish input and output data
    meta_inputs = data.metadata['combined_meta']
    imageclass_inputs = data.image_class['detections']
    nima_inputs = data.nima['images']
    labels = data.labels

    # combine modules
    metaNN = metadata_nn.Graph(input_length=len(meta_inputs[0]), intermediate_layer=128,
        dropout=0.25, hidden_layers=5, hidden_sizes=[256, 256, 256, 256, 256])

    imageNN = imageclass_nn.Graph(input_length=len(imageclass_inputs[0]), vocab_size=data.image_class['vocab_size'], embed_size=100,
        dropout=0.25, hidden_layers=3, hidden_sizes=[400, 400, 400])

    nima = nima_nn.Graph(dropout=0.25)

    combineNN = combined_nn.Graph(input_tensor=keras.layers.concatenate([metaNN.outputs, imageNN.outputs, nima.outputs], axis=1), 
        intermediate_layer=250, dropout=0.25, hidden_layers=5, hidden_sizes=[200, 300, 400, 300, 200])

    learning_rate, decay_rate = 0.001, 1e-05
    epochs, batch_size = 5, 100
    test_size = 0.33
    validation_split = 0

    combined_model = keras.Model(inputs=[metaNN.inputs, imageNN.inputs, nima.inputs], outputs=combineNN.outputs)

    optimizer = keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)
    combined_model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])

    # create data generators
    inputs = (meta_inputs, imageclass_inputs, nima_inputs)
    training_generator, testing_generator = utils.create_data_generators(inputs, labels, train_batch_size=batch_size, test_size=test_size)
    
    # train model
    combined_model.fit_generator(generator=training_generator, use_multiprocessing=True, workers=3, epochs=epochs)
    
    # make predictions and test the model
    utils.keras_predict(testing_generator, combined_model)


if __name__ == '__main__':
    main()

