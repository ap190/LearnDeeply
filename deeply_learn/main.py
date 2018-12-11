#!/usr/bin/env python3.6.5

'''
=========================

Main models file that calls on the other individual models.
Serves as a rendezvous point for all the individual models of the multi-modal model for
training predicting using all the modes altogether.

=========================
'''
import argparse

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

combined_nn_arg = 'combined_nn'
meta_nn_arg = 'meta_nn'
image_class_nn_arg = 'image_class_nn'
combined_basic_nn_arg = 'combined_basic_nn'

parser = argparse.ArgumentParser(description='Predict instagram virality')
parser.add_argument('-model', type=str, default=combined_nn_arg,
                    help='Specify a model to run.\n Options include: \n\tcombined_nn, \n\tmeta_nn')

# establish input and output data
meta_inputs = data.metadata['combined_meta']
imageclass_inputs = data.image_class['detections']
nima_inputs = data.nima['images']
labels = data.labels

# hyperparameters 
learning_rate, decay_rate = 0.001, 1e-05
epochs, batch_size = 5, 30
test_size = 0.33
validation_split = 0

# create data generators
inputs = (meta_inputs, imageclass_inputs, nima_inputs)
optimizer = keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)

def meta_nn():
    """
    Simple model that only trains on the user's metadata:
    ~1374 / 5985 within a 10% distance
    val_loss: 0.3539 - val_mean_absolute_error: 0.3539 - val_mean_squared_error: 0.2958
    """
    model_type = 1
    metaNN = metadata_nn.Graph(input_length=len(meta_inputs[0]), output_length=1, intermediate_layer=128,
        dropout=0.0, hidden_layers=5, hidden_sizes=[256, 256, 256, 256, 256])

    model = keras.Model(inputs=metaNN.inputs, outputs=metaNN.outputs)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse', 'mape'])

    training_generator, testing_generator = utils.create_data_generators(inputs, labels, train_batch_size=batch_size, test_size=test_size, model_type=model_type)
    # train model
    model.fit_generator(generator=training_generator, validation_data=testing_generator, use_multiprocessing=True, workers=3, epochs=30)
    
    # make predictions and test the model
    utils.keras_predict(testing_generator, model)


def image_class_nn():
    """
    Model that trains exclusively on the objects classified in the image using InceptionV3 trained on ImageNet weights
    380 / 5985
    val_loss: 1.0987 - val_mean_absolute_error: 1.0987 - val_mean_squared_error: 2.0790 - val_mean_absolute_percentage_error: 12.6747
    """
    model_type = 2
    imageNN = imageclass_nn.Graph(input_length=len(imageclass_inputs[0]), vocab_size=data.image_class['vocab_size'], output_length=1, embed_size=100,
        dropout=0.0, hidden_layers=5, hidden_sizes=[300, 400, 500, 400, 300])

    model = keras.Model(inputs=imageNN.inputs, outputs=imageNN.outputs)
    model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse', 'mape'])

    training_generator, testing_generator = utils.create_data_generators(inputs, labels, train_batch_size=batch_size, test_size=test_size, model_type=model_type)
    # train model
    model.fit_generator(generator=training_generator, validation_data=testing_generator, use_multiprocessing=True, workers=3, epochs=30)
    
    # make predictions and test the model
    utils.keras_predict(testing_generator, model)


def combined_basic_nn():
    """
    Basic combined model that integrates meta_nn and image_class_nn
    val_loss: 0.3592 - val_mean_absolute_error: 0.3592 - val_mean_squared_error: 0.2830
    val_loss: 0.3479 - val_mean_absolute_error: 0.3479 - val_mean_squared_error: 0.3137
    ~1315 / 5985
    """
    model_type = 3
    metaNN = metadata_nn.Graph(input_length=len(meta_inputs[0]), output_length=1, intermediate_layer=128,
        dropout=0.0, hidden_layers=5, hidden_sizes=[400, 400, 400, 256, 256])

    imageNN = imageclass_nn.Graph(input_length=len(imageclass_inputs[0]), vocab_size=data.image_class['vocab_size'], embed_size=100,
        dropout=0.0, hidden_layers=0, hidden_sizes=[300, 400, 500, 400, 300])

    combineNN = combined_nn.Graph(input_tensor=keras.layers.concatenate([metaNN.outputs, imageNN.outputs], axis=1), 
        intermediate_layer=250, dropout=0.0, hidden_layers=5, hidden_sizes=[200, 400, 800, 1600, 800, 400, 200])

    combined_model = keras.Model(inputs=[metaNN.inputs, imageNN.inputs], outputs=combineNN.outputs)
    combined_model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])

    training_generator, testing_generator = utils.create_data_generators(inputs, labels, train_batch_size=batch_size, test_size=test_size, model_type=model_type)
    # train model
    combined_model.fit_generator(generator=training_generator, validation_data=testing_generator, use_multiprocessing=True, workers=3, epochs=10)
    
    # make predictions and test the model
    utils.keras_predict(testing_generator, combined_model)



def combined_nn_model():
    model_type = 0
    # combine modules
    metaNN = metadata_nn.Graph(input_length=len(meta_inputs[0]), intermediate_layer=128,
        dropout=0.0, hidden_layers=5, hidden_sizes=[64, 128, 256, 512, 1024])

    imageNN = imageclass_nn.Graph(input_length=len(imageclass_inputs[0]), vocab_size=data.image_class['vocab_size'], embed_size=100,
        dropout=0.0, hidden_layers=5, hidden_sizes=[300, 400, 500, 400, 300])

    nima = nima_nn.Graph(dropout=0.75)

    combineNN = combined_nn.Graph(input_tensor=keras.layers.concatenate([metaNN.outputs, imageNN.outputs, nima.outputs], axis=1), 
        intermediate_layer=250, dropout=0.0, hidden_layers=7, hidden_sizes=[200, 400, 800, 1600, 800, 400, 200])

    combined_model = keras.Model(inputs=[metaNN.inputs, imageNN.inputs, nima.inputs], outputs=combineNN.outputs)

    combined_model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    
    training_generator, testing_generator = utils.create_data_generators(inputs, labels, train_batch_size=batch_size, test_size=test_size, model_type=model_type)

    # train model
    combined_model.fit_generator(generator=training_generator, validation_data=testing_generator, use_multiprocessing=True, workers=3, epochs=epochs)
    
    # make predictions and test the model
    utils.keras_predict(testing_generator, combined_model)


if __name__ == '__main__':
    args = parser.parse_args()
    model = args.model 
    if model is combined_nn_arg:
        combined_nn_model()
    elif model == meta_nn_arg:
        meta_nn()
    elif model == image_class_nn_arg:
        image_class_nn()
    elif model == combined_basic_nn_arg:
        combined_basic_nn()


