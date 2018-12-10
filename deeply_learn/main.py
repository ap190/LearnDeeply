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
import sys

sys.path.append(sys.path[0]+'/modules/')
import imageclass_nn
import metadata_nn
import nima_nn
import combined_nn
import preprocess_data

def main():
    print('splitting data . . . ')
    # shuffle data indices and establish test/train data split
    indices = utils.shuffle_indices(len(data.labels))
    test_size = 0.33
    spl = np.int(len(indices) * (1-test_size))

    # split data into training and testing sets
    print(' . . . splitting image classification data')
    imageclass_inputs = data.image_class['detections']
    imageclass_train = [imageclass_inputs[i] for i in indices[:spl]]
    imageclass_test  = [imageclass_inputs[i] for i in indices[spl:]]
    print(' . . . splitting meta data')
    metadata_inputs = data.metadata['combined_meta']
    metadata_train = [metadata_inputs[i] for i in indices[:spl]]
    metadata_test  = [metadata_inputs[i] for i in indices[spl:]]
    print(' . . . splitting nima data')
    nima_inputs = data.nima['images']
    nima_train = [nima_inputs[i] for i in indices[:spl]]
    nima_test = [nima_inputs[i] for i in indices[spl:]]
    print(' . . . splitting labels data')
    labels = data.labels
    labels_train = [labels[i] for i in indices[:spl]]
    labels_test  = [labels[i] for i in indices[spl:]]

    print('data splitting finished')
    print('starting modules . . . ')

    # combine modules
    metaNN = metadata_nn.Graph(input_length=len(metadata_inputs[0]), intermediate_layer=128,
        dropout=0.0, hidden_layers=5, hidden_sizes=[256, 256, 256, 256, 256])

    imageNN = imageclass_nn.Graph(input_length=len(imageclass_inputs[0]), vocab_size=data.image_class['vocab_size'], embed_size=100,
        dropout=0.0, hidden_layers=3, hidden_sizes=[400, 400, 400])

    nima = nima_nn.Graph(input_shape=len(nima_inputs[0]), dropout=0.25)

    combineNN = combined_nn.Graph(input_tensor=keras.layers.concatenate([metaNN.inputs, imageNN.inputs, nima.inputs]), 
        intermediate_layer=250, dropout=0.0, hidden_layers=5, hidden_sizes=[200, 300, 400, 300, 200])

    learning_rate, decay_rate = 0.001, 1e-05
    epochs, batch_size = 25, 10
    validation_split = 0

    combined_model = keras.Model(inputs=[metaNN.inputs, imageNN.inputs, nima.inputs], outputs=combineNN.outputs)
    nima.layers.load_weights('mobilenet_weights.h5')

    optimizer = keras.optimizers.Adam(lr=learning_rate, decay=decay_rate)
    combined_model.compile(loss='mae', optimizer=optimizer, metrics=['mae', 'mse'])
    combined_model.fit(x=[metadata_train, imageclass_train], y=labels_train,
        epochs=epochs, batch_size=batch_size, verbose=1, validation_split=validation_split)

    utils.keras_predict(combined_model, [metadata_test, imageclass_test], labels_test)


if __name__ == '__main__':
    main()


