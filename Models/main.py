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

#import image_CNN
import imageclass_NN
import metadata_NN
import combined_NN

# ==================== TEST
def test():
    image_classifications = keras.layers.Input(shape=(5, ))
    imageNN_layer1 = keras.layers.Dense(units=100)(image_classifications)
    imageNN_layer2 = keras.layers.Dense(units=5)(imageNN_layer1)

# ====================


# ==================== METADATA
def metadata():
    inputs, labels = data.metadata['inputs'], data.metadata['labels']
    metaNN = metadata_NN.Model(inputs=inputs, labels=labels, learning_rate=0.001, dropout=0, hidden_layers=3, hidden_sizes=[256, 256, 256])
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
    # shuffle indices of the data
    indices = utils.shuffle_indices(len(data.metadata['labels']))
    test_size = 0.33
    split = np.int(len(indices) * (1-test_size))

    # split meta data into training and testing
    meta_inputs, meta_labels = data.metadata['inputs'], data.metadata['labels']
    
    meta_train_inputs = [meta_inputs[i] for i in indices[:split]]
    meta_train_labels = [meta_labels[i] for i in indices[:split]]
    
    meta_test_inputs = [meta_inputs[i] for i in indices[split:]]
    meta_test_labels = [meta_labels[i] for i in indices[split:]]

    # split image classification data into training and testing
    imageclass_inputs, imageclass_labels, imageclass_probs = data.image_class['inputs'], data.image_class['labels'], data.image_class['probabilities']
    imageclass_vocab_size = data.image_class['vocab_size']

    imageclass_train_inputs = [imageclass_inputs[i] for i in indices[:split]]
    imageclass_train_labels = [imageclass_labels[i] for i in indices[:split]]
    imageclass_train_probabilities = [imageclass_probs[i] for i in indices[:split]]
    
    imageclass_test_inputs = [imageclass_inputs[i] for i in indices[split:]]
    imageclass_test_labels = [imageclass_labels[i] for i in indices[split:]]
    imageclass_test_probabilities = [imageclass_probs[i] for i in indices[split:]]

    # Metadata Neural Network
    metaNN = metadata_NN.Model(train_inputs=meta_train_inputs, train_labels=meta_test_labels, 
        test_inputs=meta_test_inputs, test_labels=meta_test_labels,
        learning_rate=0.001, dropout=0.0,
        hidden_layers=3, hidden_sizes=[400, 200, 400])

    # Image Classification Neural Network
    imageNN = imageclass_NN.Model(train_inputs=imageclass_train_inputs, train_labels=imageclass_train_labels, train_probabilities=imageclass_train_probabilities,
        test_inputs=imageclass_test_inputs, test_labels=imageclass_test_labels, test_probabilities=imageclass_test_probabilities,
        learning_rate=0.001, embed_size=100, vocab_size=imageclass_vocab_size, dropout=0.0, hidden_layers=3, hidden_sizes=[400, 200, 400])

    combined_inputs = keras.layers.concatenate([metaNN.model_inputs, imageNN.model_inputs])

    # combined NN
    # inputs = data.captions['inputs']
    # combined_inputs = np.concatenate((meta_output, image_output, np.reshape(inputs, (inputs.shape[0], 1))), axis=1)

    combinedNN = combined_NN.Model(train_inputs=combined_inputs, train_labels=meta_train_labels, 
        learning_rate=0.001, dropout=0.0, 
        hidden_layers=5, hidden_sizes=[150, 250, 400, 250, 150], epochs=5)
    combined_model = keras.Model(inputs=[metaNN.model_inputs, imageNN.model_inputs], outputs=combinedNN.model_outputs)
    adam = keras.optimizers.Adam(lr=0.001, epsilon=None, decay=1e-5, amsgrad=False)
    combined_model.compile(loss='mae', optimizer=adam, metrics=['mae', 'mse'])
    combined_model.fit(x=[meta_train_inputs, imageclass_train_inputs], y=meta_train_labels,
        epochs=50, batch_size=30, verbose=1, validation_split=0.0)

    predictions = combined_model.predict([meta_test_inputs, imageclass_test_inputs])

    # combinedNN.test_model(verbose=1, correct_only=0)
    correct, tolerance = 0, 0.10 
    for prediction, target in zip(predictions, meta_test_labels):
        prediction, target = np.exp(prediction), np.exp(target)
        mark = ''

        if np.abs(prediction-target) <= target*tolerance:
            correct += 1
            mark = 'O'

        print('prediction: %20.3f  |  target: %20.3f  |  %s' % (prediction, target, mark))

    print('Total number correct = %d / %d' % (correct, len(meta_test_inputs)), end='\n')
# ==================== 

if __name__ == '__main__':
    main()
