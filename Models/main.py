#!/usr/bin/env python3.7

'''
Main models file that calls on the other individual models.
Serves as a rendezvous point for all the individual models of the multi-modal model for
training predicting using all the modes altogether.
'''

# import utility functions and initialize global data preprocessing object
import utils
utils.initialize_globals()
data = utils.preprocess

# import packages, libraries, and python files
import imageclass_NN


# ==================== Main.py
def main():
    inputs, labels = data.image_class['inputs'], data.image_class['labels']
    probabilities, vocab_size = data.image_class['probabilities'], data.image_class['vocab_size']
    imageNN = imageclass_NN.Model(inputs=inputs, labels=labels, probabilities=probabilities, vocab_size=vocab_size,
        embed_size=50, hidden_layers=0, hidden_sizes=[250, 250, 250, 250])
    imageNN.train_model(verbose=1)
    imageNN.test_model(verbose=1)

if __name__ == '__main__':
    main()

# ====================