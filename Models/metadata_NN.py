#!/usr/bin/env python3.6.5
'''
Neural Network for processing and learning weights of the the metadata of 
user's instagram post.
'''

import json
import math
import numpy as np
import keras
import utils
# from hashtag.popular_hashtags import *

# ==================== DATA PREPROCESSING FOR META DATA NN (WILL AUTO RUN ON IMPORT)
#hash_weights = popular_hashtags()
json_data = utils.preprocess.json_data

max_num_following, max_num_followers,      max_num_posts    = 0, 0, 0
max_num_tags,      max_description_length, max_num_mentions = 0, 0, 0
max_tag_weight,    max_num_likes,          max_average_likes= 0, 0, 0

for user in json_data:
    if not user['images']:
        continue

    max_num_following = max(max_num_following, utils.to_int(user['following']))
    max_num_followers = max(max_num_followers, utils.to_int(user['followers']))
    max_num_posts = max(max_num_posts, utils.to_int(user['posts']))
    max_average_likes = max(max_average_likes, utils.to_int(user['avg_likes']))

    for post in user['images']:
        max_num_tags = max(max_num_tags, len(post['tags']))
        max_description_length = max(max_description_length, len(post['description']))
        max_num_mentions = max(max_num_mentions, len(post['mentions']))

        max_num_likes = max(max_num_likes, utils.to_int(post['likes']))
        # tag_weight = 0
        # for tag in post['tags']:
        #     if tag[1:] in hash_weights:
        #         tag_weight += hash_weights[tag[1:]]
        # max_tag_weight = max(max_tag_weight, tag_weight)

metadata, labels = [], []
for user in json_data:
    if not user['images']:
        continue

    num_following, num_followers = utils.to_int(user['following']), utils.to_int(user['followers'])
    num_posts = utils.to_int(user['posts'])

    userdata, likes = [], []
    for post in user['images']:
        if not utils.to_int(post['likes']) > 0:
            continue

        # tag_weight = 0
        # for tag in post['tags']:
        #     if tag[1:] in hash_weights:
        #         tag_weight += hash_weights[tag[1:]]


        # using 24 hours for post hour
        # postinfo = [utils.to_int(user['following']) / max_num_following,
        #             utils.to_int(user['followers']) / max_num_followers,
        #             utils.to_int(user['posts']) / max_num_posts,
        #             len(post['tags']) / max_num_tags,
        #             len(post['description']) / max_description_length,
        #             len(post['mentions']) / max_num_mentions,
        #             post['weekday'] / 6,
        #             post['hour'] / 23,
        #             user['avg_likes'] / max_average_likes]
        #             # tag_weight / max_tag_weight]

        # using 6 sections for hour instead of 24
        # hour = post['hour']
        # hour_ind = 0
        # if hour >= 0 and hour < 8:
        #     hour_ind = 0
        # elif hour >= 8 and hour < 12:
        #     hour_ind = 1
        # elif hour >= 12 and hour < 14:
        #     hour_ind = 2
        # elif hour >= 14 and hour < 17:
        #     hour_ind = 3
        # elif hour >= 17 and hour < 20:
        #     hour_ind = 4
        # else:
        #     hour_ind = 5
        # postinfo = [utils.to_int(user['following']) / max_num_following,
        #             utils.to_int(user['followers']) / max_num_followers,
        #             utils.to_int(user['posts']) / max_num_posts,
        #             len(post['tags']) / max_num_tags,
        #             len(post['description']) / max_description_length,
        #             len(post['mentions']) / max_num_mentions,
        #             post['weekday'] / 6,
        #             hour_ind / 5,
        #             user['avg_likes']]
        #             #tag_weight / max_tag_weight]                

        # using the one_hot_weekday and one_hot_hour
        one_hot_weekday = [0]*7
        one_hot_weekday[post['weekday']] = 1
        one_hot_hour = [0]*6
        hour = post['hour']
        hour_ind = 0
        if hour >= 0 and hour < 8:
            hour_ind = 0
        elif hour >= 8 and hour < 12:
            hour_ind = 1
        elif hour >= 12 and hour < 14:
            hour_ind = 2
        elif hour >= 14 and hour < 17:
            hour_ind = 3
        elif hour >= 17 and hour < 20:
            hour_ind = 4
        else:
            hour_ind = 5
        one_hot_hour[hour_ind] = 1
        postinfo = [utils.to_int(user['following']) / max_num_following,
                    utils.to_int(user['followers']) / max_num_followers,
                    utils.to_int(user['posts']) / max_num_posts,
                    len(post['tags']) / max_num_tags,
                    len(post['description']) / max_description_length,
                    len(post['mentions']) / max_num_mentions,
                    user['avg_likes']] + one_hot_weekday + one_hot_hour
                    #tag_weight / max_tag_weight]

        likes.append(utils.to_int(post['likes']))
        userdata.append(postinfo)

    metadata += userdata
    labels += likes

model_data = {
    'inputs': np.array(metadata),
    'labels': utils.log(np.asarray(labels))
}
utils.preprocess.add_model_data('metadata', model_data)
# ====================


# ====================
''' Neural Network for learning the weights of the meta data of users and their posts' effect on likes recieved '''
class Model:
    def __init__(self, train_inputs, train_labels, test_inputs, test_labels, 
        learning_rate=0.001, dropout=0.0,
        hidden_layers=0, hidden_sizes=[], batch_size=30, epochs=10):

        # required parameters
        self.train_inputs = train_inputs
        self.train_labels = train_labels

        self.test_inputs = test_inputs
        self.test_labels = test_labels

        # derived parameters
        self.input_length = len(self.train_inputs[0])
        print(self.input_length)

        # optional parameters that have defaulted values
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.hidden_sizes = hidden_sizes
        self.epochs = epochs
        self.batch_size = batch_size

        # computation graph construction
        self.model_inputs, self.model_outputs = self.construct_graph()

    ''' Construct computation graph with keras '''
    def construct_graph(self):
        # instantiate input tensors
        metadata_inputs = keras.layers.Input(shape=(self.input_length, ))
        # inputs = keras.layers.BatchNormalization()(metadata_inputs)


        # always include at least 1 layer before output
        inputs = keras.layers.Dense(units=128, kernel_initializer='random_normal', activation='softplus', name='iam1')(metadata_inputs)
        print("here1")
        inputs = keras.layers.Dropout(rate=self.dropout)(inputs)
        print(inputs)

        # if number of hidden layers and their respective sizes are specified
        if self.hidden_layers:
            # iterate through hidden layer construction
            for layer in range(self.hidden_layers):
                inputs = keras.layers.Dense(units=self.hidden_sizes[layer], kernel_initializer='random_normal', activation='softplus')(inputs)
                inputs = keras.layers.Dropout(rate=self.dropout)(inputs)

        output = keras.layers.Dense(units=self.input_length, kernel_initializer='random_normal', activation='linear')(inputs)
        
        return metadata_inputs, output

    ''' Train constructed model '''
    def train_model(self, verbose=0):
        if not verbose:
            print('Training meta data NN . . . ')

        self.model.fit(x=self.train_inputs, y=self.train_labels, 
            batch_size=self.batch_size, epochs=self.epochs, verbose=verbose)

        if not verbose:
            print(' . . . Finished training.')

    ''' Make predictions with constructed model '''
    def predict(self, inputs, verbose=0):
        return self.model.predict(x=inputs, verbose=verbose)

    ''' Test constructed model '''
    def test_model(self, tolerance=0.10, verbose=0):
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
                print('prediction: %20.3f  |  target: %20.3f  |  %s' % (prediction, target, mark))

        print('Total number correct = %d / %d' % (correct, len(self.test_labels)), end='\n')

# ====================