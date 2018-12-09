#!/usr/bin/env python3.6.5
'''
Neural Network that utilizes NIMA.
'''
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from keras import backend as K 
import utils

# ==================== DATA PREPROCESSING FOR IMAGE CNN (WILL AUTO RUN ON IMPORT)
json_data = utils.preprocess.json_data
image_data, likes = [], []
for user in json_data:
    for image in user['images']:
        num_likes = utils.to_int(image['likes'])
        if not num_likes > 0:
            continue

        image_data.append(utils.preprocess.data_path + image['picture'])
        likes.append(num_likes)

model_data = {
    'inputs': image_data,
    'labels': utils.log(likes)
}

# add processed data structure for this NN model into global pre-processed data class
utils.preprocess.add_model_data('nima', model_data)
# ====================
class Model:
    def __init__(self):
        # computation graph construction
        self.model_inputs, self.model_outputs, self.layers = self.construct_graph()

    ''' Construct computation graph with keras '''
    def construct_graph(self):
        image_size = 224
        base_model = MobileNet((image_size, image_size, 3), alpha=1, include_top=False, pooling='avg')
        for layer in base_model.layers:
            layer.trainable = False
        x = Dropout(0.75)(base_model.output)
        x = Dense(10, activation='softmax')(x)
        return base_model.input, x, base_model.layers




