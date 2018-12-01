#!/usr/bin/env python3.5

import cv2
import datetime
import dateutil.parser as dateparser
import numpy as np
import os
import json
import random
import string
import sys
import urllib.request

from PIL import Image
from skimage import io

from keras.applications import InceptionResNetV2
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

# instantiate InceptionResNet model and associated parameters
model = InceptionResNetV2(weights='imagenet')
in_shape = [299, 299]

# obtain parent directory for JSON data files
fileprefix = sys.argv[1]

# import all JSON file names
filenames = [file for file in os.listdir(fileprefix) if file.endswith('.json')]

# iterate through json files and extract data
# each entry in the list will be a dictionary for each insta user we've collected
data, counter = [], 1

startIndex = int(sys.argv[2])
endIndex = int(sys.argv[3])

for filename in filenames[startIndex:endIndex]:
    filename = fileprefix + filename
    
    try:
        with open(filename) as file:
            # load json file
            jsonfile = json.load(file)

            # extact user specific data
            user_data = {
                'user'     : jsonfile['alias'],
                'posts'    : jsonfile['numberPosts'],
                'followers': jsonfile['numberFollowers'],
                'following': jsonfile['numberFollowing'],
                'images'   : []
            }

            # extract and compress images from the user's instagram page
            for image in jsonfile['posts']:
                if not image['isVideo']:
                    try:
                        # open image from URL and obtain compressed array representation for keras models
                        res = urllib.request.urlopen(image['urlImage'])

                        img = img_to_array(load_img(res, target_size=in_shape))
                        img = np.expand_dims(img_to_array(img), axis=0)
                        processed_img = preprocess_input(img)

                        # save images in image directory
                        savename =  'images/' + user_data['user'] + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=15)) + '.jpg'
                        io.imsave(savename, np.reshape(img, (in_shape[0], in_shape[1], 3)))

                        # conduct image classification with Inception ResNet model
                        predictions = model.predict(processed_img)
                        predictions = imagenet_utils.decode_predictions(predictions)
                        predictions = np.asarray(predictions)

                        # preprocess and restructure date information into more readable format
                        # weekdays ordered from start at monday = 0 and sunday = 6
                        date = dateparser.parse(image['date'])

                        # store metadata into dictionary
                        image_data = {
                            'picture'       : savename,
                            'classification': np.reshape(predictions, (predictions.shape[1], predictions.shape[2])).tolist(),
                            'tags'          : image['tags'],
                            'mentions'      : image['mentions'],
                            'description'   : image['description'],
                            'year'          : int(date.year),
                            'month'         : int(date.month),
                            'weekday'       : int(date.weekday()),
                            'hour'          : int(date.hour),
                            'likes'         : image['numberLikes']
                        }

                        # append image data dictionary into the instagram user's image posts
                        user_data['images'].append(image_data)
                    except:
                        continue


            # append instagram user's data and list of image posts into overall list
            data.append(user_data)

        # print user alias that was processing was just completed for
        print('({:d}) completed processing for user {:s}'.format(counter, user_data['user']))
        counter += 1
    except: 
        continue

# write a new data.json file with the produced dictionary object
now = datetime.datetime.now()
savename = 'data_' + str(now.month) + '-' + str(now.day) + '-' + str(now.hour) + str(now.minute) + '.json'
with open(savename, 'w') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=2)