import cv2
import dateutil.parser as dateparser
import numpy as np
import os
from PIL import Image
import json
import urllib

from keras.applications import InceptionResNetV2
from keras.applications import imagenet_utils
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img

# instantiate InceptionResNet model and associated parameters
model = InceptionResNetV2(weights='imagenet')
in_shape = [299, 299]

# import all JSON file names
filenames = [file for file in os.listdir() if file.endswith('.json')]

# iterate through json files and extract data
# each entry in the list will be a dictionary for each insta user we've collected
data = []
for filename in filenames:
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

                        # conduct image classification with Inception ResNet model
                        predictions = model.predict(processed_img)
                        predictions = imagenet_utils.decode_predictions(predictions)
                        predictions = np.delete(np.asarray(predictions[0]), np.s_[0], axis=1)

                        # preprocess and restructure date information into more readable format
                        date = dateparser.parse(image['date'])

                        # store metadata into dictionary
                        image_data = {
                            'picture'       : processed_img,
                            'classification': predictions,
                            'tags'          : image['tags'],
                            'mentions'      : image['mentions'],
                            'description'   : image['description'],
                            'weekday'       : date.weekday(),
                            'hour'          : date.hour,
                            'likes'         : image['numberLikes']
                        }

                        # append image data dictionary into the instagram user's image posts
                        user_data['images'].append(image_data)
                    except:
                        continue

            # append instagram user's data and list of image posts into overall list
            data.append(user_data)

        # close open file in preparation for next iteration
        file.close()
    except: 
        continue

# write a new data.json file with the produced dictionary object
with open('../data.json', 'w') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=2)