import cv2
import numpy as np
import os
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

        # extract and compress images
        for image in jsonfile['posts']:
            if not image['isVideo']:
                res = urllib.request.urlopen(image['urlImage'])
                img = np.asarray(bytearray(res.read()), dtype='uint8')

                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = load_img(img, target_size=in_shape)
                img = np.expand_dims(img_to_array(img), axis=0)
                processed_img = preprocess_input(img)

                print(processed_img)
                stop

                image_data = {
                    'picture'    : processed_img,
                    'tags'       : image['tags'],
                    'mentions'   : image['mentions'],
                    'description': image['description'],
                    'date'       : image['date'],
                    'likes'      : image['numberLikes']
                }

                user_data['images'].append(image_data)

        data.append(user_data)
    file.close()

with open('../data.json', 'w') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=2)