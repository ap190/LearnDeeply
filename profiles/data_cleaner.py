import os
import json

filenames = [file for file in os.listdir() if file.endswith('.json')]

data = []
for filename in filenames:
    with open(filename) as file:
        jsonfile = json.load(file)

        user_data = {
            'user'     : jsonfile['alias'],
            'posts'    : jsonfile['numberPosts'],
            'followers': jsonfile['numberFollowers'],
            'following': jsonfile['numberFollowing'],
            'images'   : []
        }

        for image in jsonfile['posts']:
            if not image['isVideo']:
                image_data = {
                    'picture'    : image['urlImage'],
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