#!/usr/bin/env python3

import json
import requests
import os
import dateutil.parser as dateparser
import re

api_endpoint = "http://api.instagram.com/oembed"


data, counter = [], 1
def cleanData(filepath):
    try:
        with open(filepath) as file:
            jsonfile = json.load(file)

            # extact user specific data
            user_data = {
                'user'     : jsonfile['username'],
                'posts'    : jsonfile['num_of_posts'],
                'followers': jsonfile['followers'],
                'following': jsonfile['following'],
                'images'   : []
            }
            cc = 1
            for image in jsonfile['posts']:
                date = dateparser.parse(image['date'])
                # Clean old posts because we don't want to repeat data
                if int(date.year) <= 2017 and int(date.month) <= 5:
                    continue
                # Fix null urls for images
                imageUrl = image['img']
                need_to_pull = False
                if image['img'] is None:
                    need_to_pull = True
                    imageUrl = {"url" : image["url"]}
                    request = requests.get(url = api_endpoint, params = imageUrl)
                    try:
                        result = request.json()
                        updated_image_url = result["thumbnail_url"]
                        # replace urls
                        imageUrl = updated_image_url
                    except:
                        # Remove this post from the dataset
                        print("inner loop broken")
                        continue

                caption = image['caption']
                tags = image['tags']
                mentions = image['mentions']
                if image['caption'] is "":
                    comments = image['comments']['list']
                    print(comments)
                    if len(comments) > 0:
                        caption = comments[0]['comment']
                        cleaned_cap = remove_emoji(caption)
                        split_cap = cleaned_cap.split()
                        tags = re.findall(r"#(\w+)", split_cap)
                        mentions = re.findall(r"@(\w+)", split_cap)

                # store metadata into dictionary
                image_data = { 
                    'url_image'     : imageUrl,
                    'url'           : image['url'], # backup url 
                    'tags'          : tags,
                    'mentions'      : mentions,
                    'description'   : caption,
                    'month'         : date.month,
                    'weekday'       : date.weekday(),
                    'hour'          : date.hour,
                    'likes'         : image['likes']
                }
                # append image data dictionary into the instagram user's image posts
                user_data['images'].append(image_data)

            data.append(user_data)

            # print user alias that was processing was just completed for
            print('({:d}) completed processing for user {:s}'.format(counter, user_data['user']))
            return len(user_data['images'])
    except:
        return 0


def remove_emoji(string):
    print("here")
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)



total_num_posts = 0
for root, dirs, files in os.walk("."):  
  print('last user: {:s}'.format(files[-1])) 
  for file in files:
      if ".json" in file: 
          total_num_posts += cleanData(file)
  print(total_num_posts)

# write a new data.json file with the produced dictionary object
with open('../new_data.json', 'w') as outfile:
    json.dump(data, outfile, ensure_ascii=False, indent=2)
