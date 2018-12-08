#!/usr/bin/env python3.5
"""
	Adds the average number of likes to each user 
"""
import json
import numpy as np

seen_users = {}
with open('compiled_data.json') as f:
	data = json.load(f)
	for entry in data:
		username = entry['user']
		posts = entry['posts']
		followers = entry['followers']
		following = entry['following']

		# Unique key for user (naive approach)
		user_key =  username + str(posts) + str(followers) + str(following)
		if user_key in seen_users:
			seen_users[user_key][1].extend(entry['images'])
		else:

			seen_users[user_key] = (
				{
				'user': username,
				'posts': posts,
				'followers': followers,
				'following': following
				},
				entry['images'])


data_to_dump = []
for _, value in seen_users.items():
	user_blob = value[0]
	images = value[1]
	likes = []
	for image in images:
		likes.append(image['likes'])

	user_blob['avg_likes'] = np.mean(likes)
	user_blob['images'] = images

	data_to_dump.append(user_blob)

with open("compiled_data.json", 'w') as outfile:
    json.dump(data_to_dump, outfile, ensure_ascii=False, indent=2)


