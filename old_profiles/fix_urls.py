#!/usr/bin/env python3
"""
 This code is used to regenerate the images url with instagram oembed 
 and update our dataset to have updated image urls. (functional images)
"""
import json
import requests
from pprint import pprint
import os

api_endpoint = "http://api.instagram.com/oembed"

def regenerateUrls(filepath):
	data = {}
	rewrite_file_path = "../profiles/" + filepath
	try: 
		with open(filepath)  as file:
			data = json.load(file)
			posts = data["posts"]

			total_num_posts = 0
			for post in posts:
				imageUrl = {"url" : post["url"]}
				request = requests.get(url = api_endpoint, params = imageUrl)
				try:
					result = request.json()
					updated_image_url = result["thumbnail_url"]
					# replace urls
					post["urlImage"] = updated_image_url
				except:
					# Remove this post from the dataset
					posts.remove(post)
	except:
		print("Removing file:")
		print(filepath)
		try:
			os.remove(rewrite_file_path)
			return 0
		except:
			return 0


	# rewrite data
	out = json.dumps(data, ensure_ascii=False, indent=2)
	try:
		with open(rewrite_file_path, 'w', encoding='utf8') as file:
			file.write(out)
			total_num_posts += len(posts)
	except IOError:
		print("Removing file:")
		print(filepath)
		os.remove(rewrite_file_path)



	return total_num_posts


total_num_posts = 0
for root, dirs, files in os.walk("."):  
    for file in files:
        num_posts = regenerateUrls(file)
        total_num_posts += num_posts
print(total_num_posts)
